"""Удалённые агенты (отдельные контейнеры) и гости: шлюз пересылает им апдейты по HTTP."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SetMyCommands, TelegramMethod
from aiogram.types import BotCommandScopeChat, BotCommandScopeDefault, Update, User

from lab_manager.gateway import SECRET_HEADER, Gateway, RemoteAgent
from tests.telegram.fake_telegram import (
    BOT_TOKEN,
    OWNER_ID,
    STRANGER_ID,
    FakeTelegram,
    callback_update,
    message_update,
)
from tests.telegram.test_gateway import DummyAgent

GUEST = User(id=777, is_bot=False, first_name="Гостья")
STRANGER = User(id=STRANGER_ID, is_bot=False, first_name="Stranger")
SECRET = "shared-secret"  # noqa: S105 — тестовый
LABEL = "🔮 Телепатия"


class RemoteService:
    """Контейнер удалённого агента: принимает POST и запоминает апдейты."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.status != 200:
            return httpx.Response(self.status)
        self.headers.append(request.headers)
        self.updates.append(json.loads(request.content))
        return httpx.Response(200, json={})

    def texts(self) -> list[str | None]:
        return [
            (u.get("message") or {}).get("text") or (u.get("callback_query") or {}).get("data")
            for u in self.updates
        ]


@pytest.fixture
def remote() -> RemoteService:
    return RemoteService()


@pytest.fixture
async def agent(remote: RemoteService) -> AsyncIterator[RemoteAgent]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(remote.handler))
    agent = RemoteAgent(
        name="telepathy",
        title="Телепатия",
        description="Совпадения мыслей пары",
        url="http://telepathy:9010",
        secret=SECRET,
        commands=[("pair", "Код для партнёра"), ("stats", "Статистика"), ("help", "Справка")],
        menu=[LABEL],
        client=client,
    )
    yield agent
    await agent.aclose()


@pytest.fixture
def session() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture
def bot(session: FakeTelegram) -> Bot:
    return Bot(BOT_TOKEN, session=session)


def make(local: DummyAgent, agent: RemoteAgent) -> Gateway:
    return Gateway(
        [local, agent], owner_id=OWNER_ID, guests={GUEST.id: "telepathy"}, burst_window=0.01
    )


def sent_texts(session: FakeTelegram) -> list[str]:
    return session.texts()


async def test_forward_posts_update_json_with_secret(
    agent: RemoteAgent, remote: RemoteService, bot: Bot
) -> None:
    update = message_update("хочу пиццу", user=GUEST)

    assert await agent.forward(update) is True

    [headers] = remote.headers
    assert headers[SECRET_HEADER] == SECRET
    assert headers["content-type"] == "application/json"
    [body] = remote.updates
    assert body["message"]["from"]["id"] == GUEST.id  # имена полей как в Bot API
    assert Update.model_validate(body).message.text == "хочу пиццу"


async def test_forward_reports_failure(agent: RemoteAgent, remote: RemoteService) -> None:
    remote.status = 502
    assert await agent.forward(message_update("x", user=GUEST)) is False


async def test_guest_reaches_only_their_remote_agent(
    agent: RemoteAgent, remote: RemoteService, bot: Bot, session: FakeTelegram
) -> None:
    local = DummyAgent("lab")
    gateway = make(local, agent)
    dp = gateway.dispatcher()

    for text in ("/start", "/help", "/lab_ping", "/agent lab", "мысль", LABEL):
        await dp.feed_update(bot, message_update(text, user=GUEST))
    await dp.feed_update(bot, callback_update("lab:yes", user=GUEST))
    await gateway.drain()

    assert remote.texts() == [
        "/start",
        "/help",
        "/lab_ping",
        "/agent lab",
        "мысль",
        LABEL,
        "lab:yes",
    ]
    assert local.bursts == []
    assert local.callbacks == []
    assert not [t for t in sent_texts(session) if "pong" in t]


async def test_stranger_is_not_forwarded(
    agent: RemoteAgent, remote: RemoteService, bot: Bot, session: FakeTelegram
) -> None:
    dp = make(DummyAgent("lab"), agent).dispatcher()

    await dp.feed_update(bot, message_update("/pair", user=STRANGER))
    await dp.feed_update(bot, message_update("мысль", user=STRANGER))

    assert remote.updates == []
    assert session.requests == []


async def test_owner_remote_commands_go_remote_local_stay_local(
    agent: RemoteAgent, remote: RemoteService, bot: Bot, session: FakeTelegram
) -> None:
    local = DummyAgent("lab")
    gateway = make(local, agent)
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update("/pair"))
    await dp.feed_update(bot, message_update("/stats@agents_test_bot"))
    await dp.feed_update(bot, message_update("/lab_ping"))
    await dp.feed_update(bot, message_update("/help"))  # команда шлюза, а не агента
    await dp.feed_update(bot, message_update("просто текст"))
    await gateway.drain()

    assert remote.texts() == ["/pair", "/stats@agents_test_bot"]
    assert "pong from lab" in sent_texts(session)
    assert "Телепатия" in sent_texts(session)[-1]  # справка шлюза перечисляет агентов
    assert [b.messages[0].text for b in local.bursts] == ["просто текст"]


async def test_owner_switches_with_menu_buttons(
    agent: RemoteAgent, remote: RemoteService, bot: Bot, session: FakeTelegram
) -> None:
    local = DummyAgent("lab")
    local.menu = lambda: ["📚 Лабы"]  # type: ignore[attr-defined]
    gateway = make(local, agent)
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update(LABEL))
    assert "Телепатия" in session.last_text()
    await dp.feed_update(bot, message_update("хочу пиццу"))
    await dp.feed_update(bot, callback_update("telepathy:x"))
    await dp.feed_update(bot, message_update("/lab_ping"))  # команда лаб — всё равно лабам
    await dp.feed_update(bot, message_update("📚 Лабы"))
    await dp.feed_update(bot, message_update("про лабу"))
    await gateway.drain()

    assert remote.texts() == ["хочу пиццу", "telepathy:x"]
    assert [m.text for b in local.bursts for m in b.messages] == ["📚 Лабы", "про лабу"]


async def test_agent_command_switches_to_remote(
    agent: RemoteAgent, remote: RemoteService, bot: Bot
) -> None:
    gateway = make(DummyAgent("lab"), agent)
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update("/agent telepathy"))
    await dp.feed_update(bot, message_update("мысль"))
    await dp.feed_update(bot, message_update("/unknown_cmd"))

    assert remote.texts() == ["мысль", "/unknown_cmd"]


async def test_unavailable_remote_agent_is_reported(
    agent: RemoteAgent, remote: RemoteService, bot: Bot, session: FakeTelegram
) -> None:
    remote.status = 503
    dp = make(DummyAgent("lab"), agent).dispatcher()

    await dp.feed_update(bot, message_update("мысль", user=GUEST))

    assert "недоступ" in session.last_text()
    assert session.sent()[-1].chat_id == GUEST.id


async def test_set_commands_per_user(agent: RemoteAgent, bot: Bot, session: FakeTelegram) -> None:
    gateway = make(DummyAgent("lab"), agent)

    await gateway.set_commands(bot)

    requests = [r for r in session.requests if isinstance(r, SetMyCommands)]
    by_scope = {
        type(r.scope).__name__ if r.scope else "default": [c.command for c in r.commands]
        for r in requests
    }
    owner = by_scope["default"]
    assert owner[:3] == ["start", "help", "agent"]
    assert {"lab_ping", "pair", "stats"} <= set(owner)
    assert len(owner) == len(set(owner))  # /help агента не дублирует /help шлюза
    [guest] = [r for r in requests if isinstance(r.scope, BotCommandScopeChat)]
    assert guest.scope.chat_id == GUEST.id
    assert [c.command for c in guest.commands] == ["start", "pair", "stats", "help"]
    assert not any(isinstance(r.scope, BotCommandScopeDefault) for r in requests)


class ChatNotFoundTelegram(FakeTelegram):
    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[Any],
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> Any:  # noqa: ANN401
        if isinstance(method, SetMyCommands) and isinstance(method.scope, BotCommandScopeChat):
            self.requests.append(method)
            if len([r for r in self.requests if isinstance(r, SetMyCommands)]) <= 2:
                raise TelegramBadRequest(method=method, message="chat not found")
            return True
        return await super().make_request(bot, method, timeout)


async def test_guest_commands_retried_on_first_contact(agent: RemoteAgent) -> None:
    session = ChatNotFoundTelegram()
    bot = Bot(BOT_TOKEN, session=session)
    gateway = make(DummyAgent("lab"), agent)
    dp = gateway.dispatcher()

    await gateway.set_commands(bot)  # гостья ещё не писала боту — Telegram не знает чат
    await dp.feed_update(bot, message_update("/start", user=GUEST))
    await gateway.drain()
    await dp.feed_update(bot, message_update("ещё", user=GUEST))
    await gateway.drain()

    guest_scopes = [
        r
        for r in session.requests
        if isinstance(r, SetMyCommands) and isinstance(r.scope, BotCommandScopeChat)
    ]
    assert len(guest_scopes) == 2  # при старте (ошибка) и один раз при первом контакте


def test_guest_must_map_to_remote_agent(agent: RemoteAgent) -> None:
    with pytest.raises(ValueError, match="удалённ"):
        Gateway([DummyAgent("lab"), agent], owner_id=OWNER_ID, guests={GUEST.id: "lab"})
    with pytest.raises(ValueError, match="нет агента"):
        Gateway([DummyAgent("lab"), agent], owner_id=OWNER_ID, guests={GUEST.id: "nope"})
    with pytest.raises(ValueError, match="владел"):
        Gateway([DummyAgent("lab"), agent], owner_id=OWNER_ID, guests={OWNER_ID: "telepathy"})
