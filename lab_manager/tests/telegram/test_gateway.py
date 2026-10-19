"""Шлюз: доступ только владельцу, маршрутизация к агентам, сборка «пачек» сообщений."""

from collections.abc import AsyncIterator

import pytest
from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.methods import SetMyCommands
from aiogram.types import CallbackQuery, Message, User

from lab_manager.gateway import Burst, Gateway
from tests.telegram.fake_telegram import (
    BOT_TOKEN,
    STRANGER_ID,
    FakeTelegram,
    callback_update,
    message_update,
)

STRANGER = User(id=STRANGER_ID, is_bot=False, first_name="Stranger")


class DummyAgent:
    def __init__(self, name: str = "echo", *, fail: bool = False) -> None:
        self.name = name
        self.title = f"Агент {name}"
        self.description = f"Тестовый агент {name}"
        self.bursts: list[Burst] = []
        self.fail = fail
        self.callbacks: list[str] = []

    def commands(self) -> list[tuple[str, str]]:
        return [(f"{self.name}_ping", "Проверка")]

    def router(self) -> Router:
        router = Router(name=self.name)

        @router.message(Command(f"{self.name}_ping"))
        async def ping(message: Message) -> None:
            await message.answer(f"pong from {self.name}")

        @router.callback_query(lambda query: (query.data or "").startswith(f"{self.name}:"))
        async def callback(query: CallbackQuery) -> None:
            self.callbacks.append(query.data or "")
            await query.answer()

        return router

    async def on_burst(self, bot: Bot, burst: Burst) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.bursts.append(burst)


@pytest.fixture
def session() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture
async def bot(session: FakeTelegram) -> AsyncIterator[Bot]:
    bot = Bot(BOT_TOKEN, session=session)
    yield bot


def make_gateway(*agents: DummyAgent) -> Gateway:
    from tests.telegram.fake_telegram import OWNER_ID

    return Gateway(list(agents), owner_id=OWNER_ID, burst_window=0.01)


async def test_stranger_is_ignored(bot: Bot, session: FakeTelegram) -> None:
    agent = DummyAgent()
    gateway = make_gateway(agent)
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update("/start", user=STRANGER))
    await dp.feed_update(bot, message_update("привет", user=STRANGER))
    await dp.feed_update(bot, callback_update("echo:x", user=STRANGER))
    await gateway.drain()

    assert session.requests == []
    assert agent.bursts == []
    assert agent.callbacks == []


async def test_start_lists_agents_and_commands(bot: Bot, session: FakeTelegram) -> None:
    dp = make_gateway(DummyAgent()).dispatcher()

    await dp.feed_update(bot, message_update("/start"))

    text = session.last_text()
    assert "Агент echo" in text
    assert "/echo_ping" in text


async def test_agent_command_is_handled_immediately(bot: Bot, session: FakeTelegram) -> None:
    agent = DummyAgent()
    gateway = make_gateway(agent)

    await gateway.dispatcher().feed_update(bot, message_update("/echo_ping"))

    assert session.last_text() == "pong from echo"
    await gateway.drain()
    assert agent.bursts == []


async def test_agent_callbacks_are_routed(bot: Bot) -> None:
    agent = DummyAgent()

    await make_gateway(agent).dispatcher().feed_update(bot, callback_update("echo:yes"))

    assert agent.callbacks == ["echo:yes"]


async def test_messages_close_in_time_form_one_burst(bot: Bot) -> None:
    agent = DummyAgent()
    gateway = make_gateway(agent)
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update("Нейронные сети, ЛР 4"))
    await dp.feed_update(bot, message_update("пост", forwarded_from_channel="Кафедра"))
    await gateway.drain()

    [burst] = agent.bursts
    assert [m.text for m in burst.messages] == ["Нейронные сети, ЛР 4", "пост"]
    assert burst.messages[1].forward_origin is not None


async def test_separate_bursts(bot: Bot) -> None:
    agent = DummyAgent()
    gateway = make_gateway(agent)
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update("один"))
    await gateway.drain()
    await dp.feed_update(bot, message_update("два"))
    await gateway.drain()

    assert [[m.text for m in b.messages] for b in agent.bursts] == [["один"], ["два"]]


async def test_switch_active_agent(bot: Bot, session: FakeTelegram) -> None:
    first, second = DummyAgent("lab"), DummyAgent("notes")
    gateway = make_gateway(first, second)
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update("в первый"))
    await gateway.drain()
    await dp.feed_update(bot, message_update("/agent notes"))
    await dp.feed_update(bot, message_update("во второй"))
    await gateway.drain()

    assert [b.messages[0].text for b in first.bursts] == ["в первый"]
    assert [b.messages[0].text for b in second.bursts] == ["во второй"]
    assert "notes" in session.texts()[0]


async def test_agent_command_lists_and_rejects_unknown(bot: Bot, session: FakeTelegram) -> None:
    dp = make_gateway(DummyAgent("lab"), DummyAgent("notes")).dispatcher()

    await dp.feed_update(bot, message_update("/agent"))
    listing = session.last_text()
    await dp.feed_update(bot, message_update("/agent nope"))

    assert "lab" in listing and "notes" in listing
    assert "nope" in session.last_text()


async def test_failing_agent_reports_error_and_keeps_working(
    bot: Bot, session: FakeTelegram
) -> None:
    gateway = make_gateway(DummyAgent(fail=True))
    dp = gateway.dispatcher()

    await dp.feed_update(bot, message_update("сломай"))
    await gateway.drain()

    assert "ошибка" in session.last_text().lower()
    assert "boom" not in session.last_text()  # детали — в лог, не пользователю


async def test_set_commands_aggregates_agents(bot: Bot, session: FakeTelegram) -> None:
    gateway = make_gateway(DummyAgent("lab"), DummyAgent("notes"))

    await gateway.set_commands(bot)

    [request] = [r for r in session.requests if isinstance(r, SetMyCommands)]
    names = [command.command for command in request.commands]
    assert names[:3] == ["start", "help", "agent"]
    assert {"lab_ping", "notes_ping"} <= set(names)


@pytest.mark.parametrize("bad", ["Lab", "lab-1", "", "лаба"])
def test_agent_names_are_validated(bad: str) -> None:
    with pytest.raises(ValueError):
        make_gateway(DummyAgent(bad))


def test_duplicate_agent_names_rejected() -> None:
    with pytest.raises(ValueError):
        make_gateway(DummyAgent("lab"), DummyAgent("lab"))
