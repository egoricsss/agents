"""Telegram-шлюз для нескольких агентов владельца.

Шлюз ничего не знает о предметной области. Агенты бывают двух видов:
- локальные (`BotAgent`) — живут в этом процессе со своими роутерами;
- удалённые (`RemoteAgent`) — отдельные контейнеры: шлюз пересылает им сырые апдейты по HTTP
  в формате вебхука Telegram, а отвечают они сами через Bot API тем же токеном
  (один токен = один процесс `getUpdates`, но отправлять может кто угодно).

Доступ:
- владельцу доступны все агенты; гостю — ровно один удалённый агент, и шлюз для него
  прозрачен: все его апдейты, включая /start, уходят этому агенту;
- остальные апдейты молча отбрасываются.

Маршрутизация для владельца:
- команды и inline-кнопки агентов — сразу их агенту (`callback_data` начинается с `<имя>:`);
- кнопка меню агента переключает активного агента чата;
- остальные сообщения — активному агенту; для локального они собираются в «пачки» (`Burst`):
  всё, что пришло почти одновременно — подпись + пересланный пост (Telegram шлёт комментарий
  раньше пересылки), альбом файлов, — отдаётся агенту одним вызовом.
"""

import asyncio
import contextlib
import html
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    TelegramObject,
    Update,
    User,
)

log = logging.getLogger(__name__)

AGENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
DEFAULT_BURST_WINDOW = 1.5
ERROR_TEXT = "⚠️ Произошла ошибка при обработке. Подробности — в логах бота."
GATEWAY_COMMANDS = ("start", "help", "agent")
REMOTE_UPDATE_PATH = "/telegram/update"
SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"  # noqa: S105 — имя заголовка вебхука


@dataclass(frozen=True, slots=True)
class Burst:
    """Сообщения, пришедшие почти одновременно, в порядке поступления."""

    chat_id: int
    messages: tuple[Message, ...]


class BotAgent(Protocol):
    name: str  # латиница, префикс команд и callback_data
    title: str
    description: str

    def commands(self) -> list[tuple[str, str]]: ...

    def router(self) -> Router: ...

    async def on_burst(self, bot: Bot, burst: Burst) -> None: ...


class RemoteAgent:
    """Агент в отдельном контейнере: получает апдейты POST-ом (как вебхук Telegram)."""

    def __init__(
        self,
        *,
        name: str,
        title: str,
        description: str,
        url: str,
        secret: str,
        commands: Sequence[tuple[str, str]],
        menu: Sequence[str] = (),
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.title = title
        self.description = description
        self.endpoint = url.rstrip("/") + REMOTE_UPDATE_PATH
        self._secret = secret
        self._commands = list(commands)
        self._menu = list(menu)
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def commands(self) -> list[tuple[str, str]]:
        return list(self._commands)

    def menu(self) -> list[str]:
        return list(self._menu)

    async def forward(self, update: Update) -> bool:
        """Переслать апдейт; False — агент недоступен (подробности в логе)."""
        body = update.model_dump_json(by_alias=True, exclude_none=True)
        try:
            response = await self._client.post(
                self.endpoint,
                content=body,
                headers={SECRET_HEADER: self._secret, "Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("remote agent %s unavailable: %s", self.name, type(exc).__name__)
            return False
        return True

    async def aclose(self) -> None:
        await self._client.aclose()


AnyAgent = BotAgent | RemoteAgent


class AccessRouter(BaseMiddleware):
    """Доступ и маршрутизация на уровне апдейта: чужие — мимо, удалённым — пересылка."""

    def __init__(self, gateway: "Gateway") -> None:
        self.gateway = gateway

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:  # noqa: ANN401 — сигнатура aiogram
        user: User | None = data.get("event_from_user")
        gateway = self.gateway
        if user is None or not gateway.has_access(user.id):
            log.info("ignored update from user %s", user.id if user else None)
            return None
        assert isinstance(event, Update)  # noqa: S101 — middleware висит на dp.update
        bot: Bot = data["bot"]
        if user.id in gateway.guests:
            await gateway.ensure_guest_commands(bot, user.id)
        target = gateway.route(event, user.id)
        if target is None:
            return await handler(event, data)
        await gateway.forward(bot, target, event, user.id)
        return None


BurstHandler = Callable[[Bot, Burst], Awaitable[None]]


class BurstCollector:
    """Дебаунс по чату: пачка отдаётся через `window` секунд тишины."""

    def __init__(self, window: float, handler: BurstHandler) -> None:
        self.window = window
        self.handler = handler
        self._pending: dict[int, list[Message]] = {}
        self._timers: dict[int, asyncio.Task[None]] = {}
        self._running: set[asyncio.Task[None]] = set()

    def add(self, bot: Bot, message: Message) -> None:
        chat_id = message.chat.id
        self._pending.setdefault(chat_id, []).append(message)
        if timer := self._timers.get(chat_id):
            timer.cancel()
        self._timers[chat_id] = asyncio.create_task(self._flush_later(bot, chat_id))

    async def _flush_later(self, bot: Bot, chat_id: int) -> None:
        await asyncio.sleep(self.window)
        self._timers.pop(chat_id, None)
        messages = tuple(self._pending.pop(chat_id, ()))
        if not messages:
            return
        # Обработка — отдельной задачей: новое сообщение отменяет только таймер, не её.
        task = asyncio.create_task(self.handler(bot, Burst(chat_id, messages)))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def drain(self) -> None:
        """Дождаться всех таймеров и обработчиков (тесты, остановка бота)."""
        while self._timers or self._running:
            tasks = [*self._timers.values(), *self._running]
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)


class Gateway:
    def __init__(
        self,
        agents: Sequence[AnyAgent],
        *,
        owner_id: int,
        guests: Mapping[int, str] | None = None,
        burst_window: float = DEFAULT_BURST_WINDOW,
    ) -> None:
        if not agents:
            raise ValueError("нужен хотя бы один агент")
        names = [agent.name for agent in agents]
        for name in names:
            if not AGENT_NAME.match(name):
                raise ValueError(f"некорректное имя агента: {name!r} (латиница, a-z0-9_)")
        if len(set(names)) != len(names):
            raise ValueError(f"имена агентов повторяются: {names}")
        self.agents: dict[str, AnyAgent] = {agent.name: agent for agent in agents}
        self.remote = {a.name: a for a in agents if isinstance(a, RemoteAgent)}
        self.local: dict[str, BotAgent] = {
            a.name: a for a in agents if not isinstance(a, RemoteAgent)
        }
        self.default_agent = names[0]
        self.owner_id = owner_id
        self.guests = dict(guests or {})
        for guest_id, name in self.guests.items():
            if guest_id == owner_id:
                raise ValueError("владельцу и так доступны все агенты")
            if name not in self.agents:
                raise ValueError(f"нет агента {name!r} для гостя {guest_id}")
            if name not in self.remote:
                raise ValueError(f"гостю {guest_id} можно выдать только удалённого агента")
        self._command_owner: dict[str, str] = {}
        self._menu_owner: dict[str, str] = {}
        for agent in agents:
            for command, _ in agent.commands():
                if command in GATEWAY_COMMANDS:
                    continue
                if self._command_owner.setdefault(command, agent.name) != agent.name:
                    raise ValueError(f"команда /{command} есть у нескольких агентов")
            for label in _menu(agent):
                self._menu_owner[label] = agent.name
        self._active: dict[int, str] = {}
        self._guest_commands_tried: set[int] = set()
        self.collector = BurstCollector(burst_window, self._dispatch_burst)

    def has_access(self, user_id: int) -> bool:
        return user_id == self.owner_id or user_id in self.guests

    def active_agent(self, chat_id: int) -> AnyAgent:
        return self.agents[self._active.get(chat_id, self.default_agent)]

    def route(self, update: Update, user_id: int) -> RemoteAgent | None:
        """Удалённый агент, которому переслать апдейт, или None — обработать здесь.

        Кнопка меню локального агента заодно делает его активным в чате.
        """
        if (name := self.guests.get(user_id)) is not None:
            return self.remote[name]
        if update.callback_query is not None:
            return self.remote.get((update.callback_query.data or "").split(":", 1)[0])
        message = update.message
        if message is None:
            return None
        text = message.text or ""
        if text.startswith("/"):
            command = text[1:].split(maxsplit=1)[0].split("@", 1)[0].lower() if text[1:] else ""
            if command in GATEWAY_COMMANDS:
                return None
            if (owner := self._command_owner.get(command)) is not None:
                return self.remote.get(owner)
        elif (owner := self._menu_owner.get(text)) is not None:
            if owner in self.local:
                self._active[message.chat.id] = owner
            return None  # переключение на удалённого агента делает роутер шлюза
        active = self._active.get(message.chat.id, self.default_agent)
        return self.remote.get(active)

    async def forward(self, bot: Bot, agent: RemoteAgent, update: Update, user_id: int) -> None:
        if await agent.forward(update):
            return
        with contextlib.suppress(TelegramAPIError):
            await bot.send_message(
                user_id, f"⚠️ «{html.escape(agent.title)}» сейчас недоступен. Попробуйте чуть позже."
            )

    def dispatcher(self) -> Dispatcher:
        dp = Dispatcher()
        dp.update.outer_middleware(AccessRouter(self))
        dp.include_router(self._own_router())
        for agent in self.local.values():
            dp.include_router(agent.router())
        if self.local:
            dp.include_router(self._burst_router())
        return dp

    def owner_commands(self) -> list[tuple[str, str]]:
        commands = [
            ("start", "Что умеет бот"),
            ("help", "Справка по командам"),
            ("agent", "Выбрать агента"),
        ]
        seen = set(GATEWAY_COMMANDS)
        for agent in self.agents.values():
            for command, about in agent.commands():
                if command not in seen:
                    seen.add(command)
                    commands.append((command, about))
        return commands

    def guest_commands(self, user_id: int) -> list[tuple[str, str]]:
        commands = self.remote[self.guests[user_id]].commands()
        if all(command != "start" for command, _ in commands):
            commands.insert(0, ("start", "Начать"))
        return commands

    async def set_commands(self, bot: Bot) -> None:
        """Меню команд: по умолчанию — владельца, у каждого гостя — своего агента."""
        await bot.set_my_commands(_bot_commands(self.owner_commands()))
        for guest_id in self.guests:
            await self._set_guest_commands(bot, guest_id)

    async def ensure_guest_commands(self, bot: Bot, user_id: int) -> None:
        """При первом контакте гостя: при старте Telegram мог ещё не знать его чат."""
        if user_id in self._guest_commands_tried:
            return
        self._guest_commands_tried.add(user_id)
        await self._set_guest_commands(bot, user_id)

    async def _set_guest_commands(self, bot: Bot, user_id: int) -> None:
        try:
            await bot.set_my_commands(
                _bot_commands(self.guest_commands(user_id)),
                scope=BotCommandScopeChat(chat_id=user_id),
            )
        except TelegramAPIError as exc:
            log.info("commands for guest %s not set yet: %s", user_id, exc)

    async def drain(self) -> None:
        await self.collector.drain()

    async def aclose(self) -> None:
        for agent in self.remote.values():
            await agent.aclose()

    def _menu_keyboard(self) -> ReplyKeyboardMarkup | None:
        """Постоянные кнопки внизу экрана: агенты могут предложить свои через `menu()`."""
        buttons = [label for agent in self.agents.values() for label in _menu(agent)]
        if not buttons:
            return None
        rows = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
        return ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text=label) for label in row] for row in rows],
            resize_keyboard=True,
            is_persistent=True,
        )

    def _help_text(self) -> str:
        lines = [
            "<b>Личный бот-фронтенд для агентов.</b>",
            "Кнопки внизу экрана — быстрая навигация; можно просто писать текстом.",
            "",
        ]
        for agent in self.agents.values():
            lines.append(f"<b>{html.escape(agent.title)}</b> (<code>{agent.name}</code>)")
            lines.append(html.escape(agent.description))
            lines += [
                f"/{command} — {html.escape(about)}"
                for command, about in agent.commands()
                if command not in GATEWAY_COMMANDS
            ]
            lines.append("")
        if len(self.agents) > 1:
            lines.append("/agent &lt;имя&gt; — кому отправлять сообщения без команды")
        return "\n".join(lines).strip()

    def _own_router(self) -> Router:
        router = Router(name="gateway")

        @router.message(CommandStart())
        @router.message(Command("help"))
        async def start(message: Message) -> None:
            await message.answer(self._help_text(), reply_markup=self._menu_keyboard())

        @router.message(Command("agent"))
        async def switch(message: Message, command: CommandObject) -> None:
            chat_id = message.chat.id
            name = (command.args or "").strip()
            if not name:
                current = self.active_agent(chat_id).name
                listing = "\n".join(
                    f"{'▶️' if agent.name == current else '•'} <code>{agent.name}</code> — "
                    f"{html.escape(agent.title)}"
                    for agent in self.agents.values()
                )
                await message.answer(f"Агенты:\n{listing}\n\n/agent &lt;имя&gt; — переключить")
                return
            if name not in self.agents:
                await message.answer(f"Нет агента <code>{html.escape(name)}</code>. Список: /agent")
                return
            self._active[chat_id] = name
            await message.answer(f"Теперь сообщения получает: <b>{name}</b>")

        remote_labels = {
            label: name for label, name in self._menu_owner.items() if name in self.remote
        }

        @router.message(F.text.in_(remote_labels))
        async def switch_by_button(message: Message) -> None:
            agent = self.remote[remote_labels[message.text or ""]]
            self._active[message.chat.id] = agent.name
            await message.answer(
                f"Сообщения без команды теперь получает «{html.escape(agent.title)}».\n"
                f"Вернуться — кнопкой меню или /agent {self.default_agent}"
            )

        return router

    def _burst_router(self) -> Router:
        router = Router(name="gateway-bursts")

        @router.message()
        async def collect(message: Message, bot: Bot) -> None:
            self.collector.add(bot, message)

        return router

    async def _dispatch_burst(self, bot: Bot, burst: Burst) -> None:
        # Пока собиралась пачка, чат могли переключить на удалённого агента — пачка
        # всё равно адресовалась локальному.
        agent = self.active_agent(burst.chat_id)
        if isinstance(agent, RemoteAgent):
            agent = next(iter(self.local.values()))
        try:
            await agent.on_burst(bot, burst)
        except Exception:
            log.exception("agent %s failed on burst in chat %s", agent.name, burst.chat_id)
            with contextlib.suppress(Exception):
                await bot.send_message(burst.chat_id, ERROR_TEXT)


def _menu(agent: AnyAgent) -> list[str]:
    return list(getattr(agent, "menu", lambda: [])() or [])


def _bot_commands(commands: Sequence[tuple[str, str]]) -> list[BotCommand]:
    return [BotCommand(command=command, description=about) for command, about in commands]
