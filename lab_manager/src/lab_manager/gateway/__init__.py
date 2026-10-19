"""Telegram-шлюз для нескольких агентов владельца.

Шлюз ничего не знает о предметной области: агенты подключаются через протокол `BotAgent`.
- Доступ только владельцу (остальные апдейты молча отбрасываются).
- Команды и inline-кнопки агентов обрабатываются сразу (своими роутерами; `callback_data`
  агента начинается с `<имя>:`).
- Остальные сообщения собираются в «пачки» (`Burst`): всё, что пришло почти одновременно —
  подпись + пересланный пост (Telegram шлёт комментарий раньше пересылки), альбом файлов, —
  отдаётся активному агенту чата одним вызовом.
"""

import asyncio
import contextlib
import html
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BotCommand, Message, TelegramObject, User

log = logging.getLogger(__name__)

AGENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
DEFAULT_BURST_WINDOW = 1.5
ERROR_TEXT = "⚠️ Произошла ошибка при обработке. Подробности — в логах бота."


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


class OwnerOnly(BaseMiddleware):
    """Бот личный: апдейты не от владельца не доходят до обработчиков."""

    def __init__(self, owner_id: int) -> None:
        self.owner_id = owner_id

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:  # noqa: ANN401 — сигнатура aiogram
        user: User | None = data.get("event_from_user")
        if user is None or user.id != self.owner_id:
            log.info("ignored update from user %s", user.id if user else None)
            return None
        return await handler(event, data)


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
        agents: Sequence[BotAgent],
        *,
        owner_id: int,
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
        self.agents = {agent.name: agent for agent in agents}
        self.default_agent = names[0]
        self.owner_id = owner_id
        self._active: dict[int, str] = {}
        self.collector = BurstCollector(burst_window, self._dispatch_burst)

    def active_agent(self, chat_id: int) -> BotAgent:
        return self.agents[self._active.get(chat_id, self.default_agent)]

    def dispatcher(self) -> Dispatcher:
        dp = Dispatcher()
        dp.update.outer_middleware(OwnerOnly(self.owner_id))
        dp.include_router(self._own_router())
        for agent in self.agents.values():
            dp.include_router(agent.router())
        dp.include_router(self._burst_router())
        return dp

    async def set_commands(self, bot: Bot) -> None:
        commands = [
            BotCommand(command="start", description="Что умеет бот"),
            BotCommand(command="help", description="Справка по командам"),
            BotCommand(command="agent", description="Выбрать агента"),
        ]
        for agent in self.agents.values():
            commands += [BotCommand(command=c, description=d) for c, d in agent.commands()]
        await bot.set_my_commands(commands)

    async def drain(self) -> None:
        await self.collector.drain()

    def _help_text(self) -> str:
        lines = ["<b>Личный бот-фронтенд для агентов.</b>", ""]
        for agent in self.agents.values():
            lines.append(f"<b>{html.escape(agent.title)}</b> (<code>{agent.name}</code>)")
            lines.append(html.escape(agent.description))
            lines += [f"/{command} — {html.escape(about)}" for command, about in agent.commands()]
            lines.append("")
        if len(self.agents) > 1:
            lines.append("/agent &lt;имя&gt; — кому отправлять сообщения без команды")
        return "\n".join(lines).strip()

    def _own_router(self) -> Router:
        router = Router(name="gateway")

        @router.message(CommandStart())
        @router.message(Command("help"))
        async def start(message: Message) -> None:
            await message.answer(self._help_text())

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

        return router

    def _burst_router(self) -> Router:
        router = Router(name="gateway-bursts")

        @router.message()
        async def collect(message: Message, bot: Bot) -> None:
            self.collector.add(bot, message)

        return router

    async def _dispatch_burst(self, bot: Bot, burst: Burst) -> None:
        agent = self.active_agent(burst.chat_id)
        try:
            await agent.on_burst(bot, burst)
        except Exception:
            log.exception("agent %s failed on burst in chat %s", agent.name, burst.chat_id)
            with contextlib.suppress(Exception):
                await bot.send_message(burst.chat_id, ERROR_TEXT)
