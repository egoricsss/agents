"""Хендлеры aiogram: только разбор сообщений и ответы, вся логика — в сервисах.

Сервисы приходят из `workflow_data` диспетчера (`pairing`, `matching`, `stats`).
"""

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BotCommand, Message, TelegramObject

from telepathy.bot import texts
from telepathy.services.errors import TelepathyError
from telepathy.services.matching import FoundMatch, MatchingService
from telepathy.services.pairing import PairingService
from telepathy.services.stats import StatsService

log = logging.getLogger(__name__)

Handler = Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]]


class UserGate(BaseMiddleware):
    """Пускает только разрешённых (если список задан) и регистрирует автора сообщения."""

    def __init__(self, allowed_user_ids: Iterable[int]) -> None:
        self.allowed = frozenset(allowed_user_ids)

    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:  # noqa: ANN401 — сигнатура aiogram
        user = data.get("event_from_user")
        if user is None or (self.allowed and user.id not in self.allowed):
            log.info("ignored update from user %s", user.id if user else None)
            return None
        pairing: PairingService = data["pairing"]
        await pairing.register(user.id, user.first_name)
        return await handler(event, data)


async def _send(bot: Bot, chat_id: int, text: str) -> None:
    """Сообщение партнёру: если он заблокировал бота — не повод ронять обработку."""
    try:
        await bot.send_message(chat_id, text)
    except TelegramAPIError as exc:
        log.warning("cannot message user %s: %s", chat_id, exc)


async def notify_match(bot: Bot, found: FoundMatch, pairing: PairingService) -> None:
    first, second = found.user_ids
    names = {first: await pairing.display_name(first), second: await pairing.display_name(second)}
    for viewer, partner in ((first, second), (second, first)):
        await _send(bot, viewer, texts.match(found, viewer, names[partner]))


def create_router() -> Router:
    router = Router(name="telepathy")
    router.message.filter(F.chat.type == "private")

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def start(message: Message, pairing: PairingService) -> None:
        assert message.from_user is not None  # noqa: S101 — гарантирует UserGate
        status = await pairing.status(message.from_user.id)
        partner = await pairing.display_name(status.partner_id) if status.partner_id else None
        await message.answer(texts.start(status, partner))

    @router.message(Command("pair"))
    async def pair(message: Message, pairing: PairingService) -> None:
        assert message.from_user is not None  # noqa: S101
        try:
            code = await pairing.invite(message.from_user.id)
        except TelepathyError as exc:
            await message.answer(str(exc))
            return
        await message.answer(texts.invite(code))

    @router.message(Command("join"))
    async def join(
        message: Message, command: CommandObject, bot: Bot, pairing: PairingService
    ) -> None:
        assert message.from_user is not None  # noqa: S101
        code = (command.args or "").strip()
        if not code:
            await message.answer(texts.JOIN_USAGE)
            return
        try:
            joined = await pairing.join(message.from_user.id, code)
        except TelepathyError as exc:
            await message.answer(str(exc))
            return
        await message.answer(texts.joined(await pairing.display_name(joined.partner_id)))
        me = await pairing.display_name(message.from_user.id)
        await _send(bot, joined.partner_id, texts.partner_joined(me))

    @router.message(Command("stats"))
    async def stats(message: Message, stats: StatsService) -> None:
        assert message.from_user is not None  # noqa: S101
        try:
            value = await stats.for_user(message.from_user.id, message.date)
        except TelepathyError as exc:
            await message.answer(str(exc))
            return
        await message.answer(texts.stats(value))

    @router.message(F.text.startswith("/"))
    async def unknown_command(message: Message) -> None:
        await message.answer(texts.UNKNOWN_COMMAND)

    @router.message(F.text)
    async def thought(
        message: Message, bot: Bot, matching: MatchingService, pairing: PairingService
    ) -> None:
        assert message.from_user is not None  # noqa: S101
        assert message.text is not None  # noqa: S101 — фильтр F.text
        try:
            saved = await matching.save(message.from_user.id, message.text, message.date)
        except TelepathyError as exc:
            await message.answer(str(exc))
            return
        await message.answer(texts.THOUGHT_ACCEPTED)
        found = await matching.find_match(saved)
        if found is not None:
            await notify_match(bot, found, pairing)

    @router.message()
    async def not_text(message: Message) -> None:
        await message.answer(texts.ONLY_TEXT)

    return router


def create_dispatcher(
    *,
    pairing: PairingService,
    matching: MatchingService,
    stats: StatsService,
    allowed_user_ids: Iterable[int] = (),
) -> Dispatcher:
    dp = Dispatcher(pairing=pairing, matching=matching, stats=stats)
    dp.message.outer_middleware(UserGate(allowed_user_ids))
    dp.include_router(create_router())
    return dp


async def set_commands(bot: Bot) -> None:
    """Меню команд — только в самостоятельном режиме: в общем боте его ведёт шлюз."""
    commands = [BotCommand(command=c, description=d) for c, d in texts.COMMANDS]
    try:
        await bot.set_my_commands(commands)
    except TelegramAPIError as exc:
        log.warning("cannot set bot commands: %s", exc)
