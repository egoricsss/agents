"""Фейковая сессия Bot API: записывает запросы бота, наружу ничего не уходит."""

import itertools
from datetime import datetime
from typing import Any

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import DeleteWebhook, GetMe, SendMessage, SetMyCommands, TelegramMethod
from aiogram.types import Chat, Message, PhotoSize, Update, User

from tests.conftest import NOW

BOT_TOKEN = "42:TEST-TOKEN"  # noqa: S105 — фиктивный
BOT_USER = User(id=42, is_bot=True, first_name="Telepathy", username="telepathy_test_bot")


class FakeTelegram(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TelegramMethod[Any]] = []
        self.blocked: set[int] = set()
        self._ids = itertools.count(1000)

    def texts(self, chat_id: int) -> list[str]:
        return [
            r.text for r in self.requests if isinstance(r, SendMessage) and r.chat_id == chat_id
        ]

    def last(self, chat_id: int) -> str:
        return self.texts(chat_id)[-1]

    def reset(self) -> None:
        self.requests.clear()

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[Any],
        timeout: int | None = None,  # noqa: ASYNC109 — сигнатура aiogram
    ) -> Any:  # noqa: ANN401 — тип ответа зависит от метода
        if isinstance(method, SendMessage) and method.chat_id in self.blocked:
            from aiogram.exceptions import TelegramForbiddenError

            raise TelegramForbiddenError(method=method, message="bot was blocked by the user")
        self.requests.append(method)
        if isinstance(method, SendMessage):
            return Message(
                message_id=next(self._ids),
                date=NOW,
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=BOT_USER,
                text=method.text,
            )
        if isinstance(method, SetMyCommands | DeleteWebhook):
            return True
        if isinstance(method, GetMe):
            return BOT_USER
        raise NotImplementedError(type(method).__name__)

    async def stream_content(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError

    async def close(self) -> None:
        return None


_update_ids = itertools.count(1)
_message_ids = itertools.count(1)


def message_update(
    user: User, text: str | None = None, *, at: datetime = NOW, photo: bool = False
) -> Update:
    return Update(
        update_id=next(_update_ids),
        message=Message(
            message_id=next(_message_ids),
            date=at,
            chat=Chat(id=user.id, type="private"),
            from_user=user,
            text=text,
            photo=[PhotoSize(file_id="p", file_unique_id="p", width=1, height=1)]
            if photo
            else None,
        ),
    )
