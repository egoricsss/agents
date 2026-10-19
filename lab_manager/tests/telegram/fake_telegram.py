"""Фейковая сессия Bot API: записывает запросы бота и отвечает правдоподобными объектами.

Обработчики aiogram работают по-настоящему; наружу ничего не уходит.
"""

import itertools
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import (
    AnswerCallbackQuery,
    DeleteWebhook,
    EditMessageReplyMarkup,
    EditMessageText,
    GetFile,
    GetMe,
    SendDocument,
    SendMessage,
    SetMyCommands,
    TelegramMethod,
)
from aiogram.types import (
    CallbackQuery,
    Chat,
    Document,
    File,
    Message,
    MessageOriginChannel,
    MessageOriginUser,
    Update,
    User,
)

OWNER_ID = 4242
STRANGER_ID = 999
BOT_TOKEN = "42:TEST-TOKEN"  # noqa: S105 — фиктивный
BOT_USER = User(id=42, is_bot=True, first_name="Agents", username="agents_test_bot")
OWNER = User(id=OWNER_ID, is_bot=False, first_name="Owner")
NOW = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)


class FakeTelegram(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TelegramMethod[Any]] = []
        self.files: dict[str, bytes] = {}
        self._ids = itertools.count(1000)

    # --- то, что видит тест ---
    def sent(self) -> list[SendMessage | SendDocument | EditMessageText]:
        return [
            r for r in self.requests if isinstance(r, SendMessage | SendDocument | EditMessageText)
        ]

    def texts(self) -> list[str]:
        out = []
        for request in self.sent():
            if isinstance(request, SendDocument):
                out.append(request.caption or "")
            else:
                out.append(request.text)
        return out

    def last_text(self) -> str:
        return self.texts()[-1]

    def reset(self) -> None:
        self.requests.clear()

    # --- BaseSession ---
    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[Any],
        timeout: int | None = None,  # noqa: ASYNC109 — сигнатура aiogram
    ) -> Any:  # noqa: ANN401 — тип ответа зависит от метода
        self.requests.append(method)
        if isinstance(method, SendMessage | EditMessageText):
            return self._message(int(method.chat_id or OWNER_ID), text=method.text)
        if isinstance(method, SendDocument):
            return self._message(int(method.chat_id), text=None)
        if isinstance(method, EditMessageReplyMarkup):
            return self._message(int(method.chat_id or OWNER_ID), text="")
        if isinstance(method, AnswerCallbackQuery | SetMyCommands | DeleteWebhook):
            return True
        if isinstance(method, GetMe):
            return BOT_USER
        if isinstance(method, GetFile):
            data = self.files[method.file_id]
            return File(
                file_id=method.file_id,
                file_unique_id=f"u-{method.file_id}",
                file_size=len(data),
                file_path=f"documents/{method.file_id}",
            )
        raise NotImplementedError(type(method).__name__)

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,  # noqa: ASYNC109 — сигнатура aiogram
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        yield self.files[url.rsplit("/", 1)[-1]]

    async def close(self) -> None:
        return None

    def _message(self, chat_id: int, *, text: str | None) -> Message:
        return Message(
            message_id=next(self._ids),
            date=NOW,
            chat=Chat(id=chat_id, type="private"),
            from_user=BOT_USER,
            text=text,
        )


# --- конструкторы апдейтов ---

_update_ids = itertools.count(1)
_message_ids = itertools.count(1)


def message_update(
    text: str | None = None,
    *,
    user: User = OWNER,
    caption: str | None = None,
    document: Document | None = None,
    forwarded_from_channel: str | None = None,
    forwarded_from_user: str | None = None,
    reply_to_message_id: int | None = None,
    message_id: int | None = None,
) -> Update:
    origin: Any = None
    if forwarded_from_channel:
        origin = MessageOriginChannel(
            type="channel",
            date=NOW,
            chat=Chat(id=-100500, type="channel", title=forwarded_from_channel),
            message_id=77,
        )
    elif forwarded_from_user:
        origin = MessageOriginUser(
            type="user",
            date=NOW,
            sender_user=User(id=555, is_bot=False, first_name=forwarded_from_user),
        )
    reply_to = None
    if reply_to_message_id is not None:
        reply_to = Message(
            message_id=reply_to_message_id,
            date=NOW,
            chat=Chat(id=user.id, type="private"),
            from_user=BOT_USER,
            text="…",
        )
    return Update(
        update_id=next(_update_ids),
        message=Message(
            message_id=message_id or next(_message_ids),
            date=NOW,
            chat=Chat(id=user.id, type="private"),
            from_user=user,
            text=text,
            caption=caption,
            document=document,
            forward_origin=origin,
            reply_to_message=reply_to,
        ),
    )


def document(
    session: FakeTelegram, data: bytes, file_name: str, mime_type: str = "application/pdf"
) -> Document:
    file_id = f"file{len(session.files) + 1}"
    session.files[file_id] = data
    return Document(
        file_id=file_id,
        file_unique_id=f"u-{file_id}",
        file_name=file_name,
        mime_type=mime_type,
        file_size=len(data),
    )


def callback_update(data: str, *, user: User = OWNER, message_id: int = 1) -> Update:
    return Update(
        update_id=next(_update_ids),
        callback_query=CallbackQuery(
            id=str(next(_update_ids)),
            from_user=user,
            chat_instance="ci",
            data=data,
            message=Message(
                message_id=message_id,
                date=NOW,
                chat=Chat(id=user.id, type="private"),
                from_user=BOT_USER,
                text="…",
            ),
        ),
    )
