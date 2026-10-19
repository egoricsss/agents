"""Агент «Лабораторные» для Telegram-шлюза.

Приём заданий (файлы, альбомы, пересланные посты с подписью владельца), переспрос недостающих
полей, подтверждение сомнительного разбора кнопками, команды /list /week /lab /status /req /prompt.
Вся логика — в сервисах; здесь только разбор сообщений и ответы.
"""

import contextlib
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from html import escape
from io import BytesIO
from zoneinfo import ZoneInfo

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from lab_manager.bot import format as fmt
from lab_manager.db import Database
from lab_manager.domain import RequirementScope, Source, Status
from lab_manager.errors import (
    ConversionError,
    FileTooLargeError,
    NotFoundError,
    UnsupportedFormatError,
)
from lab_manager.extraction import LabExtraction
from lab_manager.gateway import Burst
from lab_manager.services.executor_prompt import PROMPT_FILE, ExecutorPromptBuilder
from lab_manager.services.ingest import (
    MAX_FILE_BYTES,
    TEXT_MESSAGE_FILENAME,
    IngestOutcome,
    IngestService,
)
from lab_manager.services.labs import (
    STATUS_TITLES,
    LabQueries,
    LabView,
    parse_status,
    subject_matches,
)
from lab_manager.services.requirements import RequirementsService

log = logging.getLogger(__name__)

NAME = "lab"
PENDING_TTL = 3600.0
GLOBAL_REFS = {"все", "всё", "общие", "global", "*"}

TextHandler = Callable[[Bot, Message, str], Awaitable[None]]


@dataclass
class PendingIngest:
    """Задание, ждущее подпись или подтверждение владельца."""

    chat_id: int
    data: bytes
    filename: str
    mime_type: str | None
    prompt: str | None
    extractions: tuple[LabExtraction, ...] | None
    created: float


class LabAgent:
    name = NAME
    title = "Лабораторные"
    description = (
        "Раскладывает задания по папкам, ведёт статусы и готовит промпт для исполнителя. "
        "Пришлите файл задания или перешлите пост преподавателя с подписью «Предмет, ЛР N»."
    )

    def __init__(
        self,
        *,
        db: Database,
        ingest: IngestService,
        queries: LabQueries,
        requirements: RequirementsService,
        builder: ExecutorPromptBuilder,
        tz: ZoneInfo,
        disk_root: str = "app:/Лабы",
        max_file_bytes: int = MAX_FILE_BYTES,
        on_text: TextHandler | None = None,
    ) -> None:
        self.db = db
        self.ingest = ingest
        self.queries = queries
        self.requirements = requirements
        self.builder = builder
        self.tz = tz
        self.disk_root = disk_root
        self.max_file_bytes = max_file_bytes
        self.on_text = on_text  # свободный текст → диалоговый агент (шаг 7b)
        self._pending: dict[str, PendingIngest] = {}
        self._questions: dict[tuple[int, int], str] = {}  # (chat, message_id) → token
        self._last_question: dict[int, str] = {}  # chat → token

    def commands(self) -> list[tuple[str, str]]:
        return [
            ("list", "Открытые лабы"),
            ("week", "Что сдать на этой неделе"),
            ("lab", "Карточка лабы: /lab ОС 3"),
            ("status", "Статус: /status ОС 3 сделано"),
            ("req", "Требования: /req ОС 3: вариант 7"),
            ("prompt", "Промпт для исполнителя: /prompt ОС 3"),
        ]

    # ------------------------------------------------------------------ роутер

    def router(self) -> Router:
        router = Router(name=NAME)
        router.message.register(self.cmd_list, Command("list"))
        router.message.register(self.cmd_week, Command("week"))
        router.message.register(self.cmd_lab, Command("lab"))
        router.message.register(self.cmd_status, Command("status"))
        router.message.register(self.cmd_req, Command("req"))
        router.message.register(self.cmd_prompt, Command("prompt"))
        router.callback_query.register(self.on_callback, F.data.startswith(f"{NAME}:"))
        return router

    # ------------------------------------------------------------------ пачки сообщений

    async def on_burst(self, bot: Bot, burst: Burst) -> None:
        self._purge()
        documents = [m for m in burst.messages if m.document]
        forwards = [m for m in burst.messages if m.forward_origin and not m.document]
        own_texts = [
            m for m in burst.messages if m.text and not m.forward_origin and not m.document
        ]
        prompt = "\n".join(m.text for m in own_texts if m.text) or None

        if documents or forwards:
            if prompt is None:
                prompt = next(
                    (m.caption for m in documents if m.caption and not m.forward_origin), None
                )
            for message in documents:
                await self._intake_document(bot, burst.chat_id, message, prompt)
            if forwards:
                text = "\n\n".join(m.text or m.caption or "" for m in forwards).strip()
                if text:
                    await self._intake(
                        bot, burst.chat_id, text.encode(), TEXT_MESSAGE_FILENAME, None, prompt
                    )
            return

        if not own_texts:
            await bot.send_message(burst.chat_id, "Такие сообщения я пока не понимаю. /help")
            return
        token = self._answered_question(burst.chat_id, own_texts)
        if token and prompt:
            pending = self._pending.pop(token)
            self._forget_questions(token)
            await self._intake(
                bot, burst.chat_id, pending.data, pending.filename, pending.mime_type, prompt
            )
            return
        if self.on_text:
            await self.on_text(bot, own_texts[-1], prompt or "")
            return
        await bot.send_message(
            burst.chat_id,
            "Пришлите файл задания или перешлите пост преподавателя с подписью"
            f" {fmt.PROMPT_EXAMPLE}. Команды — /help",
        )

    async def _intake_document(
        self, bot: Bot, chat_id: int, message: Message, prompt: str | None
    ) -> None:
        document = message.document
        assert document is not None  # noqa: S101
        name = document.file_name or "задание"
        if document.file_size and document.file_size > self.max_file_bytes:
            await bot.send_message(
                chat_id,
                f"Файл «{escape(name)}» больше {self.max_file_bytes // (1024 * 1024)} МБ —"
                " Telegram не даст его скачать боту.",
            )
            return
        buffer = await bot.download(document.file_id, destination=BytesIO())
        assert isinstance(buffer, BytesIO)  # noqa: S101
        await self._intake(bot, chat_id, buffer.getvalue(), name, document.mime_type, prompt)

    async def _intake(
        self,
        bot: Bot,
        chat_id: int,
        data: bytes,
        filename: str,
        mime_type: str | None,
        prompt: str | None,
    ) -> None:
        try:
            preview = await self.ingest.preview(data=data, filename=filename, prompt=prompt)
        except (FileTooLargeError, UnsupportedFormatError, ConversionError) as exc:
            await bot.send_message(
                chat_id, f"Не могу принять «{escape(filename)}»: {escape(str(exc))}"
            )
            return

        if preview.duplicate:
            await bot.send_message(chat_id, fmt.duplicate(preview.duplicate.items))
            return
        pending = PendingIngest(chat_id, data, filename, mime_type, prompt, None, time.monotonic())
        if preview.missing:
            token = self._remember(pending)
            sent = await bot.send_message(chat_id, fmt.missing(preview.missing))
            self._ask(chat_id, sent.message_id, token)
            return
        if preview.needs_confirmation:
            pending.extractions = preview.extractions
            token = self._remember(pending)
            sent = await bot.send_message(
                chat_id,
                fmt.confirmation(preview.extractions, self.tz),
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="✅ Сохранить", callback_data=f"{NAME}:ok:{token}"
                            ),
                            InlineKeyboardButton(
                                text="✖️ Отмена", callback_data=f"{NAME}:no:{token}"
                            ),
                        ]
                    ]
                ),
            )
            self._ask(chat_id, sent.message_id, token)
            return
        await self._store(bot, pending, preview.extractions)

    async def _store(
        self, bot: Bot, pending: PendingIngest, extractions: Sequence[LabExtraction]
    ) -> None:
        result = await self.ingest.ingest(
            data=pending.data,
            filename=pending.filename,
            prompt=pending.prompt,
            mime_type=pending.mime_type,
            source=Source.TELEGRAM,
            extractions=extractions,
        )
        if result.outcome is IngestOutcome.DUPLICATE:
            text = fmt.duplicate(result.items)
        elif result.outcome is IngestOutcome.NEEDS_INFO:
            text = fmt.missing(result.missing)
        else:
            text = "\n\n".join(fmt.ingested(item, self.tz, self.disk_root) for item in result.items)
        await bot.send_message(pending.chat_id, text)

    # ------------------------------------------------------------------ кнопки

    async def on_callback(self, query: CallbackQuery, bot: Bot) -> None:
        parts = (query.data or "").split(":")
        action = parts[1] if len(parts) > 1 else ""
        chat_id = query.message.chat.id if query.message else query.from_user.id
        await query.answer()

        if action in {"ok", "no"}:
            token = parts[2] if len(parts) > 2 else ""
            pending = self._pending.pop(token, None)
            self._forget_questions(token)
            if pending is None:
                await bot.send_message(chat_id, "Запрос устарел — пришлите задание ещё раз.")
            elif action == "no":
                await bot.send_message(chat_id, "Отменено, ничего не сохранил.")
            else:
                await self._store(bot, pending, pending.extractions or ())
            await self._drop_buttons(query)
            return

        if action == "st" and len(parts) == 4:
            try:
                status = Status(parts[3])
                view = await self.queries.view(int(parts[2]))
            except (ValueError, NotFoundError):
                await bot.send_message(chat_id, "Лаба или статус не найдены.")
                return
            await self._set_status(bot, chat_id, view, status)
            await self._drop_buttons(query)

    async def _drop_buttons(self, query: CallbackQuery) -> None:
        if isinstance(query.message, Message):
            with contextlib.suppress(Exception):
                await query.message.edit_reply_markup(reply_markup=None)

    # ------------------------------------------------------------------ команды

    async def cmd_list(self, message: Message) -> None:
        views = await self.queries.open_labs()
        total = len(await self.queries.all_labs())
        empty = (
            "Открытых лаб нет 🎉"
            if total
            else "Пока нет ни одной лабы. Пришлите файл задания или перешлите пост"
            f" с подписью {fmt.PROMPT_EXAMPLE}."
        )
        await message.answer(fmt.lab_list(views, self.tz, empty=empty))

    async def cmd_week(self, message: Message) -> None:
        await message.answer(fmt.week(await self.queries.week(), self.tz))

    async def cmd_lab(self, message: Message, command: CommandObject) -> None:
        view = await self._resolve_one(message, command.args, "lab")
        if view is None:
            return
        lab = view.lab
        requirements = [
            ("к лабе", await self.db.requirements.current(RequirementScope.LAB, lab.id)),
            (
                "к предмету",
                await self.db.requirements.current(RequirementScope.SUBJECT, lab.subject_id),
            ),
            ("общие", await self.db.requirements.current(RequirementScope.GLOBAL)),
        ]
        await message.answer(fmt.lab_card(view, self.tz, self.disk_root, requirements))

    async def cmd_status(self, message: Message, command: CommandObject) -> None:
        args = (command.args or "").strip()
        if not args:
            await message.answer(
                "Пример: <code>/status ОС 3 сделано</code> или <code>/status ОС 3</code>"
            )
            return
        ref, status = _split_status(args)
        views = await self.queries.resolve(ref)
        if not views and status is None:
            # возможно, последнее слово — непонятный статус: «/status ОС 3 почти»
            head, _, tail = args.rpartition(" ")
            if head and await self.queries.resolve(head):
                await message.answer(
                    f"Не понял статус «{escape(tail)}». Варианты: "
                    + ", ".join(STATUS_TITLES.values())
                )
                return
        view = await self._pick(message, views, ref, "status")
        if view is None:
            return
        if status is None:
            await message.answer(
                f"{fmt.lab_header(view.subject, view.lab)}\nВыберите статус:",
                reply_markup=_status_keyboard(view.lab.id),
            )
            return
        await self._set_status(message.bot, message.chat.id, view, status)  # type: ignore[arg-type]

    async def cmd_req(self, message: Message, command: CommandObject) -> None:
        args = (command.args or "").strip()
        if not args:
            await message.answer(
                "Пример: <code>/req ОС 3: вариант 7</code> (к лабе), "
                "<code>/req ОС: только Linux</code> (к предмету), "
                "<code>/req все: отчёт по ГОСТ</code> (общие). Без «:» — показать текущие."
            )
            return
        ref, sep, text = (part.strip() for part in args.partition(":"))
        target = await self._requirement_target(message, ref)
        if target is None:
            return
        scope, target_id, label = target
        if not sep:
            current = await self.requirements.current(scope, target_id)
            body = escape(current.text) if current and current.text else "пока нет"
            await message.answer(f"<b>Требования {label}:</b>\n{body}")
            return
        update = await self.requirements.set(scope, text, target_id)
        await message.answer(
            f"✅ Требования {label} сохранены (версия {update.requirement.version})."
            f" Промптов пересобрано: {len(update.rebuilt_lab_ids)}."
        )

    async def cmd_prompt(self, message: Message, command: CommandObject) -> None:
        view = await self._resolve_one(message, command.args, "prompt")
        if view is None:
            return
        await self.builder.write(view.lab.id)
        _, prompt, _ = await self.builder.build(view.lab.id)
        await message.answer_document(
            BufferedInputFile(prompt.encode("utf-8"), filename=PROMPT_FILE),
            caption=f"Промпт для исполнителя: {fmt.lab_header(view.subject, view.lab)}",
        )

    # ------------------------------------------------------------------ помощники

    async def _set_status(self, bot: Bot, chat_id: int, view: LabView, status: Status) -> None:
        old = view.lab.status
        lab = await self.queries.set_status(view.lab.id, status, Source.TELEGRAM)
        await bot.send_message(
            chat_id,
            f"{fmt.lab_header(view.subject, lab)}\n"
            f"{STATUS_TITLES[old]} → <b>{STATUS_TITLES[lab.status]}</b>",
        )

    async def _resolve_one(
        self, message: Message, args: str | None, command: str
    ) -> LabView | None:
        ref = (args or "").strip()
        if not ref:
            await message.answer(
                f"Пример: <code>/{command} ОС 3</code> или <code>/{command} #12</code>"
            )
            return None
        return await self._pick(message, await self.queries.resolve(ref), ref, command)

    async def _pick(
        self, message: Message, views: list[LabView], ref: str, command: str
    ) -> LabView | None:
        if not views:
            await message.answer(
                f"Не нашёл лабу «{escape(ref)}». Пример: <code>/{command} ОС 3</code>."
                " Все лабы — /list"
            )
            return None
        if len(views) > 1:
            await message.answer(fmt.ambiguous(views, self.tz, command))
            return None
        return views[0]

    async def _requirement_target(
        self, message: Message, ref: str
    ) -> tuple[RequirementScope, int | None, str] | None:
        if ref.casefold() in GLOBAL_REFS:
            return RequirementScope.GLOBAL, None, "общие"
        if any(char.isdigit() for char in ref):
            view = await self._pick(message, await self.queries.resolve(ref), ref, "req")
            if view is None:
                return None
            return (
                RequirementScope.LAB,
                view.lab.id,
                f"к {escape(view.subject)} · ЛР {view.lab.number}",
            )
        subjects = [s for s in await self.db.subjects.list() if subject_matches(ref, s.name)]
        if len(subjects) != 1:
            names = ", ".join(escape(s.name) for s in subjects) or "нет совпадений"
            await message.answer(f"Не понял, какой предмет «{escape(ref)}»: {names}.")
            return None
        return RequirementScope.SUBJECT, subjects[0].id, f"к предмету {escape(subjects[0].name)}"

    def _remember(self, pending: PendingIngest) -> str:
        token = secrets.token_urlsafe(6)
        self._pending[token] = pending
        return token

    def _ask(self, chat_id: int, message_id: int, token: str) -> None:
        self._questions[(chat_id, message_id)] = token
        self._last_question[chat_id] = token

    def _answered_question(self, chat_id: int, texts: Sequence[Message]) -> str | None:
        """Ответ на вопрос бота: reply на него или просто следующее текстовое сообщение."""
        for message in texts:
            if message.reply_to_message:
                token = self._questions.get((chat_id, message.reply_to_message.message_id))
                if token in self._pending:
                    return token
        token = self._last_question.get(chat_id)
        return token if token in self._pending else None

    def _forget_questions(self, token: str) -> None:
        self._questions = {key: value for key, value in self._questions.items() if value != token}
        self._last_question = {
            chat: value for chat, value in self._last_question.items() if value != token
        }

    def _purge(self) -> None:
        now = time.monotonic()
        for token in [t for t, p in self._pending.items() if now - p.created > PENDING_TTL]:
            self._pending.pop(token, None)
            self._forget_questions(token)


def _split_status(args: str) -> tuple[str, Status | None]:
    """«ОС 3 в работе» → («ОС 3», IN_PROGRESS); статус — последние одно-два слова."""
    words = args.split()
    for size in (2, 1):
        if len(words) > size and (status := parse_status(" ".join(words[-size:]))):
            return " ".join(words[:-size]), status
    return args, None


def _status_keyboard(lab_id: int) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=title, callback_data=f"{NAME}:st:{lab_id}:{status.value}")
        for status, title in STATUS_TITLES.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[:3], buttons[3:]])
