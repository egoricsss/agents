"""Агент «Лабораторные» для Telegram-шлюза.

Приём заданий (файлы, альбомы, пересланные посты с подписью владельца), переспрос недостающих
полей, подтверждение сомнительного разбора кнопками, команды /list /week /lab /status /req /prompt.
Вся логика — в сервисах; здесь только разбор сообщений и ответы.
"""

import contextlib
import logging
import secrets
import time
from collections.abc import Sequence
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

from lab_manager.assistant import LabAssistant, Proposal
from lab_manager.bot import format as fmt
from lab_manager.db import Database
from lab_manager.domain import RequirementScope, Source, Status
from lab_manager.errors import (
    AlreadyExistsError,
    ConversionError,
    FileTooLargeError,
    NotFoundError,
    UnsupportedFormatError,
)
from lab_manager.extraction import LabExtraction
from lab_manager.gateway import Burst
from lab_manager.services.editing import LabEditor
from lab_manager.services.executor_prompt import PROMPT_FILE, ExecutorPromptBuilder
from lab_manager.services.ingest import (
    ASSIGNMENT_MARKDOWN,
    MAX_FILE_BYTES,
    TEXT_MESSAGE_FILENAME,
    IngestOutcome,
    IngestService,
)
from lab_manager.services.labs import (
    OPEN_STATUSES,
    STATUS_ICONS,
    STATUS_TITLES,
    LabQueries,
    LabView,
    parse_status,
    subject_matches,
)
from lab_manager.services.removal import LabRemover
from lab_manager.services.requirements import RequirementsService

log = logging.getLogger(__name__)

NAME = "lab"
MENU_LABS = "📚 Лабы"
MENU_WEEK = "📅 Неделя"
PENDING_TTL = 3600.0
GLOBAL_REFS = {"все", "всё", "общие", "global", "*"}


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
        assistant: LabAssistant | None = None,
        remover: LabRemover | None = None,
        editor: LabEditor | None = None,
    ) -> None:
        self.db = db
        self.ingest = ingest
        self.queries = queries
        self.requirements = requirements
        self.builder = builder
        self.tz = tz
        self.disk_root = disk_root
        self.max_file_bytes = max_file_bytes
        self.assistant = assistant  # свободный текст → диалоговый агент
        self._proposals: dict[str, Proposal] = {}
        self._batches: dict[str, list[str]] = {}  # «ничего не надо» убирает все кнопки ответа
        self.editor = editor
        self._current: dict[int, int] = {}  # chat → открытая лаба (контекст для свободного текста)
        self.remover = remover
        self._pending: dict[str, PendingIngest] = {}
        self._questions: dict[tuple[int, int], str] = {}  # (chat, message_id) → token
        self._last_question: dict[int, str] = {}  # chat → token

    def menu(self) -> list[str]:
        """Постоянные кнопки внизу экрана (шлюз показывает их по /start)."""
        return [MENU_LABS, MENU_WEEK]

    def commands(self) -> list[tuple[str, str]]:
        return [
            ("list", "Открытые лабы"),
            ("week", "Что сдать на этой неделе"),
            ("lab", "Карточка лабы: /lab ОС 3"),
            ("status", "Статус: /status ОС 3 сделано"),
            ("req", "Требования: /req ОС 3: вариант 7"),
            ("prompt", "Промпт для исполнителя: /prompt ОС 3"),
            ("delete", "Удалить ошибочную лабу: /delete #1"),
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
        router.message.register(self.cmd_delete, Command("delete"))
        router.message.register(self.cmd_list, F.text == MENU_LABS)
        router.message.register(self.cmd_week, F.text == MENU_WEEK)
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
        log.info(
            "burst chat=%s: %d docs, %d forwards, %d own texts (prompt %d chars)",
            burst.chat_id,
            len(documents),
            len(forwards),
            len(own_texts),
            len(prompt or ""),
        )

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
        if self.assistant and prompt:
            await self._converse(bot, burst.chat_id, prompt)
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

        log.info(
            "preview %s (%d bytes): duplicate=%s missing=%s confirm=%s labs=%s",
            filename,
            len(data),
            preview.duplicate is not None,
            preview.missing,
            preview.needs_confirmation,
            [(e.subject, e.number, round(e.confidence, 2)) for e in preview.extractions],
        )
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

        if action == "nv":
            await self._navigate(query, chat_id, parts[2:])
            return
        if action in {"ss", "pf", "tf", "rq", "ed", "dl"} and len(parts) >= 3:
            await self._card_action(query, bot, chat_id, action, parts[2:])
            return

        if action == "ap" and len(parts) == 3:
            proposal = self._proposals.pop(parts[2], None)
            if proposal is None:
                await bot.send_message(chat_id, "⌛ Это предложение уже неактуально.")
                return
            await self._remove_button(query, f"{NAME}:ap:{parts[2]}")
            log.info("proposal applied: %s on lab %s", proposal.kind, proposal.lab_id)
            await self._apply(bot, chat_id, proposal)
            return
        if action == "rj" and len(parts) == 3:
            for token in self._batches.pop(parts[2], []):
                self._proposals.pop(token, None)
            await self._drop_buttons(query)
            await bot.send_message(chat_id, "👌 Хорошо, ничего не трогаю.")
            return

        if action in {"del", "keep"} and len(parts) == 3:
            await self._drop_buttons(query)
            if action == "keep":
                await bot.send_message(chat_id, "Оставил как есть.")
                return
            if self.remover is None:
                await bot.send_message(chat_id, "Удаление не настроено.")
                return
            try:
                view = await self.queries.view(int(parts[2]))
                await self.remover.remove(view.lab.id)
            except (ValueError, NotFoundError):
                await bot.send_message(chat_id, "Лаба уже удалена.")
                return
            log.info("lab %s deleted by owner", view.lab.id)
            await bot.send_message(
                chat_id,
                f"🗑 Удалена: {fmt.lab_header(view.subject, view.lab)}. "
                "Теперь можно прислать задание заново с правильной подписью.",
            )
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
        log.info("command list")
        text, markup = await self._root_view()
        await message.answer(text, reply_markup=markup)

    async def cmd_week(self, message: Message) -> None:
        await message.answer(fmt.week(await self.queries.week(), self.tz))

    async def cmd_lab(self, message: Message, command: CommandObject) -> None:
        log.info("command lab")
        view = await self._resolve_one(message, command.args, "lab")
        if view is None:
            return
        self._current[message.chat.id] = view.lab.id
        text, markup = await self._lab_view(view.lab.id)
        await message.answer(text, reply_markup=markup)

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

    async def _converse(self, bot: Bot, chat_id: int, text: str) -> None:
        assert self.assistant is not None  # noqa: S101
        with contextlib.suppress(Exception):
            await bot.send_chat_action(chat_id, "typing")
        try:
            reply = await self.assistant.answer(chat_id, text, self._current.get(chat_id))
        except Exception:
            log.exception("assistant failed")
            await bot.send_message(
                chat_id,
                "😔 Не получилось ответить: языковая модель сейчас недоступна или ответила"
                " странно. Кнопки внизу и команды /list /week работают как обычно.",
            )
            return
        log.info(
            "assistant answered: %d chars, %d proposals", len(reply.text), len(reply.proposals)
        )
        rows: list[list[InlineKeyboardButton]] = []
        batch = secrets.token_urlsafe(6)
        tokens = []
        for proposal in reply.proposals:
            token = secrets.token_urlsafe(6)
            self._proposals[token] = proposal
            tokens.append(token)
            mark = "⚠️ " if proposal.from_document else ""
            rows.append([_button(f"{mark}{proposal.label}", f"ap:{token}")])
        if rows:
            self._batches[batch] = tokens
            rows.append([_button("✖️ Ничего не надо", f"rj:{batch}")])
        text_out = escape(reply.text) or "…"
        if rows:
            text_out += "\n\n👇 <i>Без нажатия ничего не меняется.</i>"
        if any(p.from_document for p in reply.proposals):
            text_out += (
                "\n\n⚠️ Кнопки с ⚠️ появились после чтения текста задания — это может быть"
                " инструкция из методички, а не ваша просьба. Нажимайте, только если согласны."
            )
        await bot.send_message(
            chat_id,
            text_out,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows) if rows else None,
        )

    async def _apply(self, bot: Bot, chat_id: int, proposal: Proposal) -> None:
        """Выполнить подтверждённое предложение агента — ровно то, что было на кнопке."""
        try:
            view = await self.queries.view(proposal.lab_id) if proposal.lab_id else None
        except NotFoundError:
            await bot.send_message(chat_id, "Лаба уже удалена 🤷")
            return
        if view is not None:
            self._current[chat_id] = view.lab.id
        params = proposal.params
        if proposal.kind == "status" and view:
            await self._set_status(bot, chat_id, view, Status(params["status"]), undo=True)
        elif proposal.kind == "edit" and view:
            if self.editor is None:
                await bot.send_message(chat_id, "Правка не настроена.")
                return
            try:
                result = await self.editor.edit(view.lab.id, **params)
            except AlreadyExistsError:
                await bot.send_message(chat_id, "🙅 Такая лаба (предмет + номер) уже есть.")
                return
            await bot.send_message(
                chat_id,
                f"✏️ <b>Готово!</b> {fmt.lab_header(result.subject, result.lab)}\n"
                + "\n".join(f"• {escape(change)}" for change in result.changes),
                reply_markup=_open_card(result.lab.id),
            )
        elif proposal.kind == "requirements":
            update = await self.requirements.set(
                RequirementScope(params["scope"]), params["text"], params["target_id"]
            )
            await bot.send_message(
                chat_id,
                f"📋 <b>Требования сохранены</b> (версия {update.requirement.version})\n"
                f"{_quote_text(params['text'])}\n"
                f"Промптов пересобрано: {len(update.rebuilt_lab_ids)}",
            )
        elif proposal.kind == "prompt" and view:
            await self._send_prompt(bot, chat_id, view)
        elif proposal.kind == "assignment" and view:
            await self._send_assignment(bot, chat_id, view)
        elif proposal.kind == "card" and view:
            text, markup = await self._lab_view(view.lab.id)
            await bot.send_message(chat_id, text, reply_markup=markup)
        elif proposal.kind == "delete" and view:
            await self._ask_delete(bot, chat_id, view)

    async def _remove_button(self, query: CallbackQuery, data: str) -> None:
        """Убрать из клавиатуры применённую кнопку; если остались только «ничего не надо» — всё."""
        message = query.message
        if not isinstance(message, Message) or message.reply_markup is None:
            return
        rows = [
            [b for b in row if b.callback_data != data]
            for row in message.reply_markup.inline_keyboard
        ]
        rows = [row for row in rows if row]
        useful = [row for row in rows if not (row[0].callback_data or "").startswith(f"{NAME}:rj:")]
        with contextlib.suppress(Exception):
            await message.edit_reply_markup(
                reply_markup=InlineKeyboardMarkup(inline_keyboard=rows) if useful else None
            )

    async def cmd_delete(self, message: Message, command: CommandObject) -> None:
        view = await self._resolve_one(message, command.args, "delete")
        if view is None:
            return
        await message.answer(
            f"Удалить {fmt.lab_header(view.subject, view.lab)}?\n"
            "Удалятся запись, локальная папка и папка на Диске (вместе с отчётами).",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="🗑 Удалить", callback_data=f"{NAME}:del:{view.lab.id}"
                        ),
                        InlineKeyboardButton(
                            text="Оставить", callback_data=f"{NAME}:keep:{view.lab.id}"
                        ),
                    ]
                ]
            ),
        )

    # ------------------------------------------------------------------ навигация

    async def _root_view(self) -> tuple[str, InlineKeyboardMarkup | None]:
        views = await self.queries.all_labs()
        if not views:
            return (
                "Пока нет ни одной лабы. Пришлите файл задания или перешлите пост"
                f" с подписью {fmt.PROMPT_EXAMPLE}.",
                None,
            )
        open_views = [v for v in views if v.lab.status in OPEN_STATUSES]
        text = (
            "📚 <b>Лабы</b>\n\n"
            + fmt.subjects_overview(views)
            + "\n\n🔜 <b>Впереди</b>\n\n"
            + fmt.lab_list(open_views, self.tz, empty="Всё сдано — красота 🎉")
            + "\n\nВыберите предмет 👇"
        )
        by_subject: dict[int, list[LabView]] = {}
        for view in views:
            by_subject.setdefault(view.lab.subject_id, []).append(view)
        rows = []
        for subject_id, items in sorted(
            by_subject.items(), key=lambda kv: kv[1][0].subject.casefold()
        ):
            opened = sum(1 for v in items if v.lab.status in OPEN_STATUSES)
            rows.append(
                [_button(f"{items[0].subject} · {opened}/{len(items)}", f"nv:s:{subject_id}")]
            )
        rows.append([_button(MENU_WEEK, "nv:w")])
        return text, InlineKeyboardMarkup(inline_keyboard=rows)

    async def _subject_view(self, subject_id: int) -> tuple[str, InlineKeyboardMarkup]:
        views = [v for v in await self.queries.all_labs() if v.lab.subject_id == subject_id]
        views.sort(key=lambda v: v.lab.number)
        name = views[0].subject if views else "Предмет"
        rows = [[_button(_lab_button_text(v, self.tz), f"nv:l:{v.lab.id}")] for v in views]
        rows.append([_button("⬅️ Все предметы", "nv:r")])
        text = f"<b>{escape(name)}</b> — выберите лабу" if views else "В этом предмете лаб нет."
        return text, InlineKeyboardMarkup(inline_keyboard=rows)

    async def _lab_view(self, lab_id: int) -> tuple[str, InlineKeyboardMarkup]:
        view = await self.queries.view(lab_id)
        lab = view.lab
        requirements = [
            ("к лабе", await self.db.requirements.current(RequirementScope.LAB, lab.id)),
            (
                "к предмету",
                await self.db.requirements.current(RequirementScope.SUBJECT, lab.subject_id),
            ),
            ("общие", await self.db.requirements.current(RequirementScope.GLOBAL)),
        ]
        text = fmt.lab_card(view, self.tz, self.disk_root, requirements)
        text += "\n\n✏️ Чтобы поправить, просто напишите: «срок 30.10», «это ЛР 4», «тема — …»."
        statuses = [s for s in CARD_STATUSES if s is not lab.status]
        rows = [
            [
                _button(f"{STATUS_ICONS[s]} {STATUS_TITLES[s]}", f"ss:{lab.id}:{s.value}")
                for s in statuses
            ],
            [
                _button("📄 Промпт", f"pf:{lab.id}"),
                _button("📝 Задание", f"tf:{lab.id}"),
                _button("📋 Требования", f"rq:{lab.id}"),
            ],
            [_button("✏️ Изменить", f"ed:{lab.id}"), _button("🗑 Удалить", f"dl:{lab.id}")],
            [_button("⬅️ К предмету", f"nv:s:{lab.subject_id}"), _button("📚 Все", "nv:r")],
        ]
        return text, InlineKeyboardMarkup(inline_keyboard=rows)

    async def _navigate(self, query: CallbackQuery, chat_id: int, target: list[str]) -> None:
        try:
            if target[:1] == ["s"] and len(target) == 2:
                text, markup = await self._subject_view(int(target[1]))
            elif target[:1] == ["l"] and len(target) == 2:
                self._current[chat_id] = int(target[1])
                text, markup = await self._lab_view(int(target[1]))
            elif target[:1] == ["w"]:
                text = fmt.week(await self.queries.week(), self.tz)
                markup = InlineKeyboardMarkup(inline_keyboard=[[_button("⬅️ Все предметы", "nv:r")]])
            else:
                text, markup = await self._root_view()  # type: ignore[assignment]
        except (ValueError, NotFoundError):
            text, markup = await self._root_view()  # type: ignore[assignment]
        await _show(query, text, markup)

    async def _card_action(
        self, query: CallbackQuery, bot: Bot, chat_id: int, action: str, args: list[str]
    ) -> None:
        try:
            view = await self.queries.view(int(args[0]))
        except (ValueError, NotFoundError):
            await bot.send_message(chat_id, "Лаба не найдена — возможно, удалена. /list")
            return
        lab = view.lab
        self._current[chat_id] = lab.id
        log.info("card action %s on lab %s", action, lab.id)
        if action == "ss" and len(args) == 2:
            with contextlib.suppress(ValueError):
                await self.queries.set_status(lab.id, Status(args[1]), Source.TELEGRAM)
            text, markup = await self._lab_view(lab.id)
            await _show(query, text, markup)
        elif action == "pf":
            await self._send_prompt(bot, chat_id, view)
        elif action == "tf":
            await self._send_assignment(bot, chat_id, view)
        elif action == "rq":
            current = await self.requirements.current(RequirementScope.LAB, lab.id)
            body = escape(current.text) if current and current.text else "пока нет"
            await bot.send_message(
                chat_id,
                f"📋 <b>Требования к {escape(view.subject)} · ЛР {lab.number}:</b>\n{body}\n\n"
                "Чтобы задать или заменить — напишите, например:"
                " «требования: вариант 7, код на C11».",
            )
        elif action == "ed":
            await bot.send_message(
                chat_id,
                f"✏️ Правим {fmt.lab_header(view.subject, lab)}.\n"
                "Напишите, что изменить, например:\n"
                "• «это ЛР 4, а не 3»\n• «перенеси в предмет Нейронные сети»\n"
                "• «срок 30.10 18:00» или «убери срок»\n• «тема — RL-агент для торговли»",
            )
        elif action == "dl":
            await self._ask_delete(bot, chat_id, view)

    async def _send_prompt(self, bot: Bot, chat_id: int, view: LabView) -> None:
        await self.builder.write(view.lab.id)
        _, prompt, _ = await self.builder.build(view.lab.id)
        await bot.send_document(
            chat_id,
            BufferedInputFile(prompt.encode("utf-8"), filename=PROMPT_FILE),
            caption=f"📄 Промпт для исполнителя\n{fmt.lab_header(view.subject, view.lab)}\n"
            "Отдайте его Claude или другому агенту целиком.",
        )

    async def _send_assignment(self, bot: Bot, chat_id: int, view: LabView) -> None:
        lab = view.lab
        text = (
            self.builder.storage.read_text(f"{lab.folder}/{ASSIGNMENT_MARKDOWN}")
            if lab.folder
            else None
        )
        if not text:
            await bot.send_message(
                chat_id, "📄 Текст задания не извлёкся (скан) — оригинал на Диске."
            )
            return
        await bot.send_document(
            chat_id,
            BufferedInputFile(text.encode("utf-8"), filename="assignment.md"),
            caption=f"📝 Задание\n{fmt.lab_header(view.subject, lab)}",
        )

    async def _ask_delete(self, bot: Bot, chat_id: int, view: LabView) -> None:
        await bot.send_message(
            chat_id,
            f"🗑 Удалить {fmt.lab_header(view.subject, view.lab)}?\n"
            "Удалятся запись, локальная папка и папка на Диске — вместе с отчётами.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        _button("🗑 Да, удалить", f"del:{view.lab.id}"),
                        _button("Оставить", f"keep:{view.lab.id}"),
                    ]
                ]
            ),
        )

    # ------------------------------------------------------------------ помощники

    async def _set_status(
        self, bot: Bot, chat_id: int, view: LabView, status: Status, *, undo: bool = False
    ) -> None:
        old = view.lab.status
        lab = await self.queries.set_status(view.lab.id, status, Source.TELEGRAM)
        markup = None
        if undo and old is not status:
            markup = InlineKeyboardMarkup(
                inline_keyboard=[[_button("↩️ Вернуть как было", f"st:{lab.id}:{old.value}")]]
            )
        await bot.send_message(
            chat_id, fmt.status_changed(view.subject, lab, old), reply_markup=markup
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


CARD_STATUSES = (Status.TODO, Status.IN_PROGRESS, Status.DONE, Status.SUBMITTED, Status.DEFENDED)


def _button(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text[:64], callback_data=f"{NAME}:{data}")


def _lab_button_text(view: LabView, tz: ZoneInfo) -> str:
    lab = view.lab
    due = f" · {fmt.deadline(lab.deadline, tz, short=True)}" if lab.deadline else ""
    return f"{STATUS_ICONS[lab.status]} ЛР {lab.number} · {lab.title[:30]}{due}"


async def _show(query: CallbackQuery, text: str, markup: InlineKeyboardMarkup | None) -> None:
    """Навигация — в том же сообщении; если отредактировать нельзя, присылаем новое."""
    message = query.message
    if isinstance(message, Message):
        try:
            await message.edit_text(text, reply_markup=markup)
            return
        except Exception as exc:  # «message is not modified» и т. п.
            if "not modified" in str(exc):
                return
        await message.answer(text, reply_markup=markup)


def _open_card(lab_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[_button("📂 Открыть карточку", f"nv:l:{lab_id}")]]
    )


def _quote_text(text: str) -> str:
    return f"<blockquote>{escape(text)}</blockquote>"
