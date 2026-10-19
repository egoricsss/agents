"""Агент лаб в боте: команды, приём файлов и пересланных постов, переспрос, подтверждения."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, SendDocument, SendMessage
from aiogram.types import InlineKeyboardMarkup, Update

from lab_manager.bot import LabAgent
from lab_manager.converter import DocumentConverter
from lab_manager.db import Database
from lab_manager.domain import RequirementScope, Status
from lab_manager.extraction.rules import RuleBasedExtractor
from lab_manager.gateway import Gateway
from lab_manager.services.executor_prompt import ExecutorPromptBuilder
from lab_manager.services.ingest import IngestService
from lab_manager.services.labs import LabQueries
from lab_manager.services.requirements import RequirementsService
from lab_manager.storage import LabStorage
from tests.fixtures import ASSIGNMENT_TEXT, RL_POST, make_pdf
from tests.telegram.fake_telegram import (
    BOT_TOKEN,
    OWNER_ID,
    FakeTelegram,
    callback_update,
    document,
    message_update,
)

MSK = ZoneInfo("Europe/Moscow")


@dataclass
class Env:
    db: Database
    session: FakeTelegram
    bot: Bot
    dp: Dispatcher
    gateway: Gateway
    storage: LabStorage

    async def send(self, update: Update) -> None:
        await self.dp.feed_update(self.bot, update)
        await self.gateway.drain()

    async def say(self, text: str, **kwargs: Any) -> None:  # noqa: ANN401
        await self.send(message_update(text, **kwargs))

    def last(self) -> str:
        return self.session.last_text()

    def last_message(self) -> SendMessage:
        return [r for r in self.session.requests if isinstance(r, SendMessage)][-1]

    def buttons(self) -> list[tuple[str, str]]:
        markup = self.last_message().reply_markup
        assert isinstance(markup, InlineKeyboardMarkup)
        return [(b.text, b.callback_data or "") for row in markup.inline_keyboard for b in row]


@pytest.fixture
async def env(db: Database, tmp_path: Path) -> AsyncIterator[Env]:
    storage = LabStorage(tmp_path / "Лабы")
    builder = ExecutorPromptBuilder(db=db, storage=storage, disk_root="app:/Лабы", tz=MSK)
    agent = LabAgent(
        db=db,
        ingest=IngestService(
            db=db,
            storage=storage,
            converter=DocumentConverter(),
            extractor=RuleBasedExtractor(),
            tz=MSK,
            max_bytes=1024 * 1024,
        ),
        queries=LabQueries(db, MSK),
        requirements=RequirementsService(db=db, builder=builder),
        builder=builder,
        tz=MSK,
    )
    gateway = Gateway([agent], owner_id=OWNER_ID, burst_window=0.01)
    session = FakeTelegram()
    bot = Bot(BOT_TOKEN, session=session)
    yield Env(db, session, bot, gateway.dispatcher(), gateway, storage)


async def add_lab(env: Env, prompt: str = "ОС, лаба 3, до 25.10") -> None:
    await env.send(
        message_update(
            caption=prompt, document=document(env.session, ASSIGNMENT_TEXT.encode(), "ЛР3.txt")
        )
    )


# --- приём заданий ---


async def test_document_with_caption_creates_lab(env: Env) -> None:
    await add_lab(env)

    text = env.last()
    assert "🆕" in text
    assert "ОС" in text and "ЛР 3" in text and "Процессы и потоки" in text
    assert "25.10.2026 23:59" in text
    [lab] = await env.db.labs.list()
    assert lab.number == 3


async def test_same_file_again_is_reported_as_duplicate(env: Env) -> None:
    await add_lab(env)
    await add_lab(env)

    assert "Уже есть" in env.last()
    assert len(await env.db.labs.list()) == 1


async def test_forwarded_post_with_owner_caption_before_it(env: Env) -> None:
    # Telegram присылает комментарий к пересылке раньше самой пересылки.
    await env.dp.feed_update(env.bot, message_update("Нейронные сети, ЛР 3"))
    await env.dp.feed_update(env.bot, message_update(RL_POST, forwarded_from_channel="Кафедра ИИ"))
    await env.gateway.drain()

    assert "🆕" in env.last()
    [lab] = await env.db.labs.list()
    assert lab.number == 3
    subject = await env.db.subjects.get(lab.subject_id)
    assert subject is not None and subject.name == "Нейронные сети"


async def test_forwarded_post_without_caption_asks_and_reply_completes(env: Env) -> None:
    await env.say(RL_POST, forwarded_from_user="Преподаватель")

    question = env.last_message()
    assert "Не хватает" in question.text and "предмет" in question.text
    assert await env.db.labs.list() == []

    await env.say("Нейронные сети, ЛР 3", reply_to_message_id=1000)  # первый ответ фейка — id 1000

    assert "🆕" in env.last()
    assert len(await env.db.labs.list()) == 1


async def test_pending_question_is_answered_by_next_plain_text(env: Env) -> None:
    await env.say(RL_POST, forwarded_from_user="Преподаватель")

    await env.say("Нейронные сети, ЛР 3")

    assert "🆕" in env.last()


async def test_low_confidence_asks_confirmation_before_saving(env: Env) -> None:
    # в посте «ЛР 3», в подписи — «ЛР 4»
    await env.dp.feed_update(env.bot, message_update("Нейронные сети, ЛР 4"))
    await env.dp.feed_update(env.bot, message_update(RL_POST, forwarded_from_channel="Кафедра"))
    await env.gateway.drain()

    assert "Проверьте" in env.last()
    assert await env.db.labs.list() == []
    buttons = env.buttons()
    ok = next(data for text, data in buttons if data.startswith("lab:ok:"))
    assert any(data.startswith("lab:no:") for _, data in buttons)

    await env.send(callback_update(ok))

    assert "🆕" in env.last()
    [lab] = await env.db.labs.list()
    assert lab.number == 4
    assert any(isinstance(r, AnswerCallbackQuery) for r in env.session.requests)


async def test_low_confidence_cancel(env: Env) -> None:
    await env.dp.feed_update(env.bot, message_update("Нейронные сети, ЛР 4"))
    await env.dp.feed_update(env.bot, message_update(RL_POST, forwarded_from_channel="Кафедра"))
    await env.gateway.drain()
    cancel = next(data for _, data in env.buttons() if data.startswith("lab:no:"))

    await env.send(callback_update(cancel))

    assert "Отменено" in env.last()
    assert await env.db.labs.list() == []


async def test_expired_confirmation(env: Env) -> None:
    await env.send(callback_update("lab:ok:unknown"))

    assert "устарел" in env.last()


async def test_album_with_one_caption_creates_each_lab(env: Env) -> None:
    first = document(env.session, b"Lab text number one, long enough", "a.txt")
    second = document(env.session, b"Lab text number two, long enough", "b.txt")
    await env.dp.feed_update(env.bot, message_update(caption="ОС, лабы 1 и 2", document=first))
    await env.dp.feed_update(env.bot, message_update(document=second))
    await env.gateway.drain()

    labs = await env.db.labs.list()
    assert sorted(lab.number for lab in labs) == [1, 2]


async def test_scan_warns_about_missing_text(env: Env) -> None:
    await env.send(
        message_update(
            caption="ОС, лаба 1", document=document(env.session, make_pdf([]), "scan.pdf")
        )
    )

    assert "скан" in env.last().lower()


async def test_unsupported_and_too_large_files(env: Env) -> None:
    await env.send(
        message_update(caption="ОС, лаба 1", document=document(env.session, b"MZ", "x.exe"))
    )
    assert "не поддерживается" in env.last()

    big = document(env.session, b"x" * (1024 * 1024 + 1), "big.txt")
    await env.send(message_update(caption="ОС, лаба 1", document=big))
    assert "МБ" in env.last()


# --- команды ---


async def test_list_and_week(env: Env) -> None:
    await env.say("/list")
    assert "Пока нет" in env.last()

    await add_lab(env)
    await env.say("/list")
    assert "ОС" in env.last() and "ЛР 3" in env.last()

    await env.say("/week")
    assert "ЛР 3" in env.last()  # срок 25.10 — на этой неделе (19–25.10)


async def test_lab_card(env: Env) -> None:
    await add_lab(env)
    [lab] = await env.db.labs.list()
    await env.db.requirements.add(RequirementScope.LAB, "Вариант 7", target_id=lab.id)

    await env.say("/lab ОС 3")

    text = env.last()
    assert "Процессы и потоки" in text and "Вариант 7" in text and "к выполнению" in text


async def test_lab_not_found_and_ambiguous(env: Env) -> None:
    await env.say("/lab ОС 9")
    assert "Не нашёл" in env.last()

    await add_lab(env)
    await add_lab(env, "БД, лаба 3")  # тот же файл — дубликат, поэтому другой текст:
    await env.send(
        message_update(
            caption="БД, лаба 3", document=document(env.session, b"Another lab, long text", "x.txt")
        )
    )
    await env.say("/lab 3")
    assert "Уточните" in env.last()


async def test_set_status_by_command_and_buttons(env: Env) -> None:
    await add_lab(env)
    [lab] = await env.db.labs.list()

    await env.say("/status ОС 3 сделано")
    assert "сделана" in env.last()
    assert (await env.db.labs.get(lab.id)).status is Status.DONE  # type: ignore[union-attr]

    await env.say("/status ОС 3")
    in_progress = next(data for text, data in env.buttons() if "в работе" in text)
    await env.send(callback_update(in_progress))
    assert (await env.db.labs.get(lab.id)).status is Status.IN_PROGRESS  # type: ignore[union-attr]
    history = await env.db.labs.history(lab.id)
    assert [h.source.value for h in history][-1] == "telegram"


async def test_unknown_status_word(env: Env) -> None:
    await add_lab(env)

    await env.say("/status ОС 3 почти")

    assert "Не понял статус" in env.last()


async def test_requirements_levels(env: Env) -> None:
    await add_lab(env)
    [lab] = await env.db.labs.list()

    await env.say("/req ОС 3: Вариант 7")
    await env.say("/req ОС: Только Linux")
    await env.say("/req все: Отчёт по ГОСТ 7.32")

    prompt = env.storage.read_text(f"{lab.folder}/prompt.md")
    assert prompt is not None
    assert "Вариант 7" in prompt and "Только Linux" in prompt and "ГОСТ 7.32" in prompt

    await env.say("/req ОС 3")
    assert "Вариант 7" in env.last()


async def test_prompt_is_sent_as_file(env: Env) -> None:
    await add_lab(env)

    await env.say("/prompt ОС 3")

    [request] = [r for r in env.session.requests if isinstance(r, SendDocument)]
    assert request.document.filename == "prompt.md"  # type: ignore[union-attr]


async def test_commands_without_arguments_explain_usage(env: Env) -> None:
    for command in ("/lab", "/status", "/req", "/prompt"):
        await env.say(command)
        assert "Пример" in env.last()


async def test_html_in_user_data_is_escaped(env: Env) -> None:
    await add_lab(env, "<b>ОС</b>, лаба 3, <i>Тема</i>")

    assert "&lt;b&gt;ОС&lt;/b&gt;" in env.last()


async def test_plain_text_without_context_gets_hint(env: Env) -> None:
    await env.say("привет")

    assert "/help" in env.last()
