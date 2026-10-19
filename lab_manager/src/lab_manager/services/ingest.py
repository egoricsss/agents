"""Приём задания: sha256 → markitdown → извлечение полей → БД + папки (атомарно).

Медленные шаги (конвертация, LLM) выполняются до транзакции; дубликат перепроверяется
внутри неё, поэтому параллельная отправка одного файла создаёт лабу ровно один раз.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from zoneinfo import ZoneInfo

from lab_manager.converter import DocumentConverter
from lab_manager.db import Database
from lab_manager.domain import FileRecord, Lab, Source, Subject
from lab_manager.errors import FileTooLargeError, UnsupportedFormatError
from lab_manager.extraction import REQUIRED_FIELDS, LabExtraction, LabExtractor
from lab_manager.storage import ASSIGNMENT_DIR, LabStorage

DISK_SYNC_JOB = "disk_sync"
BUILD_PROMPT_JOB = "build_prompt"
MAX_FILE_BYTES = 20 * 1024 * 1024  # лимит Bot API на скачивание файла
ASSIGNMENT_MARKDOWN = f"{ASSIGNMENT_DIR}/assignment.md"
TEXT_MESSAGE_FILENAME = "задание.md"


class IngestOutcome(StrEnum):
    CREATED = "created"
    NEW_VERSION = "new_version"
    DUPLICATE = "duplicate"
    NEEDS_INFO = "needs_info"


class IngestWarning(StrEnum):
    NO_TEXT = "no_text"  # скан без текстового слоя: файл сохранён, текст не извлечён
    NO_DEADLINE = "no_deadline"
    LOW_CONFIDENCE = "low_confidence"  # расхождение номеров или разбор без LLM — подтвердить


LOW_CONFIDENCE = 0.5


@dataclass(frozen=True, slots=True)
class IngestedLab:
    """Результат по одной лабе из сообщения."""

    outcome: IngestOutcome
    lab: Lab
    subject: Subject | None
    file: FileRecord
    extraction: LabExtraction | None = None
    warnings: tuple[IngestWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Результат по сообщению: одна или несколько лаб (`items`).

    `lab`, `subject`, `file`, `extraction`, `warnings` — первой лабы, для простого случая.
    """

    outcome: IngestOutcome
    items: tuple[IngestedLab, ...] = ()
    extractions: tuple[LabExtraction, ...] = field(default=())

    @property
    def lab(self) -> Lab | None:
        return self.items[0].lab if self.items else None

    @property
    def subject(self) -> Subject | None:
        return self.items[0].subject if self.items else None

    @property
    def file(self) -> FileRecord | None:
        return self.items[0].file if self.items else None

    @property
    def extraction(self) -> LabExtraction | None:
        return self.extractions[0] if self.extractions else None

    @property
    def warnings(self) -> tuple[IngestWarning, ...]:
        return self.items[0].warnings if self.items else ()

    @property
    def missing(self) -> list[str]:
        fields = {name for extraction in self.extractions for name in extraction.missing_fields}
        return [name for name in REQUIRED_FIELDS if name in fields]

    @property
    def labs(self) -> list[Lab]:
        return [item.lab for item in self.items]


@dataclass(frozen=True, slots=True)
class IngestPreview:
    """Разбор без сохранения. `duplicate` — если такое содержимое уже обработано."""

    duplicate: IngestResult | None = None
    extractions: tuple[LabExtraction, ...] = ()
    has_text: bool = True

    @property
    def missing(self) -> list[str]:
        fields = {name for extraction in self.extractions for name in extraction.missing_fields}
        return [name for name in REQUIRED_FIELDS if name in fields]

    @property
    def needs_confirmation(self) -> bool:
        return any(extraction.confidence <= LOW_CONFIDENCE for extraction in self.extractions)


class IngestService:
    def __init__(
        self,
        *,
        db: Database,
        storage: LabStorage,
        converter: DocumentConverter,
        extractor: LabExtractor,
        tz: ZoneInfo,
        max_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        self.db = db
        self.storage = storage
        self.converter = converter
        self.extractor = extractor
        self.tz = tz
        self.max_bytes = max_bytes

    async def ingest_text(
        self, text: str, *, prompt: str | None, source: Source = Source.TELEGRAM
    ) -> IngestResult:
        """Задание без файла (например, пересланный пост) хранится как Markdown-файл."""
        return await self.ingest(
            data=text.encode("utf-8"),
            filename=TEXT_MESSAGE_FILENAME,
            prompt=prompt,
            source=source,
        )

    async def preview(self, *, data: bytes, filename: str, prompt: str | None) -> "IngestPreview":
        """Разобрать без сохранения: что будет создано/обновлено (для подтверждения кнопкой).

        Повторный разбор того же содержимого берётся из кеша конвертера и LLM.
        """
        self._check(data, filename)
        sha256 = hashlib.sha256(data).hexdigest()
        if duplicate := await self._duplicate(sha256):
            return IngestPreview(duplicate=duplicate)
        document = await self.converter.convert(data, filename)
        return IngestPreview(
            extractions=await self._extract(document.markdown, prompt),
            has_text=document.has_text,
        )

    async def ingest(
        self,
        *,
        data: bytes,
        filename: str,
        prompt: str | None,
        mime_type: str | None = None,
        source: Source = Source.TELEGRAM,
        extractions: Sequence[LabExtraction] | None = None,
    ) -> IngestResult:
        """Одно сообщение → одна или несколько лаб: новые создаются, существующие обновляются.

        `extractions` — уже подтверждённый владельцем разбор (из `preview`), повторно не извлекаем.
        """
        self._check(data, filename)
        sha256 = hashlib.sha256(data).hexdigest()
        if duplicate := await self._duplicate(sha256):
            return duplicate

        document = await self.converter.convert(data, filename)
        found = (
            tuple(extractions) if extractions else await self._extract(document.markdown, prompt)
        )
        if not found or any(extraction.missing_fields for extraction in found):
            return IngestResult(IngestOutcome.NEEDS_INFO, extractions=found)

        items: list[IngestedLab] = []
        async with self.db.transaction():
            if duplicate := await self._duplicate(sha256):
                return duplicate
            for extraction in found:
                items.append(
                    await self._store(
                        extraction,
                        document.markdown,
                        document.has_text,
                        data=data,
                        sha256=sha256,
                        filename=filename,
                        mime_type=mime_type,
                        source=source,
                    )
                )
        outcomes = {item.outcome for item in items}
        outcome = (
            IngestOutcome.CREATED
            if IngestOutcome.CREATED in outcomes
            else IngestOutcome.NEW_VERSION
        )
        return IngestResult(outcome, tuple(items), found)

    def _check(self, data: bytes, filename: str) -> None:
        if len(data) > self.max_bytes:
            raise FileTooLargeError(f"файл больше {self.max_bytes // (1024 * 1024)} МБ")
        if not self.converter.supports(filename):
            raise UnsupportedFormatError(f"формат файла {filename!r} не поддерживается")

    async def _extract(self, markdown: str, prompt: str | None) -> tuple[LabExtraction, ...]:
        today = self.db.clock().astimezone(self.tz).date()
        known_subjects = [subject.name for subject in await self.db.subjects.list()]
        return tuple(
            await self.extractor.extract_all(
                markdown=markdown,
                prompt=prompt,
                today=today,
                tz=self.tz,
                known_subjects=known_subjects,
            )
        )

    async def _store(
        self,
        extraction: LabExtraction,
        markdown: str,
        has_text: bool,
        *,
        data: bytes,
        sha256: str,
        filename: str,
        mime_type: str | None,
        source: Source,
    ) -> IngestedLab:
        assert extraction.subject is not None and extraction.number is not None  # noqa: S101
        subject = await self.db.subjects.get_or_create(extraction.subject)
        lab = await self.db.labs.find(subject.id, extraction.number)
        if lab is None:
            outcome = IngestOutcome.CREATED
            title = extraction.title or ""
            lab = await self.db.labs.create(
                subject_id=subject.id,
                number=extraction.number,
                title=title or f"Лабораторная работа {extraction.number}",
                deadline=extraction.deadline,
                summary=extraction.summary,
                folder=self.storage.lab_folder(subject.name, extraction.number, title),
                artifacts=extraction.requirements,
                source=source,
            )
        else:
            outcome = IngestOutcome.NEW_VERSION
            lab = await self._refresh(lab, extraction)
        folder = lab.folder or self.storage.lab_folder(subject.name, lab.number, lab.title)

        version = await self._next_version(lab.id)
        local_path = self.storage.save_original(folder, version, filename, data)
        record = await self.db.files.add(
            lab_id=lab.id,
            sha256=sha256,
            original_name=filename,
            mime_type=mime_type,
            size_bytes=len(data),
            local_path=local_path,
        )
        self.storage.write_text(folder, ASSIGNMENT_MARKDOWN, markdown)
        # Transactional outbox: задачи появляются только вместе с файлом.
        # build_prompt идёт раньше disk_sync (очередь FIFO), чтобы промпт уехал на Диск сразу.
        await self.db.jobs.enqueue(
            BUILD_PROMPT_JOB,
            {"lab_id": lab.id},
            idempotency_key=f"{BUILD_PROMPT_JOB}:file:{record.id}",
        )
        await self.db.jobs.enqueue(
            DISK_SYNC_JOB,
            {"lab_id": lab.id},
            idempotency_key=f"{DISK_SYNC_JOB}:file:{record.id}",
        )
        return IngestedLab(
            outcome, lab, subject, record, extraction, self._warnings(has_text, extraction)
        )

    async def _duplicate(self, sha256: str) -> IngestResult | None:
        """Это содержимое уже обработано (в одной или нескольких лабах) — ничего не делаем."""
        items = []
        for record in await self.db.files.list_by_sha256(sha256):
            lab = await self.db.labs.get(record.lab_id)
            if lab is None:
                continue
            subject = await self.db.subjects.get(lab.subject_id)
            items.append(IngestedLab(IngestOutcome.DUPLICATE, lab, subject, record))
        return IngestResult(IngestOutcome.DUPLICATE, tuple(items)) if items else None

    async def _next_version(self, lab_id: int) -> int:
        latest = await self.db.files.latest_for_lab(lab_id)
        return latest.version + 1 if latest else 1

    async def _refresh(self, lab: Lab, extraction: LabExtraction) -> Lab:
        """Новая версия задания: обновляем то, что извлекли. Папка не меняется."""
        changes: dict[str, object] = {}
        if extraction.title:
            changes["title"] = extraction.title
        if extraction.deadline is not None:
            changes["deadline"] = extraction.deadline
        if extraction.summary:
            changes["summary"] = extraction.summary
        if extraction.requirements:
            changes["artifacts"] = extraction.requirements
        return await self.db.labs.update(lab.id, **changes) if changes else lab  # type: ignore[arg-type]

    @staticmethod
    def _warnings(has_text: bool, extraction: LabExtraction) -> tuple[IngestWarning, ...]:
        warnings: Sequence[IngestWarning] = [
            *([] if has_text else [IngestWarning.NO_TEXT]),
            *([] if extraction.deadline else [IngestWarning.NO_DEADLINE]),
            *([IngestWarning.LOW_CONFIDENCE] if extraction.confidence <= LOW_CONFIDENCE else []),
        ]
        return tuple(warnings)
