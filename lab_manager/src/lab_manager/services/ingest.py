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
from lab_manager.extraction import LabExtraction, LabExtractor
from lab_manager.storage import ASSIGNMENT_DIR, LabStorage

DISK_SYNC_JOB = "disk_sync"
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


@dataclass(frozen=True, slots=True)
class IngestResult:
    outcome: IngestOutcome
    lab: Lab | None = None
    subject: Subject | None = None
    file: FileRecord | None = None
    extraction: LabExtraction | None = None
    warnings: tuple[IngestWarning, ...] = field(default=())

    @property
    def missing(self) -> list[str]:
        return self.extraction.missing_fields if self.extraction else []


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
        """Задание, присланное обычным сообщением, хранится как Markdown-файл."""
        return await self.ingest(
            data=text.encode("utf-8"),
            filename=TEXT_MESSAGE_FILENAME,
            prompt=prompt,
            source=source,
        )

    async def ingest(
        self,
        *,
        data: bytes,
        filename: str,
        prompt: str | None,
        mime_type: str | None = None,
        source: Source = Source.TELEGRAM,
    ) -> IngestResult:
        if len(data) > self.max_bytes:
            raise FileTooLargeError(f"файл больше {self.max_bytes // (1024 * 1024)} МБ")
        if not self.converter.supports(filename):
            raise UnsupportedFormatError(f"формат файла {filename!r} не поддерживается")

        sha256 = hashlib.sha256(data).hexdigest()
        if duplicate := await self._duplicate(sha256):
            return duplicate

        document = await self.converter.convert(data, filename)
        today = self.db.clock().astimezone(self.tz).date()
        known_subjects = [subject.name for subject in await self.db.subjects.list()]
        extraction = await self.extractor.extract(
            markdown=document.markdown,
            prompt=prompt,
            today=today,
            tz=self.tz,
            known_subjects=known_subjects,
        )
        if extraction.missing_fields:
            return IngestResult(IngestOutcome.NEEDS_INFO, extraction=extraction)
        assert extraction.subject is not None and extraction.number is not None  # noqa: S101

        warnings = self._warnings(document.has_text, extraction)
        async with self.db.transaction():
            if duplicate := await self._duplicate(sha256):
                return duplicate
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
            self.storage.write_text(folder, ASSIGNMENT_MARKDOWN, document.markdown)
            # Transactional outbox: задача на выгрузку появляется только вместе с файлом.
            await self.db.jobs.enqueue(
                DISK_SYNC_JOB,
                {"lab_id": lab.id},
                idempotency_key=f"{DISK_SYNC_JOB}:file:{record.id}",
            )

        return IngestResult(outcome, lab, subject, record, extraction, warnings)

    async def _duplicate(self, sha256: str) -> IngestResult | None:
        record = await self.db.files.get_by_sha256(sha256)
        if record is None:
            return None
        lab = await self.db.labs.get(record.lab_id)
        subject = await self.db.subjects.get(lab.subject_id) if lab else None
        return IngestResult(IngestOutcome.DUPLICATE, lab, subject, record)

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
        return await self.db.labs.update(lab.id, **changes) if changes else lab  # type: ignore[arg-type]

    @staticmethod
    def _warnings(has_text: bool, extraction: LabExtraction) -> tuple[IngestWarning, ...]:
        warnings: Sequence[IngestWarning] = [
            *([] if has_text else [IngestWarning.NO_TEXT]),
            *([] if extraction.deadline else [IngestWarning.NO_DEADLINE]),
        ]
        return tuple(warnings)
