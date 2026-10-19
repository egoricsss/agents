"""Сервис приёма заданий: идемпотентность, версии, переспрос, атомарность."""

import asyncio
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lab_manager.converter import DocumentConverter
from lab_manager.db import Database
from lab_manager.domain import Source
from lab_manager.errors import FileTooLargeError, UnsupportedFormatError
from lab_manager.extraction import LabExtraction
from lab_manager.extraction.rules import RuleBasedExtractor
from lab_manager.services.ingest import (
    IngestOutcome,
    IngestResult,
    IngestService,
    IngestWarning,
)
from lab_manager.storage import LabStorage
from tests.fixtures import ASSIGNMENT_TEXT, make_pdf

MSK = ZoneInfo("Europe/Moscow")
PROMPT = "ОС, лаба 3, до 25.10"


class CountingExtractor:
    """Оборачивает правила и считает вызовы (в проде на этом месте будет LLM)."""

    version = "counting/v1"

    def __init__(self) -> None:
        self.calls = 0
        self.known_subjects: list[str] = []
        self._inner = RuleBasedExtractor()

    async def extract(
        self,
        *,
        markdown: str,
        prompt: str | None,
        today: date,
        tz: ZoneInfo,
        known_subjects: Sequence[str] = (),
    ) -> LabExtraction:
        self.calls += 1
        self.known_subjects = list(known_subjects)
        return await self._inner.extract(markdown=markdown, prompt=prompt, today=today, tz=tz)


@pytest.fixture
def storage(tmp_path: Path) -> LabStorage:
    return LabStorage(tmp_path / "Лабы")


@pytest.fixture
def extractor() -> CountingExtractor:
    return CountingExtractor()


@pytest.fixture
def service(
    db: Database, storage: LabStorage, extractor: CountingExtractor, tmp_path: Path
) -> IngestService:
    return IngestService(
        db=db,
        storage=storage,
        converter=DocumentConverter(cache_dir=tmp_path / "cache"),
        extractor=extractor,
        tz=MSK,
        max_bytes=1024 * 1024,
    )


async def ingest(
    service: IngestService, data: bytes, prompt: str | None = PROMPT, name: str = "task.txt"
) -> IngestResult:
    return await service.ingest(data=data, filename=name, prompt=prompt, source=Source.TELEGRAM)


async def test_new_assignment_is_created_and_laid_out(
    service: IngestService, storage: LabStorage, db: Database
) -> None:
    result = await ingest(service, ASSIGNMENT_TEXT.encode())

    assert result.outcome is IngestOutcome.CREATED
    assert result.subject is not None and result.subject.name == "ОС"
    lab = result.lab
    assert lab is not None
    assert (lab.number, lab.title) == (3, "Процессы и потоки")
    assert lab.deadline == datetime(2026, 10, 25, 23, 59, tzinfo=MSK)
    assert lab.folder == "ОС/03 Процессы и потоки"
    assert result.file is not None and result.file.version == 1
    assert storage.resolve(result.file.local_path).read_bytes() == ASSIGNMENT_TEXT.encode()
    assignment = storage.read_text(f"{lab.folder}/задание/assignment.md")
    assert assignment is not None and "fork()" in assignment
    assert result.warnings == ()


async def test_same_file_twice_is_a_noop(
    service: IngestService, extractor: CountingExtractor, db: Database
) -> None:
    first = await ingest(service, ASSIGNMENT_TEXT.encode())
    second = await ingest(service, ASSIGNMENT_TEXT.encode(), prompt="Совсем другое, лаба 9")

    assert second.outcome is IngestOutcome.DUPLICATE
    assert second.lab == first.lab
    assert second.file == first.file
    assert extractor.calls == 1
    assert len(await db.labs.list()) == 1


async def test_new_file_for_existing_lab_is_a_new_version(
    service: IngestService, storage: LabStorage
) -> None:
    first = await ingest(service, ASSIGNMENT_TEXT.encode())
    updated_text = ASSIGNMENT_TEXT + "\n4. Добавлено: замерить время fork().\n"

    second = await ingest(service, updated_text.encode(), prompt="ОС, лаба 3, до 01.11")

    assert second.outcome is IngestOutcome.NEW_VERSION
    assert first.lab is not None and second.lab is not None
    assert second.lab.id == first.lab.id
    assert second.lab.folder == first.lab.folder
    assert second.lab.deadline == datetime(2026, 11, 1, 23, 59, tzinfo=MSK)
    assert second.file is not None and second.file.version == 2
    assignment = storage.read_text(f"{second.lab.folder}/задание/assignment.md")
    assert assignment is not None and "замерить время" in assignment


async def test_missing_subject_asks_and_stores_nothing(
    service: IngestService, db: Database, storage: LabStorage
) -> None:
    result = await ingest(service, ASSIGNMENT_TEXT.encode(), prompt=None)

    assert result.outcome is IngestOutcome.NEEDS_INFO
    assert result.missing == ["subject"]
    assert result.lab is None
    assert await db.labs.list() == []
    assert await db.subjects.list() == []
    assert not storage.root.exists() or not any(storage.root.rglob("*.txt"))


async def test_resubmit_after_needs_info_succeeds(service: IngestService) -> None:
    await ingest(service, ASSIGNMENT_TEXT.encode(), prompt=None)

    result = await ingest(service, ASSIGNMENT_TEXT.encode(), prompt=PROMPT)

    assert result.outcome is IngestOutcome.CREATED


async def test_scan_without_text_is_kept_with_warning(
    service: IngestService, storage: LabStorage
) -> None:
    result = await ingest(service, make_pdf([]), prompt="ОС, лаба 1", name="scan.pdf")

    assert result.outcome is IngestOutcome.CREATED
    assert IngestWarning.NO_TEXT in result.warnings
    assert IngestWarning.NO_DEADLINE in result.warnings
    assert result.file is not None
    assert storage.resolve(result.file.local_path).exists()


async def test_text_message_as_assignment(service: IngestService, storage: LabStorage) -> None:
    result = await service.ingest_text(ASSIGNMENT_TEXT, prompt=PROMPT, source=Source.TELEGRAM)

    assert result.outcome is IngestOutcome.CREATED
    assert result.file is not None
    assert result.file.original_name == "задание.md"


async def test_too_large_file_rejected_early(
    service: IngestService, extractor: CountingExtractor
) -> None:
    with pytest.raises(FileTooLargeError):
        await ingest(service, b"x" * (1024 * 1024 + 1))

    assert extractor.calls == 0


async def test_unsupported_format_rejected(service: IngestService, db: Database) -> None:
    with pytest.raises(UnsupportedFormatError):
        await ingest(service, b"MZ...", name="setup.exe")

    assert await db.labs.list() == []


async def test_storage_failure_rolls_back_db(
    service: IngestService, storage: LabStorage, db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(storage, "save_original", broken)

    with pytest.raises(OSError, match="disk full"):
        await ingest(service, ASSIGNMENT_TEXT.encode())

    assert await db.labs.list() == []
    assert await db.subjects.list() == []


async def test_concurrent_duplicates_create_once(service: IngestService, db: Database) -> None:
    results = await asyncio.gather(*(ingest(service, ASSIGNMENT_TEXT.encode()) for _ in range(3)))

    outcomes = sorted(result.outcome for result in results)
    assert outcomes == [
        IngestOutcome.CREATED,
        IngestOutcome.DUPLICATE,
        IngestOutcome.DUPLICATE,
    ]
    assert len(await db.labs.list()) == 1


async def test_ingest_enqueues_disk_sync_job(service: IngestService, db: Database) -> None:
    result = await ingest(service, ASSIGNMENT_TEXT.encode())
    await ingest(service, ASSIGNMENT_TEXT.encode())  # дубликат не ставит задачу повторно

    job = await db.jobs.claim_next(kinds=["disk_sync"])
    assert job is not None and result.lab is not None
    assert job.payload == {"lab_id": result.lab.id}
    assert await db.jobs.claim_next(kinds=["disk_sync"]) is None


async def test_needs_info_enqueues_nothing(service: IngestService, db: Database) -> None:
    await ingest(service, ASSIGNMENT_TEXT.encode(), prompt=None)

    assert await db.jobs.claim_next() is None


async def test_failed_ingest_enqueues_nothing(
    service: IngestService, storage: LabStorage, db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(storage, "write_text", broken)

    with pytest.raises(OSError, match="disk full"):
        await ingest(service, ASSIGNMENT_TEXT.encode())

    assert await db.jobs.claim_next() is None


async def test_known_subjects_are_passed_to_extractor(
    service: IngestService, extractor: CountingExtractor
) -> None:
    await ingest(service, ASSIGNMENT_TEXT.encode())
    await ingest(service, b"another assignment text that is long enough", prompt="Сети, лаба 1")

    assert extractor.known_subjects == ["ОС"]


class ArtifactsExtractor(CountingExtractor):
    async def extract(
        self,
        *,
        markdown: str,
        prompt: str | None,
        today: date,
        tz: ZoneInfo,
        known_subjects: Sequence[str] = (),
    ) -> LabExtraction:
        base = await super().extract(markdown=markdown, prompt=prompt, today=today, tz=tz)
        return base.model_copy(update={"requirements": ["Программа", "Отчёт"]})


async def test_artifacts_stored_and_prompt_build_queued_first(
    db: Database, storage: LabStorage, tmp_path: Path
) -> None:
    service = IngestService(
        db=db,
        storage=storage,
        converter=DocumentConverter(cache_dir=tmp_path / "cache"),
        extractor=ArtifactsExtractor(),
        tz=MSK,
    )

    result = await ingest(service, ASSIGNMENT_TEXT.encode())

    assert result.lab is not None and result.lab.artifacts == ["Программа", "Отчёт"]
    first = await db.jobs.claim_next()
    second = await db.jobs.claim_next()
    assert first is not None and first.kind == "build_prompt"
    assert second is not None and second.kind == "disk_sync"
