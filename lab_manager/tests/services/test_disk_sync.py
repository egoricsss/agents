"""Синхронизация лабы с Диском: оригиналы + assignment.md, отметки о выгрузке, идемпотентность."""

from collections.abc import AsyncIterator
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from lab_manager.converter import DocumentConverter
from lab_manager.db import Database
from lab_manager.disk import YandexDiskClient
from lab_manager.domain import Lab
from lab_manager.errors import NotFoundError
from lab_manager.extraction.rules import RuleBasedExtractor
from lab_manager.services.disk_sync import DiskSync
from lab_manager.services.ingest import IngestService
from lab_manager.storage import LabStorage
from tests.disk.fake_disk import TOKEN, FakeDisk
from tests.fixtures import ASSIGNMENT_TEXT

ROOT = "app:/Лабы"


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fake() -> FakeDisk:
    return FakeDisk()


@pytest.fixture
def storage(tmp_path: Path) -> LabStorage:
    return LabStorage(tmp_path / "Лабы")


@pytest.fixture
async def sync(db: Database, storage: LabStorage, fake: FakeDisk) -> AsyncIterator[DiskSync]:
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        disk = YandexDiskClient(TOKEN, http=http, sleep=no_sleep)
        yield DiskSync(db=db, storage=storage, disk=disk, root=ROOT)


@pytest.fixture
def ingest(db: Database, storage: LabStorage, tmp_path: Path) -> IngestService:
    return IngestService(
        db=db,
        storage=storage,
        converter=DocumentConverter(cache_dir=tmp_path / "cache"),
        extractor=RuleBasedExtractor(),
        tz=ZoneInfo("Europe/Moscow"),
    )


async def add_lab(ingest: IngestService, text: str = ASSIGNMENT_TEXT) -> Lab:
    result = await ingest.ingest(data=text.encode(), filename="task.txt", prompt="ОС, лаба 3")
    assert result.lab is not None
    return result.lab


async def test_sync_uploads_original_and_assignment(
    sync: DiskSync, ingest: IngestService, fake: FakeDisk, db: Database
) -> None:
    lab = await add_lab(ingest)

    report = await sync.sync_lab(lab.id)

    base = f"{ROOT}/ОС/03 Процессы и потоки"
    assert fake.files[f"{base}/задание/v1_task.txt"] == ASSIGNMENT_TEXT.encode()
    assert "fork()" in fake.files[f"{base}/задание/assignment.md"].decode()
    assert f"{base}/отчёт" in fake.dirs
    assert sorted(report.uploaded) == [
        f"{base}/задание/assignment.md",
        f"{base}/задание/v1_task.txt",
    ]
    [record] = await db.files.list_for_lab(lab.id)
    assert record.disk_path == f"{base}/задание/v1_task.txt"
    assert record.uploaded_at is not None


async def test_second_sync_uploads_nothing(
    sync: DiskSync, ingest: IngestService, fake: FakeDisk
) -> None:
    lab = await add_lab(ingest)
    await sync.sync_lab(lab.id)
    uploads = len(fake.upload_requests)

    report = await sync.sync_lab(lab.id)

    assert report.uploaded == []
    assert len(report.skipped) == 2
    assert len(fake.upload_requests) == uploads


async def test_new_version_uploads_only_new_file_and_changed_assignment(
    sync: DiskSync, ingest: IngestService, fake: FakeDisk
) -> None:
    lab = await add_lab(ingest)
    await sync.sync_lab(lab.id)
    await add_lab(ingest, ASSIGNMENT_TEXT + "\nДобавлен пункт 4.\n")

    report = await sync.sync_lab(lab.id)

    assert sorted(path.rsplit("/", 1)[-1] for path in report.uploaded) == [
        "assignment.md",
        "v2_task.txt",
    ]


async def test_reupload_when_remote_file_was_deleted(
    sync: DiskSync, ingest: IngestService, fake: FakeDisk
) -> None:
    lab = await add_lab(ingest)
    await sync.sync_lab(lab.id)
    fake.files.clear()

    report = await sync.sync_lab(lab.id)

    assert len(report.uploaded) == 2


async def test_unknown_lab(sync: DiskSync) -> None:
    with pytest.raises(NotFoundError):
        await sync.sync_lab(999)


async def test_ingest_to_disk_end_to_end_via_queue(
    sync: DiskSync, ingest: IngestService, fake: FakeDisk, db: Database
) -> None:
    from lab_manager.worker import JobWorker

    lab = await add_lab(ingest)
    worker = JobWorker(db, {sync.job_kind: sync.handle_job})

    job = await worker.run_once()

    assert job is not None and job.status.value == "done"
    assert any(path.endswith("/задание/v1_task.txt") for path in fake.files)
    [record] = await db.files.list_for_lab(lab.id)
    assert record.uploaded_at is not None


async def test_bad_job_payload_rejected(sync: DiskSync) -> None:
    with pytest.raises(ValueError, match="payload"):
        await sync.handle_job({"lab_id": "1"})
