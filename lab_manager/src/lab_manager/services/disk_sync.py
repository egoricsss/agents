"""Зеркалирование лабы на Яндекс Диск (локально → Диск).

Выгружаются все версии оригиналов и служебные Markdown-файлы лабы; создаётся папка `отчёт/`.
Повторный запуск ничего не грузит, если содержимое на Диске совпадает (sha256).
"""

from dataclasses import dataclass, field

from lab_manager.db import Database
from lab_manager.disk import UploadResult, YandexDiskClient, join_path
from lab_manager.errors import NotFoundError
from lab_manager.services.ingest import DISK_SYNC_JOB
from lab_manager.storage import ASSIGNMENT_DIR, REPORT_DIR, LabStorage

# Служебные файлы лабы, которые зеркалируются, если существуют (prompt.md, lab.json — шаг 6).
LAB_TEXT_FILES = (f"{ASSIGNMENT_DIR}/assignment.md", "requirements.md", "prompt.md", "lab.json")


@dataclass
class SyncReport:
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class DiskSync:
    job_kind = DISK_SYNC_JOB

    def __init__(
        self, *, db: Database, storage: LabStorage, disk: YandexDiskClient, root: str
    ) -> None:
        self.db = db
        self.storage = storage
        self.disk = disk
        self.root = root

    async def sync_lab(self, lab_id: int) -> SyncReport:
        lab = await self.db.labs.get(lab_id)
        if lab is None or lab.folder is None:
            raise NotFoundError(f"лаба {lab_id} не найдена или без папки")
        report = SyncReport()

        for record in await self.db.files.list_for_lab(lab_id):
            remote = join_path(self.root, record.local_path)
            data = self.storage.resolve(record.local_path).read_bytes()
            await self._upload(remote, data, report)
            if record.disk_path != remote or record.uploaded_at is None:
                await self.db.files.mark_uploaded(record.id, remote)

        for name in LAB_TEXT_FILES:
            relative = f"{lab.folder}/{name}"
            path = self.storage.resolve(relative)
            if path.is_file():
                await self._upload(join_path(self.root, relative), path.read_bytes(), report)

        await self.disk.ensure_dir(join_path(self.root, f"{lab.folder}/{REPORT_DIR}"))
        return report

    async def _upload(self, remote: str, data: bytes, report: SyncReport) -> None:
        result = await self.disk.upload(remote, data)
        (report.uploaded if result is UploadResult.UPLOADED else report.skipped).append(remote)

    async def handle_job(self, payload: dict[str, object]) -> None:
        """Обработчик задачи очереди `disk_sync`."""
        lab_id = payload.get("lab_id")
        if not isinstance(lab_id, int):
            raise ValueError(f"некорректный payload задачи {self.job_kind}: {payload!r}")
        if await self.db.labs.get(lab_id) is None:
            return  # лабу удалили, пока задача ждала в очереди
        await self.sync_lab(lab_id)
