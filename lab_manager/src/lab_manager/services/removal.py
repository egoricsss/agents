"""Удаление ошибочно созданной лабы: БД, локальная папка, папка на Диске, пустой предмет."""

import asyncio
import logging
import shutil

from lab_manager.db import Database
from lab_manager.disk import DiskError, YandexDiskClient, join_path
from lab_manager.errors import NotFoundError
from lab_manager.storage import LabStorage

log = logging.getLogger(__name__)


class LabRemover:
    def __init__(
        self,
        *,
        db: Database,
        storage: LabStorage,
        disk: YandexDiskClient | None,
        disk_root: str,
    ) -> None:
        self.db = db
        self.storage = storage
        self.disk = disk
        self.disk_root = disk_root

    async def remove(self, lab_id: int) -> None:
        lab = await self.db.labs.get(lab_id)
        if lab is None:
            raise NotFoundError(f"лаба {lab_id} не найдена")
        await self.db.labs.delete(lab_id)
        await self.db.subjects.delete_if_empty(lab.subject_id)
        if lab.folder:
            path = self.storage.resolve(lab.folder)
            await asyncio.to_thread(shutil.rmtree, path, True)
            if self.disk is not None:
                try:
                    await self.disk.delete(join_path(self.disk_root, lab.folder), wait=False)
                except DiskError as exc:  # Диск — зеркало: локальное удаление важнее
                    log.warning("не удалось удалить папку лабы %s на Диске: %s", lab_id, exc)
