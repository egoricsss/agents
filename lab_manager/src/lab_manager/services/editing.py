"""Правка лабы: номер, предмет, тема, срок. Папка переименовывается локально и на Диске."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lab_manager.db import Database
from lab_manager.disk import DiskError, YandexDiskClient, join_path
from lab_manager.domain import Lab, Subject
from lab_manager.errors import NotFoundError
from lab_manager.services.executor_prompt import ExecutorPromptBuilder
from lab_manager.services.labs import subject_matches
from lab_manager.storage import LabStorage

log = logging.getLogger(__name__)

UNSET: Any = object()


@dataclass
class EditResult:
    lab: Lab
    subject: str
    changes: list[str] = field(default_factory=list)


class LabEditor:
    def __init__(
        self,
        *,
        db: Database,
        storage: LabStorage,
        builder: ExecutorPromptBuilder,
        disk: YandexDiskClient | None,
        disk_root: str,
    ) -> None:
        self.db = db
        self.storage = storage
        self.builder = builder
        self.disk = disk
        self.disk_root = disk_root

    async def edit(
        self,
        lab_id: int,
        *,
        subject: str | None = None,
        number: int | None = None,
        title: str | None = None,
        deadline: datetime | None = UNSET,
    ) -> EditResult:
        lab = await self.db.labs.get(lab_id)
        if lab is None:
            raise NotFoundError(f"лаба {lab_id} не найдена")
        old_subject = await self.db.subjects.get(lab.subject_id)
        assert old_subject is not None  # noqa: S101
        changes: list[str] = []
        update: dict[str, Any] = {}

        new_subject = old_subject
        if subject and subject.strip() and not subject_matches(subject, old_subject.name):
            new_subject = await self._subject(subject)
            update["subject_id"] = new_subject.id
            changes.append(f"предмет: {old_subject.name} → {new_subject.name}")
        if number is not None and number != lab.number:
            update["number"] = number
            changes.append(f"номер: {lab.number} → {number}")
        if title and title.strip() and title.strip() != lab.title:
            update["title"] = title.strip()
            changes.append(f"тема: {lab.title} → {title.strip()}")
        if deadline is not UNSET and deadline != lab.deadline:
            update["deadline"] = deadline
            changes.append("срок: " + (f"{deadline:%d.%m.%Y %H:%M}" if deadline else "убран"))
        if not changes:
            return EditResult(lab, old_subject.name, [])

        old_folder = lab.folder
        new_folder = self.storage.lab_folder(
            new_subject.name, update.get("number", lab.number), update.get("title", lab.title)
        )
        moved = old_folder is not None and new_folder != old_folder
        async with self.db.transaction():
            if moved:
                update["folder"] = new_folder
            lab = await self.db.labs.update(lab_id, **update)
            if moved and old_folder is not None:
                await self.db.files.relocate(lab_id, old_folder, new_folder)
                self.storage.move(old_folder, new_folder)  # исключение откатит БД
        if new_subject.id != old_subject.id:
            await self.db.subjects.delete_if_empty(old_subject.id)
        if moved and old_folder is not None and self.disk is not None:
            try:
                await self.disk.move(
                    join_path(self.disk_root, old_folder), join_path(self.disk_root, new_folder)
                )
            except DiskError as exc:  # не страшно: disk_sync зальёт в новую папку заново
                log.warning("не удалось перенести папку лабы %s на Диске: %s", lab_id, exc)
        await self.builder.write(lab_id)  # промпт + задача на выгрузку
        return EditResult(lab, new_subject.name, changes)

    async def _subject(self, name: str) -> Subject:
        matches = [s for s in await self.db.subjects.list() if subject_matches(name, s.name)]
        if len(matches) == 1:
            return matches[0]
        return await self.db.subjects.get_or_create(name)
