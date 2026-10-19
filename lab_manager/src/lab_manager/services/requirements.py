"""Требования владельца на трёх уровнях + пересборка промптов затронутых лаб."""

from dataclasses import dataclass

from lab_manager.db import Database
from lab_manager.domain import Requirement, RequirementScope
from lab_manager.errors import NotFoundError
from lab_manager.services.executor_prompt import ExecutorPromptBuilder


@dataclass(frozen=True, slots=True)
class RequirementsUpdate:
    requirement: Requirement
    rebuilt_lab_ids: list[int]


class RequirementsService:
    def __init__(self, *, db: Database, builder: ExecutorPromptBuilder) -> None:
        self.db = db
        self.builder = builder

    async def set(
        self, scope: RequirementScope, text: str, target_id: int | None = None
    ) -> RequirementsUpdate:
        """Новая версия требований (пустой текст — очистить) и пересборка промптов."""
        requirement = await self.db.requirements.add(scope, text.strip(), target_id)
        lab_ids = await self._affected_labs(scope, target_id)
        for lab_id in lab_ids:
            await self.builder.write(lab_id)
        return RequirementsUpdate(requirement, lab_ids)

    async def current(
        self, scope: RequirementScope, target_id: int | None = None
    ) -> Requirement | None:
        return await self.db.requirements.current(scope, target_id)

    async def _affected_labs(self, scope: RequirementScope, target_id: int | None) -> list[int]:
        if scope is RequirementScope.LAB:
            assert target_id is not None  # noqa: S101 — проверено репозиторием
            lab = await self.db.labs.get(target_id)
            if lab is None:
                raise NotFoundError(f"лаба {target_id} не найдена")
            labs = [lab]
        elif scope is RequirementScope.SUBJECT:
            labs = await self.db.labs.list(subject_id=target_id)
        else:
            labs = await self.db.labs.list()
        return [lab.id for lab in labs if lab.folder is not None]
