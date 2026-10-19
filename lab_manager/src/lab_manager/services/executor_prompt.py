"""Промпт и контракт для агента-исполнителя: `prompt.md`, `lab.json`, `requirements.md`.

Собирается детерминированно из БД и файлов (без LLM и без «текущего времени»), поэтому при тех же
данных файлы побайтно совпадают и не перезаливаются на Диск. Запись ставит задачу `disk_sync`
с ключом по содержимому — изменившийся промпт выгрузится, неизменный — нет.
"""

import hashlib
import json
from datetime import datetime
from importlib.resources import files
from string import Template
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from lab_manager.db import Database
from lab_manager.disk import join_path
from lab_manager.domain import Lab, RequirementScope
from lab_manager.errors import NotFoundError
from lab_manager.prompts import escape_closing_tag
from lab_manager.services.ingest import ASSIGNMENT_MARKDOWN, BUILD_PROMPT_JOB, DISK_SYNC_JOB
from lab_manager.storage import REPORT_DIR, LabStorage

PROMPT_FILE = "prompt.md"
SPEC_FILE = "lab.json"
REQUIREMENTS_FILE = "requirements.md"
SPEC_SCHEMA_VERSION = 1
EXECUTOR_PROMPT_VERSION = "executor/v1"
EMPTY = "—"

_TEMPLATE = Template(files("lab_manager.prompts").joinpath("executor.md").read_text("utf-8"))


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True)


class OriginalFile(_Spec):
    version: int
    name: str
    path: str
    disk_path: str
    sha256: str


class LabCard(_Spec):
    id: int
    subject: str
    number: int
    title: str
    deadline: datetime | None
    summary: str
    folder: str
    disk_folder: str


class Requirements(_Spec):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    global_: str = Field(default="", alias="global")
    subject: str = ""
    lab: str = ""


class LabSpec(_Spec):
    """Контракт с исполнителем (`lab.json`). Меняется только с повышением `schema_version`."""

    schema_version: int = SPEC_SCHEMA_VERSION
    prompt_version: str = EXECUTOR_PROMPT_VERSION
    lab: LabCard
    assignment_path: str
    originals: list[OriginalFile]
    artifacts: list[str]
    requirements: Requirements
    report_dir: str


class ExecutorPromptBuilder:
    def __init__(self, *, db: Database, storage: LabStorage, disk_root: str, tz: ZoneInfo) -> None:
        self.db = db
        self.storage = storage
        self.disk_root = disk_root
        self.tz = tz

    async def build(self, lab_id: int) -> tuple[LabSpec, str, str]:
        """Возвращает (lab.json, prompt.md, requirements.md) без записи на диск."""
        lab = await self.db.labs.get(lab_id)
        if lab is None or lab.folder is None:
            raise NotFoundError(f"лаба {lab_id} не найдена или без папки")
        subject = await self.db.subjects.get(lab.subject_id)
        assert subject is not None  # noqa: S101 — FK гарантирует
        spec = LabSpec(
            lab=LabCard(
                id=lab.id,
                subject=subject.name,
                number=lab.number,
                title=lab.title,
                deadline=lab.deadline,
                summary=lab.summary,
                folder=lab.folder,
                disk_folder=join_path(self.disk_root, lab.folder),
            ),
            assignment_path=ASSIGNMENT_MARKDOWN,
            originals=[
                OriginalFile(
                    version=record.version,
                    name=record.original_name,
                    path=record.local_path.removeprefix(f"{lab.folder}/"),
                    disk_path=join_path(self.disk_root, record.local_path),
                    sha256=record.sha256,
                )
                for record in await self.db.files.list_for_lab(lab.id)
            ],
            artifacts=list(lab.artifacts),
            requirements=await self._requirements(lab),
            report_dir=f"{REPORT_DIR}/",
        )
        assignment = self.storage.read_text(f"{lab.folder}/{ASSIGNMENT_MARKDOWN}") or ""
        return spec, self._prompt(spec, assignment), self._requirements_md(spec)

    async def write(self, lab_id: int) -> list[str]:
        """Пишет три файла в папку лабы и ставит выгрузку на Диск. Возвращает относительные пути."""
        spec, prompt, requirements = await self.build(lab_id)
        folder = spec.lab.folder
        spec_json = (
            json.dumps(spec.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2)
            + "\n"
        )
        written = [
            self.storage.write_text(folder, SPEC_FILE, spec_json),
            self.storage.write_text(folder, PROMPT_FILE, prompt),
            self.storage.write_text(folder, REQUIREMENTS_FILE, requirements),
        ]
        digest = hashlib.sha256((spec_json + prompt + requirements).encode()).hexdigest()[:16]
        await self.db.jobs.enqueue(
            DISK_SYNC_JOB,
            {"lab_id": lab_id},
            idempotency_key=f"{DISK_SYNC_JOB}:prompt:{lab_id}:{digest}",
        )
        return written

    async def handle_job(self, payload: dict[str, Any]) -> None:
        lab_id = payload.get("lab_id")
        if not isinstance(lab_id, int):
            raise ValueError(f"некорректный payload задачи {BUILD_PROMPT_JOB}: {payload!r}")
        if await self.db.labs.get(lab_id) is None:
            return  # лабу удалили, пока задача ждала в очереди
        await self.write(lab_id)

    async def _requirements(self, lab: Lab) -> Requirements:
        async def text(scope: RequirementScope, target: int | None) -> str:
            current = await self.db.requirements.current(scope, target)
            return current.text.strip() if current else ""

        return Requirements(
            global_=await text(RequirementScope.GLOBAL, None),
            subject=await text(RequirementScope.SUBJECT, lab.subject_id),
            lab=await text(RequirementScope.LAB, lab.id),
        )

    def _deadline(self, spec: LabSpec) -> str:
        if spec.lab.deadline is None:
            return "не указан"
        local = spec.lab.deadline.astimezone(self.tz)
        return f"{local:%d.%m.%Y %H:%M} ({self.tz.key})"

    def _prompt(self, spec: LabSpec, assignment: str) -> str:
        originals = ", ".join(f"`{item.path}`" for item in spec.originals) or EMPTY
        if not assignment.strip():
            assignment = "Текст не извлечён (вероятно, скан) — смотри оригиналы задания."
        return _TEMPLATE.substitute(
            subject=spec.lab.subject,
            number=spec.lab.number,
            title=spec.lab.title,
            deadline=self._deadline(spec),
            disk_folder=spec.lab.disk_folder,
            assignment_path=spec.assignment_path,
            originals=originals,
            artifacts=_bullets(spec.artifacts) or "См. текст задания.",
            requirements_global=spec.requirements.global_ or EMPTY,
            requirements_subject=spec.requirements.subject or EMPTY,
            requirements_lab=spec.requirements.lab or EMPTY,
            report_dir=spec.report_dir,
            assignment=escape_closing_tag(assignment.strip(), "assignment"),
        )

    def _requirements_md(self, spec: LabSpec) -> str:
        req = spec.requirements
        return (
            f"# Требования: {spec.lab.subject}, лаба {spec.lab.number}\n\n"
            "Приоритет: к лабе > к предмету > общие.\n\n"
            f"## Общие\n\n{req.global_ or EMPTY}\n\n"
            f"## По предмету\n\n{req.subject or EMPTY}\n\n"
            f"## К этой лабе\n\n{req.lab or EMPTY}\n"
        )


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
