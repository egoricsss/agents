"""Доменные модели и перечисления. Не зависят от БД, бота и LLM."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class Status(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    SUBMITTED = "submitted"
    DEFENDED = "defended"


class Source(StrEnum):
    """Кто инициировал изменение."""

    TELEGRAM = "telegram"
    MCP = "mcp"
    EXECUTOR = "executor"
    SYSTEM = "system"


class RequirementScope(StrEnum):
    GLOBAL = "global"
    SUBJECT = "subject"
    LAB = "lab"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class Subject(_Model):
    id: int
    name: str
    created_at: datetime


class Lab(_Model):
    id: int
    subject_id: int
    number: int
    title: str
    status: Status
    deadline: datetime | None
    summary: str
    folder: str | None
    artifacts: list[str]
    created_at: datetime
    updated_at: datetime


class StatusChange(_Model):
    id: int
    lab_id: int
    old_status: Status | None
    new_status: Status
    source: Source
    changed_at: datetime


class FileRecord(_Model):
    id: int
    lab_id: int
    sha256: str
    version: int
    original_name: str
    mime_type: str | None
    size_bytes: int
    local_path: str
    disk_path: str | None
    uploaded_at: datetime | None
    created_at: datetime


class Requirement(_Model):
    id: int
    scope: RequirementScope
    subject_id: int | None
    lab_id: int | None
    version: int
    text: str
    created_at: datetime


class Job(_Model):
    id: int
    kind: str
    payload: dict[str, Any]
    status: JobStatus
    attempts: int
    error: str | None
    idempotency_key: str
    created_at: datetime
    updated_at: datetime
