"""Репозитории: единственное место с SQL. Возвращают доменные модели."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiosqlite

from lab_manager.domain import (
    FileRecord,
    Job,
    JobStatus,
    Lab,
    Requirement,
    RequirementScope,
    Source,
    Status,
    StatusChange,
    Subject,
)
from lab_manager.errors import AlreadyExistsError, NotFoundError

if TYPE_CHECKING:
    from lab_manager.db.database import Database

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNSET: Any = object()


def to_db(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("время должно быть с часовым поясом (timezone-aware)")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def from_db(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value).astimezone(UTC)


def _row(row: aiosqlite.Row, *time_fields: str) -> dict[str, Any]:
    data = dict(row)
    for field in time_fields:
        data[field] = from_db(data[field])
    return data


@contextmanager
def _integrity(what: str) -> Iterator[None]:
    """Переводит ошибки ограничений SQLite в доменные исключения."""
    try:
        yield
    except sqlite3.IntegrityError as exc:
        message = str(exc)
        if "FOREIGN KEY" in message:
            raise NotFoundError(f"{what}: связанная запись не найдена") from exc
        if "UNIQUE" in message:
            raise AlreadyExistsError(f"{what}: уже существует") from exc
        raise


class _Repo:
    def __init__(self, db: Database) -> None:
        self._db = db

    @property
    def _conn(self) -> aiosqlite.Connection:
        return self._db.conn

    def _now(self) -> str:
        now = to_db(self._db.clock())
        assert now is not None  # noqa: S101 — clock всегда возвращает время
        return now

    async def _one(self, sql: str, params: Iterable[Any] = ()) -> aiosqlite.Row | None:
        async with self._conn.execute(sql, tuple(params)) as cursor:
            return await cursor.fetchone()

    async def _all(self, sql: str, params: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        async with self._conn.execute(sql, tuple(params)) as cursor:
            return list(await cursor.fetchall())

    async def _insert(self, sql: str, params: Iterable[Any]) -> int:
        async with self._conn.execute(sql, tuple(params)) as cursor:
            row_id = cursor.lastrowid
        assert row_id is not None  # noqa: S101
        return row_id


def normalize_name(name: str) -> str:
    return " ".join(name.split())


class SubjectRepo(_Repo):
    @staticmethod
    def _model(row: aiosqlite.Row) -> Subject:
        return Subject.model_validate(_row(row, "created_at"))

    async def get(self, subject_id: int) -> Subject | None:
        row = await self._one(
            "SELECT id, name, created_at FROM subjects WHERE id = ?", [subject_id]
        )
        return self._model(row) if row else None

    async def get_by_name(self, name: str) -> Subject | None:
        row = await self._one(
            "SELECT id, name, created_at FROM subjects WHERE name_key = ?",
            [normalize_name(name).casefold()],
        )
        return self._model(row) if row else None

    async def get_or_create(self, name: str) -> Subject:
        clean = normalize_name(name)
        if not clean:
            raise ValueError("название предмета не может быть пустым")
        async with self._db.transaction():
            if existing := await self.get_by_name(clean):
                return existing
            new_id = await self._insert(
                "INSERT INTO subjects (name, name_key, created_at) VALUES (?, ?, ?)",
                [clean, clean.casefold(), self._now()],
            )
        subject = await self.get(new_id)
        assert subject is not None  # noqa: S101
        return subject

    async def list(self) -> list[Subject]:
        rows = await self._all("SELECT id, name, created_at FROM subjects ORDER BY name_key")
        return [self._model(row) for row in rows]


_LAB_COLUMNS = (
    "id, subject_id, number, title, status, deadline, summary, folder, created_at, updated_at"
)


class LabRepo(_Repo):
    @staticmethod
    def _model(row: aiosqlite.Row) -> Lab:
        return Lab.model_validate(_row(row, "deadline", "created_at", "updated_at"))

    async def get(self, lab_id: int) -> Lab | None:
        row = await self._one(f"SELECT {_LAB_COLUMNS} FROM labs WHERE id = ?", [lab_id])  # noqa: S608
        return self._model(row) if row else None

    async def _require(self, lab_id: int) -> Lab:
        lab = await self.get(lab_id)
        if lab is None:
            raise NotFoundError(f"лаба {lab_id} не найдена")
        return lab

    async def find(self, subject_id: int, number: int) -> Lab | None:
        row = await self._one(
            f"SELECT {_LAB_COLUMNS} FROM labs WHERE subject_id = ? AND number = ?",  # noqa: S608
            [subject_id, number],
        )
        return self._model(row) if row else None

    async def create(
        self,
        *,
        subject_id: int,
        number: int,
        title: str,
        source: Source,
        deadline: datetime | None = None,
        summary: str = "",
        folder: str | None = None,
    ) -> Lab:
        if number < 1:
            raise ValueError("номер лабы должен быть >= 1")
        deadline_db = to_db(deadline)
        now = self._now()
        async with self._db.transaction():
            with _integrity(f"лаба {number}"):
                lab_id = await self._insert(
                    "INSERT INTO labs (subject_id, number, title, deadline, summary, folder,"
                    " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [subject_id, number, title, deadline_db, summary, folder, now, now],
                )
            await self._log_status(lab_id, None, Status.TODO, source, now)
        return await self._require(lab_id)

    async def update(
        self,
        lab_id: int,
        *,
        title: str = _UNSET,
        deadline: datetime | None = _UNSET,
        summary: str = _UNSET,
    ) -> Lab:
        changes: dict[str, Any] = {}
        if title is not _UNSET:
            changes["title"] = title
        if deadline is not _UNSET:
            changes["deadline"] = to_db(deadline)
        if summary is not _UNSET:
            changes["summary"] = summary
        async with self._db.transaction():
            await self._require(lab_id)
            if changes:
                changes["updated_at"] = self._now()
                assignments = ", ".join(f"{column} = ?" for column in changes)
                await self._conn.execute(
                    f"UPDATE labs SET {assignments} WHERE id = ?",  # noqa: S608 — колонки фиксированы
                    [*changes.values(), lab_id],
                )
        return await self._require(lab_id)

    async def set_status(self, lab_id: int, status: Status, source: Source) -> Lab:
        async with self._db.transaction():
            lab = await self._require(lab_id)
            if lab.status is status:
                return lab
            now = self._now()
            await self._conn.execute(
                "UPDATE labs SET status = ?, updated_at = ? WHERE id = ?",
                [status.value, now, lab_id],
            )
            await self._log_status(lab_id, lab.status, status, source, now)
        return await self._require(lab_id)

    async def _log_status(
        self, lab_id: int, old: Status | None, new: Status, source: Source, at: str
    ) -> None:
        await self._conn.execute(
            "INSERT INTO status_history (lab_id, old_status, new_status, source, changed_at)"
            " VALUES (?, ?, ?, ?, ?)",
            [lab_id, old.value if old else None, new.value, source.value, at],
        )

    async def history(self, lab_id: int) -> list[StatusChange]:
        rows = await self._all(
            "SELECT id, lab_id, old_status, new_status, source, changed_at"
            " FROM status_history WHERE lab_id = ? ORDER BY id",
            [lab_id],
        )
        return [StatusChange.model_validate(_row(row, "changed_at")) for row in rows]

    async def list(
        self,
        *,
        statuses: Iterable[Status] | None = None,
        subject_id: int | None = None,
        due_from: datetime | None = None,
        due_before: datetime | None = None,
    ) -> list[Lab]:
        """Лабы по дедлайну, без дедлайна — в конце.

        `due_from` включительно, `due_before` — нет.
        """
        where: list[str] = []
        params: list[Any] = []
        if statuses is not None:
            values = [status.value for status in statuses]
            where.append(f"status IN ({', '.join('?' * len(values))})")
            params.extend(values)
        if subject_id is not None:
            where.append("subject_id = ?")
            params.append(subject_id)
        if due_from is not None:
            where.append("deadline >= ?")
            params.append(to_db(due_from))
        if due_before is not None:
            where.append("deadline < ?")
            params.append(to_db(due_before))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = await self._all(
            f"SELECT {_LAB_COLUMNS} FROM labs {clause}"  # noqa: S608 — только плейсхолдеры
            " ORDER BY deadline IS NULL, deadline, subject_id, number",
            params,
        )
        return [self._model(row) for row in rows]


_FILE_COLUMNS = (
    "id, lab_id, sha256, version, original_name, mime_type, size_bytes,"
    " local_path, disk_path, uploaded_at, created_at"
)


class FileRepo(_Repo):
    @staticmethod
    def _model(row: aiosqlite.Row) -> FileRecord:
        return FileRecord.model_validate(_row(row, "uploaded_at", "created_at"))

    async def get(self, file_id: int) -> FileRecord | None:
        row = await self._one(f"SELECT {_FILE_COLUMNS} FROM files WHERE id = ?", [file_id])  # noqa: S608
        return self._model(row) if row else None

    async def get_by_sha256(self, sha256: str) -> FileRecord | None:
        row = await self._one(f"SELECT {_FILE_COLUMNS} FROM files WHERE sha256 = ?", [sha256])  # noqa: S608
        return self._model(row) if row else None

    async def add(
        self,
        *,
        lab_id: int,
        sha256: str,
        original_name: str,
        mime_type: str | None,
        size_bytes: int,
        local_path: str,
    ) -> FileRecord:
        if not _SHA256.match(sha256):
            raise ValueError("sha256 должен быть 64 символа в нижнем регистре hex")
        async with self._db.transaction():
            row = await self._one(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM files WHERE lab_id = ?", [lab_id]
            )
            version = int(row[0]) if row else 1
            with _integrity(f"файл {original_name}"):
                file_id = await self._insert(
                    "INSERT INTO files (lab_id, sha256, version, original_name, mime_type,"
                    " size_bytes, local_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        lab_id,
                        sha256,
                        version,
                        original_name,
                        mime_type,
                        size_bytes,
                        local_path,
                        self._now(),
                    ],
                )
        record = await self.get(file_id)
        assert record is not None  # noqa: S101
        return record

    async def list_for_lab(self, lab_id: int) -> list[FileRecord]:
        rows = await self._all(
            f"SELECT {_FILE_COLUMNS} FROM files WHERE lab_id = ? ORDER BY version",  # noqa: S608
            [lab_id],
        )
        return [self._model(row) for row in rows]

    async def latest_for_lab(self, lab_id: int) -> FileRecord | None:
        row = await self._one(
            f"SELECT {_FILE_COLUMNS} FROM files WHERE lab_id = ?"  # noqa: S608
            " ORDER BY version DESC LIMIT 1",
            [lab_id],
        )
        return self._model(row) if row else None

    async def mark_uploaded(self, file_id: int, disk_path: str) -> FileRecord:
        async with self._db.transaction():
            cursor = await self._conn.execute(
                "UPDATE files SET disk_path = ?, uploaded_at = ? WHERE id = ?",
                [disk_path, self._now(), file_id],
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"файл {file_id} не найден")
        record = await self.get(file_id)
        assert record is not None  # noqa: S101
        return record


_REQ_COLUMNS = "id, scope, subject_id, lab_id, version, text, created_at"


def _target(scope: RequirementScope, target_id: int | None) -> tuple[str, int | None, int | None]:
    """Возвращает (target_key, subject_id, lab_id) и проверяет согласованность scope/target."""
    if scope is RequirementScope.GLOBAL:
        if target_id is not None:
            raise ValueError("у глобальных требований не бывает target_id")
        return "global", None, None
    if target_id is None:
        raise ValueError(f"для scope={scope.value} нужен target_id")
    if scope is RequirementScope.SUBJECT:
        return f"subject:{target_id}", target_id, None
    return f"lab:{target_id}", None, target_id


class RequirementRepo(_Repo):
    @staticmethod
    def _model(row: aiosqlite.Row) -> Requirement:
        return Requirement.model_validate(_row(row, "created_at"))

    async def add(
        self, scope: RequirementScope, text: str, target_id: int | None = None
    ) -> Requirement:
        """Новая версия требований. Пустой текст означает «требований нет»."""
        key, subject_id, lab_id = _target(scope, target_id)
        async with self._db.transaction():
            row = await self._one(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM requirements WHERE target_key = ?",
                [key],
            )
            version = int(row[0]) if row else 1
            with _integrity(f"требования {key}"):
                req_id = await self._insert(
                    "INSERT INTO requirements (scope, subject_id, lab_id, target_key, version,"
                    " text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [scope.value, subject_id, lab_id, key, version, text, self._now()],
                )
        row = await self._one(
            f"SELECT {_REQ_COLUMNS} FROM requirements WHERE id = ?",  # noqa: S608
            [req_id],
        )
        assert row is not None  # noqa: S101
        return self._model(row)

    async def current(
        self, scope: RequirementScope, target_id: int | None = None
    ) -> Requirement | None:
        key, _, _ = _target(scope, target_id)
        row = await self._one(
            f"SELECT {_REQ_COLUMNS} FROM requirements WHERE target_key = ?"  # noqa: S608
            " ORDER BY version DESC LIMIT 1",
            [key],
        )
        return self._model(row) if row else None

    async def history(
        self, scope: RequirementScope, target_id: int | None = None
    ) -> list[Requirement]:
        key, _, _ = _target(scope, target_id)
        rows = await self._all(
            f"SELECT {_REQ_COLUMNS} FROM requirements WHERE target_key = ?"  # noqa: S608
            " ORDER BY version",
            [key],
        )
        return [self._model(row) for row in rows]


_JOB_COLUMNS = "id, kind, payload, status, attempts, error, idempotency_key, created_at, updated_at"


class JobRepo(_Repo):
    @staticmethod
    def _model(row: aiosqlite.Row) -> Job:
        data = _row(row, "created_at", "updated_at")
        data["payload"] = json.loads(data["payload"])
        return Job.model_validate(data)

    async def get(self, job_id: int) -> Job | None:
        row = await self._one(f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", [job_id])  # noqa: S608
        return self._model(row) if row else None

    async def _require(self, job_id: int) -> Job:
        job = await self.get(job_id)
        if job is None:
            raise NotFoundError(f"задача {job_id} не найдена")
        return job

    async def enqueue(
        self, kind: str, payload: dict[str, Any], *, idempotency_key: str
    ) -> tuple[Job, bool]:
        """Ставит задачу. Если ключ уже был — возвращает существующую и `False`."""
        async with self._db.transaction():
            row = await self._one(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE idempotency_key = ?",  # noqa: S608
                [idempotency_key],
            )
            if row:
                return self._model(row), False
            now = self._now()
            job_id = await self._insert(
                "INSERT INTO jobs (kind, payload, idempotency_key, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [kind, json.dumps(payload, ensure_ascii=False), idempotency_key, now, now],
            )
        return await self._require(job_id), True

    async def claim_next(self, kinds: Iterable[str] | None = None) -> Job | None:
        params: list[Any] = [JobStatus.PENDING.value]
        kind_clause = ""
        if kinds is not None:
            kind_list = list(kinds)
            kind_clause = f" AND kind IN ({', '.join('?' * len(kind_list))})"
            params.extend(kind_list)
        async with self._db.transaction():
            row = await self._one(
                f"SELECT id FROM jobs WHERE status = ?{kind_clause} ORDER BY id LIMIT 1",  # noqa: S608
                params,
            )
            if row is None:
                return None
            await self._conn.execute(
                "UPDATE jobs SET status = ?, attempts = attempts + 1, updated_at = ? WHERE id = ?",
                [JobStatus.RUNNING.value, self._now(), row["id"]],
            )
        return await self._require(row["id"])

    async def complete(self, job_id: int) -> Job:
        return await self._set(job_id, JobStatus.DONE, error=None)

    async def fail(self, job_id: int, error: str, *, max_attempts: int = 3) -> Job:
        """Возвращает задачу в очередь, пока не исчерпаны попытки; иначе — `failed`."""
        async with self._db.transaction():
            job = await self._require(job_id)
            status = JobStatus.FAILED if job.attempts >= max_attempts else JobStatus.PENDING
            return await self._set(job_id, status, error=error)

    async def recover_running(self) -> int:
        """После рестарта: прерванные `running` снова становятся `pending`."""
        async with self._db.transaction():
            cursor = await self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
                [JobStatus.PENDING.value, self._now(), JobStatus.RUNNING.value],
            )
            return cursor.rowcount

    async def _set(self, job_id: int, status: JobStatus, *, error: str | None) -> Job:
        async with self._db.transaction():
            cursor = await self._conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                [status.value, error, self._now(), job_id],
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"задача {job_id} не найдена")
        return await self._require(job_id)
