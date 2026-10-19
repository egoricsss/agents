"""Подключение к SQLite, миграции и транзакции."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import aiosqlite

from lab_manager.db.migrations import load_migrations
from lab_manager.db.repos import (
    FileRepo,
    JobRepo,
    LabRepo,
    LLMRepo,
    RequirementRepo,
    SubjectRepo,
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class Database:
    """Одно соединение на процесс; записи сериализуются через `transaction()`."""

    def __init__(self, conn: aiosqlite.Connection, clock: Clock = utc_now) -> None:
        self.conn = conn
        self.clock = clock
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self.subjects = SubjectRepo(self)
        self.labs = LabRepo(self)
        self.files = FileRepo(self)
        self.requirements = RequirementRepo(self)
        self.jobs = JobRepo(self)
        self.llm = LLMRepo(self)

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path | str, clock: Clock = utc_now) -> AsyncIterator[Self]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(path, isolation_level=None)
        try:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA busy_timeout = 5000")
            database = cls(conn, clock)
            await database.migrate()
            yield database
        finally:
            await conn.close()

    async def schema_version(self) -> int:
        async with self.conn.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def migrate(self) -> int:
        current = await self.schema_version()
        for migration in load_migrations():
            if migration.version > current:
                # executescript сам завершает открытую транзакцию, поэтому BEGIN/COMMIT внутри
                await self.conn.executescript(
                    f"BEGIN;\n{migration.sql}\nPRAGMA user_version = {migration.version};\nCOMMIT;"
                )
                current = migration.version
        return current

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Транзакция с блокировкой. Вложенный вызов из той же задачи присоединяется к внешней."""
        task = asyncio.current_task()
        if task is not None and self._owner is task:
            yield
            return
        async with self._lock:
            self._owner = task
            try:
                await self.conn.execute("BEGIN IMMEDIATE")
                try:
                    yield
                except BaseException:
                    await self.conn.execute("ROLLBACK")
                    raise
                await self.conn.execute("COMMIT")
            finally:
                self._owner = None
