"""Миграции схемы: применяются один раз, по порядку, с включёнными внешними ключами."""

import sqlite3
from pathlib import Path

import pytest

from lab_manager.db import Database
from lab_manager.db.migrations import load_migrations

EXPECTED_TABLES = {"subjects", "labs", "status_history", "files", "requirements", "jobs"}


async def test_fresh_db_is_migrated_to_latest(db: Database) -> None:
    assert await db.schema_version() == load_migrations()[-1].version


async def test_expected_tables_exist(db: Database) -> None:
    rows = await db.conn.execute_fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")

    assert {row[0] for row in rows} >= EXPECTED_TABLES


def test_migrations_are_numbered_sequentially_from_one() -> None:
    versions = [migration.version for migration in load_migrations()]

    assert versions == list(range(1, len(versions) + 1))


async def test_reopen_keeps_data_and_does_not_reapply(db_path: Path) -> None:
    async with Database.open(db_path) as first:
        await first.subjects.get_or_create("Операционные системы")
        version = await first.schema_version()

    async with Database.open(db_path) as second:
        assert await second.schema_version() == version
        assert [s.name for s in await second.subjects.list()] == ["Операционные системы"]


async def test_foreign_keys_are_enforced(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        await db.conn.execute(
            "INSERT INTO labs (subject_id, number, title, created_at, updated_at)"
            " VALUES (999, 1, 'x', '', '')"
        )


async def test_failed_transaction_is_rolled_back(db: Database) -> None:
    with pytest.raises(RuntimeError):
        async with db.transaction():
            await db.subjects.get_or_create("Сети")
            raise RuntimeError("boom")

    assert await db.subjects.list() == []


async def test_nested_transaction_joins_outer(db: Database) -> None:
    async with db.transaction():
        await db.subjects.get_or_create("Сети")
        async with db.transaction():
            await db.subjects.get_or_create("Базы данных")

    assert len(await db.subjects.list()) == 2
