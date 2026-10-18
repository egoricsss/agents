"""Предметы: уникальны без учёта регистра и лишних пробелов (включая кириллицу)."""

import pytest

from lab_manager.db import Database


async def test_get_or_create_creates_once(db: Database) -> None:
    first = await db.subjects.get_or_create("Операционные системы")
    second = await db.subjects.get_or_create("Операционные системы")

    assert first.id == second.id
    assert first.name == "Операционные системы"


async def test_name_matching_ignores_case_and_spaces(db: Database) -> None:
    original = await db.subjects.get_or_create("Базы данных")
    variant = await db.subjects.get_or_create("  БАЗЫ   данных ")

    assert variant.id == original.id
    assert variant.name == "Базы данных"


async def test_get_by_name(db: Database) -> None:
    created = await db.subjects.get_or_create("ОС")

    assert await db.subjects.get_by_name("ос") == created
    assert await db.subjects.get_by_name("Сети") is None


async def test_get_by_id(db: Database) -> None:
    created = await db.subjects.get_or_create("ОС")

    assert await db.subjects.get(created.id) == created
    assert await db.subjects.get(999) is None


@pytest.mark.parametrize("name", ["", "   "])
async def test_blank_name_rejected(db: Database, name: str) -> None:
    with pytest.raises(ValueError, match="пуст"):
        await db.subjects.get_or_create(name)


async def test_list_sorted_by_name(db: Database) -> None:
    for name in ["Сети", "Алгоритмы", "Базы данных"]:
        await db.subjects.get_or_create(name)

    assert [s.name for s in await db.subjects.list()] == ["Алгоритмы", "Базы данных", "Сети"]
