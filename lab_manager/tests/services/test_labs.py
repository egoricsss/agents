"""Запросы для интерфейсов: статусы по-русски, поиск лабы по ссылке, неделя."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab_manager.db import Database
from lab_manager.domain import Lab, Source, Status
from lab_manager.services.labs import LabQueries, parse_status, subject_matches

MSK = ZoneInfo("Europe/Moscow")
# conftest.FakeClock: 2026-10-19 09:00 UTC (понедельник) + 1 c на вызов
MONDAY = datetime(2026, 10, 19, tzinfo=MSK)


async def add(db: Database, subject: str, number: int, deadline: datetime | None = None) -> Lab:
    subj = await db.subjects.get_or_create(subject)
    return await db.labs.create(
        subject_id=subj.id,
        number=number,
        title=f"{subject} {number}",
        deadline=deadline,
        source=Source.SYSTEM,
    )


@pytest.fixture
def queries(db: Database) -> LabQueries:
    return LabQueries(db, MSK)


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("сделано", Status.DONE),
        ("Готово", Status.DONE),
        ("в  работе", Status.IN_PROGRESS),
        ("in_progress", Status.IN_PROGRESS),
        ("сдана", Status.SUBMITTED),
        ("защищена", Status.DEFENDED),
        ("заблокирована", Status.BLOCKED),
        ("todo", Status.TODO),
        ("непонятно", None),
    ],
)
def test_parse_status(text: str, status: Status | None) -> None:
    assert parse_status(text) is status


@pytest.mark.parametrize(
    ("query", "name", "expected"),
    [
        ("ОС", "Операционные системы", True),
        ("ос", "ОС", True),
        ("нейронки", "Нейронные сети", True),
        ("базы", "Базы данных", True),
        ("сети", "Компьютерные сети", True),
        ("бд", "Операционные системы", False),
        ("", "Что угодно", True),
    ],
)
def test_subject_matches(query: str, name: str, expected: bool) -> None:
    assert subject_matches(query, name) is expected


@pytest.mark.parametrize(
    "ref", ["ОС 3", "ос, лаба 3", "3 по ОС", "лр 3 ОС", "Операционные системы 3", "оп сис 3"]
)
async def test_resolve_by_subject_and_number(queries: LabQueries, db: Database, ref: str) -> None:
    target = await add(db, "Операционные системы", 3)
    await add(db, "Операционные системы", 4)
    await add(db, "Базы данных", 3)

    [found] = await queries.resolve(ref)

    assert found.lab.id == target.id
    assert found.subject == "Операционные системы"


async def test_resolve_by_number_only_can_be_ambiguous(queries: LabQueries, db: Database) -> None:
    await add(db, "ОС", 3)
    await add(db, "БД", 3)

    assert len(await queries.resolve("3")) == 2
    assert len(await queries.resolve("лаба 3")) == 2


async def test_resolve_by_id_and_misses(queries: LabQueries, db: Database) -> None:
    lab = await add(db, "ОС", 3)

    assert [v.lab.id for v in await queries.resolve(f"#{lab.id}")] == [lab.id]
    assert await queries.resolve("#999") == []
    assert await queries.resolve("ОС 9") == []
    assert await queries.resolve("просто текст") == []


async def test_week_report(queries: LabQueries, db: Database) -> None:
    overdue = await add(db, "ОС", 1, MONDAY - timedelta(days=2))
    this_week = await add(db, "ОС", 2, MONDAY + timedelta(days=3))
    await add(db, "ОС", 3, MONDAY + timedelta(days=10))  # следующая неделя
    await add(db, "ОС", 4)  # без срока
    done = await add(db, "ОС", 5, MONDAY + timedelta(days=1))
    await db.labs.set_status(done.id, Status.DONE, Source.SYSTEM)

    report = await queries.week()

    assert [v.lab.id for v in report.overdue] == [overdue.id]
    assert [v.lab.id for v in report.this_week] == [this_week.id]
    assert report.start == MONDAY
    assert report.end == MONDAY + timedelta(days=7)


async def test_open_labs_exclude_finished(queries: LabQueries, db: Database) -> None:
    open_lab = await add(db, "ОС", 1)
    finished = await add(db, "ОС", 2)
    await queries.set_status(finished.id, Status.SUBMITTED, Source.TELEGRAM)

    assert [v.lab.id for v in await queries.open_labs()] == [open_lab.id]
    assert len(await queries.all_labs()) == 2


def test_clock_is_used(db: Database) -> None:
    fixed = datetime(2026, 12, 31, tzinfo=UTC)
    assert LabQueries(db, MSK, clock=lambda: fixed).clock() == fixed
