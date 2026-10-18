"""Лабораторные: создание, поиск, фильтры, статусы с историей."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from lab_manager.db import Database
from lab_manager.domain import Lab, Source, Status
from lab_manager.errors import AlreadyExistsError, NotFoundError

MSK = timezone(timedelta(hours=3))


async def make_lab(
    db: Database,
    number: int = 1,
    subject: str = "ОС",
    deadline: datetime | None = None,
    title: str | None = None,
) -> Lab:
    subj = await db.subjects.get_or_create(subject)
    return await db.labs.create(
        subject_id=subj.id,
        number=number,
        title=title or f"Лаба {number}",
        deadline=deadline,
        source=Source.TELEGRAM,
    )


async def test_create_defaults(db: Database) -> None:
    lab = await make_lab(db, title="Процессы и потоки")

    assert lab.id > 0
    assert lab.number == 1
    assert lab.title == "Процессы и потоки"
    assert lab.status is Status.TODO
    assert lab.deadline is None
    assert lab.summary == ""
    assert lab.created_at == lab.updated_at


async def test_create_records_initial_status(db: Database) -> None:
    lab = await make_lab(db)

    [change] = await db.labs.history(lab.id)
    assert change.old_status is None
    assert change.new_status is Status.TODO
    assert change.source is Source.TELEGRAM


async def test_duplicate_number_in_subject_rejected(db: Database) -> None:
    await make_lab(db, number=1)

    with pytest.raises(AlreadyExistsError):
        await make_lab(db, number=1)


async def test_same_number_in_other_subject_allowed(db: Database) -> None:
    a = await make_lab(db, number=1, subject="ОС")
    b = await make_lab(db, number=1, subject="Сети")

    assert a.id != b.id


@pytest.mark.parametrize("number", [0, -1])
async def test_number_must_be_positive(db: Database, number: int) -> None:
    with pytest.raises(ValueError):
        await make_lab(db, number=number)


async def test_unknown_subject_rejected(db: Database) -> None:
    with pytest.raises(NotFoundError):
        await db.labs.create(subject_id=999, number=1, title="x", source=Source.SYSTEM)


async def test_get_and_find(db: Database) -> None:
    lab = await make_lab(db, number=3)

    assert await db.labs.get(lab.id) == lab
    assert await db.labs.get(999) is None
    assert await db.labs.find(lab.subject_id, 3) == lab
    assert await db.labs.find(lab.subject_id, 4) is None


async def test_deadline_must_be_timezone_aware(db: Database) -> None:
    with pytest.raises(ValueError, match="час"):
        await make_lab(db, deadline=datetime(2026, 10, 25, 23, 59))


async def test_deadline_roundtrip_is_same_instant_in_utc(db: Database) -> None:
    deadline = datetime(2026, 10, 25, 23, 59, tzinfo=MSK)
    lab = await make_lab(db, deadline=deadline)

    stored = await db.labs.get(lab.id)
    assert stored is not None
    assert stored.deadline == deadline
    assert stored.deadline is not None and stored.deadline.tzinfo == UTC


async def test_set_status_updates_and_appends_history(db: Database) -> None:
    lab = await make_lab(db)

    updated = await db.labs.set_status(lab.id, Status.IN_PROGRESS, Source.TELEGRAM)
    done = await db.labs.set_status(lab.id, Status.DONE, Source.MCP)

    assert updated.status is Status.IN_PROGRESS
    assert done.status is Status.DONE
    assert done.updated_at > lab.updated_at
    history = await db.labs.history(lab.id)
    assert [(h.old_status, h.new_status, h.source) for h in history] == [
        (None, Status.TODO, Source.TELEGRAM),
        (Status.TODO, Status.IN_PROGRESS, Source.TELEGRAM),
        (Status.IN_PROGRESS, Status.DONE, Source.MCP),
    ]


async def test_any_transition_is_allowed(db: Database) -> None:
    lab = await make_lab(db)
    await db.labs.set_status(lab.id, Status.DEFENDED, Source.TELEGRAM)

    back = await db.labs.set_status(lab.id, Status.TODO, Source.TELEGRAM)

    assert back.status is Status.TODO


async def test_set_same_status_is_noop(db: Database) -> None:
    lab = await make_lab(db)

    same = await db.labs.set_status(lab.id, Status.TODO, Source.TELEGRAM)

    assert same == lab
    assert len(await db.labs.history(lab.id)) == 1


async def test_set_status_unknown_lab(db: Database) -> None:
    with pytest.raises(NotFoundError):
        await db.labs.set_status(999, Status.DONE, Source.TELEGRAM)


async def test_update_fields_partially(db: Database) -> None:
    lab = await make_lab(db, title="Старое")
    deadline = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)

    updated = await db.labs.update(lab.id, title="Новое", deadline=deadline)
    summarized = await db.labs.update(lab.id, summary="Кратко")

    assert updated.title == "Новое"
    assert updated.deadline == deadline
    assert summarized.title == "Новое"
    assert summarized.deadline == deadline
    assert summarized.summary == "Кратко"


async def test_update_can_clear_deadline(db: Database) -> None:
    lab = await make_lab(db, deadline=datetime(2026, 11, 1, tzinfo=UTC))

    cleared = await db.labs.update(lab.id, deadline=None)

    assert cleared.deadline is None


async def test_update_unknown_lab(db: Database) -> None:
    with pytest.raises(NotFoundError):
        await db.labs.update(999, title="x")


async def test_list_orders_by_deadline_nulls_last(db: Database) -> None:
    base = datetime(2026, 10, 20, tzinfo=UTC)
    await make_lab(db, number=1, deadline=None)
    await make_lab(db, number=2, deadline=base + timedelta(days=5))
    await make_lab(db, number=3, deadline=base + timedelta(days=1))

    assert [lab.number for lab in await db.labs.list()] == [3, 2, 1]


async def test_list_filters_by_status(db: Database) -> None:
    a = await make_lab(db, number=1)
    await make_lab(db, number=2)
    await db.labs.set_status(a.id, Status.DONE, Source.TELEGRAM)

    open_labs = await db.labs.list(statuses=[Status.TODO, Status.IN_PROGRESS])

    assert [lab.number for lab in open_labs] == [2]


async def test_list_filters_by_subject(db: Database) -> None:
    os_lab = await make_lab(db, subject="ОС")
    await make_lab(db, subject="Сети")

    assert await db.labs.list(subject_id=os_lab.subject_id) == [os_lab]


async def test_list_filters_by_deadline_range(db: Database) -> None:
    monday = datetime(2026, 10, 19, tzinfo=MSK)
    await make_lab(db, number=1, deadline=monday - timedelta(days=1))
    await make_lab(db, number=2, deadline=monday + timedelta(days=2))
    await make_lab(db, number=3, deadline=monday + timedelta(days=7))
    await make_lab(db, number=4, deadline=None)

    week = await db.labs.list(due_from=monday, due_before=monday + timedelta(days=7))

    assert [lab.number for lab in week] == [2]
