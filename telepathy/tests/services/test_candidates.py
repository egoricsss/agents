"""Кандидаты: только мысли партнёра в окне времени, ещё не совпавшие, из своей пары."""

from datetime import timedelta

import numpy as np
import pytest

from telepathy.db import Database
from telepathy.db.repos import Pairs, Thoughts, Users
from tests.conftest import NOW

ME, PARTNER, STRANGER_A, STRANGER_B = 1, 2, 3, 4
WINDOW = timedelta(hours=24)
VEC = np.array([1.0, 0.0, 0.0])


@pytest.fixture
async def pairs(db: Database) -> tuple[int, int]:
    async with db.session() as session, session.begin():
        for uid in (ME, PARTNER, STRANGER_A, STRANGER_B):
            await Users(session).upsert(uid, None, NOW)
        ours = await Pairs(session).add(ME, "OURS", NOW)
        await Pairs(session).join(ours, PARTNER, NOW)
        other = await Pairs(session).add(STRANGER_A, "OTHER", NOW)
        await Pairs(session).join(other, STRANGER_B, NOW)
        return ours.id, other.id


async def add(
    db: Database, pair_id: int, user_id: int, ago: timedelta, vec: np.ndarray = VEC
) -> int:
    async with db.session() as session, session.begin():
        thought = await Thoughts(session).add(
            user_id=user_id, pair_id=pair_id, text="…", embedding=vec, at=NOW - ago
        )
        return thought.id


async def candidates(db: Database, pair_id: int) -> list[int]:
    async with db.session() as session:
        found = await Thoughts(session).candidates(
            pair_id=pair_id, author_id=ME, embedding=VEC, since=NOW - WINDOW
        )
    return [c.thought_id for c in found]


async def test_time_window(db: Database, pairs: tuple[int, int]) -> None:
    ours, _ = pairs
    inside = await add(db, ours, PARTNER, timedelta(hours=23, minutes=59))
    edge = await add(db, ours, PARTNER, WINDOW)
    await add(db, ours, PARTNER, timedelta(hours=24, seconds=1))
    await add(db, ours, PARTNER, timedelta(days=3))

    assert await candidates(db, ours) == [edge, inside]


async def test_own_thoughts_excluded(db: Database, pairs: tuple[int, int]) -> None:
    ours, _ = pairs
    await add(db, ours, ME, timedelta(minutes=1))
    partner = await add(db, ours, PARTNER, timedelta(minutes=2))

    assert await candidates(db, ours) == [partner]


async def test_matched_thoughts_excluded(db: Database, pairs: tuple[int, int]) -> None:
    ours, _ = pairs
    matched = await add(db, ours, PARTNER, timedelta(minutes=10))
    free = await add(db, ours, PARTNER, timedelta(minutes=5))
    async with db.session() as session, session.begin():
        assert await Thoughts(session).mark_matched([matched], NOW) == 1

    assert await candidates(db, ours) == [free]


async def test_other_pairs_excluded(db: Database, pairs: tuple[int, int]) -> None:
    ours, other = pairs
    await add(db, other, STRANGER_B, timedelta(minutes=1))

    assert await candidates(db, ours) == []


async def test_cosine_comes_from_sqlite_vec(db: Database, pairs: tuple[int, int]) -> None:
    ours, _ = pairs
    await add(db, ours, PARTNER, timedelta(minutes=1), vec=np.array([1.0, 1.0, 0.0]))
    await add(db, ours, PARTNER, timedelta(minutes=1), vec=np.array([0.0, 0.0, 1.0]))

    async with db.session() as session:
        found = await Thoughts(session).candidates(
            pair_id=ours, author_id=ME, embedding=VEC, since=NOW - WINDOW
        )

    assert [round(c.cosine, 4) for c in found] == [round(1 / np.sqrt(2), 4), 0.0]


async def test_mark_matched_skips_already_matched(db: Database, pairs: tuple[int, int]) -> None:
    ours, _ = pairs
    first = await add(db, ours, PARTNER, timedelta(minutes=1))
    second = await add(db, ours, ME, timedelta(minutes=1))
    async with db.session() as session, session.begin():
        assert await Thoughts(session).mark_matched([first], NOW) == 1
        assert await Thoughts(session).mark_matched([first, second], NOW) == 1
