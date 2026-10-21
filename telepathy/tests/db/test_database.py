from datetime import timedelta, timezone

import numpy as np
from sqlalchemy import text

from telepathy.db import Database
from telepathy.db.repos import Pairs, Thoughts, Users
from tests.conftest import NOW


async def test_sqlite_vec_is_loaded(db: Database) -> None:
    async with db.session() as session:
        version = await session.scalar(text("select vec_version()"))
    assert str(version).startswith("v")


async def test_foreign_keys_enforced(db: Database) -> None:
    async with db.session() as session:
        enabled = await session.scalar(text("pragma foreign_keys"))
    assert enabled == 1


async def test_file_database_persists(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "sub" / "t.db"
    first = Database.file(path)
    await first.create_schema()
    async with first.session() as session, session.begin():
        await Users(session).upsert(1, "Аня", NOW)
    await first.close()

    second = Database.file(path)
    async with second.session() as session:
        user = await Users(session).get(1)
    await second.close()
    assert user is not None
    assert user.first_name == "Аня"


async def test_times_roundtrip_as_utc(db: Database) -> None:
    moscow = timezone(timedelta(hours=3))
    async with db.session() as session, session.begin():
        await Users(session).upsert(1, None, NOW.astimezone(moscow))
    async with db.session() as session:
        user = await Users(session).get(1)
    assert user is not None
    assert user.created_at == NOW
    assert user.created_at.tzinfo is not None


async def test_upsert_updates_name(db: Database) -> None:
    async with db.session() as session, session.begin():
        await Users(session).upsert(1, "Аня", NOW)
        await Users(session).upsert(1, "Анна", NOW)
        await Users(session).upsert(1, None, NOW)
    async with db.session() as session:
        user = await Users(session).get(1)
    assert user is not None
    assert user.first_name == "Анна"


async def test_second_seat_taken_once(db: Database) -> None:
    async with db.session() as session, session.begin():
        users = Users(session)
        for uid in (1, 2, 3):
            await users.upsert(uid, None, NOW)
        pair = await Pairs(session).add(1, "ABC123", NOW)

        assert await Pairs(session).join(pair, 2, NOW) is True
        assert await Pairs(session).join(pair, 3, NOW) is False

        assert pair.user_b_id == 2
        assert pair.partner_of(1) == 2
        assert pair.partner_of(2) == 1


async def test_thought_embedding_stored_as_float32(db: Database) -> None:
    async with db.session() as session, session.begin():
        await Users(session).upsert(1, None, NOW)
        await Users(session).upsert(2, None, NOW)
        pair = await Pairs(session).add(1, "X", NOW)
        thought = await Thoughts(session).add(
            user_id=1, pair_id=pair.id, text="мысль", embedding=np.array([0.6, 0.8]), at=NOW
        )
    assert np.frombuffer(thought.embedding, dtype=np.float32).tolist() == [
        np.float32(0.6),
        np.float32(0.8),
    ]
