"""Матчинг целиком: сохранение, кандидаты, порог, судья, пометка совпавших."""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import func, select

from telepathy.db import Database
from telepathy.db.models import Thought, ThoughtMatch
from telepathy.db.repos import Pairs, Thoughts, Users
from telepathy.services.errors import NotPairedError, ThoughtLimitError
from telepathy.services.matching import MatchingConfig, MatchingService
from tests.conftest import NOW
from tests.fakes import FakeEmbedder, ScriptedJudge

ME, PARTNER, LONELY, WAITING = 1, 2, 3, 4
CONFIG = MatchingConfig(window=timedelta(hours=24), tau_hours=24, threshold=0.75)


@pytest.fixture
async def paired(db: Database) -> None:
    async with db.session() as session, session.begin():
        for uid in (ME, PARTNER, LONELY, WAITING):
            await Users(session).upsert(uid, None, NOW)
        pair = await Pairs(session).add(ME, "OURS", NOW)
        await Pairs(session).join(pair, PARTNER, NOW)
        await Pairs(session).add(WAITING, "WAIT", NOW)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder({"пицц": [1, 0], "кино": [0, 1], "фильм": [0.2, 1]})


@pytest.fixture
def judge() -> ScriptedJudge:
    return ScriptedJudge()


@pytest.fixture
def service(
    db: Database, paired: None, embedder: FakeEmbedder, judge: ScriptedJudge
) -> MatchingService:
    return MatchingService(db=db, embedder=embedder, judge=judge, config=CONFIG, clock=lambda: NOW)


async def matched_ids(db: Database) -> set[int]:
    async with db.session() as session:
        rows = await session.scalars(select(Thought.id).where(Thought.matched_at.is_not(None)))
        return set(rows)


async def match_count(db: Database) -> int:
    async with db.session() as session:
        return int(await session.scalar(select(func.count(ThoughtMatch.id))) or 0)


@pytest.mark.parametrize("user", [LONELY, WAITING])
async def test_thoughts_need_a_complete_pair(
    service: MatchingService, embedder: FakeEmbedder, user: int
) -> None:
    with pytest.raises(NotPairedError):
        await service.save(user, "мысль", NOW)
    assert embedder.calls == []


async def test_thought_saved_with_embedding_and_time(
    service: MatchingService, db: Database
) -> None:
    saved = await service.save(ME, "хочу пиццу", NOW)

    async with db.session() as session:
        thought = await Thoughts(session).get(saved.id)
    assert thought is not None
    assert (thought.user_id, thought.text, thought.created_at) == (ME, "хочу пиццу", NOW)
    assert len(thought.embedding) == 16 * 4  # float32
    assert saved.partner_id == PARTNER


async def test_same_thoughts_match(
    service: MatchingService, db: Database, judge: ScriptedJudge
) -> None:
    first = await service.save(ME, "хочу пиццу", NOW - timedelta(minutes=10))
    assert await service.find_match(first) is None

    second = await service.save(PARTNER, "Пиццу бы сейчас", NOW)
    found = await service.find_match(second)

    assert found is not None
    assert judge.calls == [("хочу пиццу", "Пиццу бы сейчас")]
    assert found.delta == timedelta(minutes=10)
    assert found.own_thought(PARTNER).text == "Пиццу бы сейчас"
    assert found.partner_thought(PARTNER).text == "хочу пиццу"
    assert found.message
    assert await matched_ids(db) == {first.id, second.id}
    assert await match_count(db) == 1


async def test_judge_says_different(
    service: MatchingService, db: Database, judge: ScriptedJudge
) -> None:
    judge.decide = lambda first, second: False
    await service.save(ME, "хочу пиццу", NOW - timedelta(minutes=1))
    second = await service.save(PARTNER, "не хочу пиццу", NOW)

    assert await service.find_match(second) is None
    assert len(judge.calls) == 1
    assert await matched_ids(db) == set()


async def test_below_threshold_does_not_reach_judge(
    service: MatchingService, judge: ScriptedJudge
) -> None:
    await service.save(ME, "пойдём в кино", NOW - timedelta(minutes=1))
    second = await service.save(PARTNER, "хочу пиццу", NOW)

    assert await service.find_match(second) is None
    assert judge.calls == []


async def test_time_decay_blocks_old_thoughts(
    service: MatchingService, judge: ScriptedJudge
) -> None:
    await service.save(ME, "хочу пиццу", NOW - timedelta(hours=23))  # 1 · e^(-23/24) ≈ 0.38
    second = await service.save(PARTNER, "хочу пиццу", NOW)

    assert await service.find_match(second) is None
    assert judge.calls == []


async def test_close_paraphrase_passes_threshold(
    service: MatchingService, judge: ScriptedJudge
) -> None:
    await service.save(ME, "посмотрим кино?", NOW - timedelta(minutes=2))  # cos ≈ 0.98
    second = await service.save(PARTNER, "хочу фильм", NOW)

    assert await service.find_match(second) is not None


async def test_llm_down_keeps_thought_and_it_can_match_later(
    service: MatchingService, db: Database, judge: ScriptedJudge
) -> None:
    first = await service.save(ME, "хочу пиццу", NOW - timedelta(minutes=5))
    judge.fail = True
    second = await service.save(PARTNER, "пиццу!", NOW - timedelta(minutes=4))

    assert await service.find_match(second) is None
    assert await matched_ids(db) == set()

    judge.fail = False
    third = await service.save(ME, "пицца, срочно", NOW)
    found = await service.find_match(third)

    assert found is not None
    assert {found.earlier.id, found.later.id} == {second.id, third.id}
    assert first.id not in await matched_ids(db)


async def test_matched_thoughts_are_not_reused(
    service: MatchingService, judge: ScriptedJudge
) -> None:
    await service.save(ME, "хочу пиццу", NOW - timedelta(minutes=2))
    assert await service.find_match(await service.save(PARTNER, "пиццу!", NOW)) is not None

    again = await service.save(ME, "всё ещё хочу пиццу", NOW)

    assert await service.find_match(again) is None
    assert len(judge.calls) == 1


async def test_simultaneous_thoughts_match_once(
    service: MatchingService, db: Database, judge: ScriptedJudge
) -> None:
    mine = await service.save(ME, "пиццу?", NOW)
    theirs = await service.save(PARTNER, "пиццу!", NOW)

    results = await asyncio.gather(service.find_match(mine), service.find_match(theirs))

    assert sum(result is not None for result in results) == 1
    assert len(judge.calls) == 1
    assert await match_count(db) == 1


async def test_hourly_limit_per_user(db: Database, paired: None, embedder: FakeEmbedder) -> None:
    service = MatchingService(
        db=db,
        embedder=embedder,
        judge=ScriptedJudge(),
        config=MatchingConfig(
            window=timedelta(hours=24), tau_hours=24, threshold=0.75, max_per_hour=3
        ),
        clock=lambda: NOW,
    )
    for minutes in (0, 10, 20):
        await service.save(ME, f"мысль {minutes}", NOW + timedelta(minutes=minutes))
    # партнёра лимит не касается
    await service.save(PARTNER, "моя мысль", NOW + timedelta(minutes=30))

    with pytest.raises(ThoughtLimitError) as error:
        await service.save(ME, "четвёртая", NOW + timedelta(minutes=30))

    assert error.value.retry_after == timedelta(minutes=30)  # освободится через час после первой
    assert "3" in str(error.value)
    assert "30 мин" in str(error.value)
    assert embedder.calls.count("четвёртая") == 0  # лишняя мысль даже не считается моделью

    # скользящее окно: через час после первой мысли можно снова
    await service.save(ME, "уже можно", NOW + timedelta(hours=1))


async def test_default_limit_is_30_per_hour() -> None:
    assert CONFIG.max_per_hour == 30
