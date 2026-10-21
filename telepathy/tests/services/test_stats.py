from datetime import timedelta

import pytest

from telepathy.db import Database
from telepathy.services.errors import NotPairedError
from telepathy.services.matching import MatchingConfig, MatchingService
from telepathy.services.pairing import PairingService
from telepathy.services.stats import StatsService
from tests.conftest import NOW
from tests.fakes import FakeEmbedder, ScriptedJudge

ANYA, BORIS = 1, 2


@pytest.fixture
async def matching(db: Database) -> MatchingService:
    pairing = PairingService(db, clock=lambda: NOW)
    await pairing.register(ANYA, "Аня")
    await pairing.register(BORIS, "Борис")
    await pairing.join(BORIS, await pairing.invite(ANYA))
    return MatchingService(
        db=db,
        embedder=FakeEmbedder({"пицц": [1, 0], "ча": [0, 1]}),
        judge=ScriptedJudge(),
        config=MatchingConfig(window=timedelta(days=30), tau_hours=1000, threshold=0.75),
        clock=lambda: NOW,
    )


async def send(matching: MatchingService, user: int, text: str, ago: timedelta) -> None:
    await matching.find_match(await matching.save(user, text, NOW - ago))


async def test_empty_stats(db: Database, matching: MatchingService) -> None:
    stats = await StatsService(db).for_user(ANYA, NOW)

    assert (stats.matches, stats.week_thoughts, stats.fastest) == (0, 0, None)
    assert stats.week_percent is None


async def test_counts_matches_week_share_and_fastest(
    db: Database, matching: MatchingService
) -> None:
    # Старое совпадение (10 дней назад): в счётчик совпадений входит, в неделю — нет.
    await send(matching, ANYA, "пицца", timedelta(days=10, minutes=30))
    await send(matching, BORIS, "пиццу!", timedelta(days=10))
    # За неделю: совпадение с Δt = 2 мин и одна мысль без пары.
    await send(matching, BORIS, "чаю бы", timedelta(hours=1, minutes=2))
    await send(matching, ANYA, "хочу чай", timedelta(hours=1))
    await send(matching, ANYA, "что-то своё", timedelta(minutes=5))

    stats = await StatsService(db).for_user(BORIS, NOW)

    assert stats.matches == 2
    assert (stats.week_thoughts, stats.week_matched) == (3, 2)
    assert stats.week_percent == pytest.approx(66.67, abs=0.01)
    assert stats.fastest == timedelta(minutes=2)


async def test_stats_need_pair(db: Database) -> None:
    with pytest.raises(NotPairedError):
        await StatsService(db).for_user(42, NOW)
