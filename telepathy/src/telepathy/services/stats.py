"""Статистика пары: число совпадений, доля совпавших мыслей за неделю, лучшая синхронность."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from telepathy.db import Database
from telepathy.db.repos import Matches, Pairs, Thoughts
from telepathy.services.errors import NotPairedError

WEEK = timedelta(days=7)


@dataclass(frozen=True, slots=True)
class PairStats:
    matches: int
    week_thoughts: int
    week_matched: int
    fastest: timedelta | None

    @property
    def week_percent(self) -> float | None:
        if self.week_thoughts == 0:
            return None
        return 100.0 * self.week_matched / self.week_thoughts


class StatsService:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def for_user(self, user_id: int, now: datetime) -> PairStats:
        async with self.db.session() as session:
            pair = await Pairs(session).of_user(user_id)
            if pair is None or not pair.complete:
                raise NotPairedError
            summary = await Matches(session).summary(pair.id)
            week = await Thoughts(session).counts_since(pair.id, now - WEEK)
        fastest = (
            None if summary.fastest_seconds is None else timedelta(seconds=summary.fastest_seconds)
        )
        return PairStats(
            matches=summary.count,
            week_thoughts=week.total,
            week_matched=week.matched,
            fastest=fastest,
        )
