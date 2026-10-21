"""Отладка для участников пары: мысли, журнал проверок, совпадения — с цифрами.

Правило приватности (как у самого бота): текст мысли виден, только если она своя или уже
совпала. У несовпавших мыслей партнёра — только время и цифры, без текста.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from telepathy.db import Database
from telepathy.db.models import MatchCheck, Pair, Thought
from telepathy.db.repos import Checks, Matches, Pairs, Thoughts
from telepathy.services.errors import NotPairedError


@dataclass(frozen=True, slots=True)
class ThoughtRef:
    at: datetime
    mine: bool
    matched: bool
    text: str | None  # None — мысль партнёра без совпадения, текст скрыт


@dataclass(frozen=True, slots=True)
class CheckView:
    thought: ThoughtRef
    candidate: ThoughtRef | None
    candidates: int
    cosine: float | None
    score: float | None
    delta: timedelta | None
    threshold: float
    outcome: str


@dataclass(frozen=True, slots=True)
class MatchView:
    earlier: ThoughtRef
    later: ThoughtRef
    delta: timedelta
    score: float
    cosine: float | None
    common_idea: str
    at: datetime


def _ref(thought: Thought, viewer_id: int) -> ThoughtRef:
    mine = thought.user_id == viewer_id
    matched = thought.matched_at is not None
    return ThoughtRef(
        at=thought.created_at,
        mine=mine,
        matched=matched,
        text=thought.text if mine or matched else None,
    )


class DebugService:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    async def _pair(session: AsyncSession, user_id: int) -> Pair:
        pair = await Pairs(session).of_user(user_id)
        if pair is None or not pair.complete:
            raise NotPairedError
        return pair

    async def thoughts(self, user_id: int, limit: int) -> list[ThoughtRef]:
        async with self.db.session() as session:
            pair = await self._pair(session, user_id)
            rows = await Thoughts(session).recent(pair.id, limit)
        return [_ref(thought, user_id) for thought in rows]

    async def checks(self, user_id: int, limit: int) -> list[CheckView]:
        async with self.db.session() as session:
            pair = await self._pair(session, user_id)
            checks = await Checks(session).recent(pair.id, limit)
            ids = {c.thought_id for c in checks} | {
                c.candidate_id for c in checks if c.candidate_id
            }
            thoughts = await Thoughts(session).by_ids(sorted(ids))
        return [self._check_view(check, thoughts, user_id) for check in checks]

    @staticmethod
    def _check_view(check: MatchCheck, thoughts: dict[int, Thought], viewer: int) -> CheckView:
        candidate = thoughts.get(check.candidate_id) if check.candidate_id else None
        return CheckView(
            thought=_ref(thoughts[check.thought_id], viewer),
            candidate=_ref(candidate, viewer) if candidate else None,
            candidates=check.candidates,
            cosine=check.cosine,
            score=check.score,
            delta=None if check.delta_seconds is None else timedelta(seconds=check.delta_seconds),
            threshold=check.threshold,
            outcome=check.outcome,
        )

    async def matches(self, user_id: int, limit: int) -> list[MatchView]:
        async with self.db.session() as session:
            pair = await self._pair(session, user_id)
            matches = await Matches(session).recent(pair.id, limit)
            ids = sorted({m.thought_a_id for m in matches} | {m.thought_b_id for m in matches})
            thoughts = await Thoughts(session).by_ids(ids)
            cosines = {
                c.thought_id: c.cosine
                for c in await Checks(session).recent(pair.id, 10_000)
                if c.outcome == "matched"
            }
        return [
            MatchView(
                earlier=_ref(thoughts[m.thought_a_id], user_id),
                later=_ref(thoughts[m.thought_b_id], user_id),
                delta=timedelta(seconds=m.delta_seconds),
                score=m.score,
                cosine=cosines.get(m.thought_b_id, cosines.get(m.thought_a_id)),
                common_idea=m.common_idea,
                at=m.created_at,
            )
            for m in matches
        ]
