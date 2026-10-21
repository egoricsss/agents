"""Приём мысли и поиск совпадения с мыслями партнёра.

Сохранение и поиск разделены: мысль сохраняется (и подтверждается пользователю) до обращения
к LLM, поэтому недоступность судьи ничего не теряет. Поиск для одной пары сериализован
замком: если оба партнёра пишут одновременно, совпадение найдётся и запишется один раз.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, Self

import numpy as np

from telepathy.config import Settings
from telepathy.db import Database
from telepathy.db.models import MatchCheck, Thought
from telepathy.db.repos import Checks, Matches, Pairs, Thoughts
from telepathy.services.embeddings import Embedder
from telepathy.services.errors import NotPairedError
from telepathy.services.judge import Match
from telepathy.services.scoring import ScoredCandidate, top_candidate

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ThoughtJudge(Protocol):
    async def compare(self, first: str, second: str) -> Match | None: ...


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    window: timedelta
    tau_hours: float
    threshold: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            window=timedelta(hours=settings.match_window_hours),
            tau_hours=settings.tau_hours,
            threshold=settings.score_threshold,
        )


@dataclass(frozen=True, slots=True)
class SavedThought:
    id: int
    user_id: int
    partner_id: int
    pair_id: int
    text: str
    created_at: datetime
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class ThoughtView:
    id: int
    user_id: int
    text: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class FoundMatch:
    match_id: int
    pair_id: int
    score: float
    delta: timedelta
    common_idea: str
    message: str
    earlier: ThoughtView
    later: ThoughtView

    @property
    def user_ids(self) -> tuple[int, int]:
        return self.earlier.user_id, self.later.user_id

    def own_thought(self, user_id: int) -> ThoughtView:
        return self.earlier if self.earlier.user_id == user_id else self.later

    def partner_thought(self, user_id: int) -> ThoughtView:
        return self.later if self.earlier.user_id == user_id else self.earlier


def _view(thought: Thought | SavedThought) -> ThoughtView:
    return ThoughtView(
        id=thought.id, user_id=thought.user_id, text=thought.text, created_at=thought.created_at
    )


class MatchingService:
    def __init__(
        self,
        *,
        db: Database,
        embedder: Embedder,
        judge: ThoughtJudge,
        config: MatchingConfig,
        clock: Clock = utc_now,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.judge = judge
        self.config = config
        self.clock = clock
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def save(self, user_id: int, text: str, at: datetime) -> SavedThought:
        """Сохранить мысль участника полной пары (текст, эмбеддинг, время)."""
        async with self.db.session() as session:
            pair = await Pairs(session).of_user(user_id)
        if pair is None or not pair.complete:
            raise NotPairedError
        partner_id = pair.partner_of(user_id)
        assert partner_id is not None  # noqa: S101 — пара полная, проверено выше
        embedding = await self.embedder.embed(text)
        async with self.db.session() as session, session.begin():
            thought = await Thoughts(session).add(
                user_id=user_id, pair_id=pair.id, text=text, embedding=embedding, at=at
            )
        log.info("thought %s saved (pair %s)", thought.id, pair.id)
        return SavedThought(
            id=thought.id,
            user_id=user_id,
            partner_id=partner_id,
            pair_id=pair.id,
            text=text,
            created_at=at,
            embedding=embedding,
        )

    async def find_match(self, thought: SavedThought) -> FoundMatch | None:
        async with self._locks[thought.pair_id]:
            return await self._find_match(thought)

    async def _find_match(self, thought: SavedThought) -> FoundMatch | None:
        async with self.db.session() as session:
            thoughts = Thoughts(session)
            current = await thoughts.get(thought.id)
            if current is None or current.matched_at is not None:
                return None  # уже совпала с мыслью, которую партнёр прислал одновременно
            candidates = await thoughts.candidates(
                pair_id=thought.pair_id,
                author_id=thought.user_id,
                embedding=thought.embedding,
                since=thought.created_at - self.config.window,
            )
        best = top_candidate(candidates, at=thought.created_at, tau_hours=self.config.tau_hours)
        if best is None or best.score < self.config.threshold:
            log.info("thought %s: no candidate of %d passed threshold", thought.id, len(candidates))
            outcome = "no_candidates" if best is None else "below_threshold"
            await self._record(thought, best, len(candidates), outcome)
            return None

        async with self.db.session() as session:
            other = await Thoughts(session).get(best.candidate.thought_id)
        assert other is not None  # noqa: S101 — мысли не удаляются
        earlier, later = sorted((_view(thought), _view(other)), key=lambda t: t.created_at)
        log.info(
            "thought %s: candidate %s score=%.3f, asking judge", thought.id, other.id, best.score
        )
        verdict = await self.judge.compare(earlier.text, later.text)
        if verdict is None or not verdict.is_same:
            outcome = "judge_unavailable" if verdict is None else "judge_rejected"
            await self._record(thought, best, len(candidates), outcome)
            return None

        async with self.db.session() as session:
            if await Thoughts(session).mark_matched([thought.id, other.id], self.clock()) != 2:
                await session.rollback()
                log.warning("thoughts %s/%s already matched, skipping", thought.id, other.id)
                return None
            match = await Matches(session).add(
                pair_id=thought.pair_id,
                thought_a_id=earlier.id,
                thought_b_id=later.id,
                score=best.score,
                delta_seconds=best.delta.total_seconds(),
                common_idea=verdict.common_idea,
                at=self.clock(),
            )
            await Checks(session).add(self._check(thought, best, len(candidates), "matched"))
            await session.commit()
        log.info("match %s in pair %s (Δt=%s)", match.id, thought.pair_id, best.delta)
        return FoundMatch(
            match_id=match.id,
            pair_id=thought.pair_id,
            score=best.score,
            delta=best.delta,
            common_idea=verdict.common_idea,
            message=verdict.message,
            earlier=earlier,
            later=later,
        )

    def _check(
        self, thought: SavedThought, best: ScoredCandidate | None, count: int, outcome: str
    ) -> MatchCheck:
        return MatchCheck(
            pair_id=thought.pair_id,
            thought_id=thought.id,
            candidate_id=best.candidate.thought_id if best else None,
            candidates=count,
            cosine=best.candidate.cosine if best else None,
            score=best.score if best else None,
            delta_seconds=best.delta.total_seconds() if best else None,
            threshold=self.config.threshold,
            outcome=outcome,
            checked_at=self.clock(),
        )

    async def _record(
        self, thought: SavedThought, best: ScoredCandidate | None, count: int, outcome: str
    ) -> None:
        async with self.db.session() as session, session.begin():
            await Checks(session).add(self._check(thought, best, count, outcome))
