"""Репозитории: весь SQL здесь. Транзакциями управляет вызывающий сервис."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from telepathy.db.models import Pair, Thought, ThoughtMatch, User


def to_blob(vector: np.ndarray) -> bytes:
    """Вектор → float32-блоб, который понимают функции sqlite-vec."""
    return np.asarray(vector, dtype=np.float32).tobytes()


@dataclass(frozen=True, slots=True)
class Candidate:
    thought_id: int
    user_id: int
    created_at: datetime
    cosine: float


@dataclass(frozen=True, slots=True)
class MatchSummary:
    count: int
    fastest_seconds: float | None


@dataclass(frozen=True, slots=True)
class ThoughtCounts:
    total: int
    matched: int


class Users:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, user_id: int, first_name: str | None, now: datetime) -> User:
        user = await self.session.get(User, user_id)
        if user is None:
            user = User(id=user_id, first_name=first_name, created_at=now)
            self.session.add(user)
        elif first_name and user.first_name != first_name:
            user.first_name = first_name
        await self.session.flush()
        return user

    async def get(self, user_id: int) -> User | None:
        return await self.session.get(User, user_id)


class Pairs:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def of_user(self, user_id: int) -> Pair | None:
        query = select(Pair).where(or_(Pair.user_a_id == user_id, Pair.user_b_id == user_id))
        return await self.session.scalar(query)

    async def by_code(self, code: str) -> Pair | None:
        return await self.session.scalar(select(Pair).where(Pair.invite_code == code))

    async def add(self, user_a_id: int, invite_code: str, now: datetime) -> Pair:
        pair = Pair(user_a_id=user_a_id, invite_code=invite_code, created_at=now)
        self.session.add(pair)
        await self.session.flush()
        return pair

    async def join(self, pair: Pair, user_b_id: int, now: datetime) -> bool:
        """Занять второе место; False, если его успели занять (условный UPDATE)."""
        result = await self.session.execute(
            update(Pair)
            .where(Pair.id == pair.id, Pair.user_b_id.is_(None))
            .values(user_b_id=user_b_id, joined_at=now)
        )
        await self.session.refresh(pair)
        return result.rowcount == 1  # type: ignore[attr-defined]

    async def remove_pending(self, pair: Pair) -> None:
        await self.session.execute(delete(Pair).where(Pair.id == pair.id, Pair.user_b_id.is_(None)))


class Thoughts:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self, *, user_id: int, pair_id: int, text: str, embedding: np.ndarray, at: datetime
    ) -> Thought:
        thought = Thought(
            user_id=user_id, pair_id=pair_id, text=text, embedding=to_blob(embedding), created_at=at
        )
        self.session.add(thought)
        await self.session.flush()
        return thought

    async def get(self, thought_id: int) -> Thought | None:
        return await self.session.get(Thought, thought_id)

    async def candidates(
        self, *, pair_id: int, author_id: int, embedding: np.ndarray, since: datetime
    ) -> list[Candidate]:
        """Мысли партнёра в паре с момента `since`, ещё ни с чем не совпавшие, с косинусом."""
        cosine = 1 - func.vec_distance_cosine(Thought.embedding, to_blob(embedding))
        query = (
            select(Thought.id, Thought.user_id, Thought.created_at, cosine)
            .where(
                Thought.pair_id == pair_id,
                Thought.user_id != author_id,
                Thought.matched_at.is_(None),
                Thought.created_at >= since,
            )
            .order_by(Thought.created_at)
        )
        rows = await self.session.execute(query)
        return [Candidate(row[0], row[1], row[2], float(row[3])) for row in rows]

    async def mark_matched(self, thought_ids: Sequence[int], at: datetime) -> int:
        """Пометить совпавшими только ещё свободные мысли; вернуть число помеченных."""
        result = await self.session.execute(
            update(Thought)
            .where(Thought.id.in_(thought_ids), Thought.matched_at.is_(None))
            .values(matched_at=at)
        )
        return result.rowcount  # type: ignore[attr-defined]

    async def counts_since(self, pair_id: int, since: datetime) -> ThoughtCounts:
        query = select(func.count(Thought.id), func.count(Thought.matched_at)).where(
            Thought.pair_id == pair_id, Thought.created_at >= since
        )
        total, matched = (await self.session.execute(query)).one()
        return ThoughtCounts(total=total, matched=matched)


class Matches:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        pair_id: int,
        thought_a_id: int,
        thought_b_id: int,
        score: float,
        delta_seconds: float,
        common_idea: str,
        at: datetime,
    ) -> ThoughtMatch:
        match = ThoughtMatch(
            pair_id=pair_id,
            thought_a_id=thought_a_id,
            thought_b_id=thought_b_id,
            score=score,
            delta_seconds=delta_seconds,
            common_idea=common_idea,
            created_at=at,
        )
        self.session.add(match)
        await self.session.flush()
        return match

    async def summary(self, pair_id: int) -> MatchSummary:
        query = select(func.count(ThoughtMatch.id), func.min(ThoughtMatch.delta_seconds)).where(
            ThoughtMatch.pair_id == pair_id
        )
        count, fastest = (await self.session.execute(query)).one()
        return MatchSummary(count=count, fastest_seconds=fastest)
