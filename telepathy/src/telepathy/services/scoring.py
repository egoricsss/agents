"""Оценка кандидата: score = cos(a, b) · exp(−Δt / τ). Чистые функции без I/O."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from telepathy.db.repos import Candidate

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    delta: timedelta  # модуль разницы во времени между мыслями


def time_decay(delta: timedelta, tau_hours: float) -> float:
    """exp(−|Δt| / τ): 1 для одновременных мыслей, 1/e через τ часов."""
    hours = abs(delta.total_seconds()) / SECONDS_PER_HOUR
    return math.exp(-hours / tau_hours)


def score(cosine: float, delta: timedelta, tau_hours: float) -> float:
    return cosine * time_decay(delta, tau_hours)


def top_candidate(
    candidates: Iterable[Candidate], *, at: datetime, tau_hours: float
) -> ScoredCandidate | None:
    """Кандидат с наибольшим score, без учёта порога (для решения и для журнала)."""
    best: ScoredCandidate | None = None
    for candidate in candidates:
        delta = abs(at - candidate.created_at)
        value = score(candidate.cosine, delta, tau_hours)
        if best is None or value > best.score:
            best = ScoredCandidate(candidate=candidate, score=value, delta=delta)
    return best


def best_candidate(
    candidates: Iterable[Candidate], *, at: datetime, tau_hours: float, threshold: float
) -> ScoredCandidate | None:
    """Кандидат с наибольшим score, если он не ниже порога (порог включительно)."""
    best = top_candidate(candidates, at=at, tau_hours=tau_hours)
    if best is None or best.score < threshold:
        return None
    return best
