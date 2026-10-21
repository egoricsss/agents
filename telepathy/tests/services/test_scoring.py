"""score = cos(a, b) · exp(−Δt / τ): затухание по времени и граница порога."""

import math
from datetime import timedelta

import pytest

from telepathy.db.repos import Candidate
from telepathy.services.scoring import best_candidate, score, time_decay
from tests.conftest import NOW

TAU = 24.0


def candidate(thought_id: int, cosine: float, ago: timedelta) -> Candidate:
    return Candidate(thought_id=thought_id, user_id=2, created_at=NOW - ago, cosine=cosine)


def test_no_decay_for_simultaneous_thoughts() -> None:
    assert time_decay(timedelta(0), TAU) == 1.0
    assert score(0.9, timedelta(0), TAU) == pytest.approx(0.9)


def test_decay_after_one_tau_is_one_over_e() -> None:
    assert score(0.9, timedelta(hours=TAU), TAU) == pytest.approx(0.9 / math.e)


def test_decay_is_monotonic_and_ignores_order() -> None:
    values = [time_decay(timedelta(minutes=m), TAU) for m in (0, 1, 30, 60, 600, 1440)]
    assert values == sorted(values, reverse=True)
    assert time_decay(timedelta(hours=-3), TAU) == time_decay(timedelta(hours=3), TAU)


def test_smaller_tau_decays_faster() -> None:
    delta = timedelta(hours=2)
    assert time_decay(delta, 1) < time_decay(delta, 24)


def test_threshold_is_inclusive() -> None:
    exact = candidate(1, 0.75, timedelta(0))
    best = best_candidate([exact], at=NOW, tau_hours=TAU, threshold=0.75)
    assert best is not None
    assert best.score == pytest.approx(0.75)


def test_just_below_threshold_is_rejected() -> None:
    # Через секунду затухание уже опускает 0.75 ниже порога.
    late = candidate(1, 0.75, timedelta(seconds=1))
    assert best_candidate([late], at=NOW, tau_hours=TAU, threshold=0.75) is None


def test_old_similar_thought_loses_to_recent_one() -> None:
    old = candidate(1, 0.95, timedelta(hours=12))  # 0.95 · e^-0.5 ≈ 0.58
    recent = candidate(2, 0.85, timedelta(minutes=5))  # ≈ 0.847
    best = best_candidate([old, recent], at=NOW, tau_hours=TAU, threshold=0.5)
    assert best is not None
    assert best.candidate.thought_id == 2
    assert best.delta == timedelta(minutes=5)


def test_no_candidates() -> None:
    assert best_candidate([], at=NOW, tau_hours=TAU, threshold=0.0) is None
