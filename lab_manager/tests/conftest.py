"""Общие фикстуры: управляемые часы и БД во временной папке."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lab_manager.db import Database


class FakeClock:
    """Детерминированные часы: каждый вызов сдвигает время на 1 секунду."""

    def __init__(self, start: datetime | None = None) -> None:
        self.current = start or datetime(2026, 10, 19, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "lab_manager.db"


@pytest.fixture
async def db(db_path: Path, clock: FakeClock) -> AsyncIterator[Database]:
    async with Database.open(db_path, clock=clock) as database:
        yield database
