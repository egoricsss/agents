from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from telepathy.db import Database

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database.memory()
    await database.create_schema()
    yield database
    await database.close()
