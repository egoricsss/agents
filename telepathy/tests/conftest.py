from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from pydantic_ai import models

from telepathy.db import Database

# Ни один тест не ходит в настоящую LLM.
models.ALLOW_MODEL_REQUESTS = False

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database.memory()
    await database.create_schema()
    yield database
    await database.close()
