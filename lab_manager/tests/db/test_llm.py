"""Кеш LLM-результатов и журнал вызовов."""

from datetime import UTC, datetime

from lab_manager.db import Database


async def test_cache_roundtrip_and_overwrite(db: Database) -> None:
    assert await db.llm.get_cached("k") is None

    await db.llm.put_cached("k", "extract_lab", {"number": 3, "subject": "ОС"})
    await db.llm.put_cached("k", "extract_lab", {"number": 4})

    assert await db.llm.get_cached("k") == {"number": 4}


async def test_usage_sums_calls_tokens_and_errors(db: Database) -> None:
    for ok in (True, True, False):
        await db.llm.record_call(
            purpose="extract_lab",
            model="GigaChat-2",
            prompt_version="v1",
            latency_ms=100,
            ok=ok,
            prompt_tokens=10,
            completion_tokens=5,
            error=None if ok else "timeout",
        )

    usage = await db.llm.usage()

    assert (usage.calls, usage.errors, usage.prompt_tokens, usage.completion_tokens) == (
        3,
        1,
        30,
        15,
    )


async def test_usage_since(db: Database) -> None:
    await db.llm.record_call(
        purpose="x", model="m", prompt_version="v", latency_ms=1, ok=True, prompt_tokens=7
    )

    assert (await db.llm.usage(since=datetime(2000, 1, 1, tzinfo=UTC))).calls == 1
    assert (await db.llm.usage(since=datetime(2100, 1, 1, tzinfo=UTC))).calls == 0
