"""Устойчивая очередь задач: идемпотентная постановка, захват, повторы, восстановление."""

import asyncio

from lab_manager.db import Database
from lab_manager.domain import JobStatus


async def test_enqueue_is_idempotent_by_key(db: Database) -> None:
    job, created = await db.jobs.enqueue("ingest", {"file_id": "abc"}, idempotency_key="tg:1:2")
    same, created_again = await db.jobs.enqueue("ingest", {"other": 1}, idempotency_key="tg:1:2")

    assert created is True
    assert created_again is False
    assert same == job
    assert job.status is JobStatus.PENDING
    assert job.attempts == 0
    assert job.payload == {"file_id": "abc"}


async def test_claim_takes_oldest_pending_and_marks_running(db: Database) -> None:
    first, _ = await db.jobs.enqueue("ingest", {"n": 1}, idempotency_key="a")
    await db.jobs.enqueue("ingest", {"n": 2}, idempotency_key="b")

    claimed = await db.jobs.claim_next()

    assert claimed is not None
    assert claimed.id == first.id
    assert claimed.status is JobStatus.RUNNING
    assert claimed.attempts == 1


async def test_claim_returns_none_when_queue_empty(db: Database) -> None:
    assert await db.jobs.claim_next() is None


async def test_claim_filters_by_kind(db: Database) -> None:
    await db.jobs.enqueue("upload", {}, idempotency_key="u")
    ingest, _ = await db.jobs.enqueue("ingest", {}, idempotency_key="i")

    claimed = await db.jobs.claim_next(kinds=["ingest"])

    assert claimed is not None
    assert claimed.id == ingest.id


async def test_completed_job_is_not_claimed_again(db: Database) -> None:
    job, _ = await db.jobs.enqueue("ingest", {}, idempotency_key="a")
    await db.jobs.claim_next()

    done = await db.jobs.complete(job.id)

    assert done.status is JobStatus.DONE
    assert await db.jobs.claim_next() is None


async def test_fail_retries_until_max_attempts(db: Database) -> None:
    job, _ = await db.jobs.enqueue("ingest", {}, idempotency_key="a")

    await db.jobs.claim_next()
    retry = await db.jobs.fail(job.id, "timeout", max_attempts=2)
    assert retry.status is JobStatus.PENDING
    assert retry.error == "timeout"

    await db.jobs.claim_next()
    final = await db.jobs.fail(job.id, "timeout again", max_attempts=2)
    assert final.status is JobStatus.FAILED
    assert final.attempts == 2
    assert await db.jobs.claim_next() is None


async def test_recover_running_requeues_interrupted_jobs(db: Database) -> None:
    await db.jobs.enqueue("ingest", {}, idempotency_key="a")
    await db.jobs.enqueue("ingest", {}, idempotency_key="b")
    await db.jobs.claim_next()

    recovered = await db.jobs.recover_running()

    assert recovered == 1
    assert (await db.jobs.claim_next()) is not None
    assert (await db.jobs.claim_next()) is not None


async def test_concurrent_claims_never_share_a_job(db: Database) -> None:
    for i in range(5):
        await db.jobs.enqueue("ingest", {"n": i}, idempotency_key=str(i))

    claimed = await asyncio.gather(*(db.jobs.claim_next() for _ in range(8)))

    ids = [job.id for job in claimed if job is not None]
    assert len(ids) == 5
    assert len(set(ids)) == 5
