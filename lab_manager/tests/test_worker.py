"""Обработчик очереди задач: выполнение, повторы, восстановление, остановка."""

import asyncio
from typing import Any

from lab_manager.db import Database
from lab_manager.domain import JobStatus
from lab_manager.worker import JobWorker


async def test_run_once_executes_handler_and_completes(db: Database) -> None:
    seen: list[dict[str, Any]] = []

    async def handler(payload: dict[str, Any]) -> None:
        seen.append(payload)

    job, _ = await db.jobs.enqueue("disk_sync", {"lab_id": 1}, idempotency_key="a")
    worker = JobWorker(db, {"disk_sync": handler})

    processed = await worker.run_once()

    assert processed is not None and processed.id == job.id
    assert seen == [{"lab_id": 1}]
    done = await db.jobs.get(job.id)
    assert done is not None and done.status is JobStatus.DONE


async def test_run_once_returns_none_when_idle(db: Database) -> None:
    worker = JobWorker(db, {"disk_sync": lambda payload: asyncio.sleep(0)})

    assert await worker.run_once() is None


async def test_failing_handler_is_retried_then_failed(db: Database) -> None:
    calls = 0

    async def handler(payload: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("disk is down")

    job, _ = await db.jobs.enqueue("disk_sync", {}, idempotency_key="a")
    worker = JobWorker(db, {"disk_sync": handler}, max_attempts=2)

    await worker.run_once()
    await worker.run_once()
    await worker.run_once()

    assert calls == 2
    failed = await db.jobs.get(job.id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.error == "RuntimeError: disk is down"


async def test_jobs_of_unknown_kind_are_left_alone(db: Database) -> None:
    other, _ = await db.jobs.enqueue("other_agent", {}, idempotency_key="x")
    worker = JobWorker(db, {"disk_sync": lambda payload: asyncio.sleep(0)})

    assert await worker.run_once() is None
    untouched = await db.jobs.get(other.id)
    assert untouched is not None and untouched.status is JobStatus.PENDING


async def test_run_recovers_interrupted_jobs_and_stops(db: Database) -> None:
    done = asyncio.Event()

    async def handler(payload: dict[str, Any]) -> None:
        done.set()

    await db.jobs.enqueue("disk_sync", {}, idempotency_key="a")
    await db.jobs.claim_next()  # «упали» посреди выполнения
    stop = asyncio.Event()
    worker = JobWorker(db, {"disk_sync": handler}, poll_interval=0.01)

    task = asyncio.create_task(worker.run(stop))
    await asyncio.wait_for(done.wait(), timeout=2)
    stop.set()
    await asyncio.wait_for(task, timeout=2)
