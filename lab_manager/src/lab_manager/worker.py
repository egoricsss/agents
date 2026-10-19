"""Обработчик устойчивой очереди задач (таблица `jobs`).

Берёт только задачи известных ему типов, поэтому несколько агентов могут делить одну очередь.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from lab_manager.db import Database
from lab_manager.domain import Job

Handler = Callable[[dict[str, Any]], Awaitable[None]]

log = logging.getLogger(__name__)
MAX_ERROR_LENGTH = 500


class JobWorker:
    def __init__(
        self,
        db: Database,
        handlers: Mapping[str, Handler],
        *,
        max_attempts: int = 3,
        poll_interval: float = 1.0,
    ) -> None:
        self.db = db
        self.handlers = dict(handlers)
        self.max_attempts = max_attempts
        self.poll_interval = poll_interval

    async def run_once(self) -> Job | None:
        job = await self.db.jobs.claim_next(kinds=self.handlers)
        if job is None:
            return None
        try:
            await self.handlers[job.kind](job.payload)
        except Exception as exc:  # задача не должна ронять обработчик
            error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_LENGTH]
            failed = await self.db.jobs.fail(job.id, error, max_attempts=self.max_attempts)
            log.warning("job %s (%s) failed: %s -> %s", job.id, job.kind, error, failed.status)
            return failed
        return await self.db.jobs.complete(job.id)

    async def run(self, stop: asyncio.Event) -> None:
        recovered = await self.db.jobs.recover_running()
        if recovered:
            log.info("requeued %d interrupted jobs", recovered)
        while not stop.is_set():
            if await self.run_once() is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.poll_interval)
