"""Точка входа.

`python -m lab_manager` — бот и обработчик очереди; `python -m lab_manager health` — healthcheck.
"""

import asyncio
import contextlib
import logging
import sys
import time
from pathlib import Path

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from lab_manager.bot import LabAgent
from lab_manager.config import Settings
from lab_manager.converter import DocumentConverter
from lab_manager.db import Database
from lab_manager.disk import YandexDiskClient
from lab_manager.extraction.llm import LLMExtractor
from lab_manager.gateway import Gateway
from lab_manager.llm.gigachat import GigaChatClient
from lab_manager.services.disk_sync import DiskSync
from lab_manager.services.executor_prompt import ExecutorPromptBuilder
from lab_manager.services.ingest import BUILD_PROMPT_JOB, IngestService
from lab_manager.services.labs import LabQueries
from lab_manager.services.requirements import RequirementsService
from lab_manager.storage import LabStorage
from lab_manager.worker import JobWorker

log = logging.getLogger("lab_manager")

HEARTBEAT_FILE = "heartbeat"
HEARTBEAT_INTERVAL = 30.0
HEARTBEAT_MAX_AGE = 120.0


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # Библиотеки HTTP на INFO пишут каждый запрос — это шум (секретов в URL нет, но всё же).
    for noisy in ("httpx", "httpcore", "aiogram.event"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def heartbeat(path: Path, stop: asyncio.Event) -> None:
    """Пульс для healthcheck: файл обновляется, пока жив цикл событий."""
    while not stop.is_set():
        await asyncio.to_thread(path.write_text, str(time.time()), encoding="utf-8")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_INTERVAL)


async def run(settings: Settings) -> None:
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tz = settings.zone

    async with (
        Database.open(data_dir / "lab_manager.db") as db,
        GigaChatClient(
            settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            model=settings.gigachat_model,
            ca_bundle=settings.gigachat_ca_bundle_file,
        ) as gigachat,
        YandexDiskClient(settings.yandex_disk_token) as disk,
    ):
        storage = LabStorage(data_dir / "Лабы")
        builder = ExecutorPromptBuilder(
            db=db, storage=storage, disk_root=settings.yandex_disk_root, tz=tz
        )
        ingest = IngestService(
            db=db,
            storage=storage,
            converter=DocumentConverter(cache_dir=data_dir / "cache" / "markdown"),
            extractor=LLMExtractor(gigachat, db),  # при сбое GigaChat сам откатывается к правилам
            tz=tz,
        )
        sync = DiskSync(db=db, storage=storage, disk=disk, root=settings.yandex_disk_root)
        worker = JobWorker(
            db, {BUILD_PROMPT_JOB: builder.handle_job, sync.job_kind: sync.handle_job}
        )

        agent = LabAgent(
            db=db,
            ingest=ingest,
            queries=LabQueries(db, tz),
            requirements=RequirementsService(db=db, builder=builder),
            builder=builder,
            tz=tz,
            disk_root=settings.yandex_disk_root,
        )
        gateway = Gateway([agent], owner_id=settings.telegram_owner_id)
        bot = Bot(
            settings.telegram_bot_token.get_secret_value(),
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        dispatcher = gateway.dispatcher()

        stop = asyncio.Event()
        background = [
            asyncio.create_task(worker.run(stop), name="worker"),
            asyncio.create_task(heartbeat(data_dir / HEARTBEAT_FILE, stop), name="heartbeat"),
        ]
        try:
            me = await bot.get_me()
            log.info("bot @%s started, owner %s", me.username, settings.telegram_owner_id)
            await bot.delete_webhook(drop_pending_updates=False)
            await gateway.set_commands(bot)
            # start_polling сам ловит SIGINT/SIGTERM и завершается штатно.
            await dispatcher.start_polling(bot, allowed_updates=["message", "callback_query"])
        finally:
            log.info("shutting down")
            await gateway.drain()
            stop.set()
            await asyncio.gather(*background, return_exceptions=True)
            await bot.session.close()


def health(settings: Settings) -> int:
    """Код возврата для Docker healthcheck: 0 — пульс свежий."""
    path = settings.data_dir / HEARTBEAT_FILE
    try:
        age = time.time() - float(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    return 0 if age < HEARTBEAT_MAX_AGE else 1


def main() -> None:
    settings = Settings()
    if sys.argv[1:] == ["health"]:
        raise SystemExit(health(settings))
    setup_logging(settings.log_level)
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
