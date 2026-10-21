"""Точка входа: `python -m telepathy`.

Режим задаёт TELEGRAM_MODE: `polling` — самостоятельный бот со своим токеном,
`gateway` — апдейты присылает шлюз lab_manager (общий бот), см. `bot/gateway.py`.
"""

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # без рекламы внешней трассировки
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import asyncio
import contextlib
import logging
import signal
import sys
import time
from pathlib import Path

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from telepathy.bot import create_dispatcher, set_commands
from telepathy.bot.gateway import serve_gateway
from telepathy.config import Settings
from telepathy.db import Database
from telepathy.health import HEARTBEAT_FILE
from telepathy.services.embeddings import SentenceEmbedder
from telepathy.services.judge import Judge, build_model
from telepathy.services.matching import MatchingConfig, MatchingService
from telepathy.services.pairing import PairingService
from telepathy.services.stats import StatsService

log = logging.getLogger("telepathy")

HEARTBEAT_INTERVAL = 30.0


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # На INFO эти библиотеки пишут каждый запрос — шум.
    for noisy in ("httpx", "httpcore", "aiogram.event", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def heartbeat(path: Path, stop: asyncio.Event) -> None:
    """Пульс для healthcheck: файл обновляется, пока жив цикл событий."""
    while not stop.is_set():
        await asyncio.to_thread(path.write_text, str(time.time()), encoding="utf-8")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_INTERVAL)


async def run(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database.file(settings.database_path)
    await db.create_schema()
    bot = Bot(
        settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    stop = asyncio.Event()
    beat = asyncio.create_task(heartbeat(settings.data_dir / HEARTBEAT_FILE, stop))
    try:
        # Модель грузится один раз, до приёма сообщений; цикл событий при этом свободен.
        embedder = await SentenceEmbedder.load(
            settings.embedding_model,
            threads=settings.embedding_threads,
            cache_folder=settings.data_dir / "models",
        )
        judge = Judge(
            build_model(
                settings.llm_model, api_key=settings.llm_api_key, base_url=settings.llm_base_url
            ),
            timeout=settings.llm_timeout_seconds,
        )
        dp = create_dispatcher(
            pairing=PairingService(db),
            matching=MatchingService(
                db=db,
                embedder=embedder,
                judge=judge,
                config=MatchingConfig.from_settings(settings),
            ),
            stats=StatsService(db),
            allowed_user_ids=settings.allowed_user_ids,
        )
        me = await bot.get_me()
        log.info(
            "bot @%s ready, mode %s, judge %s",
            me.username,
            settings.telegram_mode,
            settings.llm_model,
        )
        if settings.telegram_mode == "polling":
            await bot.delete_webhook(drop_pending_updates=False)
            await set_commands(bot)
            # start_polling сам ловит SIGINT/SIGTERM и завершается штатно.
            await dp.start_polling(bot, allowed_updates=["message"])
        else:
            assert settings.gateway_secret is not None  # noqa: S101 — проверено в Settings
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await serve_gateway(
                dp,
                bot,
                secret=settings.gateway_secret.get_secret_value(),
                host=settings.gateway_host,
                port=settings.gateway_port,
                stop=stop,
            )
    finally:
        log.info("shutting down")
        stop.set()
        await asyncio.gather(beat, return_exceptions=True)
        await bot.session.close()
        await db.close()


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_level)
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
