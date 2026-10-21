"""Режим gateway: Telegram опрашивает общий шлюз (lab_manager), а он пересылает сюда апдейты
пользователей «Телепатии» по внутренней docker-сети. Формат — как у вебхука Telegram,
поэтому приём делает штатный обработчик вебхуков aiogram; ответы уходят в Bot API напрямую
тем же токеном (опрашивать `getUpdates` может только шлюз, отправлять — кто угодно).
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

log = logging.getLogger(__name__)

UPDATE_PATH = "/telegram/update"
SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"  # noqa: S105 — имя заголовка


def create_gateway_app(
    dp: Dispatcher, bot: Bot, *, secret: str, background: bool = True
) -> web.Application:
    """`background=True`: шлюз получает 200 сразу, не дожидаясь ответа LLM-судьи."""
    app = web.Application()
    SimpleRequestHandler(
        dispatcher=dp, bot=bot, handle_in_background=background, secret_token=secret
    ).register(app, path=UPDATE_PATH)
    setup_application(app, dp, bot=bot)
    return app


async def serve_gateway(
    dp: Dispatcher, bot: Bot, *, secret: str, host: str, port: int, stop: asyncio.Event
) -> None:
    runner = web.AppRunner(create_gateway_app(dp, bot, secret=secret), handle_signals=False)
    await runner.setup()
    try:
        await web.TCPSite(runner, host, port).start()
        log.info("waiting for gateway updates on %s:%s%s", host, port, UPDATE_PATH)
        await stop.wait()
    finally:
        await runner.cleanup()
