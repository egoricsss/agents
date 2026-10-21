"""Сборка приложения для тестов: aiogram-диспетчер, фейковый Telegram, in-memory SQLite."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import User

from telepathy.bot import create_dispatcher
from telepathy.db import Database
from telepathy.services.matching import MatchingConfig, MatchingService
from telepathy.services.pairing import PairingService
from telepathy.services.stats import StatsService
from tests.bot.fake_telegram import BOT_TOKEN, FakeTelegram, message_update
from tests.conftest import NOW
from tests.fakes import FakeEmbedder, ScriptedJudge

ANYA = User(id=101, is_bot=False, first_name="Аня")
BORIS = User(id=102, is_bot=False, first_name="Борис")
CARL = User(id=103, is_bot=False, first_name="Карл")
ACCEPTED = "🔮 мысль принята"


@dataclass
class App:
    dp: Dispatcher
    bot: Bot
    tg: FakeTelegram
    judge: ScriptedJudge
    db: Database

    async def say(self, user: User, text: str | None, *, at: datetime = NOW, **kw: bool) -> None:
        await self.dp.feed_update(self.bot, message_update(user, text, at=at, **kw))

    async def pair(self, first: User, second: User) -> None:
        await self.say(first, "/pair")
        code = re.search(r"/join ([A-Z0-9]{6})", self.tg.last(first.id))
        assert code is not None
        await self.say(second, f"/join {code.group(1)}")


def make_app(db: Database, *, allowed: tuple[int, ...] = ()) -> App:
    tg = FakeTelegram()
    judge = ScriptedJudge()
    matching = MatchingService(
        db=db,
        embedder=FakeEmbedder({"пицц": [1, 0], "кино": [0, 1]}),
        judge=judge,
        config=MatchingConfig(window=timedelta(hours=24), tau_hours=24, threshold=0.75),
        clock=lambda: NOW,
    )
    dp = create_dispatcher(
        pairing=PairingService(db, clock=lambda: NOW),
        matching=matching,
        stats=StatsService(db),
        allowed_user_ids=allowed,
    )
    bot = Bot(BOT_TOKEN, session=tg, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    return App(dp=dp, bot=bot, tg=tg, judge=judge, db=db)
