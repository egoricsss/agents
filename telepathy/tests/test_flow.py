"""Интеграционный тест: весь флоу через aiogram на in-memory SQLite, без сети."""

import logging
import re
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from telepathy.db import Database
from telepathy.services.pairing import PairingService
from tests.bot.harness import ACCEPTED, ANYA, BORIS, CARL, App, make_app
from tests.conftest import NOW


@pytest.fixture
async def app(db: Database) -> AsyncIterator[App]:
    yield make_app(db)


async def test_full_flow(app: App) -> None:
    tg = app.tg
    await app.say(ANYA, "/start")
    assert "Телепатия" in tg.last(ANYA.id)
    await app.say(BORIS, "/start")

    await app.say(ANYA, "/pair")
    code = re.search(r"/join ([A-Z0-9]{6})", tg.last(ANYA.id))
    assert code is not None

    await app.say(ANYA, f"/join {code.group(1)}")
    assert "собственный код" in tg.last(ANYA.id)

    await app.say(BORIS, f"/join {code.group(1)}")
    assert "Аня" in tg.last(BORIS.id)
    assert "Борис" in tg.last(ANYA.id)

    await app.say(CARL, f"/join {code.group(1)}")
    assert "уже двое" in tg.last(CARL.id)

    tg.reset()
    await app.say(ANYA, "хочу <пиццу>", at=NOW)
    assert tg.texts(ANYA.id) == [ACCEPTED]
    assert tg.texts(BORIS.id) == []

    await app.say(BORIS, "пойдём в кино", at=NOW + timedelta(minutes=1))
    assert tg.texts(BORIS.id) == [ACCEPTED]
    assert tg.texts(ANYA.id) == [ACCEPTED]  # мысль без совпадения партнёру не раскрыта

    await app.say(BORIS, "пиццу бы", at=NOW + timedelta(minutes=3))
    boris_match = tg.last(BORIS.id)
    anya_match = tg.last(ANYA.id)
    for text in (boris_match, anya_match):
        assert "Вы подумали об одном и том же" in text
        assert "хочу &lt;пиццу&gt;" in text
        assert "пиццу бы" in text
        assert "кино" not in text
        assert "3 мин" in text
    assert "Аня" in boris_match
    assert "Борис" in anya_match
    assert app.judge.calls == [("хочу <пиццу>", "пиццу бы")]
    assert all("кино" not in text for text in tg.texts(ANYA.id))

    await app.say(ANYA, "/stats", at=NOW + timedelta(minutes=5))
    stats = tg.last(ANYA.id)
    assert "Совпадений: 1" in stats
    assert "67%" in stats
    assert "3 мин" in stats


async def test_llm_down_keeps_bot_alive(app: App, caplog: pytest.LogCaptureFixture) -> None:
    await app.pair(ANYA, BORIS)
    app.judge.fail = True

    with caplog.at_level(logging.INFO):
        await app.say(ANYA, "хочу пиццу")
        await app.say(BORIS, "пиццу!")

    assert app.tg.last(BORIS.id) == ACCEPTED
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "пицц" not in caplog.text  # тексты мыслей не попадают в логи

    app.judge.fail = False
    await app.say(ANYA, "пицца-пицца")
    assert "Вы подумали об одном и том же" in app.tg.last(ANYA.id)


async def test_thought_without_pair(app: App) -> None:
    await app.say(ANYA, "просто мысль")
    assert "/pair" in app.tg.last(ANYA.id)


async def test_non_text_and_unknown_command(app: App) -> None:
    await app.say(ANYA, None, photo=True)
    assert "текст" in app.tg.last(ANYA.id)

    await app.say(ANYA, "/wat")
    assert "/help" in app.tg.last(ANYA.id)


async def test_join_without_code(app: App) -> None:
    await app.say(ANYA, "/join")
    assert "/join КОД" in app.tg.last(ANYA.id)


async def test_stats_before_pair(app: App) -> None:
    await app.say(ANYA, "/stats")
    assert "/pair" in app.tg.last(ANYA.id)


async def test_blocked_partner_does_not_break_match(app: App) -> None:
    await app.pair(ANYA, BORIS)
    await app.say(ANYA, "хочу пиццу")
    app.tg.blocked.add(ANYA.id)

    await app.say(BORIS, "пиццу!")

    assert "Вы подумали об одном и том же" in app.tg.last(BORIS.id)


async def test_allowed_users_only(db: Database) -> None:
    app = make_app(db, allowed=(ANYA.id,))

    await app.say(CARL, "/start")
    await app.say(CARL, "мысль")
    await app.say(ANYA, "/start")

    assert app.tg.texts(CARL.id) == []
    assert app.tg.texts(ANYA.id)
    assert await PairingService(db).display_name(CARL.id) == "Партнёр"  # не зарегистрирован
