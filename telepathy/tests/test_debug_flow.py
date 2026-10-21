"""Отладка в боте: цифры видны обоим, несовпавшие мысли партнёра — нет."""

from datetime import timedelta

import pytest
from sqlalchemy import select

from telepathy.db import Database
from telepathy.db.models import MatchCheck
from tests.bot.harness import ANYA, BORIS, App, make_app
from tests.conftest import NOW

BORIS_PRIVATE = "пойдём в кино"
ANYA_PRIVATE = "кино?"


@pytest.fixture
async def app(db: Database) -> App:
    app = make_app(db)
    await app.pair(ANYA, BORIS)
    # кино у судьи — «разные мысли», пицца — совпадение
    app.judge.decide = lambda first, second: "пицц" in first
    await app.say(ANYA, "хочу пиццу", at=NOW)  # нет кандидатов
    await app.say(BORIS, BORIS_PRIVATE, at=NOW + timedelta(minutes=1))  # ниже порога
    await app.say(ANYA, ANYA_PRIVATE, at=NOW + timedelta(minutes=2))  # судья: разные
    await app.say(BORIS, "пиццу бы", at=NOW + timedelta(minutes=3))  # совпало
    app.tg.reset()
    return app


async def outcomes(db: Database) -> list[str]:
    async with db.session() as session:
        return list(await session.scalars(select(MatchCheck.outcome).order_by(MatchCheck.id)))


async def test_every_search_is_journaled(app: App) -> None:
    assert await outcomes(app.db) == [
        "no_candidates",
        "below_threshold",
        "judge_rejected",
        "matched",
    ]


async def test_judge_outage_is_journaled(app: App) -> None:
    app.judge.fail = True
    await app.say(ANYA, "пицца снова", at=NOW + timedelta(minutes=4))
    await app.say(BORIS, "и я про пиццу", at=NOW + timedelta(minutes=5))

    assert (await outcomes(app.db))[-1] == "judge_unavailable"


@pytest.mark.parametrize("command", ["/thoughts", "/debug", "/matches"])
async def test_partner_unmatched_thoughts_stay_hidden(app: App, command: str) -> None:
    await app.say(ANYA, command)
    await app.say(BORIS, command)

    anya = "\n".join(app.tg.texts(ANYA.id))
    boris = "\n".join(app.tg.texts(BORIS.id))
    assert BORIS_PRIVATE not in anya
    assert ANYA_PRIVATE not in boris


async def test_thoughts_show_own_matched_and_hidden(app: App) -> None:
    await app.say(ANYA, "/thoughts")

    text = app.tg.last(ANYA.id)
    assert "Вы: «хочу пиццу» ✨" in text
    assert "Вы: «кино?»" in text
    assert "Борис: «пиццу бы» ✨" in text
    assert "Борис: 🔒 скрыто" in text
    assert "21.10 15:00:00" in text  # время по Москве


async def test_debug_shows_numbers_and_decisions(app: App) -> None:
    await app.say(BORIS, "/debug")

    text = app.tg.last(BORIS.id)
    for expected in (
        "✨ совпало",
        "судья: мысли разные",
        "ниже порога",
        "нет мыслей партнёра",
        "cos 1.000",
        "порог 0.75",
        "Δt 1 мин",
    ):
        assert expected in text, expected
    assert "Вы: «пойдём в кино»" in text  # свою несовпавшую мысль Борис видит


async def test_matches_show_both_texts_and_cosine(app: App) -> None:
    await app.say(ANYA, "/matches")

    text = app.tg.last(ANYA.id)
    assert "«хочу пиццу»" in text
    assert "«пиццу бы»" in text
    assert "cos 1.000" in text
    assert "Δt 3 мин" in text


async def test_limit_argument(app: App) -> None:
    await app.say(ANYA, "/thoughts 1")

    assert app.tg.last(ANYA.id).count("·") == 1


async def test_debug_needs_pair(db: Database) -> None:
    app = make_app(db)
    await app.say(ANYA, "/debug")

    assert "/pair" in app.tg.last(ANYA.id)
