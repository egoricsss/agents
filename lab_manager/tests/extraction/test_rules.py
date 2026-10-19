"""Детерминированный экстрактор.

Формат: «<Предмет>, лаба <N>[, до ДД.ММ[.ГГГГ] [ЧЧ:ММ]][, <название>]».
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from lab_manager.extraction import LabExtraction
from lab_manager.extraction.rules import RuleBasedExtractor

MSK = ZoneInfo("Europe/Moscow")
TODAY = date(2026, 10, 19)


async def extract(prompt: str | None, markdown: str = "") -> LabExtraction:
    return await RuleBasedExtractor().extract(markdown=markdown, prompt=prompt, today=TODAY, tz=MSK)


async def test_short_prompt() -> None:
    result = await extract("ОС, лаба 3, до 25.10")

    assert result.subject == "ОС"
    assert result.number == 3
    assert result.deadline == datetime(2026, 10, 25, 23, 59, tzinfo=MSK)
    assert result.missing_fields == []


async def test_semicolons_number_sign_and_full_year() -> None:
    result = await extract("Базы данных; ЛР №2; сдать до 01.11.2026")

    assert result.subject == "Базы данных"
    assert result.number == 2
    assert result.deadline == datetime(2026, 11, 1, 23, 59, tzinfo=MSK)


@pytest.mark.parametrize(
    "token", ["лаба 7", "Лабораторная 7", "лабораторная работа №7", "лр7", "LAB #7", "лаб. 7"]
)
async def test_number_spellings(token: str) -> None:
    assert (await extract(f"ОС, {token}")).number == 7


async def test_explicit_time() -> None:
    result = await extract("ОС, лаба 1, до 25.10 18:30")

    assert result.deadline == datetime(2026, 10, 25, 18, 30, tzinfo=MSK)


async def test_date_far_in_past_rolls_to_next_year() -> None:
    result = await extract("ОС, лаба 1, до 10.01")

    assert result.deadline == datetime(2027, 1, 10, 23, 59, tzinfo=MSK)


async def test_recently_passed_date_stays_this_year() -> None:
    result = await extract("ОС, лаба 1, до 15.10")

    assert result.deadline == datetime(2026, 10, 15, 23, 59, tzinfo=MSK)


async def test_two_digit_year() -> None:
    result = await extract("ОС, лаба 1, до 05.02.27")

    assert result.deadline == datetime(2027, 2, 5, 23, 59, tzinfo=MSK)


@pytest.mark.parametrize("bad", ["до 31.02", "до 99.99", "до 25.13", "до 25.10 25:00"])
async def test_invalid_date_is_ignored(bad: str) -> None:
    result = await extract(f"ОС, лаба 1, {bad}")

    assert result.deadline is None
    assert result.number == 1


async def test_title_from_prompt_segment() -> None:
    result = await extract("ОС, лаба 3, до 25.10, Процессы и потоки")

    assert result.title == "Процессы и потоки"


async def test_number_and_title_from_document() -> None:
    markdown = "# Лабораторная работа № 5. Сокеты Беркли\n\nНаписать TCP-сервер."

    result = await extract("Сети", markdown)

    assert result.subject == "Сети"
    assert result.number == 5
    assert result.title == "Сокеты Беркли"


async def test_title_from_topic_line() -> None:
    markdown = "Лабораторная работа 2\n\nТема: Индексы в PostgreSQL\n\nЦель: ..."

    result = await extract("БД", markdown)

    assert result.number == 2
    assert result.title == "Индексы в PostgreSQL"


async def test_deadline_from_document() -> None:
    markdown = "Лабораторная работа 1\nСрок сдачи: 30.10.2026"

    result = await extract("ОС", markdown)

    assert result.deadline == datetime(2026, 10, 30, 23, 59, tzinfo=MSK)


async def test_prompt_has_priority_over_document() -> None:
    markdown = "# Лабораторная работа №5. Из документа\nСрок сдачи: 30.10.2026"

    result = await extract("ОС, лаба 6, до 01.11, Из промпта", markdown)

    assert result.number == 6
    assert result.title == "Из промпта"
    assert result.deadline == datetime(2026, 11, 1, 23, 59, tzinfo=MSK)


async def test_missing_subject_reported() -> None:
    result = await extract(None, "Лабораторная работа №4")

    assert result.subject is None
    assert result.number == 4
    assert result.missing_fields == ["subject"]


async def test_nothing_found() -> None:
    result = await extract(None, "")

    assert result.missing_fields == ["subject", "number"]
    assert result.title is None
    assert result.deadline is None


async def test_extractor_has_version_for_cache_keys() -> None:
    assert RuleBasedExtractor().version.startswith("rules/")


def test_extraction_number_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LabExtraction(number=0)
