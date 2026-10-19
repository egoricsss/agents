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


async def extract_all(prompt: str | None, markdown: str = "") -> list[LabExtraction]:
    return await RuleBasedExtractor().extract_all(
        markdown=markdown, prompt=prompt, today=TODAY, tz=MSK
    )


@pytest.mark.parametrize(
    ("prompt", "numbers"),
    [
        ("Нейросети, ЛР 3 и ЛР4", [3, 4]),
        ("Нейросети, лабы 3, 4, до 25.10", [3, 4]),
        ("Нейросети; лр 3-5", [3, 4, 5]),
        ("Нейросети, лабораторные 2 и 3", [2, 3]),
        ("Нейросети, лаба 3, 25.10", [3]),  # дата без «до» — не номер
    ],
)
async def test_several_numbers_in_prompt(prompt: str, numbers: list[int]) -> None:
    labs = await extract_all(prompt)

    assert [lab.number for lab in labs] == numbers
    assert {lab.subject for lab in labs} == {"Нейросети"}


async def test_shared_deadline_and_no_shared_title() -> None:
    labs = await extract_all("Нейросети, лабы 3 и 4, до 25.10")

    assert all(lab.deadline == datetime(2026, 10, 25, 23, 59, tzinfo=MSK) for lab in labs)
    assert all(lab.title is None for lab in labs)


async def test_several_headings_in_document() -> None:
    markdown = (
        "# Лабораторная работа №3. Свёртки\n\nСделать CNN.\n\n"
        "# Лабораторная работа №4. Трансформеры\n\nСделать attention.\n"
    )

    labs = await extract_all("Нейросети", markdown)

    assert [lab.number for lab in labs] == [3, 4]


async def test_mention_list_in_post() -> None:
    labs = await extract_all("Нейросети", "Коллеги, публикую задания на ЛР 3 и ЛР 4.")

    assert [lab.number for lab in labs] == [3, 4]


async def test_prompt_number_conflicting_with_post_lowers_confidence() -> None:
    post = "📎 Коллеги, публикую задание на ЛР 3.\n\nЛабораторная посвящена RL."

    [lab] = await extract_all("Нейронные сети, ЛР 4", post)

    assert lab.number == 4
    assert lab.confidence == 0.5


async def test_matching_numbers_keep_confidence() -> None:
    [lab] = await extract_all("Нейронные сети, ЛР 3", "Публикую задание на ЛР 3.")

    assert lab.confidence == 1.0


@pytest.mark.parametrize("bad_range", ["лр 3-99", "лр 5-3"])
async def test_suspicious_ranges_are_not_expanded(bad_range: str) -> None:
    labs = await extract_all(f"ОС, {bad_range}")

    assert len(labs) == 2


@pytest.mark.parametrize(
    "text", ["Срок сдачи: 23.10.", "Сдать до 23.10. Удачи!", "Срок: 23.10.2026."]
)
async def test_date_at_end_of_sentence(text: str) -> None:
    [lab] = await extract_all("ОС, ЛР 3", text)

    assert lab.deadline is not None and lab.deadline.date() == date(2026, 10, 23)
