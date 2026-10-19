"""Экстрактор на правилах: работает без сети, служит запасным вариантом для LLM.

Формат промпта: `<Предмет>, лаба <N>[, до ДД.ММ[.ГГГГ] [ЧЧ:ММ]][, <название>]`
(разделители — запятая, точка с запятой или перевод строки). Промпт важнее документа.
Несколько лаб: «ЛР 3 и ЛР4», «лабы 3, 4», «ЛР 3-5» или несколько заголовков в документе.
"""

import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from lab_manager.extraction import LabExtraction

_LAB_WORD = r"(?:лабораторн\w*(?:\s+работ\w*)?|лаб\w*\.?|лр|lab)\s*(?:№|#)?\s*"
_NUM = r"\d{1,3}(?!\d|\.\d)"  # не часть даты «25.10»
_NUMBER_LIST = re.compile(
    rf"{_LAB_WORD}({_NUM}(?:\s*(?:,|\bи\b|\band\b|-|–|—)\s*(?:{_LAB_WORD})?{_NUM})*)",
    re.IGNORECASE,
)
MAX_RANGE = 10
MAX_SUBJECT_WORDS = 4
MAX_TITLE_WORDS = 8
# После даты допустима точка конца предложения («Срок: 23.10.»), но не продолжение числа.
_DATE = re.compile(
    r"(?<![\d.])(\d{1,2})\.(\d{1,2})(?:\.(\d{4}|\d{2}))?(?!\d|\.\d)"
    r"(?:\s+(?:в\s+)?(\d{1,2}):(\d{2}))?"
)
_DEADLINE_LINE = re.compile(r"срок|дедлайн|сдать|сдача|\bдо\b", re.IGNORECASE)
_LAB_HEADING = re.compile(
    r"^[\s#*]*лабораторн\w*\s+работ\w*\s*(?:№|#)?\s*(\d{1,3})\s*[.:—–-]?\s*(.*)$",
    re.IGNORECASE,
)
_TOPIC = re.compile(r"^[\s#*]*тема\s*[:—–-]\s*(.+)$", re.IGNORECASE)
_SEGMENTS = re.compile(r"[,;\n]")

DEFAULT_TIME = time(23, 59)
PAST_TOLERANCE = timedelta(days=60)  # недавно прошедшая дата — просрочка, а не следующий год
HEAD_LINES = 30


def _is_short(text: str, max_words: int) -> bool:
    return len(text.split()) <= max_words and len(text) <= max_words * 15


def _clean(text: str) -> str:
    return text.strip().strip("«»\"'*_`").strip(" .:—–-")


class RuleBasedExtractor:
    version = "rules/v2"

    async def extract(
        self,
        *,
        markdown: str,
        prompt: str | None,
        today: date,
        tz: ZoneInfo,
        known_subjects: Sequence[str] = (),
    ) -> LabExtraction:
        labs = await self.extract_all(markdown=markdown, prompt=prompt, today=today, tz=tz)
        return labs[0]

    async def extract_all(
        self,
        *,
        markdown: str,
        prompt: str | None,
        today: date,
        tz: ZoneInfo,
        known_subjects: Sequence[str] = (),
    ) -> list[LabExtraction]:
        """Одна или несколько лаб. Общие поля (предмет, срок) — на все; название — только одной."""
        from_prompt = self.from_prompt(prompt or "", today, tz)
        from_doc = self.from_document(markdown, today, tz)
        numbers: list[int | None] = list(
            from_prompt.get("numbers") or from_doc.get("numbers") or [None]
        )
        single = len(numbers) == 1
        conflict = numbers_conflict(from_prompt.get("numbers"), markdown)
        labs = []
        for number in numbers:
            lab = LabExtraction(
                subject=from_prompt.get("subject"),
                number=number,
                title=from_prompt.get("title") or (from_doc.get("title") if single else None),
                deadline=from_prompt.get("deadline") or from_doc.get("deadline"),
            )
            uncertain = bool(lab.missing_fields) or conflict
            labs.append(lab.model_copy(update={"confidence": 0.5 if uncertain else 1.0}))
        return labs

    def from_prompt(self, prompt: str, today: date, tz: ZoneInfo) -> dict[str, Any]:
        found: dict[str, Any] = {}
        numbers: list[int] = []

        def take(match: re.Match[str]) -> str:
            numbers.extend(_parse_numbers(match[1]))
            return " "

        rest = _NUMBER_LIST.sub(take, prompt)
        if numbers:
            found["numbers"] = _unique(numbers)
            found["number"] = found["numbers"][0]
        free: list[str] = []
        for raw in _SEGMENTS.split(rest):
            segment = raw.strip()
            if not segment:
                continue
            if date_match := _DATE.search(segment):
                deadline = _deadline(date_match, today, tz)
                if deadline is not None:
                    found.setdefault("deadline", deadline)
                continue
            free.append(_clean(segment))
        free = [text for text in free if text]
        # Предмет и тема — только из коротких фрагментов подписи («Нейронные сети, ЛР 4»).
        # Длинную фразу («задание по лабораторной по нейронным сетям…») разбирает LLM.
        if free and _is_short(free[0], MAX_SUBJECT_WORDS):
            found["subject"] = free[0]
            if len(free) > 1 and _is_short(free[1], MAX_TITLE_WORDS):
                found["title"] = free[1]
        return found

    def from_document(self, markdown: str, today: date, tz: ZoneInfo) -> dict[str, Any]:
        found: dict[str, Any] = {}
        if headings := heading_numbers(markdown):
            found["numbers"] = headings
        for line in markdown.splitlines()[:HEAD_LINES]:
            if "title" not in found and (heading := _LAB_HEADING.match(line)):
                if title := _clean(heading[2]):
                    found["title"] = title
            elif "title" not in found and (topic := _TOPIC.match(line)):
                found["title"] = _clean(topic[1])
            if "numbers" not in found and (mention := _NUMBER_LIST.search(line)):
                found["numbers"] = _unique(_parse_numbers(mention[1]))
            if (
                "deadline" not in found
                and _DEADLINE_LINE.search(line)
                and (date_match := _DATE.search(line))
                and (deadline := _deadline(date_match, today, tz))
            ):
                found["deadline"] = deadline
        if found.get("numbers"):
            found["number"] = found["numbers"][0]
        return found


def _deadline(match: re.Match[str], today: date, tz: ZoneInfo) -> datetime | None:
    day, month, year_raw, hour, minute = match.groups()
    try:
        at = time(int(hour), int(minute)) if hour is not None else DEFAULT_TIME
        if year_raw is None:
            candidate = date(today.year, int(month), int(day))
            if candidate < today - PAST_TOLERANCE:
                candidate = candidate.replace(year=today.year + 1)
        else:
            year = int(year_raw) + (2000 if len(year_raw) == 2 else 0)
            candidate = date(year, int(month), int(day))
    except ValueError:
        return None
    return datetime.combine(candidate, at, tzinfo=tz)


def parse_deadline(text: str, today: date, tz: ZoneInfo) -> datetime | None:
    """`ДД.ММ[.ГГГГ] [ЧЧ:ММ]` → срок с часовым поясом.

    Год без явного указания выводится из `today`.
    """
    match = _DATE.search(text)
    return _deadline(match, today, tz) if match else None


def heading_number(markdown: str) -> int | None:
    """Номер из первого заголовка «Лабораторная работа №N» (строгий признак)."""
    headings = heading_numbers(markdown)
    return headings[0] if headings else None


def heading_numbers(markdown: str) -> list[int]:
    """Номера всех заголовков «Лабораторная работа №N» в документе, по порядку, без повторов."""
    return _unique(
        int(heading[1]) for line in markdown.splitlines() if (heading := _LAB_HEADING.match(line))
    )


def mentioned_numbers(markdown: str) -> set[int]:
    """Все номера лаб, упомянутые в тексте (заголовки и «ЛР 3», «лабы 3 и 4»)."""
    found = set(heading_numbers(markdown))
    for match in _NUMBER_LIST.finditer(markdown):
        found.update(_parse_numbers(match[1]))
    return found


def numbers_conflict(prompt_numbers: Sequence[int] | None, markdown: str) -> bool:
    """Владелец назвал одни номера, а текст говорит о других (подпись «ЛР 4», в посте «ЛР 3»)."""
    mentioned = mentioned_numbers(markdown)
    return bool(prompt_numbers and mentioned and not mentioned & set(prompt_numbers))


def _parse_numbers(text: str) -> list[int]:
    """«3, 4» / «3 и ЛР4» / «3-5» → [3, 4] / [3, 4] / [3, 4, 5]."""
    tokens = re.findall(r"\d+|[-–—]", text)
    numbers: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isdigit():
            if numbers and index >= 2 and tokens[index - 1] in "-–—":
                start, end = numbers[-1], int(token)
                if start < end <= start + MAX_RANGE:
                    numbers.extend(range(start + 1, end + 1))
                else:
                    numbers.append(end)
            else:
                numbers.append(int(token))
        index += 1
    return [number for number in numbers if number >= 1]


def _unique(numbers: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(numbers))
