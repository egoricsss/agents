"""Экстрактор на правилах: работает без сети, служит запасным вариантом для LLM.

Формат промпта: `<Предмет>, лаба <N>[, до ДД.ММ[.ГГГГ] [ЧЧ:ММ]][, <название>]`
(разделители — запятая, точка с запятой или перевод строки). Промпт важнее документа.
"""

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from lab_manager.extraction import LabExtraction

_NUMBER = re.compile(
    r"(?:лабораторн\w*(?:\s+работ\w*)?|лаб\w*\.?|лр|lab)\s*(?:№|#)?\s*(\d{1,3})(?!\d)",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"(?<![\d.])(\d{1,2})\.(\d{1,2})(?:\.(\d{4}|\d{2}))?(?![\d.])"
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


def _clean(text: str) -> str:
    return text.strip().strip("«»\"'*_`").strip(" .:—–-")


class RuleBasedExtractor:
    version = "rules/v1"

    async def extract(
        self, *, markdown: str, prompt: str | None, today: date, tz: ZoneInfo
    ) -> LabExtraction:
        from_prompt = self._from_prompt(prompt or "", today, tz)
        from_doc = self._from_document(markdown, today, tz)
        merged = {key: from_prompt.get(key) or from_doc.get(key) for key in _FIELDS}
        result = LabExtraction(**merged)
        return result.model_copy(update={"confidence": 0.5 if result.missing_fields else 1.0})

    def _from_prompt(self, prompt: str, today: date, tz: ZoneInfo) -> dict[str, object]:
        found: dict[str, object] = {}
        free: list[str] = []
        for raw in _SEGMENTS.split(prompt):
            segment = raw.strip()
            if not segment:
                continue
            consumed = False
            if number := _NUMBER.search(segment):
                found.setdefault("number", int(number[1]))
                consumed = True
            if date_match := _DATE.search(segment):
                deadline = _deadline(date_match, today, tz)
                if deadline is not None:
                    found.setdefault("deadline", deadline)
                consumed = True
            if not consumed:
                free.append(_clean(segment))
        free = [text for text in free if text]
        if free:
            found["subject"] = free[0]
        if len(free) > 1:
            found["title"] = free[1]
        return found

    def _from_document(self, markdown: str, today: date, tz: ZoneInfo) -> dict[str, object]:
        found: dict[str, object] = {}
        for line in markdown.splitlines()[:HEAD_LINES]:
            if "number" not in found and (heading := _LAB_HEADING.match(line)):
                found["number"] = int(heading[1])
                if title := _clean(heading[2]):
                    found["title"] = title
            elif "title" not in found and (topic := _TOPIC.match(line)):
                found["title"] = _clean(topic[1])
            if "number" not in found and (number := _NUMBER.search(line)):
                found["number"] = int(number[1])
            if (
                "deadline" not in found
                and _DEADLINE_LINE.search(line)
                and (date_match := _DATE.search(line))
                and (deadline := _deadline(date_match, today, tz))
            ):
                found["deadline"] = deadline
        return found


_FIELDS = ("subject", "number", "title", "deadline")


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
