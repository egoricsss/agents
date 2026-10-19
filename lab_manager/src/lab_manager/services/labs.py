"""Запросы и действия над лабами для интерфейсов (бот, позже MCP).

Списки, неделя, поиск лабы по ссылке, статусы.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from lab_manager.db import Database
from lab_manager.domain import Lab, Source, Status
from lab_manager.errors import NotFoundError

OPEN_STATUSES = (Status.TODO, Status.IN_PROGRESS, Status.BLOCKED)

STATUS_TITLES = {
    Status.TODO: "к выполнению",
    Status.IN_PROGRESS: "в работе",
    Status.BLOCKED: "заблокирована",
    Status.DONE: "сделана",
    Status.SUBMITTED: "сдана",
    Status.DEFENDED: "защищена",
}

STATUS_ICONS = {
    Status.TODO: "⚪",
    Status.IN_PROGRESS: "🔵",
    Status.BLOCKED: "🟠",
    Status.DONE: "✅",
    Status.SUBMITTED: "📤",
    Status.DEFENDED: "🏁",
}

_STATUS_ALIASES = {
    Status.TODO: ["todo", "сделать", "к выполнению", "новая", "не начата", "не начал"],
    Status.IN_PROGRESS: ["in_progress", "in progress", "в работе", "делаю", "начал", "начата"],
    Status.BLOCKED: ["blocked", "блок", "заблокирована", "застряла", "стоп"],
    Status.DONE: ["done", "сделано", "сделана", "готово", "готова", "выполнена", "сделал"],
    Status.SUBMITTED: ["submitted", "сдана", "сдал", "отправлена", "отправил"],
    Status.DEFENDED: ["defended", "защищена", "защитил", "принята", "зачтена"],
}
_ALIAS_TO_STATUS = {
    alias: status for status, aliases in _STATUS_ALIASES.items() for alias in aliases
}


def parse_status(text: str) -> Status | None:
    return _ALIAS_TO_STATUS.get(" ".join(text.casefold().replace("ё", "е").split()))


@dataclass(frozen=True, slots=True)
class LabView:
    lab: Lab
    subject: str


@dataclass(frozen=True, slots=True)
class WeekReport:
    start: datetime
    end: datetime
    overdue: list[LabView]
    this_week: list[LabView]


_REF_ID = re.compile(r"^#(\d+)$")
_REF_TAIL_NUMBER = re.compile(
    r"^(?P<subject>.*?)[\s,]*(?:(?:лаб\w*\.?|лр|lab)\s*(?:№|#)?\s*)?(?P<number>\d{1,3})$",
    re.IGNORECASE,
)
_REF_HEAD_NUMBER = re.compile(
    r"^(?:(?:лаб\w*\.?|лр|lab)\s*(?:№|#)?\s*)?(?P<number>\d{1,3})\s+(?:по\s+)?(?P<subject>.+)$",
    re.IGNORECASE,
)


def _abbreviation(name: str) -> str:
    return "".join(word[0] for word in re.findall(r"\w+", name)).casefold()


def subject_matches(query: str, name: str) -> bool:
    """«ОС» ~ «Операционные системы», «нейронки» ~ «Нейронные сети», «базы» ~ «Базы данных»."""
    q = query.casefold().replace("ё", "е").strip()
    n = name.casefold().replace("ё", "е")
    if not q:
        return True
    if q == n or q == _abbreviation(name) or n.startswith(q) or q in n:
        return True
    q_words, n_words = q.split(), n.split()
    # каждое слово запроса — начало соответствующего слова: «оп сис» ~ «операционные системы»
    if len(q_words) == len(n_words) and all(
        nw.startswith(qw) for qw, nw in zip(q_words, n_words, strict=True)
    ):
        return True
    # общий корень первых слов: «нейронки» / «нейронные»
    return len(q_words[0]) >= 5 and n_words[0][:5] == q_words[0][:5]


class LabQueries:
    def __init__(
        self, db: Database, tz: ZoneInfo, clock: Callable[[], datetime] | None = None
    ) -> None:
        self.db = db
        self.tz = tz
        self.clock = clock or db.clock

    async def views(self, labs: Iterable[Lab]) -> list[LabView]:
        names = {subject.id: subject.name for subject in await self.db.subjects.list()}
        return [LabView(lab, names.get(lab.subject_id, "?")) for lab in labs]

    async def view(self, lab_id: int) -> LabView:
        lab = await self.db.labs.get(lab_id)
        if lab is None:
            raise NotFoundError(f"лаба {lab_id} не найдена")
        return (await self.views([lab]))[0]

    async def open_labs(self) -> list[LabView]:
        return await self.views(await self.db.labs.list(statuses=OPEN_STATUSES))

    async def all_labs(self) -> list[LabView]:
        return await self.views(await self.db.labs.list())

    async def week(self) -> WeekReport:
        """Открытые лабы со сроком на текущей неделе (пн–вс) и просроченные."""
        now = self.clock().astimezone(self.tz)
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=7)
        overdue = await self.db.labs.list(statuses=OPEN_STATUSES, due_before=now)
        this_week = await self.db.labs.list(statuses=OPEN_STATUSES, due_from=now, due_before=end)
        return WeekReport(start, end, await self.views(overdue), await self.views(this_week))

    async def resolve(self, ref: str) -> list[LabView]:
        """Поиск лабы по ссылке: «#12», «ОС 3», «3 по ОС», «нейронки лр 4», «3»."""
        ref = " ".join(ref.split())
        if match := _REF_ID.match(ref):
            lab = await self.db.labs.get(int(match[1]))
            return await self.views([lab]) if lab else []
        match = _REF_TAIL_NUMBER.match(ref) or _REF_HEAD_NUMBER.match(ref)
        if match is None:
            return []
        number = int(match["number"])
        subject_query = re.sub(r"^по\s+", "", match["subject"].strip(" ,"), flags=re.IGNORECASE)
        candidates = [view for view in await self.all_labs() if view.lab.number == number]
        return [view for view in candidates if subject_matches(subject_query, view.subject)]

    async def set_status(self, lab_id: int, status: Status, source: Source) -> Lab:
        return await self.db.labs.set_status(lab_id, status, source)
