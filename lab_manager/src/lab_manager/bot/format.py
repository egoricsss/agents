"""Тексты сообщений бота (HTML). Все данные пользователя и документов экранируются.

Стиль: коротко, по-человечески, с понятной иерархией — заголовок, срок с обратным отсчётом,
суть цитатой, длинные списки — в сворачиваемом блоке.
"""

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

from lab_manager.domain import Lab, Requirement, Status
from lab_manager.extraction import LabExtraction
from lab_manager.services.ingest import IngestedLab, IngestOutcome, IngestWarning
from lab_manager.services.labs import STATUS_ICONS, STATUS_TITLES, LabView, WeekReport

FIELD_TITLES = {"subject": "предмет", "number": "номер лабы"}
PROMPT_EXAMPLE = "«ОС, ЛР 3, до 25.10»"
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
FINISHED = (Status.DONE, Status.SUBMITTED, Status.DEFENDED)
LONG_LIST = 4  # длиннее — прячем в сворачиваемый блок


# ------------------------------------------------------------------ мелкие помощники


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def deadline(value: datetime | None, tz: ZoneInfo, *, short: bool = False) -> str:
    if value is None:
        return "не указан"
    local = value.astimezone(tz)
    return f"{local:%d.%m}" if short else f"{local:%d.%m.%Y %H:%M}"


def countdown(value: datetime | None, tz: ZoneInfo, now: datetime | None = None) -> str:
    """«сегодня», «завтра», «через 3 дня», «⏰ просрочено на 2 дня»."""
    if value is None:
        return ""
    today = (now or datetime.now(tz)).astimezone(tz).date()
    days = (value.astimezone(tz).date() - today).days
    if days < 0:
        return f"⏰ просрочено на {-days} {plural(days, 'день', 'дня', 'дней')}"
    if days == 0:
        return "🔥 сегодня"
    if days == 1:
        return "завтра"
    return f"через {days} {plural(days, 'день', 'дня', 'дней')}"


def due(value: datetime | None, tz: ZoneInfo, now: datetime | None = None) -> str:
    """Строка срока: «⏳ до 01.11.2026 23:59 · через 13 дней» или «⏳ срок не указан»."""
    if value is None:
        return "⏳ срок не указан"
    return f"⏳ до {deadline(value, tz)} · {countdown(value, tz, now)}"


def progress(done: int, total: int, width: int = 5) -> str:
    filled = round(width * done / total) if total else 0
    return "▰" * filled + "▱" * (width - filled)


def disk_path(root: str, folder: str | None) -> str:
    """Человекочитаемый путь на Диске: без технических префиксов `app:/`, `disk:/`."""
    if not folder:
        return "папка ещё не создана"
    base = root.split(":/", 1)[-1].strip("/")
    return f"{base}/{folder}" if base else folder


def _quote(text: str, *, expandable: bool = False) -> str:
    tag = "blockquote expandable" if expandable else "blockquote"
    return f"<{tag}>{escape(text)}</blockquote>"


def _bullets(items: Sequence[str]) -> str:
    body = "\n".join(f"• {escape(item)}" for item in items)
    if len(items) > LONG_LIST:
        return f"<blockquote expandable>{body}</blockquote>"
    return body


# ------------------------------------------------------------------ строки лаб


def lab_line(
    view: LabView, tz: ZoneInfo, *, with_subject: bool = False, now: datetime | None = None
) -> str:
    lab = view.lab
    subject = f"{escape(view.subject)} · " if with_subject else ""
    when = f" — {countdown(lab.deadline, tz, now)}" if lab.deadline else ""
    return f"{STATUS_ICONS[lab.status]} {subject}<b>ЛР {lab.number}</b> {escape(lab.title)}{when}"


def lab_header(subject: str, lab: Lab) -> str:
    return f"<b>{escape(subject)} · ЛР {lab.number} — {escape(lab.title)}</b>"


# ------------------------------------------------------------------ приём заданий


def ingested(item: IngestedLab, tz: ZoneInfo, disk_root: str) -> str:
    lab = item.lab
    subject = item.subject.name if item.subject else "?"
    head = (
        "🆕 <b>Новая лаба — разложил по полочкам!</b>"
        if item.outcome is IngestOutcome.CREATED
        else "🔄 <b>Обновил задание</b> — сохранил новую версию"
    )
    lines = [
        head,
        "",
        f"📘 {escape(subject)}",
        f"<b>ЛР {lab.number} · {escape(lab.title)}</b>",
        due(lab.deadline, tz),
    ]
    if lab.summary:
        lines += ["", _quote(lab.summary)]
    artifacts = item.extraction.requirements if item.extraction else []
    if artifacts:
        lines += ["", "📦 <b>Что сдать</b>", _bullets(artifacts)]
    lines += ["", f"☁️ {escape(disk_path(disk_root, lab.folder))}"]
    if IngestWarning.NO_TEXT in item.warnings:
        lines.append("⚠️ Похоже на скан: текст не извлёк, файл сохранил как есть.")
    if IngestWarning.NO_DEADLINE in item.warnings:
        lines.append("💡 Срок не нашёл — напишите, например, «срок 30.10».")
    return "\n".join(lines)


def duplicate(items: Sequence[IngestedLab]) -> str:
    labs = ", ".join(
        f"{escape(item.subject.name if item.subject else '?')} · ЛР {item.lab.number}"
        for item in items
    )
    return f"♻️ Уже есть: это задание я уже видел ({labs}). Ничего не менял."


def missing(fields: Iterable[str]) -> str:
    names = ", ".join(FIELD_TITLES.get(name, name) for name in fields)
    return (
        f"🤔 Почти готово! Не хватает: <b>{names}</b>.\n"
        f"Ответьте на это сообщение подписью, например {PROMPT_EXAMPLE}."
    )


def confirmation(extractions: Sequence[LabExtraction], tz: ZoneInfo) -> str:
    lines = ["🔎 <b>Проверьте, правильно ли я понял</b>", ""]
    for extraction in extractions:
        title = f" · {escape(extraction.title)}" if extraction.title else ""
        lines.append(
            f"📘 {escape(extraction.subject or '?')} — <b>ЛР {extraction.number}</b>{title}\n"
            f"    {due(extraction.deadline, tz)}"
        )
    lines += [
        "",
        "Номер в подписи и в тексте расходятся (или я не уверен).",
        "Если не так — ответьте на это сообщение правильной подписью.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ карточка и списки


def lab_card(
    view: LabView,
    tz: ZoneInfo,
    disk_root: str,
    requirements: Sequence[tuple[str, Requirement | None]],
) -> str:
    lab = view.lab
    lines = [
        f"📘 {escape(view.subject)}",
        f"<b>ЛР {lab.number} · {escape(lab.title)}</b>",
        "",
        f"{STATUS_ICONS[lab.status]} {STATUS_TITLES[lab.status].capitalize()}",
        due(lab.deadline, tz),
        f"☁️ {escape(disk_path(disk_root, lab.folder))}",
    ]
    if lab.summary:
        lines += ["", _quote(lab.summary)]
    if lab.artifacts:
        lines += ["", "📦 <b>Что сдать</b>", _bullets(lab.artifacts)]
    reqs = [(title, req.text) for title, req in requirements if req and req.text]
    if reqs:
        lines += ["", "📋 <b>Ваши требования</b>"]
        lines += [f"<i>{title}</i> — {escape(text)}" for title, text in reqs]
    return "\n".join(lines)


def lab_list(views: Sequence[LabView], tz: ZoneInfo, *, empty: str) -> str:
    if not views:
        return empty
    by_subject: dict[str, list[LabView]] = {}
    for view in views:
        by_subject.setdefault(view.subject, []).append(view)
    blocks = []
    for subject in sorted(by_subject, key=str.casefold):
        rows = sorted(by_subject[subject], key=lambda v: v.lab.number)
        blocks.append(f"📘 <b>{escape(subject)}</b>\n" + "\n".join(lab_line(v, tz) for v in rows))
    count = len(views)
    return (
        "\n\n".join(blocks)
        + f"\n\nВ работе и впереди: {count} {plural(count, 'лаба', 'лабы', 'лаб')}"
    )


def subjects_overview(views: Sequence[LabView]) -> str:
    """Прогресс по предметам: «📘 ОС ▰▰▱▱▱ 2/5 сдано»."""
    by_subject: dict[str, list[LabView]] = {}
    for view in views:
        by_subject.setdefault(view.subject, []).append(view)
    lines = []
    for subject in sorted(by_subject, key=str.casefold):
        items = by_subject[subject]
        finished = sum(1 for v in items if v.lab.status in FINISHED)
        lines.append(
            f"📘 <b>{escape(subject)}</b>  {progress(finished, len(items))} {finished}/{len(items)}"
        )
    return "\n".join(lines)


def week(report: WeekReport, tz: ZoneInfo) -> str:
    last_day = (report.end - timedelta(days=1)).astimezone(tz)
    title = f"🗓 <b>Неделя {report.start.astimezone(tz):%d.%m} – {last_day:%d.%m}</b>"
    if not report.overdue and not report.this_week:
        return f"{title}\n\n🌴 На этой неделе сроков нет — можно выдохнуть."
    parts = [title]
    if report.overdue:
        parts.append(
            "🔥 <b>Просрочено</b>\n"
            + "\n".join(lab_line(v, tz, with_subject=True) for v in report.overdue)
        )
    by_day: dict[date, list[LabView]] = {}
    for view in report.this_week:
        assert view.lab.deadline is not None  # noqa: S101 — в недельном отчёте срок есть всегда
        by_day.setdefault(view.lab.deadline.astimezone(tz).date(), []).append(view)
    for day in sorted(by_day):
        rows = "\n".join(
            f"{STATUS_ICONS[v.lab.status]} {escape(v.subject)} · <b>ЛР {v.lab.number}</b>"
            f" {escape(v.lab.title)} · до {v.lab.deadline.astimezone(tz):%H:%M}"  # type: ignore[union-attr]
            for v in by_day[day]
        )
        parts.append(f"<b>{WEEKDAYS[day.weekday()]} {day:%d.%m}</b>\n{rows}")
    total = len(report.overdue) + len(report.this_week)
    parts.append(f"Всего: {total} {plural(total, 'лаба', 'лабы', 'лаб')}. Справитесь 💪")
    return "\n\n".join(parts)


def ambiguous(views: Sequence[LabView], tz: ZoneInfo, command: str) -> str:
    rows = "\n".join(
        f"{lab_line(v, tz, with_subject=True)} → <code>/{command} #{v.lab.id}</code>" for v in views
    )
    return f"🤔 Нашлось несколько. Уточните, какую лабу:\n\n{rows}"


def status_changed(subject: str, lab: Lab, old: Status) -> str:
    cheer = {
        Status.IN_PROGRESS: "Вперёд! 🚀",
        Status.DONE: "Отличная работа! 🎉",
        Status.SUBMITTED: "Сдано — осталось дождаться оценки 🤞",
        Status.DEFENDED: "Защищена! Можно забыть про неё 🏁",
        Status.BLOCKED: "Застряли — бывает. Разберёмся 🛠",
    }.get(lab.status, "")
    return (
        f"{STATUS_ICONS[lab.status]} {lab_header(subject, lab)}\n"
        f"{STATUS_TITLES[old]} → <b>{STATUS_TITLES[lab.status]}</b>"
        + (f"\n{cheer}" if cheer else "")
    )
