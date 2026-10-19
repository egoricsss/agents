"""Тексты сообщений бота (HTML). Все данные пользователя и документов экранируются."""

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

from lab_manager.domain import Lab, Requirement
from lab_manager.extraction import LabExtraction
from lab_manager.services.ingest import IngestedLab, IngestOutcome, IngestWarning
from lab_manager.services.labs import STATUS_ICONS, STATUS_TITLES, LabView, WeekReport

FIELD_TITLES = {"subject": "предмет", "number": "номер лабы"}
PROMPT_EXAMPLE = "«ОС, ЛР 3, до 25.10»"


def deadline(value: datetime | None, tz: ZoneInfo, *, short: bool = False) -> str:
    if value is None:
        return "не указан"
    local = value.astimezone(tz)
    return f"{local:%d.%m}" if short else f"{local:%d.%m.%Y %H:%M}"


def lab_line(view: LabView, tz: ZoneInfo, *, with_subject: bool = False) -> str:
    lab = view.lab
    subject = f"{escape(view.subject)} · " if with_subject else ""
    due = f" · до {deadline(lab.deadline, tz, short=True)}" if lab.deadline else ""
    return f"{STATUS_ICONS[lab.status]} {subject}ЛР {lab.number} — {escape(lab.title)}{due}"


def lab_header(subject: str, lab: Lab) -> str:
    return f"<b>{escape(subject)} · ЛР {lab.number} — {escape(lab.title)}</b>"


def ingested(item: IngestedLab, tz: ZoneInfo, disk_root: str) -> str:
    icon = "🆕 Создана" if item.outcome is IngestOutcome.CREATED else "🔄 Обновлена"
    lab = item.lab
    lines = [
        f"{icon}: {lab_header(item.subject.name if item.subject else '?', lab)}",
        f"📅 Срок: {deadline(lab.deadline, tz)}",
        f"📁 {escape(_disk_folder(disk_root, lab.folder))}",
    ]
    if lab.summary:
        lines.append(f"📝 {escape(lab.summary)}")
    if item.extraction and item.extraction.requirements:
        lines.append("Сдать: " + escape("; ".join(item.extraction.requirements[:6])))
    if IngestWarning.NO_TEXT in item.warnings:
        lines.append("⚠️ Похоже на скан: текст не извлечён, файл сохранён как есть.")
    if IngestWarning.NO_DEADLINE in item.warnings:
        lines.append("ℹ️ Срок не найден — укажите его в подписи при следующей версии.")
    return "\n".join(lines)


def duplicate(items: Sequence[IngestedLab]) -> str:
    labs = ", ".join(
        f"{escape(item.subject.name if item.subject else '?')} · ЛР {item.lab.number}"
        for item in items
    )
    return f"♻️ Уже есть: {labs}. Ничего не менял."


def missing(fields: Iterable[str]) -> str:
    names = ", ".join(FIELD_TITLES.get(name, name) for name in fields)
    return (
        f"❓ Не хватает: {names}.\nОтветьте на это сообщение подписью, например {PROMPT_EXAMPLE}."
    )


def confirmation(extractions: Sequence[LabExtraction], tz: ZoneInfo) -> str:
    lines = ["🔎 <b>Проверьте разбор</b> — номер в подписи и в тексте расходятся или я не уверен:"]
    for extraction in extractions:
        title = f" — {escape(extraction.title)}" if extraction.title else ""
        lines.append(
            f"• {escape(extraction.subject or '?')} · ЛР {extraction.number}{title},"
            f" срок: {deadline(extraction.deadline, tz)}"
        )
    lines.append("\nЕсли не так — ответьте на это сообщение правильной подписью.")
    return "\n".join(lines)


def lab_card(
    view: LabView,
    tz: ZoneInfo,
    disk_root: str,
    requirements: Sequence[tuple[str, Requirement | None]],
) -> str:
    lab = view.lab
    lines = [
        lab_header(view.subject, lab),
        f"{STATUS_ICONS[lab.status]} Статус: {STATUS_TITLES[lab.status]}",
        f"📅 Срок: {deadline(lab.deadline, tz)}",
        f"📁 {escape(_disk_folder(disk_root, lab.folder))}",
    ]
    if lab.summary:
        lines.append(f"\n📝 {escape(lab.summary)}")
    if lab.artifacts:
        lines.append("\n<b>Сдать:</b>\n" + "\n".join(f"• {escape(a)}" for a in lab.artifacts))
    reqs = [(title, req.text) for title, req in requirements if req and req.text]
    if reqs:
        lines.append("\n<b>Требования:</b>")
        lines += [f"<i>{title}:</i> {escape(text)}" for title, text in reqs]
    lines.append(f"\n<code>/prompt #{lab.id}</code> — промпт для исполнителя")
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
        blocks.append(f"<b>{escape(subject)}</b>\n" + "\n".join(lab_line(v, tz) for v in rows))
    return "\n\n".join(blocks) + f"\n\nВсего открытых: {len(views)}"


def week(report: WeekReport, tz: ZoneInfo) -> str:
    last_day = (report.end - timedelta(days=1)).astimezone(tz)
    title = f"<b>Неделя {report.start.astimezone(tz):%d.%m}–{last_day:%d.%m}</b>"
    if not report.overdue and not report.this_week:
        return f"{title}\nСроков на этой неделе нет 🎉"
    parts = [title]
    if report.overdue:
        parts.append(
            "⏰ <b>Просрочены:</b>\n"
            + "\n".join(lab_line(v, tz, with_subject=True) for v in report.overdue)
        )
    if report.this_week:
        parts.append(
            "📅 <b>Сдать на этой неделе:</b>\n"
            + "\n".join(lab_line(v, tz, with_subject=True) for v in report.this_week)
        )
    return "\n\n".join(parts)


def ambiguous(views: Sequence[LabView], tz: ZoneInfo, command: str) -> str:
    rows = "\n".join(
        f"{lab_line(v, tz, with_subject=True)} → <code>/{command} #{v.lab.id}</code>" for v in views
    )
    return f"Уточните, какую лабу:\n{rows}"


def _disk_folder(root: str, folder: str | None) -> str:
    return f"{root.rstrip('/')}/{folder}" if folder else "папка ещё не создана"
