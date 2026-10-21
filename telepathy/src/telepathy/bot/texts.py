"""Тексты ответов (HTML). Всё, что ввёл пользователь, экранируется."""

from datetime import timedelta
from html import escape
from zoneinfo import ZoneInfo

from telepathy.services.debug import CheckView, MatchView, ThoughtRef
from telepathy.services.matching import FoundMatch
from telepathy.services.pairing import PairStatus
from telepathy.services.stats import PairStats

THOUGHT_ACCEPTED = "🔮 мысль принята"
ONLY_TEXT = "Пока принимаю только текстовые мысли ✍️"
UNKNOWN_COMMAND = "Не знаю такой команды. Список — /help"
JOIN_USAGE = "Укажите код партнёра: <code>/join КОД</code>"

COMMANDS: list[tuple[str, str]] = [
    ("pair", "Код-приглашение для партнёра"),
    ("join", "Вступить в пару по коду"),
    ("stats", "Статистика совпадений"),
    ("thoughts", "Отладка: последние мысли пары"),
    ("debug", "Отладка: проверки совпадений с цифрами"),
    ("matches", "Отладка: совпадения с цифрами"),
    ("help", "Как это работает"),
]

HELP = (
    "🔮 <b>Телепатия</b>\n\n"
    "Пишите сюда любые мысли — партнёр их не увидит. Если ваши мысли совпадут "
    "по смыслу, я сообщу вам обоим и покажу обе мысли.\n\n"
    "/pair — код-приглашение для партнёра\n"
    "/join КОД — вступить в пару\n"
    "/stats — статистика совпадений\n"
    "/help — эта справка\n\n"
    "<b>Отладка</b> (чужие мысли без совпадения скрыты — видно только время и цифры):\n"
    "/thoughts [N] — последние мысли пары\n"
    "/debug [N] — проверки: лучший кандидат, cos, Δt, score, решение\n"
    "/matches [N] — совпадения с цифрами"
)


def duration(delta: timedelta) -> str:
    seconds = round(abs(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds} сек"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} ч {minutes} мин" if minutes else f"{hours} ч"
    days, hours = divmod(hours, 24)
    return f"{days} дн {hours} ч" if hours else f"{days} дн"


def start(status: PairStatus, partner_name: str | None) -> str:
    if status.complete:
        tail = f"Вы в паре с <b>{escape(partner_name or '')}</b>. Просто напишите мысль."
    elif status.invite_code:
        tail = f"Ждём партнёра. Ваш код: <code>{status.invite_code}</code>"
    else:
        tail = "Начните с /pair или вступите в пару партнёра: <code>/join КОД</code>."
    return f"{HELP}\n\n{tail}"


def invite(code: str) -> str:
    return f"Ваш код: <code>{code}</code>\n\nПусть партнёр отправит мне:\n<code>/join {code}</code>"


def joined(partner_name: str) -> str:
    return (
        f"💞 Готово! Вы в паре с <b>{escape(partner_name)}</b>.\n"
        "Пишите мысли — я сообщу, когда они совпадут."
    )


def partner_joined(partner_name: str) -> str:
    return f"💞 <b>{escape(partner_name)}</b> теперь в вашей паре. Можно начинать!"


def match(found: FoundMatch, viewer_id: int, partner_name: str) -> str:
    own = found.own_thought(viewer_id)
    other = found.partner_thought(viewer_id)
    lines = [
        escape(found.message),
        "",
        f"💭 Вы: «{escape(own.text)}»",
        f"💭 {escape(partner_name)}: «{escape(other.text)}»",
        "",
    ]
    if found.common_idea:
        lines.append(f"🧠 Общая идея: {escape(found.common_idea)}")
    lines.append(f"⏱ Разница во времени: {duration(found.delta)}")
    return "\n".join(lines)


def stats(value: PairStats) -> str:
    if value.week_percent is None:
        week = "мыслей пока не было"
    else:
        week = f"{value.week_percent:.0f}% ({value.week_matched} из {value.week_thoughts})"
    fastest = duration(value.fastest) if value.fastest is not None else "ещё впереди"
    return (
        "📊 <b>Статистика пары</b>\n\n"
        f"Совпадений: {value.matches}\n"
        f"Совпавших мыслей за 7 дней: {week}\n"
        f"Самая быстрая синхронность: {fastest}"
    )


DEBUG_TEXT_LIMIT = 120
OUTCOMES = {
    "no_candidates": "нет мыслей партнёра в окне",
    "below_threshold": "ниже порога, судью не спрашивали",
    "judge_unavailable": "⚠️ судья недоступен",
    "judge_rejected": "судья: мысли разные",
    "matched": "✨ совпало",
}


def _when(ref: ThoughtRef, zone: ZoneInfo) -> str:
    return ref.at.astimezone(zone).strftime("%d.%m %H:%M:%S")


def _who(ref: ThoughtRef, partner_name: str) -> str:
    return "Вы" if ref.mine else escape(partner_name)


def _what(ref: ThoughtRef) -> str:
    if ref.text is None:
        return "🔒 скрыто"
    text = ref.text if len(ref.text) <= DEBUG_TEXT_LIMIT else ref.text[:DEBUG_TEXT_LIMIT] + "…"
    return f"«{escape(text)}»" + (" ✨" if ref.matched else "")


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def debug_thoughts(refs: list[ThoughtRef], partner_name: str, zone: ZoneInfo) -> str:
    if not refs:
        return "Мыслей пока нет."
    lines = [f"🗂 <b>Последние мысли пары</b> ({len(refs)}, новые сверху)", ""]
    lines += [f"{_when(r, zone)} · {_who(r, partner_name)}: {_what(r)}" for r in refs]
    return "\n".join(lines)


def debug_checks(checks: list[CheckView], partner_name: str, zone: ZoneInfo) -> str:
    if not checks:
        return "Проверок пока не было."
    lines = [f"🔬 <b>Проверки совпадений</b> ({len(checks)}, новые сверху)"]
    for c in checks:
        lines += [
            "",
            f"{_when(c.thought, zone)} · {_who(c.thought, partner_name)}: {_what(c.thought)}",
        ]
        if c.candidate is not None:
            lines.append(
                f"  ↔ {_who(c.candidate, partner_name)} {_when(c.candidate, zone)}: "
                f"{_what(c.candidate)}"
            )
            lines.append(
                f"  cos {_num(c.cosine)} · Δt {duration(c.delta) if c.delta else '—'} · "
                f"score {_num(c.score)} / порог {c.threshold:.2f} · кандидатов {c.candidates}"
            )
        lines.append(f"  → {OUTCOMES.get(c.outcome, c.outcome)}")
    return "\n".join(lines)


def debug_matches(matches: list[MatchView], partner_name: str, zone: ZoneInfo) -> str:
    if not matches:
        return "Совпадений пока не было."
    lines = [f"✨ <b>Совпадения</b> ({len(matches)}, новые сверху)"]
    for m in matches:
        lines += [
            "",
            f"{_when(m.later, zone)} · Δt {duration(m.delta)} · cos {_num(m.cosine)} · "
            f"score {_num(m.score)}",
            f"  {_who(m.earlier, partner_name)}: {_what(m.earlier)}",
            f"  {_who(m.later, partner_name)}: {_what(m.later)}",
        ]
        if m.common_idea:
            lines.append(f"  🧠 {escape(m.common_idea)}")
    return "\n".join(lines)
