"""Тексты ответов (HTML). Всё, что ввёл пользователь, экранируется."""

from datetime import timedelta
from html import escape

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
    ("help", "Как это работает"),
]

HELP = (
    "🔮 <b>Телепатия</b>\n\n"
    "Пишите сюда любые мысли — партнёр их не увидит. Если ваши мысли совпадут "
    "по смыслу, я сообщу вам обоим и покажу обе мысли.\n\n"
    "/pair — код-приглашение для партнёра\n"
    "/join КОД — вступить в пару\n"
    "/stats — статистика совпадений\n"
    "/help — эта справка"
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
