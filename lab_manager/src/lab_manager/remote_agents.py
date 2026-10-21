"""Удалённые агенты общего бота: что шлюзу нужно знать о соседних контейнерах.

Команды здесь должны совпадать с командами самого агента (telepathy/bot/texts.py):
по ним шлюз решает, кому отдать апдейт владельца, и строит меню команд.
"""

from lab_manager.config import Settings
from lab_manager.gateway import RemoteAgent

TELEPATHY_NAME = "telepathy"
TELEPATHY_MENU = "🔮 Телепатия"
TELEPATHY_COMMANDS = [
    ("pair", "Код-приглашение для партнёра"),
    ("join", "Вступить в пару по коду"),
    ("stats", "Статистика совпадений"),
    ("help", "Как это работает"),
]


def remote_agents(settings: Settings) -> tuple[list[RemoteAgent], dict[int, str]]:
    """Удалённые агенты и гости (Telegram ID → имя единственного доступного агента)."""
    if settings.telepathy_url is None or settings.telepathy_secret is None:
        return [], {}
    telepathy = RemoteAgent(
        name=TELEPATHY_NAME,
        title="Телепатия",
        description="Пишите мысли — бот скажет, когда они совпали с мыслями партнёра.",
        url=str(settings.telepathy_url),
        secret=settings.telepathy_secret.get_secret_value(),
        commands=TELEPATHY_COMMANDS,
        menu=[TELEPATHY_MENU],
    )
    return [telepathy], {user_id: TELEPATHY_NAME for user_id in settings.telepathy_users}
