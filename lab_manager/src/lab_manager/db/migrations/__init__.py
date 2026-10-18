"""SQL-миграции: файлы `NNNN_<name>.sql`, применяются по возрастанию номера."""

import re
from dataclasses import dataclass
from importlib.resources import files

_NAME = re.compile(r"^(\d{4})_(\w+)\.sql$")


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


def load_migrations() -> list[Migration]:
    migrations = []
    for entry in files(__package__).iterdir():
        if match := _NAME.match(entry.name):
            migrations.append(Migration(int(match[1]), match[2], entry.read_text(encoding="utf-8")))
    return sorted(migrations, key=lambda m: m.version)
