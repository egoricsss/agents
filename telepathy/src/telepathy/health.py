"""Healthcheck для Docker: `python -m telepathy.health` → 0, если пульс свежий.

Только стандартная библиотека: проверка раз в минуту не должна грузить torch и тратить
память контейнера, где уже живёт модель эмбеддингов.
"""

import os
import time
from pathlib import Path

HEARTBEAT_FILE = "heartbeat"
HEARTBEAT_MAX_AGE = 120.0


def check(data_dir: Path, now: float | None = None) -> int:
    try:
        beat = float((data_dir / HEARTBEAT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    return 0 if (now or time.time()) - beat < HEARTBEAT_MAX_AGE else 1


if __name__ == "__main__":
    raise SystemExit(check(Path(os.environ.get("DATA_DIR", "data"))))
