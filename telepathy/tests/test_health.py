import subprocess
import sys
import time
from pathlib import Path

from telepathy.health import HEARTBEAT_FILE, HEARTBEAT_MAX_AGE, check


def test_fresh_heartbeat_is_healthy(tmp_path: Path) -> None:
    (tmp_path / HEARTBEAT_FILE).write_text(str(time.time()), encoding="utf-8")
    assert check(tmp_path) == 0


def test_stale_or_missing_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    assert check(tmp_path) == 1
    (tmp_path / HEARTBEAT_FILE).write_text("100", encoding="utf-8")
    assert check(tmp_path, now=100 + HEARTBEAT_MAX_AGE + 1) == 1
    (tmp_path / HEARTBEAT_FILE).write_text("мусор", encoding="utf-8")
    assert check(tmp_path) == 1


def test_healthcheck_does_not_import_heavy_modules(tmp_path: Path) -> None:
    (tmp_path / HEARTBEAT_FILE).write_text(str(time.time()), encoding="utf-8")
    code = "\n".join(
        [
            "import runpy, sys",
            "try:",
            "    runpy.run_module('telepathy.health', run_name='__main__')",
            "except SystemExit as exc:",
            "    status = exc.code",
            "heavy = [m for m in ('torch', 'pydantic_ai', 'aiogram') if m in sys.modules]",
            "print(status, heavy)",
        ]
    )
    result = subprocess.run(  # noqa: S603 — фиксированная команда
        [sys.executable, "-c", code],
        env={"DATA_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "0 []"
