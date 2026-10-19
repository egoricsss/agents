"""Локальное хранилище лаб: `<Предмет>/<NN> <Название>/{задание,отчёт}/…`.

Все пути — относительные к корню, с `/` как разделителем; та же структура зеркалируется на Диск.
Любой путь проверяется на выход за пределы корня.
"""

import os
import re
from pathlib import Path, PurePosixPath

ASSIGNMENT_DIR = "задание"
REPORT_DIR = "отчёт"
MAX_NAME_LENGTH = 80

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_FORBIDDEN = re.compile(r'[/\\:*?"<>|]+')


def safe_name(raw: str) -> str:
    """Имя файла/папки, безопасное для ФС и Яндекс Диска."""
    name = _CONTROL.sub(" ", raw)
    name = _FORBIDDEN.sub("_", name)
    name = " ".join(name.split())[:MAX_NAME_LENGTH].rstrip(". ")
    return name if name.strip(".") else "_"


class LabStorage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def lab_folder(self, subject: str, number: int, title: str) -> str:
        lab = f"{number:02d} {safe_name(title)}" if title.strip() else f"{number:02d}"
        return f"{safe_name(subject)}/{lab}"

    def resolve(self, relative: str) -> Path:
        if PurePosixPath(relative).is_absolute() or Path(relative).is_absolute():
            raise ValueError(f"ожидался относительный путь: {relative}")
        root = self.root.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"путь выходит за пределы хранилища: {relative}")
        return path

    def ensure_lab_dirs(self, folder: str) -> Path:
        path = self.resolve(folder)
        for sub in (ASSIGNMENT_DIR, REPORT_DIR):
            (path / sub).mkdir(parents=True, exist_ok=True)
        return path

    def save_original(self, folder: str, version: int, filename: str, data: bytes) -> str:
        self.ensure_lab_dirs(folder)
        relative = f"{folder}/{ASSIGNMENT_DIR}/v{version}_{safe_name(filename)}"
        _atomic_write(self.resolve(relative), data)
        return relative

    def write_text(self, folder: str, name: str, text: str) -> str:
        relative = f"{folder}/{name}"
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, text.encode("utf-8"))
        return relative

    def move(self, old_folder: str, new_folder: str) -> None:
        """Переименовать папку лабы (вместе с заданием и отчётами)."""
        source, target = self.resolve(old_folder), self.resolve(new_folder)
        if source == target or not source.exists():
            return
        if target.exists():
            raise FileExistsError(f"папка уже существует: {new_folder}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        parent = source.parent  # убрать опустевшую папку предмета
        if parent != self.root.resolve() and not any(parent.iterdir()):
            parent.rmdir()

    def read_text(self, relative: str) -> str | None:
        path = self.resolve(relative)
        return path.read_text(encoding="utf-8") if path.is_file() else None


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
