"""Локальная раскладка лаб по папкам: имена, безопасность путей, атомарная запись."""

from pathlib import Path

import pytest

from lab_manager.storage import ASSIGNMENT_DIR, REPORT_DIR, LabStorage, safe_name


@pytest.fixture
def storage(tmp_path: Path) -> LabStorage:
    return LabStorage(tmp_path / "Лабы")


def test_lab_folder_name(storage: LabStorage) -> None:
    assert storage.lab_folder("Операционные системы", 3, "Процессы и потоки") == (
        "Операционные системы/03 Процессы и потоки"
    )


def test_lab_folder_without_title(storage: LabStorage) -> None:
    assert storage.lab_folder("ОС", 12, "") == "ОС/12"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ОС/Linux", "ОС_Linux"),
        ('a:b*c?"d<e>f|g\\h', "a_b_c_d_e_f_g_h"),
        ("  много    пробелов  ", "много пробелов"),
        ("точка в конце...", "точка в конце"),
        ("..", "_"),
        (".", "_"),
        ("", "_"),
        ("tab\tnew\nline\x00", "tab new line"),
    ],
)
def test_safe_name(raw: str, expected: str) -> None:
    assert safe_name(raw) == expected


def test_safe_name_is_length_limited() -> None:
    assert len(safe_name("я" * 500)) <= 80


def test_traversal_in_names_stays_inside_root(storage: LabStorage) -> None:
    folder = storage.lab_folder("../../etc", 1, "../passwd")
    path = storage.ensure_lab_dirs(folder)

    assert path.resolve().is_relative_to(storage.root.resolve())


def test_resolve_rejects_paths_outside_root(storage: LabStorage) -> None:
    with pytest.raises(ValueError):
        storage.resolve("../outside.txt")
    with pytest.raises(ValueError):
        storage.resolve("/etc/passwd")


def test_ensure_lab_dirs_creates_layout(storage: LabStorage) -> None:
    path = storage.ensure_lab_dirs("ОС/01 Лаба")

    assert (path / ASSIGNMENT_DIR).is_dir()
    assert (path / REPORT_DIR).is_dir()


def test_save_original_is_versioned(storage: LabStorage) -> None:
    rel1 = storage.save_original("ОС/01 Лаба", 1, "Методичка.PDF", b"v1")
    rel2 = storage.save_original("ОС/01 Лаба", 2, "Методичка.PDF", b"v2")

    assert rel1 == "ОС/01 Лаба/задание/v1_Методичка.PDF"
    assert rel2 == "ОС/01 Лаба/задание/v2_Методичка.PDF"
    assert storage.resolve(rel1).read_bytes() == b"v1"
    assert storage.resolve(rel2).read_bytes() == b"v2"


def test_save_original_sanitizes_filename(storage: LabStorage) -> None:
    rel = storage.save_original("ОС/01 Лаба", 1, "../../evil.txt", b"x")

    assert rel == "ОС/01 Лаба/задание/v1_.._.._evil.txt"
    assert storage.resolve(rel).read_bytes() == b"x"


def test_write_text_overwrites_atomically(storage: LabStorage) -> None:
    rel = storage.write_text("ОС/01 Лаба", "задание/assignment.md", "старое")
    storage.write_text("ОС/01 Лаба", "задание/assignment.md", "новое")

    path = storage.resolve(rel)
    assert path.read_text(encoding="utf-8") == "новое"
    assert not list(path.parent.glob("*.tmp"))


def test_read_text(storage: LabStorage) -> None:
    storage.write_text("ОС/01 Лаба", "requirements.md", "ГОСТ")

    assert storage.read_text("ОС/01 Лаба/requirements.md") == "ГОСТ"
    assert storage.read_text("ОС/01 Лаба/missing.md") is None
