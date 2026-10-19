"""Файлы заданий: уникальность по sha256, версии внутри лабы, отметка о выгрузке."""

import hashlib

import pytest

from lab_manager.db import Database
from lab_manager.domain import FileRecord, Lab, Source
from lab_manager.errors import AlreadyExistsError, NotFoundError


def sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


async def make_lab(db: Database, number: int = 1) -> Lab:
    subject = await db.subjects.get_or_create("ОС")
    return await db.labs.create(
        subject_id=subject.id, number=number, title="Лаба", source=Source.TELEGRAM
    )


async def add(db: Database, lab: Lab, content: str, name: str = "task.pdf") -> FileRecord:
    return await db.files.add(
        lab_id=lab.id,
        sha256=sha(content),
        original_name=name,
        mime_type="application/pdf",
        size_bytes=len(content),
        local_path=f"ОС/01/задание/{name}",
    )


async def test_versions_increment_per_lab(db: Database) -> None:
    lab = await make_lab(db)
    other = await make_lab(db, number=2)

    v1 = await add(db, lab, "first")
    v2 = await add(db, lab, "second")
    other_v1 = await add(db, other, "third")

    assert (v1.version, v2.version, other_v1.version) == (1, 2, 1)
    assert await db.files.latest_for_lab(lab.id) == v2
    assert await db.files.list_for_lab(lab.id) == [v1, v2]


async def test_lookup_by_sha256(db: Database) -> None:
    lab = await make_lab(db)
    record = await add(db, lab, "content")

    assert await db.files.get_by_sha256(sha("content")) == record
    assert await db.files.get_by_sha256(sha("unknown")) is None


async def test_same_content_cannot_be_added_twice_to_one_lab(db: Database) -> None:
    lab = await make_lab(db)
    await add(db, lab, "content")

    with pytest.raises(AlreadyExistsError):
        await add(db, lab, "content", name="copy.pdf")


async def test_same_content_can_belong_to_several_labs(db: Database) -> None:
    lab3 = await make_lab(db, number=3)
    lab4 = await make_lab(db, number=4)

    first = await add(db, lab3, "post about labs 3 and 4")
    second = await add(db, lab4, "post about labs 3 and 4")

    assert await db.files.list_by_sha256(first.sha256) == [first, second]
    assert await db.files.get_by_sha256(first.sha256) == first


@pytest.mark.parametrize("bad", ["", "abc", "Z" * 64, sha("x").upper()])
async def test_sha256_format_validated(db: Database, bad: str) -> None:
    lab = await make_lab(db)

    with pytest.raises(ValueError, match="sha256"):
        await db.files.add(
            lab_id=lab.id,
            sha256=bad,
            original_name="a.pdf",
            mime_type=None,
            size_bytes=1,
            local_path="a.pdf",
        )


async def test_unknown_lab_rejected(db: Database) -> None:
    with pytest.raises(NotFoundError):
        await db.files.add(
            lab_id=999,
            sha256=sha("x"),
            original_name="a.pdf",
            mime_type=None,
            size_bytes=1,
            local_path="a.pdf",
        )


async def test_latest_for_lab_without_files(db: Database) -> None:
    lab = await make_lab(db)

    assert await db.files.latest_for_lab(lab.id) is None


async def test_mark_uploaded(db: Database) -> None:
    lab = await make_lab(db)
    record = await add(db, lab, "content")
    assert record.disk_path is None
    assert record.uploaded_at is None

    uploaded = await db.files.mark_uploaded(record.id, "disk:/Лабы/ОС/01 Лаба/задание/task.pdf")

    assert uploaded.disk_path == "disk:/Лабы/ОС/01 Лаба/задание/task.pdf"
    assert uploaded.uploaded_at is not None
    assert await db.files.get_by_sha256(record.sha256) == uploaded


async def test_mark_uploaded_unknown_file(db: Database) -> None:
    with pytest.raises(NotFoundError):
        await db.files.mark_uploaded(999, "disk:/x")
