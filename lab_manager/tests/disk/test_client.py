"""Клиент Яндекс Диска на фейковом сервере: метаданные, папки, идемпотентная загрузка, ошибки."""

import hashlib
from collections.abc import AsyncIterator

import httpx
import pytest

from lab_manager.disk import DiskAuthError, DiskError, UploadResult, YandexDiskClient, join_path
from tests.disk.fake_disk import TOKEN, FakeDisk


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fake() -> FakeDisk:
    return FakeDisk()


@pytest.fixture
async def disk(fake: FakeDisk) -> AsyncIterator[YandexDiskClient]:
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        yield YandexDiskClient(TOKEN, http=http, sleep=no_sleep)


@pytest.mark.parametrize(
    ("root", "relative", "expected"),
    [
        ("app:/Лабы", "ОС/03 x/a.pdf", "app:/Лабы/ОС/03 x/a.pdf"),
        ("app:/Лабы/", "/ОС/a.pdf", "app:/Лабы/ОС/a.pdf"),
        ("app:/", "a.pdf", "app:/a.pdf"),
    ],
)
def test_join_path(root: str, relative: str, expected: str) -> None:
    assert join_path(root, relative) == expected


async def test_meta_of_missing_resource_is_none(disk: YandexDiskClient) -> None:
    assert await disk.get_meta("app:/nothing") is None


async def test_ensure_dir_creates_parents_and_is_idempotent(
    disk: YandexDiskClient, fake: FakeDisk
) -> None:
    await disk.ensure_dir("app:/Лабы/ОС/03 Процессы")
    await disk.ensure_dir("app:/Лабы/ОС/03 Процессы")

    assert {"app:/Лабы", "app:/Лабы/ОС", "app:/Лабы/ОС/03 Процессы"} <= fake.dirs
    meta = await disk.get_meta("app:/Лабы/ОС")
    assert meta is not None and meta.type == "dir"


async def test_upload_creates_dirs_and_stores_content(
    disk: YandexDiskClient, fake: FakeDisk
) -> None:
    data = "задание".encode()

    result = await disk.upload("app:/Лабы/ОС/03/задание/v1_a.txt", data)

    assert result is UploadResult.UPLOADED
    assert fake.files["app:/Лабы/ОС/03/задание/v1_a.txt"] == data
    meta = await disk.get_meta("app:/Лабы/ОС/03/задание/v1_a.txt")
    assert meta is not None and meta.sha256 == sha(data) and meta.size == len(data)


async def test_upload_same_content_is_skipped(disk: YandexDiskClient, fake: FakeDisk) -> None:
    await disk.upload("app:/a.txt", b"same")
    uploads_before = len(fake.upload_requests)

    result = await disk.upload("app:/a.txt", b"same")

    assert result is UploadResult.SKIPPED
    assert len(fake.upload_requests) == uploads_before


async def test_upload_changed_content_overwrites(disk: YandexDiskClient, fake: FakeDisk) -> None:
    await disk.upload("app:/a.txt", b"old")

    result = await disk.upload("app:/a.txt", b"new")

    assert result is UploadResult.UPLOADED
    assert fake.files["app:/a.txt"] == b"new"


async def test_token_is_not_sent_to_upload_host(disk: YandexDiskClient, fake: FakeDisk) -> None:
    await disk.upload("app:/a.txt", b"x")

    [request] = fake.upload_requests
    assert "Authorization" not in request.headers


async def test_delete_is_idempotent(disk: YandexDiskClient, fake: FakeDisk) -> None:
    await disk.upload("app:/dir/a.txt", b"x")

    await disk.delete("app:/dir")
    await disk.delete("app:/dir")

    assert "app:/dir/a.txt" not in fake.files
    assert "app:/dir" not in fake.dirs


@pytest.mark.parametrize("status", [423, 429, 500, 503])
async def test_transient_errors_are_retried(
    disk: YandexDiskClient, fake: FakeDisk, status: int
) -> None:
    fake.fail_next = [status, status]

    assert await disk.get_meta("app:/") is not None


async def test_retries_are_bounded(fake: FakeDisk) -> None:
    fake.fail_next = [503] * 10
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        disk = YandexDiskClient(TOKEN, http=http, sleep=no_sleep, max_retries=2)

        with pytest.raises(DiskError) as error:
            await disk.get_meta("app:/")

    assert error.value.status == 503
    assert len(fake.api_calls) == 3


async def test_bad_token_raises_auth_error_without_retry(fake: FakeDisk) -> None:
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        disk = YandexDiskClient("wrong-secret-token", http=http, sleep=no_sleep)

        with pytest.raises(DiskAuthError) as error:
            await disk.get_meta("app:/")

    assert len(fake.api_calls) == 1
    assert "wrong-secret-token" not in str(error.value)
    assert "wrong-secret-token" not in repr(disk)


async def test_client_error_is_not_retried(disk: YandexDiskClient, fake: FakeDisk) -> None:
    fake.fail_next = [400]

    with pytest.raises(DiskError) as error:
        await disk.get_meta("app:/")

    assert error.value.status == 400
    assert len(fake.api_calls) == 1


async def test_owns_http_client_when_not_given() -> None:
    async with YandexDiskClient(TOKEN) as disk:
        assert disk.http is not None


async def test_delete_directory_waits_for_async_operation(
    disk: YandexDiskClient, fake: FakeDisk
) -> None:
    fake.operation_polls_before_done = 3
    await disk.ensure_dir("app:/dir/sub")

    await disk.delete("app:/dir")

    assert await disk.get_meta("app:/dir/sub") is None


async def test_failed_async_operation_raises(disk: YandexDiskClient, fake: FakeDisk) -> None:
    fake.fail_operations = True
    await disk.ensure_dir("app:/dir")

    with pytest.raises(DiskError, match="failed"):
        await disk.delete("app:/dir")


async def test_async_operation_wait_is_bounded(fake: FakeDisk) -> None:
    fake.operation_polls_before_done = 100
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        disk = YandexDiskClient(TOKEN, http=http, sleep=no_sleep, operation_max_polls=5)
        await disk.ensure_dir("app:/dir")

        with pytest.raises(DiskError, match="timeout"):
            await disk.delete("app:/dir")


async def test_delete_without_waiting_returns_immediately(
    disk: YandexDiskClient, fake: FakeDisk
) -> None:
    fake.operation_polls_before_done = 100
    await disk.ensure_dir("app:/dir")
    calls_before = len(fake.api_calls)

    await disk.delete("app:/dir", wait=False)

    assert len(fake.api_calls) == calls_before + 1  # только DELETE, без опроса операции


async def test_operation_polling_backs_off(fake: FakeDisk) -> None:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    fake.operation_polls_before_done = 6
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        disk = YandexDiskClient(TOKEN, http=http, sleep=record)
        await disk.ensure_dir("app:/dir")
        await disk.delete("app:/dir")

    assert delays == sorted(delays)
    assert delays[0] == 0.5
    assert delays[-1] > delays[0]
    assert max(delays) <= 5.0
