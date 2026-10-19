"""Живые тесты против настоящего Яндекс Диска. Запуск: `uv run pytest -m live`.

Работают во временной папке `<YANDEX_DISK_ROOT>/_live_tests/<uuid>` и удаляют её в конце.
"""

import hashlib
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lab_manager.config import Settings
from lab_manager.disk import DiskAuthError, UploadResult, YandexDiskClient, join_path

pytestmark = pytest.mark.live

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@pytest.fixture(scope="module")
def settings() -> Settings:
    if not ENV_FILE.exists():
        pytest.skip("нет .env с реальными ключами")
    return Settings(_env_file=ENV_FILE)  # type: ignore[call-arg]


@pytest.fixture
async def disk(settings: Settings) -> AsyncIterator[YandexDiskClient]:
    async with YandexDiskClient(settings.yandex_disk_token) as client:
        yield client


@pytest.fixture
async def sandbox(settings: Settings, disk: YandexDiskClient) -> AsyncIterator[str]:
    parent = join_path(settings.yandex_disk_root, "_live_tests")
    try:
        yield f"{parent}/{uuid.uuid4().hex}"
    finally:
        await disk.delete(parent)  # тесты идут последовательно — убираем папку целиком


async def test_upload_roundtrip_is_idempotent(disk: YandexDiskClient, sandbox: str) -> None:
    path = f"{sandbox}/ОС/03 Процессы/задание/v1_тест.txt"
    data = "живой тест: кириллица в пути и в содержимом".encode()

    assert await disk.upload(path, data) is UploadResult.UPLOADED
    meta = await disk.get_meta(path)
    assert meta is not None
    assert meta.type == "file"
    assert meta.size == len(data)
    assert meta.sha256 == hashlib.sha256(data).hexdigest()

    assert await disk.upload(path, data) is UploadResult.SKIPPED
    assert await disk.upload(path, data + b"!") is UploadResult.UPLOADED


async def test_ensure_dir_and_delete(disk: YandexDiskClient, sandbox: str) -> None:
    nested = f"{sandbox}/a/b/c"

    await disk.ensure_dir(nested)
    await disk.ensure_dir(nested)
    meta = await disk.get_meta(nested)
    assert meta is not None and meta.type == "dir"

    await disk.delete(f"{sandbox}/a")
    assert await disk.get_meta(nested) is None


async def test_invalid_token_is_rejected(settings: Settings) -> None:
    async with YandexDiskClient("y0_invalid") as disk:
        with pytest.raises(DiskAuthError):
            await disk.get_meta(settings.yandex_disk_root)
