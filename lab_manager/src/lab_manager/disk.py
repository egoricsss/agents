"""Асинхронный клиент REST API Яндекс Диска (только нужные менеджеру операции).

Загрузка идемпотентна: перед отправкой сверяется sha256 файла на Диске. Временные ошибки
(429, 5xx, сеть) повторяются с экспоненциальной задержкой; 401/403 — сразу ошибка.
Токен передаётся только на cloud-api.yandex.net, не на хост загрузки, и не попадает в repr/ошибки.
"""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import SecretStr

from lab_manager.errors import LabManagerError

API_URL = "https://cloud-api.yandex.net/v1/disk"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 30.0

Sleep = Callable[[float], Awaitable[None]]


class DiskError(LabManagerError):
    def __init__(self, status: int, code: str = "") -> None:
        self.status = status
        self.code = code
        super().__init__(f"Яндекс Диск ответил {status} {code}".strip())


class DiskAuthError(DiskError):
    """Токен недействителен или у приложения нет прав на путь."""


class UploadResult(StrEnum):
    UPLOADED = "uploaded"
    SKIPPED = "skipped"  # на Диске уже лежит файл с тем же sha256


@dataclass(frozen=True, slots=True)
class RemoteResource:
    path: str
    type: str
    sha256: str | None = None
    size: int | None = None


def join_path(root: str, relative: str) -> str:
    base = root if root.endswith(":/") else root.rstrip("/") + "/"
    return base + relative.lstrip("/")


def parent_path(path: str) -> str:
    head, _, _ = path.rstrip("/").rpartition("/")
    return head + "/" if head.endswith(":") else head


def _is_root(path: str) -> bool:
    return path.endswith(":/")


class YandexDiskClient:
    def __init__(
        self,
        token: SecretStr | str,
        *,
        http: httpx.AsyncClient | None = None,
        base_url: str = API_URL,
        max_retries: int = 3,
        backoff: float = 0.5,
        sleep: Sleep = asyncio.sleep,
        timeout: float = 60.0,
        operation_poll_interval: float = 0.5,
        operation_max_polls: int = 120,
    ) -> None:
        self._token = token if isinstance(token, SecretStr) else SecretStr(token)
        self._owns_http = http is None
        self.http = http or httpx.AsyncClient(timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self.operation_poll_interval = operation_poll_interval
        self.operation_max_polls = operation_max_polls

    def __repr__(self) -> str:
        return f"YandexDiskClient(base_url={self.base_url!r})"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self.http.aclose()

    async def get_meta(self, path: str) -> RemoteResource | None:
        response = await self._api(
            "GET",
            "/resources",
            {"path": path, "fields": "path,type,sha256,size"},
            ok={200, 404},
        )
        if response.status_code == 404:
            return None
        data = response.json()
        return RemoteResource(data["path"], data["type"], data.get("sha256"), data.get("size"))

    async def ensure_dir(self, path: str) -> None:
        if _is_root(path):
            return
        response = await self._api("PUT", "/resources", {"path": path}, ok={201, 409})
        if response.status_code == 201:
            return
        code = _error_code(response)
        if code == "DiskPathPointsToExistentDirectoryError":
            return
        if code == "DiskPathDoesntExistsError":
            await self.ensure_dir(parent_path(path))
            await self._api("PUT", "/resources", {"path": path}, ok={201, 409})
            return
        raise DiskError(response.status_code, code)

    async def upload(self, path: str, data: bytes) -> UploadResult:
        meta = await self.get_meta(path)
        if meta is not None and meta.sha256 == hashlib.sha256(data).hexdigest():
            return UploadResult.SKIPPED
        await self.ensure_dir(parent_path(path))
        link = await self._api("GET", "/resources/upload", {"path": path, "overwrite": "true"})
        href = link.json()["href"]
        # Ссылка на загрузку уже подписана — токен на этот хост не отправляем.
        await self._send("PUT", href, content=data, auth=False, ok={201, 202})
        return UploadResult.UPLOADED

    async def delete(self, path: str) -> None:
        """Удаляет навсегда. Папки Диск удаляет асинхронно (202) — дожидаемся операции."""
        response = await self._api(
            "DELETE", "/resources", {"path": path, "permanently": "true"}, ok={202, 204, 404}
        )
        if response.status_code == 202:
            await self._wait_operation(response.json()["href"])

    async def _wait_operation(self, href: str) -> None:
        if httpx.URL(href).host != httpx.URL(self.base_url).host:
            raise DiskError(0, "unexpected operation host")
        for _ in range(self.operation_max_polls):
            response = await self._send("GET", href, auth=True, ok={200})
            status = response.json().get("status")
            if status == "success":
                return
            if status == "failed":
                raise DiskError(response.status_code, "operation failed")
            await self._sleep(self.operation_poll_interval)
        raise DiskError(0, "operation timeout")

    async def _api(
        self,
        method: str,
        endpoint: str,
        params: dict[str, str],
        *,
        ok: set[int] | None = None,
    ) -> httpx.Response:
        return await self._send(
            method, f"{self.base_url}{endpoint}", params=params, auth=True, ok=ok or {200}
        )

    async def _send(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        content: bytes | None = None,
        auth: bool,
        ok: set[int],
    ) -> httpx.Response:
        headers = {"Accept": "application/json"}
        if auth:
            headers["Authorization"] = f"OAuth {self._token.get_secret_value()}"
        attempt = 0
        while True:
            try:
                response = await self.http.request(
                    method, url, params=params, content=content, headers=headers
                )
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise DiskError(0, type(exc).__name__) from exc
                await self._sleep(self._delay(attempt, None))
                attempt += 1
                continue
            status = response.status_code
            if status in ok:
                return response
            if status in RETRY_STATUSES and attempt < self.max_retries:
                await self._sleep(self._delay(attempt, response))
                attempt += 1
                continue
            if status in {401, 403}:
                raise DiskAuthError(status, _error_code(response))
            raise DiskError(status, _error_code(response))

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None and (retry_after := response.headers.get("Retry-After")):
            try:
                return min(float(retry_after), MAX_RETRY_AFTER)
            except ValueError:
                pass
        return float(self.backoff * 2**attempt)


def _error_code(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        return ""
    return str(body.get("error", "")) if isinstance(body, dict) else ""
