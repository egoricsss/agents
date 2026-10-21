"""Асинхронный клиент GigaChat API (OAuth + chat/completions с function calling).

- Токен доступа (живёт 30 мин) кешируется и обновляется заранее; одновременные запросы
  получают его одним OAuth-вызовом. 401 от API → одно обновление токена и повтор.
- Временные ошибки (429, 5xx, сеть) повторяются с экспоненциальной задержкой.
- TLS проверяется по корневому сертификату Минцифры (`ca_bundle`), проверку не отключаем.
- Ключ авторизации не попадает в repr, ошибки и запросы к API (туда идёт только Bearer).
"""

import asyncio
import json
import ssl
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import SecretStr

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://gigachat.devices.sberbank.ru/api/v1"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
TOKEN_REFRESH_MARGIN_MS = 60_000

Sleep = Callable[[float], Awaitable[None]]
Message = dict[str, Any]


class GigaChatError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        self.status = status
        super().__init__(f"GigaChat ответил {status} {message}".strip())


class GigaChatAuthError(GigaChatError):
    """Неверный ключ авторизации/scope или токен не принимается."""


@dataclass(frozen=True, slots=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    function_call: FunctionCall | None
    usage: Usage
    model: str
    finish_reason: str


def make_ssl_context(ca_bundle: Path | None) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(ca_bundle) if ca_bundle else None)


class GigaChatClient:
    def __init__(
        self,
        credentials: SecretStr | str,
        *,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat-2",
        ca_bundle: Path | None = None,
        http: httpx.AsyncClient | None = None,
        auth_url: str = AUTH_URL,
        api_url: str = API_URL,
        max_retries: int = 3,
        backoff: float = 1.0,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.time,
        timeout: float = 120.0,
    ) -> None:
        self._credentials = (
            credentials if isinstance(credentials, SecretStr) else SecretStr(credentials)
        )
        self.scope = scope
        self.model = model
        self._owns_http = http is None
        self.http = http or httpx.AsyncClient(verify=make_ssl_context(ca_bundle), timeout=timeout)
        self.auth_url = auth_url
        self.api_url = api_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._clock = clock
        self._token: SecretStr | None = None
        self._expires_at_ms = 0
        self._token_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"GigaChatClient(model={self.model!r}, scope={self.scope!r})"

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

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        functions: Sequence[dict[str, Any]] | None = None,
        function_call: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {"model": model or self.model, "messages": list(messages)}
        if functions:
            body["functions"] = list(functions)
            body["function_call"] = {"name": function_call} if function_call else "auto"
        if temperature is not None:
            body["temperature"] = temperature

        response = await self._authorized_post(f"{self.api_url}/chat/completions", body)
        return _parse_chat(response.json())

    async def _authorized_post(self, url: str, body: dict[str, Any]) -> httpx.Response:
        for refreshed in (False, True):
            token = await self._get_token(force=refreshed)
            response = await self._send(
                "POST",
                url,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                json_body=body,
                ok={200, 401},
            )
            if response.status_code == 200:
                return response
        raise GigaChatAuthError(401, "токен доступа не принят")

    async def _get_token(self, *, force: bool = False) -> str:
        async with self._token_lock:
            now_ms = self._clock() * 1000
            if (
                force
                or self._token is None
                or now_ms >= self._expires_at_ms - TOKEN_REFRESH_MARGIN_MS
            ):
                await self._refresh_token()
            assert self._token is not None  # noqa: S101
            return self._token.get_secret_value()

    async def _refresh_token(self) -> None:
        response = await self._send(
            "POST",
            self.auth_url,
            headers={
                "Authorization": f"Basic {self._credentials.get_secret_value()}",
                "RqUID": str(uuid.uuid4()),
                "Accept": "application/json",
            },
            form={"scope": self.scope},
            ok={200},
            auth_errors={400, 401, 403},
        )
        data = response.json()
        self._token = SecretStr(data["access_token"])
        self._expires_at_ms = int(data["expires_at"])

    async def _send(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        ok: set[int],
        json_body: dict[str, Any] | None = None,
        form: dict[str, str] | None = None,
        auth_errors: set[int] | None = None,
    ) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = await self.http.request(
                    method, url, headers=headers, json=json_body, data=form
                )
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise GigaChatError(0, type(exc).__name__) from exc
                await self._sleep(self.backoff * 2**attempt)
                attempt += 1
                continue
            status = response.status_code
            if status in ok:
                return response
            if status in RETRY_STATUSES and attempt < self.max_retries:
                await self._sleep(self.backoff * 2**attempt)
                attempt += 1
                continue
            message = _error_message(response)
            if status in (auth_errors or {401, 403}):
                raise GigaChatAuthError(status, message)
            raise GigaChatError(status, message)


def _parse_chat(data: dict[str, Any]) -> ChatResult:
    choice = data["choices"][0]
    message = choice.get("message", {})
    call = message.get("function_call")
    function_call = None
    if call:
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        function_call = FunctionCall(call["name"], arguments)
    usage = data.get("usage") or {}
    return ChatResult(
        content=message.get("content") or "",
        function_call=function_call,
        usage=Usage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        ),
        model=data.get("model", ""),
        finish_reason=choice.get("finish_reason", ""),
    )


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    return str(body.get("message", ""))[:200] if isinstance(body, dict) else ""
