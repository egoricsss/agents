"""Фейковый GigaChat: OAuth + chat/completions с function calling, как в реальном API.

Формат ответа снят с настоящего API (2026-10): `arguments` — объект, `expires_at` — мс.
"""

import itertools
import json
import time
import uuid
from typing import Any
from urllib.parse import parse_qs

import httpx

AUTH_HOST = "ngw.devices.sberbank.ru"
API_HOST = "gigachat.devices.sberbank.ru"
CREDENTIALS = "dGVzdDpzZWNyZXQ="  # base64("test:secret") — фиктивные


class FakeGigaChat:
    def __init__(self, token_ttl_ms: int = 30 * 60 * 1000) -> None:
        self.token_ttl_ms = token_ttl_ms
        self.valid_tokens: set[str] = set()
        self.oauth_calls = 0
        self.chat_requests: list[dict[str, Any]] = []
        self.chat_headers: list[httpx.Headers] = []
        self.oauth_headers: list[httpx.Headers] = []
        self.replies: list[dict[str, Any] | str] = []  # dict — function_call.arguments, str — текст
        self.fail_next: list[int] = []
        self.arguments_as_string = False
        self._ids = itertools.count(1)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def expire_all(self) -> None:
        self.valid_tokens.clear()

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == AUTH_HOST and request.url.path == "/api/v2/oauth":
            return self._oauth(request)
        if request.url.host == API_HOST and request.url.path == "/api/v1/chat/completions":
            return self._chat(request)
        return httpx.Response(404)

    def _oauth(self, request: httpx.Request) -> httpx.Response:
        self.oauth_calls += 1
        self.oauth_headers.append(request.headers)
        uuid.UUID(request.headers["RqUID"])  # реальный API требует UUID
        if request.headers.get("Authorization") != f"Basic {CREDENTIALS}":
            return httpx.Response(401, json={"code": 6, "message": "credentials doesn't match"})
        scope = parse_qs(request.content.decode())["scope"][0]
        if scope != "GIGACHAT_API_PERS":
            return httpx.Response(400, json={"code": 7, "message": "scope is invalid"})
        token = f"tok{next(self._ids)}"
        self.valid_tokens.add(token)
        expires_at = int(time.time() * 1000) + self.token_ttl_ms
        return httpx.Response(200, json={"access_token": token, "expires_at": expires_at})

    def _chat(self, request: httpx.Request) -> httpx.Response:
        self.chat_headers.append(request.headers)
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), json={"status": 0, "message": "fail"})
        auth = request.headers.get("Authorization", "")
        if auth.removeprefix("Bearer ") not in self.valid_tokens:
            return httpx.Response(401, json={"status": 401, "message": "Unauthorized"})
        body = json.loads(request.content)
        self.chat_requests.append(body)
        reply = self.replies.pop(0) if self.replies else "ok"
        if isinstance(reply, str):
            message: dict[str, Any] = {"role": "assistant", "content": reply}
            finish = "stop"
        else:
            arguments: Any = (
                json.dumps(reply, ensure_ascii=False) if self.arguments_as_string else reply
            )
            name = (body.get("function_call") or {}).get("name") or body["functions"][0]["name"]
            message = {
                "role": "assistant",
                "content": "",
                "function_call": {"name": name, "arguments": arguments},
                "functions_state_id": str(uuid.uuid4()),
            }
            finish = "function_call"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message, "index": 0, "finish_reason": finish}],
                "created": int(time.time()),
                "model": f"{body['model']}:2.0.28.2",
                "object": "chat.completion",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )
