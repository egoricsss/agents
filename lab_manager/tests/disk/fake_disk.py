"""Фейковый Яндекс Диск в памяти: повторяет поведение REST API, нужное клиенту."""

import hashlib
import itertools
import json

import httpx

API_HOST = "cloud-api.yandex.net"
UPLOAD_HOST = "uploader.fake"
TOKEN = "test-token"  # noqa: S105 — фиктивный токен для тестов


def _norm(path: str) -> str:
    return path if path.endswith(":/") else path.rstrip("/")


def _parent(path: str) -> str:
    head, _, _ = path.rpartition("/")
    return head + "/" if head.endswith(":") else head


def _error(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"error": code, "message": code, "description": code})


class FakeDisk:
    def __init__(self) -> None:
        self.dirs: set[str] = {"app:/", "disk:/"}
        self.files: dict[str, bytes] = {}
        self.fail_next: list[int] = []  # статусы, которые вернут ближайшие запросы к API
        self.upload_requests: list[httpx.Request] = []
        self.api_calls: list[tuple[str, str]] = []
        self._pending: dict[str, str] = {}
        # Как настоящий Диск: удаление папки асинхронное (202 + операция).
        self.operation_polls_before_done = 1
        self.fail_operations = False
        self._operations: dict[str, tuple[str, int]] = {}
        self._ids = itertools.count(1)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == UPLOAD_HOST:
            return self._upload(request)
        assert request.url.host == API_HOST
        self.api_calls.append((request.method, request.url.path))
        if self.fail_next:
            status = self.fail_next.pop(0)
            return _error(status, "FakeFailure")
        if request.headers.get("Authorization") != f"OAuth {TOKEN}":
            return _error(401, "UnauthorizedError")
        path = _norm(request.url.params.get("path", ""))
        route = (request.method, request.url.path)
        if route == ("GET", "/v1/disk/resources"):
            return self._meta(path)
        if route == ("PUT", "/v1/disk/resources"):
            return self._mkdir(path)
        if route == ("GET", "/v1/disk/resources/upload"):
            overwrite = request.url.params.get("overwrite") == "true"
            return self._upload_link(path, overwrite)
        if route == ("DELETE", "/v1/disk/resources"):
            return self._delete(path)
        if request.method == "GET" and request.url.path.startswith("/v1/disk/operations/"):
            return self._operation(request.url.path.rsplit("/", 1)[-1])
        return _error(405, "MethodNotAllowed")

    def _meta(self, path: str) -> httpx.Response:
        if path in self.dirs:
            return httpx.Response(200, json={"type": "dir", "path": path, "name": path})
        if path in self.files:
            data = self.files[path]
            return httpx.Response(
                200,
                json={
                    "type": "file",
                    "path": path,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "md5": hashlib.md5(data).hexdigest(),  # noqa: S324
                },
            )
        return _error(404, "DiskNotFoundError")

    def _mkdir(self, path: str) -> httpx.Response:
        if path in self.dirs:
            return _error(409, "DiskPathPointsToExistentDirectoryError")
        if _parent(path) not in self.dirs:
            return _error(409, "DiskPathDoesntExistsError")
        self.dirs.add(path)
        return httpx.Response(201, json={"href": f"https://{API_HOST}/v1/disk/resources"})

    def _upload_link(self, path: str, overwrite: bool) -> httpx.Response:
        if _parent(path) not in self.dirs:
            return _error(409, "DiskPathDoesntExistsError")
        if path in self.files and not overwrite:
            return _error(409, "DiskResourceAlreadyExistsError")
        upload_id = str(next(self._ids))
        self._pending[upload_id] = path
        body = {"href": f"https://{UPLOAD_HOST}/u/{upload_id}", "method": "PUT", "templated": False}
        return httpx.Response(200, content=json.dumps(body))

    def _upload(self, request: httpx.Request) -> httpx.Response:
        self.upload_requests.append(request)
        path = self._pending.pop(request.url.path.rsplit("/", 1)[-1], None)
        if path is None or request.method != "PUT":
            return httpx.Response(404)
        self.files[path] = request.content
        return httpx.Response(201)

    def _delete(self, path: str) -> httpx.Response:
        if path in self.files:
            del self.files[path]
            return httpx.Response(204)
        if path not in self.dirs:
            return _error(404, "DiskNotFoundError")
        operation_id = f"op{next(self._ids)}"
        self._operations[operation_id] = (path, 0)
        href = f"https://{API_HOST}/v1/disk/operations/{operation_id}"
        return httpx.Response(202, json={"href": href, "method": "GET", "templated": False})

    def _operation(self, operation_id: str) -> httpx.Response:
        if operation_id not in self._operations:
            return _error(404, "DiskOperationNotFoundError")
        path, polls = self._operations[operation_id]
        if self.fail_operations:
            return httpx.Response(200, json={"status": "failed"})
        if polls < self.operation_polls_before_done:
            self._operations[operation_id] = (path, polls + 1)
            return httpx.Response(200, json={"status": "in-progress"})
        self._remove_tree(path)
        return httpx.Response(200, json={"status": "success"})

    def _remove_tree(self, path: str) -> None:
        self.files = {
            p: d for p, d in self.files.items() if not (p == path or p.startswith(path + "/"))
        }
        self.dirs = {p for p in self.dirs if not (p == path or p.startswith(path + "/"))}
