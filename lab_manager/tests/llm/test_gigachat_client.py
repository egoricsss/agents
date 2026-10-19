"""Клиент GigaChat на фейковом сервере: токены, function calling, повторы, секреты."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from lab_manager.llm.gigachat import GigaChatAuthError, GigaChatClient, GigaChatError
from tests.llm.fake_gigachat import CREDENTIALS, FakeGigaChat

MESSAGES = [{"role": "user", "content": "привет"}]
FUNCTION = {
    "name": "save_lab",
    "description": "Сохранить поля",
    "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}},
}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fake() -> FakeGigaChat:
    return FakeGigaChat()


@pytest.fixture
async def http(fake: FakeGigaChat) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=fake.transport()) as client:
        yield client


@pytest.fixture
def client(http: httpx.AsyncClient) -> GigaChatClient:
    return GigaChatClient(CREDENTIALS, http=http, sleep=no_sleep)


async def test_plain_chat(client: GigaChatClient, fake: FakeGigaChat) -> None:
    fake.replies = ["Здравствуйте!"]

    result = await client.chat(MESSAGES)

    assert result.content == "Здравствуйте!"
    assert result.function_call is None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 20
    assert result.model.startswith("GigaChat-2")
    body = fake.chat_requests[0]
    assert body["model"] == "GigaChat-2"
    assert body["messages"] == MESSAGES


async def test_forced_function_call(client: GigaChatClient, fake: FakeGigaChat) -> None:
    fake.replies = [{"number": 3}]

    result = await client.chat(
        MESSAGES, functions=[FUNCTION], function_call="save_lab", temperature=0.1
    )

    assert result.function_call is not None
    assert result.function_call.name == "save_lab"
    assert result.function_call.arguments == {"number": 3}
    body = fake.chat_requests[0]
    assert body["functions"] == [FUNCTION]
    assert body["function_call"] == {"name": "save_lab"}
    assert body["temperature"] == 0.1


async def test_function_arguments_as_json_string_are_parsed(
    client: GigaChatClient, fake: FakeGigaChat
) -> None:
    fake.arguments_as_string = True
    fake.replies = [{"number": 3}]

    result = await client.chat(MESSAGES, functions=[FUNCTION], function_call="save_lab")

    assert result.function_call is not None
    assert result.function_call.arguments == {"number": 3}


async def test_token_is_reused(client: GigaChatClient, fake: FakeGigaChat) -> None:
    for _ in range(3):
        await client.chat(MESSAGES)

    assert fake.oauth_calls == 1


async def test_token_refreshed_before_expiry(http: httpx.AsyncClient, fake: FakeGigaChat) -> None:
    clock = Clock()
    client = GigaChatClient(CREDENTIALS, http=http, sleep=no_sleep, clock=clock)
    await client.chat(MESSAGES)

    clock.now = 10**12  # далеко за expires_at
    await client.chat(MESSAGES)

    assert fake.oauth_calls == 2


async def test_rejected_token_is_refreshed_once(client: GigaChatClient, fake: FakeGigaChat) -> None:
    await client.chat(MESSAGES)
    fake.expire_all()

    result = await client.chat(MESSAGES)

    assert result.content == "ok"
    assert fake.oauth_calls == 2


async def test_concurrent_calls_share_one_token_request(
    client: GigaChatClient, fake: FakeGigaChat
) -> None:
    await asyncio.gather(*(client.chat(MESSAGES) for _ in range(5)))

    assert fake.oauth_calls == 1


async def test_oauth_request_shape(client: GigaChatClient, fake: FakeGigaChat) -> None:
    await client.chat(MESSAGES)

    [oauth] = fake.oauth_headers
    assert oauth["Authorization"] == f"Basic {CREDENTIALS}"
    [chat] = fake.chat_headers
    assert chat["Authorization"].startswith("Bearer ")
    assert CREDENTIALS not in chat["Authorization"]


async def test_bad_credentials_raise_auth_error_without_leaking(
    http: httpx.AsyncClient,
) -> None:
    client = GigaChatClient("d3Jvbmc6c2VjcmV0", http=http, sleep=no_sleep)

    with pytest.raises(GigaChatAuthError) as error:
        await client.chat(MESSAGES)

    assert "d3Jvbmc6c2VjcmV0" not in str(error.value)
    assert "d3Jvbmc6c2VjcmV0" not in repr(client)


async def test_wrong_scope_raises_auth_error(http: httpx.AsyncClient) -> None:
    client = GigaChatClient(CREDENTIALS, scope="GIGACHAT_API_CORP", http=http, sleep=no_sleep)

    with pytest.raises(GigaChatAuthError):
        await client.chat(MESSAGES)


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_transient_errors_are_retried(
    client: GigaChatClient, fake: FakeGigaChat, status: int
) -> None:
    fake.fail_next = [status, status]

    assert (await client.chat(MESSAGES)).content == "ok"


async def test_retries_are_bounded(http: httpx.AsyncClient, fake: FakeGigaChat) -> None:
    client = GigaChatClient(CREDENTIALS, http=http, sleep=no_sleep, max_retries=1)
    fake.fail_next = [503] * 5

    with pytest.raises(GigaChatError) as error:
        await client.chat(MESSAGES)

    assert error.value.status == 503


async def test_client_error_is_not_retried(client: GigaChatClient, fake: FakeGigaChat) -> None:
    fake.fail_next = [422]

    with pytest.raises(GigaChatError) as error:
        await client.chat(MESSAGES)

    assert error.value.status == 422
    assert len(fake.chat_headers) == 1


async def test_model_override(client: GigaChatClient, fake: FakeGigaChat) -> None:
    await client.chat(MESSAGES, model="GigaChat-2-Pro")

    assert fake.chat_requests[0]["model"] == "GigaChat-2-Pro"
