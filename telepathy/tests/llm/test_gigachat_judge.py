"""Судья на GigaChat: сам по себе и как запасной для DeepSeek. Сеть не используется."""

from collections.abc import AsyncIterator

import httpx
import pytest

from telepathy.config import Settings
from telepathy.llm.gigachat import GigaChatClient
from telepathy.llm.pydantic_adapter import gigachat_model
from telepathy.services.judge import DisabledJudge, Judge, Match, judge_from_settings
from tests.llm.fake_gigachat import CREDENTIALS, FakeGigaChat

SAME = {"is_same": True, "common_idea": "пицца", "message": "🍕 Оба про пиццу!"}


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fake() -> FakeGigaChat:
    return FakeGigaChat()


@pytest.fixture
async def client(fake: FakeGigaChat) -> AsyncIterator[GigaChatClient]:
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        yield GigaChatClient(CREDENTIALS, http=http, sleep=no_sleep, model="GigaChat-2")


def settings(**env: str) -> Settings:
    return Settings(_env_file=None, telegram_bot_token="42:T", **env)  # type: ignore[call-arg]  # noqa: S106


async def test_judge_through_gigachat(client: GigaChatClient, fake: FakeGigaChat) -> None:
    fake.replies = [SAME]

    verdict = await Judge(gigachat_model(client)).compare("хочу пиццу", "пиццу бы")

    assert verdict == Match(**SAME)
    [request] = fake.chat_requests
    assert request["function_call"]["name"] == request["functions"][0]["name"]
    assert "хочу пиццу" in request["messages"][-1]["content"]


async def test_gigachat_as_primary_model(client: GigaChatClient, fake: FakeGigaChat) -> None:
    fake.replies = [SAME]
    judge = judge_from_settings(settings(llm_model="gigachat:GigaChat-2-Max"), gigachat=client)

    assert isinstance(judge, Judge)
    assert await judge.compare("a", "b") == Match(**SAME)


async def test_gigachat_backs_up_deepseek(client: GigaChatClient, fake: FakeGigaChat) -> None:
    # Запросы к настоящим моделям в тестах запрещены — DeepSeek «падает», отвечает GigaChat.
    fake.replies = [SAME]
    judge = judge_from_settings(settings(llm_api_key="sk-test"), gigachat=client)

    assert await judge.compare("хочу пиццу", "пиццу бы") == Match(**SAME)
    assert len(fake.chat_requests) == 1


async def test_gigachat_alone_when_deepseek_has_no_key(
    client: GigaChatClient, fake: FakeGigaChat, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    fake.replies = [SAME]

    judge = judge_from_settings(settings(), gigachat=client)

    assert await judge.compare("a", "b") == Match(**SAME)


def test_nothing_configured_disables_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert isinstance(judge_from_settings(settings(), gigachat=None), DisabledJudge)
    assert isinstance(
        judge_from_settings(settings(llm_model="gigachat:GigaChat-2"), gigachat=None),
        DisabledJudge,
    )
