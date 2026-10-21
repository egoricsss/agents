import pytest

from lab_manager.config import Settings
from lab_manager.remote_agents import TELEPATHY_NAME, remote_agents
from tests.test_config import REQUIRED_ENV


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # markitdown при импорте делает load_dotenv(): настоящий .env может оказаться в окружении.
    for key in Settings.model_fields:
        monkeypatch.delenv(key.upper(), raising=False)


def settings(**extra: str) -> Settings:
    env = {key.lower(): value for key, value in REQUIRED_ENV.items()}
    return Settings(_env_file=None, **{**env, **extra})  # type: ignore[call-arg]


async def test_no_telepathy_by_default() -> None:
    assert remote_agents(settings()) == ([], {})


async def test_telepathy_agent_and_guests() -> None:
    agents, guests = remote_agents(
        settings(
            telepathy_url="http://telepathy:9010",
            telepathy_secret="s",  # noqa: S106 — тестовый
            telepathy_users="777",
        )
    )

    [agent] = agents
    assert agent.name == TELEPATHY_NAME
    assert agent.endpoint == "http://telepathy:9010/telegram/update"
    assert guests == {777: TELEPATHY_NAME}
    assert {"pair", "join", "stats"} <= {command for command, _ in agent.commands()}
    await agent.aclose()


@pytest.mark.parametrize("url", ["http://telepathy:9010/", "http://telepathy:9010"])
async def test_endpoint_has_single_slash(url: str) -> None:
    [agent], _ = remote_agents(settings(telepathy_url=url, telepathy_secret="s"))  # noqa: S106
    assert agent.endpoint.count("//") == 1
    await agent.aclose()
