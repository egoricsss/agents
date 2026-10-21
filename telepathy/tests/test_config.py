from pathlib import Path

import pytest
from pydantic import ValidationError

from telepathy.config import Settings


def make(**env: str) -> Settings:
    return Settings(_env_file=None, **{"telegram_bot_token": "42:T", **env})  # type: ignore[call-arg]


def test_defaults_match_spec() -> None:
    settings = make()
    assert settings.telegram_mode == "polling"
    assert settings.llm_model == "deepseek:deepseek-chat"
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.match_window_hours == 24
    assert settings.score_threshold == 0.75
    assert settings.allowed_user_ids == ()


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "42:ENV")
    monkeypatch.setenv("SCORE_THRESHOLD", "0.8")
    monkeypatch.setenv("TAU_HOURS", "12")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1, 2,3")
    monkeypatch.setenv("LLM_API_KEY", "")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.score_threshold == 0.8
    assert settings.tau_hours == 12
    assert settings.allowed_user_ids == (1, 2, 3)
    assert settings.llm_api_key is None


def test_secrets_are_hidden() -> None:
    settings = make(llm_api_key="sk-secret")
    assert "sk-secret" not in repr(settings)


@pytest.mark.parametrize(
    "env",
    [
        {"telegram_bot_token": " "},
        {"llm_model": "deepseek-chat"},
        {"score_threshold": "1.5"},
        {"tau_hours": "0"},
        {"tz": "Mars/Olympus"},
        {"telegram_mode": "gateway"},  # без GATEWAY_SECRET
    ],
)
def test_invalid_values_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        make(**env)


def test_gateway_mode_with_secret() -> None:
    settings = make(telegram_mode="gateway", gateway_secret="s3cret")  # noqa: S106
    assert settings.gateway_secret is not None


def _env_example() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / ".env.example"
    pairs = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key] = value
    return pairs


def test_env_example_documents_every_setting() -> None:
    assert set(_env_example()) == {name.upper() for name in Settings.model_fields}


def test_env_example_has_no_secret_values() -> None:
    pairs = _env_example()
    for key in ("TELEGRAM_BOT_TOKEN", "LLM_API_KEY", "GATEWAY_SECRET", "GIGACHAT_CREDENTIALS"):
        assert pairs[key] == ""
