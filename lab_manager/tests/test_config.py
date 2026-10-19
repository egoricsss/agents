"""Тесты конфигурации: загрузка из окружения, значения по умолчанию, защита секретов."""

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from lab_manager.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN": "123456:telegram-secret",
    "TELEGRAM_OWNER_ID": "42",
    "YANDEX_DISK_TOKEN": "yandex-secret",
    "GIGACHAT_CREDENTIALS": "gigachat-secret",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in Settings.model_fields:
        monkeypatch.delenv(key.upper(), raising=False)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def load() -> Settings:
    # _env_file=None: тесты никогда не читают настоящий .env
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_loads_required_values_from_env(env: pytest.MonkeyPatch) -> None:
    settings = load()

    assert settings.telegram_bot_token.get_secret_value() == "123456:telegram-secret"
    assert settings.telegram_owner_id == 42
    assert settings.yandex_disk_token.get_secret_value() == "yandex-secret"
    assert settings.gigachat_credentials.get_secret_value() == "gigachat-secret"


def test_defaults(env: pytest.MonkeyPatch) -> None:
    settings = load()

    assert settings.yandex_disk_root == "disk:/Лабы"
    assert settings.gigachat_scope == "GIGACHAT_API_PERS"
    assert settings.gigachat_model == "GigaChat-2"
    assert settings.tz == "Europe/Moscow"
    assert settings.zone == ZoneInfo("Europe/Moscow")
    assert settings.data_dir == Path("data")
    assert settings.log_level == "INFO"
    assert settings.gigachat_ca_bundle_file is None


def test_gigachat_ca_bundle_must_exist(env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env.setenv("GIGACHAT_CA_BUNDLE_FILE", str(tmp_path / "missing.pem"))

    with pytest.raises(ValidationError, match="gigachat_ca_bundle_file"):
        load()


def test_gigachat_ca_bundle_empty_means_unset(env: pytest.MonkeyPatch) -> None:
    env.setenv("GIGACHAT_CA_BUNDLE_FILE", "")

    assert load().gigachat_ca_bundle_file is None


def test_gigachat_ca_bundle_accepts_existing_file(env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = tmp_path / "ca.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    env.setenv("GIGACHAT_CA_BUNDLE_FILE", str(bundle))

    assert load().gigachat_ca_bundle_file == bundle


def test_secrets_are_not_leaked_in_repr_or_dump(env: pytest.MonkeyPatch) -> None:
    settings = load()
    rendered = [repr(settings), str(settings), settings.model_dump_json()]

    for secret in ("telegram-secret", "yandex-secret", "gigachat-secret"):
        for text in rendered:
            assert secret not in text


@pytest.mark.parametrize("missing", sorted(REQUIRED_ENV))
def test_missing_required_value_fails(env: pytest.MonkeyPatch, missing: str) -> None:
    env.delenv(missing)

    with pytest.raises(ValidationError, match=missing.lower()):
        load()


@pytest.mark.parametrize("owner_id", ["0", "-5", "not-a-number"])
def test_owner_id_must_be_positive_int(env: pytest.MonkeyPatch, owner_id: str) -> None:
    env.setenv("TELEGRAM_OWNER_ID", owner_id)

    with pytest.raises(ValidationError):
        load()


def test_invalid_timezone_fails(env: pytest.MonkeyPatch) -> None:
    env.setenv("TZ", "Mars/Olympus")

    with pytest.raises(ValidationError, match="tz"):
        load()


def test_invalid_gigachat_scope_fails(env: pytest.MonkeyPatch) -> None:
    env.setenv("GIGACHAT_SCOPE", "GIGACHAT_API_UNKNOWN")

    with pytest.raises(ValidationError):
        load()


def test_disk_root_must_be_disk_path(env: pytest.MonkeyPatch) -> None:
    env.setenv("YANDEX_DISK_ROOT", "/Лабы")

    with pytest.raises(ValidationError, match="yandex_disk_root"):
        load()


def test_blank_secret_fails(env: pytest.MonkeyPatch) -> None:
    env.setenv("YANDEX_DISK_TOKEN", "   ")

    with pytest.raises(ValidationError, match="yandex_disk_token"):
        load()


def _env_example_keys() -> dict[str, str]:
    pairs: dict[str, str] = {}
    for raw in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key.strip()] = value.strip()
    return pairs


def test_env_example_documents_every_setting() -> None:
    expected = {name.upper() for name in Settings.model_fields}

    assert set(_env_example_keys()) == expected


def test_env_example_contains_no_secret_values() -> None:
    pairs = _env_example_keys()
    secret_fields = [
        name.upper()
        for name, field in Settings.model_fields.items()
        if "SecretStr" in str(field.annotation)
    ]

    assert secret_fields, "в Settings должны быть секретные поля"
    for key in secret_fields:
        assert pairs[key] == "", f"{key} в .env.example должен быть пустым"
