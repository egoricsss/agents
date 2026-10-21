"""Конфигурация приложения из переменных окружения и `.env`."""

from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AnyHttpUrl, Field, FilePath, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

GigaChatScope = Literal["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Все настройки сервиса. Секреты хранятся как `SecretStr` и не попадают в repr/логи."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    telegram_bot_token: SecretStr
    telegram_owner_id: int = Field(gt=0)

    yandex_disk_token: SecretStr
    yandex_disk_root: str = "disk:/Лабы"

    gigachat_credentials: SecretStr
    gigachat_scope: GigaChatScope = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2"
    # Корневой сертификат Минцифры (PEM): нужен для TLS к GigaChat.
    gigachat_ca_bundle_file: FilePath | None = None

    # Агент «Телепатия» в соседнем контейнере (пусто — не подключён).
    telepathy_url: AnyHttpUrl | None = None
    telepathy_secret: SecretStr | None = None
    # Гости: Telegram ID, которым доступна только «Телепатия».
    telepathy_users: Annotated[tuple[int, ...], NoDecode] = ()

    tz: str = "Europe/Moscow"
    data_dir: Path = Path("data")
    log_level: LogLevel = "INFO"

    @field_validator("telegram_bot_token", "yandex_disk_token", "gigachat_credentials")
    @classmethod
    def _secret_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("секрет не может быть пустым")
        return value

    @field_validator("gigachat_ca_bundle_file", "telepathy_url", "telepathy_secret", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("telepathy_users", mode="before")
    @classmethod
    def _comma_separated_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(int(part) for part in value.replace(" ", "").split(",") if part)
        return value

    @model_validator(mode="after")
    def _telepathy_needs_secret(self) -> Self:
        if self.telepathy_url is not None and self.telepathy_secret is None:
            raise ValueError("для TELEPATHY_URL нужен TELEPATHY_SECRET (общий с telepathy)")
        return self

    @field_validator("yandex_disk_root")
    @classmethod
    def _disk_path(cls, value: str) -> str:
        if not value.startswith(("disk:/", "app:/")):
            raise ValueError("путь на Диске должен начинаться с 'disk:/' или 'app:/'")
        return value.rstrip("/")

    @field_validator("tz")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"неизвестный часовой пояс: {value}") from exc
        return value

    @cached_property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)
