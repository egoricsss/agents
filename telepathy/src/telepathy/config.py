"""Конфигурация из переменных окружения и `.env`. Все пороги и ключи — только отсюда."""

from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
# polling — самостоятельный бот со своим токеном;
# gateway — апдейты присылает общий шлюз (lab_manager) по внутренней docker-сети.
TelegramMode = Literal["polling", "gateway"]


class Settings(BaseSettings):
    """Секреты хранятся как `SecretStr` и не попадают в repr/логи."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    telegram_bot_token: SecretStr
    telegram_mode: TelegramMode = "polling"
    # Пусто — бот открыт всем (пары всё равно приватны); иначе — только эти Telegram ID.
    allowed_user_ids: Annotated[tuple[int, ...], NoDecode] = ()
    gateway_secret: SecretStr | None = None
    gateway_host: str = "0.0.0.0"  # noqa: S104 — порт не публикуется, только docker-сеть
    gateway_port: int = Field(9010, ge=1, le=65535)

    # LLM-судья: строка модели PydanticAI «провайдер:модель».
    llm_model: str = "deepseek:deepseek-chat"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: float = Field(30.0, gt=0)

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_threads: int = Field(1, ge=1)

    match_window_hours: float = Field(24.0, gt=0)
    tau_hours: float = Field(24.0, gt=0)
    score_threshold: float = Field(0.75, ge=0, le=1)

    tz: str = "Europe/Moscow"
    data_dir: Path = Path("data")
    log_level: LogLevel = "INFO"

    @field_validator("telegram_bot_token")
    @classmethod
    def _secret_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("секрет не может быть пустым")
        return value

    @field_validator("llm_api_key", "gateway_secret", "llm_base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def _comma_separated_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(int(part) for part in value.replace(" ", "").split(",") if part)
        return value

    @field_validator("llm_model")
    @classmethod
    def _model_has_provider(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("модель задаётся как «провайдер:модель»: deepseek:deepseek-chat")
        return value

    @field_validator("tz")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"неизвестный часовой пояс: {value}") from exc
        return value

    @model_validator(mode="after")
    def _gateway_needs_secret(self) -> Self:
        if self.telegram_mode == "gateway" and self.gateway_secret is None:
            raise ValueError("в режиме gateway нужен GATEWAY_SECRET")
        return self

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "telepathy.db"
