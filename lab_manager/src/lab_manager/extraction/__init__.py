"""Извлечение полей лабы из промпта владельца и текста задания."""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

REQUIRED_FIELDS = ("subject", "number")


class LabExtraction(BaseModel):
    """Результат извлечения. Обязательны предмет и номер, остальное — по возможности."""

    model_config = ConfigDict(frozen=True)

    subject: str | None = None
    number: int | None = Field(default=None, ge=1)
    title: str | None = None
    deadline: datetime | None = None
    summary: str = ""
    requirements: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def missing_fields(self) -> list[str]:
        return [name for name in REQUIRED_FIELDS if getattr(self, name) is None]


class LabExtractor(Protocol):
    """Общий интерфейс: правила (без сети) и LLM-экстрактор реализуют его одинаково."""

    @property
    def version(self) -> str:
        """Версия логики/промпта — часть ключа кеша."""
        ...

    async def extract(
        self,
        *,
        markdown: str,
        prompt: str | None,
        today: date,
        tz: ZoneInfo,
        known_subjects: Sequence[str] = (),
    ) -> LabExtraction:
        """`known_subjects` — уже существующие предметы, чтобы не плодить синонимы."""
        ...
