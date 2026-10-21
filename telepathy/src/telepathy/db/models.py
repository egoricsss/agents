"""Схема БД. Время хранится в UTC; эмбеддинги — float32-блобы в формате sqlite-vec."""

from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, LargeBinary, String, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """SQLite не хранит часовой пояс: пишем наивное UTC, читаем обратно aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("ожидается datetime с часовым поясом")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    type_annotation_map = {datetime: UTCDateTime()}  # noqa: RUF012 — так задаёт SQLAlchemy


class User(Base):
    __tablename__ = "users"

    # Telegram ID пользователя — он же первичный ключ.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    first_name: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime]


class Pair(Base):
    """Пара — ровно два места: `user_a` создал приглашение, `user_b` присоединился по коду."""

    __tablename__ = "pairs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_a_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    user_b_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), unique=True)
    invite_code: Mapped[str | None] = mapped_column(String(16), unique=True)
    created_at: Mapped[datetime]
    joined_at: Mapped[datetime | None]

    @property
    def complete(self) -> bool:
        return self.user_b_id is not None

    def partner_of(self, user_id: int) -> int | None:
        if user_id == self.user_a_id:
            return self.user_b_id
        if user_id == self.user_b_id:
            return self.user_a_id
        raise ValueError(f"пользователь {user_id} не состоит в паре {self.id}")


class Thought(Base):
    __tablename__ = "thoughts"
    __table_args__ = (Index("ix_thoughts_pair_time", "pair_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    pair_id: Mapped[int] = mapped_column(ForeignKey("pairs.id"))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime]
    matched_at: Mapped[datetime | None]


class ThoughtMatch(Base):
    """Совпадение двух мыслей: `thought_a` — более ранняя."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("pairs.id"), index=True)
    thought_a_id: Mapped[int] = mapped_column(ForeignKey("thoughts.id"), unique=True)
    thought_b_id: Mapped[int] = mapped_column(ForeignKey("thoughts.id"), unique=True)
    score: Mapped[float] = mapped_column(Float)
    delta_seconds: Mapped[float] = mapped_column(Float)
    common_idea: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]
