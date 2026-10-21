"""Движок SQLAlchemy (async, aiosqlite) с расширением sqlite-vec на каждом соединении."""

from pathlib import Path
from typing import Any, Self

import sqlite_vec
from sqlalchemy import event
from sqlalchemy.engine.interfaces import AdaptedConnection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from telepathy.db.models import Base


def _on_connect(dbapi_connection: AdaptedConnection, _record: Any) -> None:  # noqa: ANN401
    # aiosqlite асинхронный, а событие connect синхронное: run_async выполняет корутину
    # драйвера в том же цикле событий.
    dbapi_connection.run_async(lambda conn: conn.enable_load_extension(True))
    dbapi_connection.run_async(lambda conn: conn.load_extension(sqlite_vec.loadable_path()))
    dbapi_connection.run_async(lambda conn: conn.enable_load_extension(False))
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Database:
    def __init__(self, engine: AsyncEngine) -> None:
        event.listen(engine.sync_engine, "connect", _on_connect)
        self.engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def file(cls, path: Path) -> Self:
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(create_async_engine(f"sqlite+aiosqlite:///{path}"))

    @classmethod
    def memory(cls) -> Self:
        # Одно соединение на всё время жизни: иначе каждая сессия видела бы свою пустую БД.
        return cls(
            create_async_engine(
                "sqlite+aiosqlite://",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
        )

    async def create_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def session(self) -> AsyncSession:
        return self._sessions()

    async def close(self) -> None:
        await self.engine.dispose()
