"""Пары: регистрация, код-приглашение, вступление. В паре строго два человека."""

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass

from telepathy.db import Database
from telepathy.db.repos import Pairs, Users
from telepathy.services.errors import (
    AlreadyPairedError,
    InviteNotFoundError,
    OwnInviteError,
    PairFullError,
)
from telepathy.services.matching import Clock, utc_now

log = logging.getLogger(__name__)

# Без похожих символов (0/O, 1/I/L), чтобы код легко продиктовать.
INVITE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
INVITE_LENGTH = 6
DEFAULT_NAME = "Партнёр"


def new_invite_code() -> str:
    return "".join(secrets.choice(INVITE_ALPHABET) for _ in range(INVITE_LENGTH))


@dataclass(frozen=True, slots=True)
class PairStatus:
    pair_id: int | None
    partner_id: int | None
    invite_code: str | None

    @property
    def complete(self) -> bool:
        return self.partner_id is not None


@dataclass(frozen=True, slots=True)
class Joined:
    pair_id: int
    partner_id: int


class PairingService:
    def __init__(
        self,
        db: Database,
        *,
        clock: Clock = utc_now,
        code_factory: Callable[[], str] = new_invite_code,
    ) -> None:
        self.db = db
        self.clock = clock
        self.code_factory = code_factory

    async def register(self, user_id: int, first_name: str | None) -> None:
        async with self.db.session() as session, session.begin():
            await Users(session).upsert(user_id, first_name, self.clock())

    async def display_name(self, user_id: int) -> str:
        async with self.db.session() as session:
            user = await Users(session).get(user_id)
        return (user.first_name if user else None) or DEFAULT_NAME

    async def status(self, user_id: int) -> PairStatus:
        async with self.db.session() as session:
            pair = await Pairs(session).of_user(user_id)
        if pair is None:
            return PairStatus(pair_id=None, partner_id=None, invite_code=None)
        return PairStatus(
            pair_id=pair.id, partner_id=pair.partner_of(user_id), invite_code=pair.invite_code
        )

    async def invite(self, user_id: int) -> str:
        """Код для партнёра (`/pair`). Повторный вызов до вступления возвращает тот же код."""
        async with self.db.session() as session, session.begin():
            pairs = Pairs(session)
            pair = await pairs.of_user(user_id)
            if pair is not None and pair.complete:
                raise AlreadyPairedError
            if pair is not None and pair.invite_code:
                return pair.invite_code
            code = self.code_factory()
            while await pairs.by_code(code) is not None:
                code = self.code_factory()
            await pairs.add(user_id, code, self.clock())
        log.info("user %s created an invite", user_id)
        return code

    async def join(self, user_id: int, code: str) -> Joined:
        """Вступить в пару по коду (`/join`)."""
        code = code.strip().upper()
        async with self.db.session() as session, session.begin():
            pairs = Pairs(session)
            target = await pairs.by_code(code)
            if target is None:
                raise InviteNotFoundError
            if target.user_a_id == user_id:
                raise OwnInviteError
            if target.complete:
                raise PairFullError
            mine = await pairs.of_user(user_id)
            if mine is not None:
                if mine.complete:
                    raise AlreadyPairedError
                await pairs.remove_pending(mine)  # своё неиспользованное приглашение не нужно
            if not await pairs.join(target, user_id, self.clock()):
                raise PairFullError
        log.info("user %s joined pair %s", user_id, target.id)
        return Joined(pair_id=target.id, partner_id=target.user_a_id)
