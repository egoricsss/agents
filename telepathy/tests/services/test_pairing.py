"""Пара: код, вступление, нельзя в свою пару и нельзя третьего."""

import asyncio
import itertools

import pytest

from telepathy.db import Database
from telepathy.services.errors import (
    AlreadyPairedError,
    InviteNotFoundError,
    OwnInviteError,
    PairFullError,
)
from telepathy.services.pairing import INVITE_ALPHABET, PairingService, new_invite_code
from tests.conftest import NOW

ANYA, BORIS, CARL = 1, 2, 3


@pytest.fixture
async def pairing(db: Database) -> PairingService:
    service = PairingService(db, clock=lambda: NOW)
    for uid, name in ((ANYA, "Аня"), (BORIS, "Борис"), (CARL, "Карл")):
        await service.register(uid, name)
    return service


def test_invite_codes_are_readable() -> None:
    code = new_invite_code()
    assert len(code) == 6
    assert set(code) <= set(INVITE_ALPHABET)
    assert not set(code) & set("01OIL")


async def test_pair_and_join(pairing: PairingService) -> None:
    code = await pairing.invite(ANYA)
    joined = await pairing.join(BORIS, f"  {code.lower()} ")

    assert joined.partner_id == ANYA
    assert (await pairing.status(ANYA)).partner_id == BORIS
    assert (await pairing.status(BORIS)).partner_id == ANYA
    assert await pairing.display_name(ANYA) == "Аня"


async def test_repeated_pair_returns_same_code(pairing: PairingService) -> None:
    assert await pairing.invite(ANYA) == await pairing.invite(ANYA)


async def test_cannot_join_own_pair(pairing: PairingService) -> None:
    code = await pairing.invite(ANYA)
    with pytest.raises(OwnInviteError):
        await pairing.join(ANYA, code)
    assert not (await pairing.status(ANYA)).complete


async def test_third_user_cannot_join(pairing: PairingService) -> None:
    code = await pairing.invite(ANYA)
    await pairing.join(BORIS, code)

    with pytest.raises(PairFullError):
        await pairing.join(CARL, code)
    assert (await pairing.status(CARL)).pair_id is None
    assert (await pairing.status(ANYA)).partner_id == BORIS


async def test_concurrent_joins_admit_only_one(pairing: PairingService, db: Database) -> None:
    await pairing.register(4, "Дина")
    code = await pairing.invite(ANYA)

    results = await asyncio.gather(
        pairing.join(BORIS, code), pairing.join(CARL, code), return_exceptions=True
    )

    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert sum(isinstance(r, PairFullError) for r in results) == 1


async def test_paired_user_cannot_pair_again(pairing: PairingService) -> None:
    await pairing.join(BORIS, await pairing.invite(ANYA))
    with pytest.raises(AlreadyPairedError):
        await pairing.invite(ANYA)
    with pytest.raises(AlreadyPairedError):
        await pairing.join(BORIS, await pairing.invite(CARL))


async def test_joining_drops_own_pending_invite(pairing: PairingService) -> None:
    own = await pairing.invite(BORIS)
    await pairing.join(BORIS, await pairing.invite(ANYA))

    with pytest.raises(InviteNotFoundError):
        await pairing.join(CARL, own)


async def test_unknown_code(pairing: PairingService) -> None:
    with pytest.raises(InviteNotFoundError):
        await pairing.join(BORIS, "NOPE42")


async def test_code_collisions_are_retried(db: Database) -> None:
    codes = itertools.chain(["SAME22", "SAME22", "OTHER3"])
    pairing = PairingService(db, clock=lambda: NOW, code_factory=lambda: next(codes))
    await pairing.register(ANYA, None)
    await pairing.register(BORIS, None)

    assert await pairing.invite(ANYA) == "SAME22"
    assert await pairing.invite(BORIS) == "OTHER3"
    assert await pairing.display_name(ANYA) == "Партнёр"
