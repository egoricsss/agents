"""Требования на трёх уровнях (глобальные / предмет / лаба) с версиями."""

import pytest

from lab_manager.db import Database
from lab_manager.domain import Lab, RequirementScope, Source, Subject
from lab_manager.errors import NotFoundError


async def setup(db: Database) -> tuple[Subject, Lab]:
    subject = await db.subjects.get_or_create("ОС")
    lab = await db.labs.create(subject_id=subject.id, number=1, title="Лаба", source=Source.SYSTEM)
    return subject, lab


async def test_global_requirements_are_versioned(db: Database) -> None:
    v1 = await db.requirements.add(RequirementScope.GLOBAL, "Оформление по ГОСТ 7.32")
    v2 = await db.requirements.add(RequirementScope.GLOBAL, "ГОСТ 7.32, язык — русский")

    assert (v1.version, v2.version) == (1, 2)
    current = await db.requirements.current(RequirementScope.GLOBAL)
    assert current == v2
    assert await db.requirements.history(RequirementScope.GLOBAL) == [v1, v2]


async def test_scopes_and_targets_are_independent(db: Database) -> None:
    subject, lab = await setup(db)
    other_subject = await db.subjects.get_or_create("Сети")

    await db.requirements.add(RequirementScope.GLOBAL, "global")
    subj_req = await db.requirements.add(RequirementScope.SUBJECT, "subject", target_id=subject.id)
    lab_req = await db.requirements.add(RequirementScope.LAB, "lab", target_id=lab.id)

    assert subj_req.version == 1
    assert subj_req.subject_id == subject.id
    assert lab_req.version == 1
    assert lab_req.lab_id == lab.id
    assert await db.requirements.current(RequirementScope.SUBJECT, subject.id) == subj_req
    assert await db.requirements.current(RequirementScope.LAB, lab.id) == lab_req
    assert await db.requirements.current(RequirementScope.SUBJECT, other_subject.id) is None


async def test_current_is_none_when_empty(db: Database) -> None:
    assert await db.requirements.current(RequirementScope.GLOBAL) is None


async def test_empty_text_clears_requirements(db: Database) -> None:
    await db.requirements.add(RequirementScope.GLOBAL, "что-то")
    cleared = await db.requirements.add(RequirementScope.GLOBAL, "")

    assert (await db.requirements.current(RequirementScope.GLOBAL)) == cleared
    assert cleared.text == ""


@pytest.mark.parametrize(
    ("scope", "target_id"),
    [
        (RequirementScope.GLOBAL, 1),
        (RequirementScope.SUBJECT, None),
        (RequirementScope.LAB, None),
    ],
)
async def test_scope_target_mismatch_rejected(
    db: Database, scope: RequirementScope, target_id: int | None
) -> None:
    await setup(db)

    with pytest.raises(ValueError):
        await db.requirements.add(scope, "x", target_id=target_id)


async def test_unknown_target_rejected(db: Database) -> None:
    with pytest.raises(NotFoundError):
        await db.requirements.add(RequirementScope.LAB, "x", target_id=999)
