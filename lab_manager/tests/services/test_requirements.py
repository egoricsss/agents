"""Сервис требований: версии и пересборка промптов только затронутых лаб."""

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lab_manager.converter import DocumentConverter
from lab_manager.db import Database
from lab_manager.domain import Lab, RequirementScope
from lab_manager.errors import NotFoundError
from lab_manager.extraction.rules import RuleBasedExtractor
from lab_manager.services.executor_prompt import ExecutorPromptBuilder
from lab_manager.services.ingest import IngestService
from lab_manager.services.requirements import RequirementsService
from lab_manager.storage import LabStorage

MSK = ZoneInfo("Europe/Moscow")


@pytest.fixture
def storage(tmp_path: Path) -> LabStorage:
    return LabStorage(tmp_path / "Лабы")


@pytest.fixture
def service(db: Database, storage: LabStorage) -> RequirementsService:
    builder = ExecutorPromptBuilder(db=db, storage=storage, disk_root="app:/Лабы", tz=MSK)
    return RequirementsService(db=db, builder=builder)


@pytest.fixture
async def labs(db: Database, storage: LabStorage) -> list[Lab]:
    ingest = IngestService(
        db=db,
        storage=storage,
        converter=DocumentConverter(),
        extractor=RuleBasedExtractor(),
        tz=MSK,
    )
    result = []
    for prompt, text in [
        ("ОС, лаба 1", "Задание по ОС номер один, достаточно длинное"),
        ("ОС, лаба 2", "Задание по ОС номер два, достаточно длинное"),
        ("Сети, лаба 1", "Задание по сетям номер один, достаточно длинное"),
    ]:
        ingested = await ingest.ingest(data=text.encode(), filename="t.txt", prompt=prompt)
        assert ingested.lab is not None
        result.append(ingested.lab)
    return result


def prompt_of(storage: LabStorage, lab: Lab) -> str:
    text = storage.read_text(f"{lab.folder}/prompt.md")
    assert text is not None
    return text


async def test_global_requirements_rebuild_all_labs(
    service: RequirementsService, labs: list[Lab], storage: LabStorage
) -> None:
    update = await service.set(RequirementScope.GLOBAL, "  Отчёт по ГОСТ 7.32  ")

    assert update.requirement.text == "Отчёт по ГОСТ 7.32"
    assert sorted(update.rebuilt_lab_ids) == sorted(lab.id for lab in labs)
    assert all("Отчёт по ГОСТ 7.32" in prompt_of(storage, lab) for lab in labs)


async def test_subject_requirements_rebuild_only_its_labs(
    service: RequirementsService, labs: list[Lab], storage: LabStorage
) -> None:
    os1, os2, net = labs

    update = await service.set(RequirementScope.SUBJECT, "Только Linux", os1.subject_id)

    assert sorted(update.rebuilt_lab_ids) == sorted([os1.id, os2.id])
    assert "Только Linux" in prompt_of(storage, os2)
    assert storage.read_text(f"{net.folder}/prompt.md") is None


async def test_lab_requirements_and_versions(
    service: RequirementsService, labs: list[Lab], storage: LabStorage
) -> None:
    lab = labs[0]

    first = await service.set(RequirementScope.LAB, "Вариант 7", lab.id)
    second = await service.set(RequirementScope.LAB, "Вариант 8", lab.id)

    assert (first.requirement.version, second.requirement.version) == (1, 2)
    assert second.rebuilt_lab_ids == [lab.id]
    prompt = prompt_of(storage, lab)
    assert "Вариант 8" in prompt and "Вариант 7" not in prompt
    current = await service.current(RequirementScope.LAB, lab.id)
    assert current is not None and current.text == "Вариант 8"


async def test_empty_text_clears(
    service: RequirementsService, labs: list[Lab], storage: LabStorage
) -> None:
    lab = labs[0]
    await service.set(RequirementScope.LAB, "Вариант 7", lab.id)

    await service.set(RequirementScope.LAB, "", lab.id)

    assert "### К этой лабе\n\n—" in prompt_of(storage, lab)


async def test_unknown_lab(service: RequirementsService) -> None:
    with pytest.raises(NotFoundError):
        await service.set(RequirementScope.LAB, "x", 999)
