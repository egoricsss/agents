"""Конвертация документов в Markdown через markitdown: форматы, пустые сканы, кеш, безопасность."""

import hashlib
from pathlib import Path

import pytest

from lab_manager.converter import ConvertedDocument, DocumentConverter
from lab_manager.errors import ConversionError, UnsupportedFormatError
from tests.fixtures import ASSIGNMENT_TEXT, make_docx, make_pdf, make_pptx, make_xlsx


@pytest.fixture
def converter(tmp_path: Path) -> DocumentConverter:
    return DocumentConverter(cache_dir=tmp_path / "cache")


async def test_plain_text_utf8_cyrillic(converter: DocumentConverter) -> None:
    doc = await converter.convert(ASSIGNMENT_TEXT.encode(), "task.txt")

    assert "Процессы и потоки" in doc.markdown
    assert "fork()" in doc.markdown
    assert doc.has_text


async def test_plain_text_cp1251_is_decoded(converter: DocumentConverter) -> None:
    doc = await converter.convert(ASSIGNMENT_TEXT.encode("cp1251"), "task.txt")

    assert "Процессы и потоки" in doc.markdown


async def test_markdown_passthrough(converter: DocumentConverter) -> None:
    doc = await converter.convert("# Лаба 1\n\nСделать **всё**.".encode(), "task.md")

    assert "# Лаба 1" in doc.markdown
    assert "**всё**" in doc.markdown


async def test_html(converter: DocumentConverter) -> None:
    html = "<html><body><h1>Лаба 2</h1><p>Написать <b>сервер</b>.</p></body></html>"

    doc = await converter.convert(html.encode(), "task.html")

    assert "# Лаба 2" in doc.markdown
    assert "**сервер**" in doc.markdown


async def test_docx(converter: DocumentConverter) -> None:
    data = make_docx("Лабораторная работа №4", ["Реализовать B-дерево.", "Оформить отчёт."])

    doc = await converter.convert(data, "task.docx")

    assert "Лабораторная работа №4" in doc.markdown
    assert "B-дерево" in doc.markdown


async def test_pdf(converter: DocumentConverter) -> None:
    data = make_pdf(["Lab 5: sockets", "Implement a TCP echo server in C.", "Write a report."])

    doc = await converter.convert(data, "task.pdf")

    assert "TCP echo server" in doc.markdown
    assert doc.has_text


async def test_pptx(converter: DocumentConverter) -> None:
    doc = await converter.convert(make_pptx("Лаба 6", "Нарисовать ER-диаграмму"), "slides.pptx")

    assert "ER-диаграмму" in doc.markdown


async def test_xlsx(converter: DocumentConverter) -> None:
    data = make_xlsx([["Вариант", "Задание"], ["7", "Сортировка слиянием"]])

    doc = await converter.convert(data, "variants.xlsx")

    assert "Сортировка слиянием" in doc.markdown


async def test_scan_without_text_layer_is_flagged(converter: DocumentConverter) -> None:
    doc = await converter.convert(make_pdf([]), "scan.pdf")

    assert not doc.has_text


async def test_extension_is_case_insensitive(converter: DocumentConverter) -> None:
    doc = await converter.convert(b"hello world, this is a lab assignment text", "TASK.TXT")

    assert "hello world" in doc.markdown


@pytest.mark.parametrize("name", ["virus.exe", "archive.zip", "noext", "image.png", "doc.doc"])
async def test_unsupported_format_rejected(converter: DocumentConverter, name: str) -> None:
    assert not converter.supports(name)

    with pytest.raises(UnsupportedFormatError):
        await converter.convert(b"data", name)


async def test_broken_document_raises_conversion_error(converter: DocumentConverter) -> None:
    with pytest.raises(ConversionError):
        await converter.convert(b"this is not a real docx", "broken.docx")


async def test_result_carries_sha256(converter: DocumentConverter) -> None:
    data = b"some text that is long enough to count as a document"

    doc = await converter.convert(data, "a.txt")

    assert doc.sha256 == hashlib.sha256(data).hexdigest()


async def test_conversion_is_cached_by_content(
    converter: DocumentConverter, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = converter._convert_sync

    def counting(data: bytes, extension: str) -> str:
        nonlocal calls
        calls += 1
        return original(data, extension)

    monkeypatch.setattr(converter, "_convert_sync", counting)
    data = ASSIGNMENT_TEXT.encode()

    first = await converter.convert(data, "a.txt")
    second = await converter.convert(data, "renamed.txt")

    assert calls == 1
    assert first == second


async def test_cache_is_invalidated_by_converter_version(tmp_path: Path) -> None:
    data = ASSIGNMENT_TEXT.encode()
    old = DocumentConverter(cache_dir=tmp_path, version="old")
    new = DocumentConverter(cache_dir=tmp_path, version="new")

    await old.convert(data, "a.txt")

    assert old.cache_path(hashlib.sha256(data).hexdigest()).exists()
    assert not new.cache_path(hashlib.sha256(data).hexdigest()).exists()


async def test_works_without_cache_dir() -> None:
    doc = await DocumentConverter().convert(ASSIGNMENT_TEXT.encode(), "a.txt")

    assert isinstance(doc, ConvertedDocument)


def test_markitdown_is_configured_without_plugins_or_remote_services() -> None:
    converter = DocumentConverter()

    assert converter.markitdown._plugins_enabled is False
    assert getattr(converter.markitdown, "_llm_client", None) is None
    assert getattr(converter.markitdown, "_docintel_endpoint", None) is None
