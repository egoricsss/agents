"""Конвертация документов в Markdown через markitdown.

Только локальные данные: без URL, плагинов, LLM-описаний картинок и облачных сервисов
(сетевая политика + защита от SSRF). Результат кешируется по sha256 и версии конвертера.
"""

import asyncio
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path, PurePath

import markitdown
from markitdown import MarkItDown, StreamInfo

from lab_manager.errors import ConversionError, UnsupportedFormatError

SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".txt", ".md", ".markdown"}
)
_PLAIN_TEXT = frozenset({".txt", ".md", ".markdown"})
_TEXT_ENCODINGS = ("utf-8-sig", "cp1251")
# Сигнатуры бинарных форматов: без них markitdown молча откатывается к plain text.
_MAGIC = {
    ".pdf": b"%PDF-",
    ".docx": b"PK\x03\x04",
    ".pptx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
}

DEFAULT_VERSION = f"markitdown-{markitdown.__version__}/v1"


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    sha256: str
    markdown: str
    has_text: bool


def _extension(filename: str) -> str:
    return PurePath(filename).suffix.lower()


def _decode_text(data: bytes) -> str:
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class DocumentConverter:
    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        version: str = DEFAULT_VERSION,
        min_text_chars: int = 20,
    ) -> None:
        self.cache_dir = cache_dir
        self.version = version
        self.min_text_chars = min_text_chars
        self.markitdown = MarkItDown(enable_plugins=False)
        self._version_tag = hashlib.sha256(version.encode()).hexdigest()[:12]

    def supports(self, filename: str) -> bool:
        return _extension(filename) in SUPPORTED_EXTENSIONS

    def cache_path(self, sha256: str) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("кеш не настроен")
        return self.cache_dir / f"{sha256}.{self._version_tag}.md"

    async def convert(self, data: bytes, filename: str) -> ConvertedDocument:
        extension = _extension(filename)
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"формат {extension or 'без расширения'} не поддерживается"
            )
        sha256 = hashlib.sha256(data).hexdigest()

        markdown = self._read_cache(sha256)
        if markdown is None:
            markdown = await asyncio.to_thread(self._convert_sync, data, extension)
            self._write_cache(sha256, markdown)

        visible = sum(1 for char in markdown if not char.isspace())
        return ConvertedDocument(sha256, markdown, visible >= self.min_text_chars)

    def _convert_sync(self, data: bytes, extension: str) -> str:
        magic = _MAGIC.get(extension)
        if magic is not None and not data.startswith(magic):
            raise ConversionError(f"содержимое файла не похоже на {extension}")
        if extension in _PLAIN_TEXT:
            # Сами определяем кодировку (Windows-1251 частая у методичек), отдаём в UTF-8.
            data = _decode_text(data).encode("utf-8")
        info = StreamInfo(
            extension=extension, charset="utf-8" if extension in _PLAIN_TEXT else None
        )
        try:
            result = self.markitdown.convert_stream(io.BytesIO(data), stream_info=info)
        except Exception as exc:  # markitdown бросает разные типы на битых файлах
            raise ConversionError(f"не удалось прочитать документ {extension}") from exc
        return result.markdown.strip() + "\n"

    def _read_cache(self, sha256: str) -> str | None:
        if self.cache_dir is None:
            return None
        path = self.cache_path(sha256)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def _write_cache(self, sha256: str, markdown: str) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(sha256)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(markdown, encoding="utf-8")
        tmp.replace(path)
