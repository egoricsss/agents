"""Локальные эмбеддинги (sentence-transformers). Модель грузится один раз при старте.

`encode` — синхронный и тяжёлый для CPU, поэтому выполняется в `asyncio.to_thread`;
вызовы сериализуются: на одном ядре параллельный encode только мешает сам себе.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np

log = logging.getLogger(__name__)

# Модели e5 обучены с префиксами; для симметричного сравнения двух мыслей — «query: » у обеих.
QUERY_PREFIX = "query: "


class Encoder(Protocol):
    def encode(self, sentences: str, **kwargs: Any) -> np.ndarray: ...  # noqa: ANN401


class Embedder(Protocol):
    async def embed(self, text: str) -> np.ndarray: ...


EncoderFactory = Callable[..., Encoder]


def _sentence_transformer(name: str, **kwargs: Any) -> Encoder:  # noqa: ANN401
    # Импорт здесь: torch тяжёлый, тестам и healthcheck он не нужен.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, **kwargs)


class SentenceEmbedder:
    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder
        self._lock = asyncio.Lock()

    @classmethod
    async def load(
        cls,
        name: str,
        *,
        threads: int,
        cache_folder: Path | None = None,
        factory: EncoderFactory = _sentence_transformer,
    ) -> Self:
        started = time.perf_counter()
        encoder = await asyncio.to_thread(
            _build, factory, name, threads=threads, cache_folder=cache_folder
        )
        log.info("embedding model %s loaded in %.1fs", name, time.perf_counter() - started)
        return cls(encoder)

    async def embed(self, text: str) -> np.ndarray:
        async with self._lock:
            return await asyncio.to_thread(self._encode, text)

    def _encode(self, text: str) -> np.ndarray:
        vector = self._encoder.encode(
            QUERY_PREFIX + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)


def _build(
    factory: EncoderFactory, name: str, *, threads: int, cache_folder: Path | None
) -> Encoder:
    if factory is _sentence_transformer:
        import torch

        torch.set_num_threads(threads)
    kwargs: dict[str, Any] = {"device": "cpu"}
    if cache_folder is not None:
        kwargs["cache_folder"] = str(cache_folder)
    try:
        # Сначала из локального кеша: перезапуск не зависит от сети и не ждёт её таймаутов.
        return factory(name, local_files_only=True, **kwargs)
    except OSError:
        log.info("embedding model %s is not cached, downloading", name)
        return factory(name, **kwargs)
