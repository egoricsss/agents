"""Эмбеддинги: модель грузится один раз, encode — вне цикла событий, префикс e5, float32."""

import asyncio
import threading
import time
from typing import Any

import numpy as np
import pytest

from telepathy.services.embeddings import QUERY_PREFIX, SentenceEmbedder


class FakeEncoder:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.threads: list[int] = []
        self.active = 0
        self.max_active = 0

    def encode(self, sentences: str, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append((sentences, kwargs))
        self.threads.append(threading.get_ident())
        time.sleep(self.delay)
        self.active -= 1
        return np.array([0.6, 0.8], dtype=np.float64)


async def test_encode_runs_off_event_loop_with_e5_prefix() -> None:
    encoder = FakeEncoder()
    embedder = SentenceEmbedder(encoder)

    vector = await embedder.embed("хочу пиццу")

    [(sentence, kwargs)] = encoder.calls
    assert sentence == f"{QUERY_PREFIX}хочу пиццу"
    assert kwargs["normalize_embeddings"] is True
    assert encoder.threads[0] != threading.get_ident()
    assert vector.dtype == np.float32
    assert vector.tolist() == pytest.approx([0.6, 0.8])


async def test_event_loop_stays_responsive_while_encoding() -> None:
    embedder = SentenceEmbedder(FakeEncoder(delay=0.2))
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await embedder.embed("мысль")
    task.cancel()

    assert ticks >= 5


async def test_concurrent_calls_are_serialized() -> None:
    encoder = FakeEncoder(delay=0.05)
    embedder = SentenceEmbedder(encoder)

    await asyncio.gather(*(embedder.embed(str(i)) for i in range(4)))

    assert len(encoder.calls) == 4
    assert encoder.max_active == 1


async def test_load_builds_model_once_in_thread() -> None:
    created: list[tuple[str, dict[str, Any], int]] = []

    def factory(name: str, **kwargs: Any) -> FakeEncoder:  # noqa: ANN401
        created.append((name, kwargs, threading.get_ident()))
        return FakeEncoder()

    embedder = await SentenceEmbedder.load(
        "intfloat/multilingual-e5-small", threads=1, factory=factory
    )
    await embedder.embed("a")
    await embedder.embed("b")

    [(name, kwargs, thread)] = created
    assert name == "intfloat/multilingual-e5-small"
    assert kwargs["device"] == "cpu"
    assert kwargs["local_files_only"] is True  # перезапуск не ходит в сеть
    assert thread != threading.get_ident()


async def test_load_downloads_only_when_not_cached() -> None:
    attempts: list[bool] = []

    def factory(name: str, **kwargs: Any) -> FakeEncoder:  # noqa: ANN401
        attempts.append(kwargs.get("local_files_only", False))
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return FakeEncoder()

    await SentenceEmbedder.load("intfloat/multilingual-e5-small", threads=1, factory=factory)

    assert attempts == [True, False]


@pytest.mark.live
async def test_real_model_ranks_paraphrase_above_unrelated() -> None:
    embedder = await SentenceEmbedder.load("intfloat/multilingual-e5-small", threads=1)

    pizza = await embedder.embed("хочу пиццу")
    paraphrase = await embedder.embed("вот бы сейчас съесть пиццу")
    unrelated = await embedder.embed("надо сдать отчёт по лабораторной")

    assert pizza.shape == (384,)
    assert float(pizza @ paraphrase) > float(pizza @ unrelated)
