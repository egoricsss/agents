"""Эмбеддинги: квантованная e5 в ONNX Runtime без torch. Модель грузится один раз,
encode — вне цикла событий, префикс e5, токены как у XLM-R, mean pooling, float32."""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from telepathy.services.embeddings import (
    MAX_TOKENS,
    QUERY_PREFIX,
    E5OnnxEncoder,
    OnnxEmbedder,
)


class FakeEncoder:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.texts: list[str] = []
        self.threads: list[int] = []
        self.active = 0
        self.max_active = 0

    def encode(self, text: str) -> np.ndarray:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.texts.append(text)
        self.threads.append(threading.get_ident())
        time.sleep(self.delay)
        self.active -= 1
        return np.array([0.6, 0.8], dtype=np.float32)


async def test_encode_runs_off_event_loop_with_e5_prefix() -> None:
    encoder = FakeEncoder()

    vector = await OnnxEmbedder(encoder).embed("  хочу пиццу ")

    assert encoder.texts == [f"{QUERY_PREFIX}хочу пиццу"]
    assert encoder.threads[0] != threading.get_ident()
    assert vector.dtype == np.float32


async def test_event_loop_stays_responsive_while_encoding() -> None:
    embedder = OnnxEmbedder(FakeEncoder(delay=0.2))
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
    embedder = OnnxEmbedder(encoder)

    await asyncio.gather(*(embedder.embed(str(i)) for i in range(4)))

    assert len(encoder.texts) == 4
    assert encoder.max_active == 1


# --- E5OnnxEncoder: токенизация и пулинг на подделках SentencePiece и ONNX Runtime ---


class FakeSentencePiece:
    """id SentencePiece: unk = 0, обычные куски — с 1."""

    def __init__(self, pieces: dict[str, int]) -> None:
        self.pieces = pieces

    def unk_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [self.pieces.get(word, 0) for word in text.split()]


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    """Скрытые состояния: у токена с id i вектор [i, 1] — так видно, что усреднилось."""

    def __init__(self) -> None:
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput(n) for n in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:  # noqa: ANN401
        self.feeds.append(feed)
        ids = feed["input_ids"].astype(np.float32)
        return [np.stack([ids, np.ones_like(ids)], axis=-1)]


def test_xlmr_ids_special_tokens_and_unknown() -> None:
    encoder = E5OnnxEncoder(FakeSentencePiece({"a": 5, "b": 7}), FakeSession())

    # <s>=0, </s>=2, unk SentencePiece → 3, остальные id SentencePiece + 1 (как у XLM-R)
    assert encoder.token_ids("a ? b") == [0, 6, 3, 8, 2]


def test_long_text_is_truncated_to_model_limit() -> None:
    encoder = E5OnnxEncoder(FakeSentencePiece({"a": 5}), FakeSession())

    ids = encoder.token_ids(" ".join(["a"] * 1000))

    assert len(ids) == MAX_TOKENS
    assert (ids[0], ids[-1]) == (0, 2)


def test_mean_pooling_and_normalization() -> None:
    session = FakeSession()
    encoder = E5OnnxEncoder(FakeSentencePiece({"a": 5}), session)

    vector = encoder.encode("a")  # токены [0, 6, 2] → среднее [8/3, 1]

    expected = np.array([8 / 3, 1.0]) / np.linalg.norm([8 / 3, 1.0])
    assert vector.dtype == np.float32
    assert vector.tolist() == pytest.approx(expected.tolist())
    [feed] = session.feeds
    assert feed["input_ids"].dtype == np.int64
    assert feed["attention_mask"].tolist() == [[1, 1, 1]]
    assert feed["token_type_ids"].tolist() == [[0, 0, 0]]


# --- загрузка: один раз, в потоке, сначала из локального кеша ---


async def test_load_once_in_thread_offline_first(tmp_path: Path) -> None:
    calls: list[tuple[str, bool, int]] = []

    def download(repo: str, filename: str, **kwargs: Any) -> str:  # noqa: ANN401
        calls.append((filename, kwargs.get("local_files_only", False), threading.get_ident()))
        return str(tmp_path / filename)

    built: list[tuple[str, str]] = []

    def build(tokenizer_path: str, model_path: str, threads: int) -> FakeEncoder:
        built.append((tokenizer_path, model_path))
        return FakeEncoder()

    embedder = await OnnxEmbedder.load(
        "intfloat/multilingual-e5-small",
        onnx_file="onnx/model_qint8_avx512_vnni.onnx",
        threads=1,
        download=download,
        build=build,
    )
    await embedder.embed("a")

    assert [(name, offline) for name, offline, _ in calls] == [
        ("onnx/sentencepiece.bpe.model", True),
        ("onnx/model_qint8_avx512_vnni.onnx", True),
    ]
    assert all(thread != threading.get_ident() for _, _, thread in calls)
    assert len(built) == 1


async def test_load_downloads_only_when_not_cached(tmp_path: Path) -> None:
    attempts: list[bool] = []

    def download(repo: str, filename: str, **kwargs: Any) -> str:  # noqa: ANN401
        offline = kwargs.get("local_files_only", False)
        attempts.append(offline)
        if offline:
            raise FileNotFoundError(filename)
        return str(tmp_path / filename)

    await OnnxEmbedder.load(
        "repo",
        onnx_file="model.onnx",
        threads=1,
        download=download,
        build=lambda tokenizer, model, threads: FakeEncoder(),
    )

    assert attempts == [True, False, True, False]


@pytest.mark.live
async def test_real_model_ranks_paraphrase_above_unrelated() -> None:
    embedder = await OnnxEmbedder.load(
        "intfloat/multilingual-e5-small", onnx_file="onnx/model_qint8_avx512_vnni.onnx", threads=1
    )

    pizza = await embedder.embed("хочу пиццу")
    paraphrase = await embedder.embed("вот бы сейчас съесть пиццу")
    unrelated = await embedder.embed("надо сдать отчёт по лабораторной")

    assert pizza.shape == (384,)
    assert float(pizza @ paraphrase) > float(pizza @ unrelated)
