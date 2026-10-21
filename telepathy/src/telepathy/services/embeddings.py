"""Локальные эмбеддинги e5 на ONNX Runtime — без torch и sentence-transformers.

Почему так: torch + sentence-transformers + e5-small в fp32 занимали ~0.8 ГБ RAM, а VPS — 2 ГБ.
Здесь — квантованная (int8) e5-small из того же репозитория Hugging Face и токенизатор
SentencePiece (≈60 МБ против ≈280 МБ у `tokenizers` на словаре XLM-R в 250 тыс. токенов).
Пулинг и нормализация — как в sentence-transformers: среднее по токенам с маской, L2.

Модель грузится один раз при старте. `encode` синхронный и нагружает CPU, поэтому выполняется
в `asyncio.to_thread`; вызовы сериализуются — на одном ядре параллельный encode только мешает.
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
MAX_TOKENS = 512
TOKENIZER_FILE = "onnx/sentencepiece.bpe.model"
# Спецтокены XLM-R и сдвиг id SentencePiece (fairseq): <s>=0, <pad>=1, </s>=2, <unk>=3.
BOS_ID, EOS_ID, UNK_ID, FAIRSEQ_OFFSET = 0, 2, 3, 1


class Encoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class Embedder(Protocol):
    async def embed(self, text: str) -> np.ndarray: ...


class SentencePieceLike(Protocol):
    def unk_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...


class SessionLike(Protocol):
    def get_inputs(self) -> list[Any]: ...

    def run(self, outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...  # noqa: ANN401


class E5OnnxEncoder:
    def __init__(self, tokenizer: SentencePieceLike, session: SessionLike) -> None:
        self.tokenizer = tokenizer
        self.session = session
        self._inputs = {item.name for item in session.get_inputs()}

    def token_ids(self, text: str) -> list[int]:
        """Те же id, что даёт XLMRobertaTokenizer: обрезка до MAX_TOKENS вместе с <s> и </s>."""
        unk = self.tokenizer.unk_id()
        pieces = [UNK_ID if i == unk else i + FAIRSEQ_OFFSET for i in self.tokenizer.encode(text)]
        return [BOS_ID, *pieces[: MAX_TOKENS - 2], EOS_ID]

    def encode(self, text: str) -> np.ndarray:
        ids = np.array([self.token_ids(text)], dtype=np.int64)
        mask = np.ones_like(ids)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]
        pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        vector = pooled[0] / np.linalg.norm(pooled[0])
        return vector.astype(np.float32)


Download = Callable[..., str]
Build = Callable[[str, str, int], Encoder]


def _hf_download(repo: str, filename: str, **kwargs: Any) -> str:  # noqa: ANN401
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, **kwargs)


def _build_onnx(tokenizer_path: str, model_path: str, threads: int) -> Encoder:
    import onnxruntime as ort
    import sentencepiece as spm

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False  # арена держит пиковую память навсегда
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    return E5OnnxEncoder(spm.SentencePieceProcessor(model_file=tokenizer_path), session)


class OnnxEmbedder:
    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder
        self._lock = asyncio.Lock()

    @classmethod
    async def load(
        cls,
        repo: str,
        *,
        onnx_file: str,
        threads: int,
        cache_folder: Path | None = None,
        download: Download = _hf_download,
        build: Build = _build_onnx,
    ) -> Self:
        started = time.perf_counter()
        encoder = await asyncio.to_thread(
            _load, repo, onnx_file, threads, cache_folder, download, build
        )
        log.info("embedding model %s loaded in %.1fs", repo, time.perf_counter() - started)
        return cls(encoder)

    async def embed(self, text: str) -> np.ndarray:
        async with self._lock:
            return await asyncio.to_thread(self._encoder.encode, QUERY_PREFIX + text.strip())


def _load(
    repo: str,
    onnx_file: str,
    threads: int,
    cache_folder: Path | None,
    download: Download,
    build: Build,
) -> Encoder:
    kwargs: dict[str, Any] = {}
    if cache_folder is not None:
        kwargs["cache_dir"] = str(cache_folder)

    def fetch(filename: str) -> str:
        try:
            # Сначала из локального кеша: перезапуск не зависит от сети и не ждёт её таймаутов.
            return download(repo, filename, local_files_only=True, **kwargs)
        except Exception:
            log.info("%s/%s is not cached, downloading", repo, filename)
            return download(repo, filename, **kwargs)

    tokenizer_path = fetch(TOKENIZER_FILE)
    model_path = fetch(onnx_file)
    return build(tokenizer_path, model_path, threads)
