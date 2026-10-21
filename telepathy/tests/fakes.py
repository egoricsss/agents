"""Подделки для тестов: детерминированные эмбеддинги и судья на FunctionModel (без сети)."""

import itertools
import json
from collections.abc import Callable, Sequence

import numpy as np
from pydantic_ai import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from telepathy.services.judge import Judge

DIM = 16


def unit(values: Sequence[float]) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Текст с ключевым словом получает вектор этого слова, остальные — свою отдельную ось."""

    def __init__(self, vectors: dict[str, Sequence[float]] | None = None) -> None:
        self.vectors = {key: unit(values) for key, values in (vectors or {}).items()}
        self.calls: list[str] = []
        self._free_axes = itertools.count(DIM - 1, -1)
        self._seen: dict[str, np.ndarray] = {}

    async def embed(self, text: str) -> np.ndarray:
        self.calls.append(text)
        for key, vector in self.vectors.items():
            if key in text.lower():
                return vector
        if text not in self._seen:
            vector = np.zeros(DIM, dtype=np.float32)
            vector[next(self._free_axes)] = 1.0
            self._seen[text] = vector
        return self._seen[text]


class ScriptedJudge:
    """Настоящий `Judge` поверх FunctionModel: решение принимает функция теста."""

    def __init__(self, decide: Callable[[str, str], bool] = lambda first, second: True) -> None:
        self.decide = decide
        self.calls: list[tuple[str, str]] = []
        self.fail = False
        self.judge = Judge(FunctionModel(self._respond))

    async def compare(self, first: str, second: str):  # noqa: ANN201 — как у Judge
        return await self.judge.compare(first, second)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if self.fail:
            raise ConnectionError("LLM недоступна")
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(p.content for p in request.parts if isinstance(p, UserPromptPart))
        payload = json.loads(str(prompt).split("\n", 1)[1])
        self.calls.append((payload["first"], payload["second"]))
        same = self.decide(payload["first"], payload["second"])
        args = {
            "is_same": same,
            "common_idea": "одна идея" if same else "",
            "message": "✨ Вы подумали об одном и том же!" if same else "",
        }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])
