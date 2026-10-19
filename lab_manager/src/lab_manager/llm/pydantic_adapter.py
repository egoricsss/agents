"""GigaChat как модель для PydanticAI через `FunctionModel`.

GigaChat использует протокол `functions`/`function_call` (а не OpenAI `tools`), поэтому
стандартный OpenAI-провайдер не подходит. `FunctionModel` — официальный способ подключить
свою модель одной функцией, без наследования от внутренних классов PydanticAI.
"""

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from lab_manager.llm.gigachat import ChatResult, GigaChatClient

OnCall = Callable[[ChatResult | None, int, str | None], Awaitable[None]]

_DROP_SCHEMA_KEYS = {"title", "additionalProperties", "$schema"}


def gigachat_model(
    client: GigaChatClient, *, temperature: float = 0.2, on_call: OnCall | None = None
) -> FunctionModel:
    async def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        chat_messages = to_gigachat_messages(messages, info.instructions)
        functions = [to_function(tool) for tool in [*info.function_tools, *info.output_tools]]
        force = (
            None
            if info.allow_text_output or len(info.output_tools) != 1
            else info.output_tools[0].name
        )
        started = time.monotonic()
        try:
            result = await client.chat(
                chat_messages,
                functions=functions or None,
                function_call=force,
                temperature=temperature,
            )
        except Exception as exc:
            if on_call:
                await on_call(None, _elapsed_ms(started), str(exc))
            raise
        if on_call:
            await on_call(result, _elapsed_ms(started), None)
        return to_model_response(result)

    return FunctionModel(call, model_name=f"gigachat:{client.model}")


def to_function(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description or tool.name,
        "parameters": _clean_schema(tool.parameters_json_schema),
    }


def to_gigachat_messages(
    messages: list[ModelMessage], instructions: str | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if instructions:
        out.append({"role": "system", "content": instructions})
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                out.extend(_request_part(part))
        elif isinstance(message, ModelResponse):
            out.extend(_response_parts(message))
    return out


def to_model_response(result: ChatResult) -> ModelResponse:
    parts: list[Any] = []
    if result.function_call is not None:
        parts.append(
            ToolCallPart(
                tool_name=result.function_call.name,
                args=result.function_call.arguments,
                tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
            )
        )
    elif result.content:
        parts.append(TextPart(content=result.content))
    else:
        parts.append(TextPart(content=""))
    return ModelResponse(
        parts=parts,
        usage=RequestUsage(
            input_tokens=result.usage.prompt_tokens, output_tokens=result.usage.completion_tokens
        ),
        model_name=result.model,
    )


def _request_part(part: Any) -> list[dict[str, Any]]:  # noqa: ANN401 — объединение типов частей
    if isinstance(part, SystemPromptPart):
        return [{"role": "system", "content": part.content}]
    if isinstance(part, UserPromptPart):
        content = part.content
        if not isinstance(content, str):
            content = "\n".join(item for item in content if isinstance(item, str))
        return [{"role": "user", "content": content}]
    if isinstance(part, ToolReturnPart):
        return [
            {
                "role": "function",
                "name": part.tool_name,
                "content": json.dumps(_jsonable(part.content), ensure_ascii=False),
            }
        ]
    if isinstance(part, RetryPromptPart):
        text = (
            part.content
            if isinstance(part.content, str)
            else json.dumps(_jsonable(part.content), ensure_ascii=False)
        )
        if part.tool_name:
            return [
                {
                    "role": "function",
                    "name": part.tool_name,
                    "content": json.dumps({"error": text}, ensure_ascii=False),
                }
            ]
        return [{"role": "user", "content": text}]
    return []


def _response_parts(message: ModelResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    texts = [part.content for part in message.parts if isinstance(part, TextPart) and part.content]
    calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
    if texts and not calls:
        out.append({"role": "assistant", "content": "\n".join(texts)})
    for call in calls:  # GigaChat возвращает не больше одного вызова за ответ
        out.append(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": call.tool_name, "arguments": call.args_as_dict()},
            }
        )
    return out


def _clean_schema(schema: Any) -> Any:  # noqa: ANN401 — JSON Schema
    """Приводит схему к подмножеству, которое принимает GigaChat.

    GigaChat не понимает необязательные типы (`anyOf: [{type: X}, {type: null}]`,
    `type: [X, "null"]`) — оставляем ненулевой вариант; служебные ключи убираем.
    """
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    variants = schema.get("anyOf") or schema.get("oneOf")
    if variants:
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(non_null) == 1:
            merged = {k: v for k, v in schema.items() if k not in {"anyOf", "oneOf"}}
            return _clean_schema({**non_null[0], **merged})
    cleaned = {k: _clean_schema(v) for k, v in schema.items() if k not in _DROP_SCHEMA_KEYS}
    if isinstance(cleaned.get("type"), list):
        types = [t for t in cleaned["type"] if t != "null"]
        cleaned["type"] = types[0] if types else "string"
    if cleaned.get("default", 0) is None:
        cleaned.pop("default")
    return cleaned


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
