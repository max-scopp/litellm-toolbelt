"""Stand-ins for the objects a LiteLLM hook is handed.

The wire shapes are duck-typed everywhere in the package, so the tests use
plain attribute-style objects rather than importing LiteLLM's models: a test
that has to build a `ModelResponse` to prove a hook reads `.choices` is testing
pydantic.
"""

from __future__ import annotations

import json
from typing import Any


class FakeFunction:
    def __init__(self, name: str, arguments: dict[str, Any] | str) -> None:
        self.name = name
        self.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments


class FakeToolCall:
    def __init__(self, id: str, name: str, arguments: dict[str, Any] | str) -> None:
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[FakeToolCall] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class FakeResponse:
    def __init__(
        self,
        message: FakeMessage,
        usage: FakeUsage | None = None,
        hidden_params: dict[str, Any] | None = None,
    ) -> None:
        self.choices = [FakeChoice(message)]
        self.usage = usage
        self._hidden_params = hidden_params if hidden_params is not None else {}


class FakeDeltaFunction:
    def __init__(self, name: str | None = None, arguments: str | None = None) -> None:
        self.name = name
        self.arguments = arguments


class FakeDeltaToolCall:
    def __init__(
        self,
        index: int,
        id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        self.index = index
        self.id = id
        self.function = FakeDeltaFunction(name, arguments)


class FakeDelta:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[FakeDeltaToolCall] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class FakeStreamChoice:
    def __init__(self, delta: FakeDelta) -> None:
        self.delta = delta


class FakeChunk:
    def __init__(self, delta: FakeDelta) -> None:
        self.choices = [FakeStreamChoice(delta)]


async def as_stream(chunks: list[Any]) -> Any:
    for chunk in chunks:
        yield chunk


async def failing_stream(exc: Exception, before: list[Any] | None = None) -> Any:
    for chunk in before or []:
        yield chunk
    raise exc


async def collect(iterator: Any) -> list[Any]:
    return [chunk async for chunk in iterator]


def stream_content(chunks: list[Any]) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices[0].delta.content)


def tool_def(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"the {name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class FakePack:
    """A minimal `ToolPack`: two tools, a prompt, and a record of what ran."""

    name = "fake"

    def __init__(self, prompt: str = "Use the fake tools.") -> None:
        self.prompt = prompt
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []
        self.result = json.dumps({"ok": True})
        self.on_execute: Any = None

    def tools(self) -> list[dict[str, Any]]:
        return [tool_def("fake_search"), tool_def("fake_list")]

    def owns(self, tool_name: str) -> bool:
        return tool_name in {"fake_search", "fake_list"}

    async def system_prompt(self) -> str:
        return self.prompt

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> str:
        self.calls.append((tool_name, arguments, actor))
        if self.on_execute is not None:
            return await self.on_execute(tool_name, arguments)
        return self.result
