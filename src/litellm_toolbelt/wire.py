"""Reading and writing the OpenAI chat wire shapes, whatever object they arrive in.

A proxy hook sees the same message three ways: as a plain dict the client sent,
as a pydantic model LiteLLM built, and — while streaming — as a sequence of
fragments that only mean something once assembled. Every helper here therefore
accepts both dicts and attribute-style objects, and tool calls get one
normalized form, `{"id", "name", "arguments"}`, that the rest of the package
passes around.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "accumulate_tool_calls",
    "as_tool_call_message",
    "assistant_message_from_response",
    "chunk_delta",
    "delta_content",
    "delta_tool_calls",
    "field",
    "followup_kwargs",
    "normalize_tool_calls",
    "notice_chunk",
    "set_field",
    "tool_calls_of",
]


def field(obj: Any, name: str) -> Any:
    """Read a field from either a dict or an attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def set_field(obj: Any, name: str, value: Any) -> None:
    """Write a field to either a dict or an attribute-style object."""
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def chunk_delta(chunk: Any) -> Any:
    choices = field(chunk, "choices") or []
    if not choices:
        return None
    return field(choices[0], "delta")


def delta_tool_calls(delta: Any) -> list[Any]:
    return list(field(delta, "tool_calls") or [])


def delta_content(delta: Any) -> str:
    return field(delta, "content") or ""


def tool_calls_of(message: dict[str, Any] | None) -> list[dict[str, str]]:
    """Normalize the `tool_calls` of an assistant message dict."""
    if not message:
        return []
    out: list[dict[str, str]] = []
    for tc in message.get("tool_calls") or []:
        fn = field(tc, "function") or {}
        out.append(
            {
                "id": field(tc, "id") or "",
                "name": field(fn, "name") or "",
                "arguments": field(fn, "arguments") or "{}",
            }
        )
    return out


def as_tool_call_message(call: dict[str, str]) -> dict[str, Any]:
    """Render a normalized call as an OpenAI assistant `tool_calls` entry."""
    return {
        "id": call["id"],
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": call["arguments"] or "{}",
        },
    }


def accumulate_tool_calls(
    acc: dict[int, dict[str, str]],
    delta_calls: list[Any],
) -> None:
    """Fold streamed tool-call fragments into complete calls, keyed by index."""
    for tc in delta_calls:
        idx = field(tc, "index")
        if idx is None:
            idx = len(acc)
        slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        tc_id = field(tc, "id")
        if tc_id:
            slot["id"] = tc_id
        fn = field(tc, "function") or {}
        name = field(fn, "name")
        if name:
            slot["name"] += name
        args = field(fn, "arguments")
        if args:
            slot["arguments"] += args


def normalize_tool_calls(
    acc: dict[int, dict[str, str]],
    *,
    id_prefix: str = "call",
) -> list[dict[str, str]]:
    """Finish accumulated fragments into calls with an id and arguments.

    A provider that streams tool calls without ids still has to be answered
    with a `tool_call_id`, so a synthetic one is minted from `id_prefix`.
    """
    calls: list[dict[str, str]] = []
    for i, idx in enumerate(sorted(acc)):
        call = dict(acc[idx])
        if not call["id"]:
            call["id"] = f"{id_prefix}_{i}"
        if not call["arguments"]:
            call["arguments"] = "{}"
        calls.append(call)
    return calls


def assistant_message_from_response(response: Any) -> dict[str, Any] | None:
    """Build an `assistant` message dict from a chat completion response.

    Includes `tool_calls` (if any) so subsequent iterations preserve the
    model's intent.
    """
    choices = field(response, "choices") or []
    if not choices:
        return None
    message = field(choices[0], "message")
    if message is None:
        return None
    msg: dict[str, Any] = {"role": "assistant"}
    content = field(message, "content")
    if content:
        msg["content"] = content
    tool_calls = field(message, "tool_calls") or []
    if tool_calls:
        normalized = tool_calls_of({"tool_calls": tool_calls})
        msg["tool_calls"] = [as_tool_call_message(c) for c in normalized]
    return msg


# Request keys a follow-up must inherit so it lands on the same model, with the
# same limits, as the call the client made. `stream` is deliberately not one of
# them: the loop decides per round whether it wants a stream.
FOLLOWUP_KEYS = ("temperature", "top_p", "max_tokens", "stop", "user", "api_base", "api_key")


def followup_kwargs(
    data: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the completion kwargs for one follow-up round of a tool loop."""
    kwargs: dict[str, Any] = {
        "model": data.get("model"),
        "messages": messages,
    }
    tools = data.get("tools")
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = data.get("tool_choice", "auto")
    for key in FOLLOWUP_KEYS:
        if data.get(key) is not None:
            kwargs[key] = data[key]
    return kwargs


def notice_chunk(template: Any, text: str) -> Any:
    """A final content chunk shaped like the stream's own chunks, or None.

    Used to end a broken stream with something the user can read. It is built
    by copying a chunk the client already accepted, so it needs no knowledge of
    which flavour of chunk object this provider streams.
    """
    if template is None:
        return None
    chunk = copy.deepcopy(template)
    delta = chunk_delta(chunk)
    if delta is None:
        return None
    for name, value in (("content", text), ("tool_calls", None)):
        set_field(delta, name, value)
    return chunk
