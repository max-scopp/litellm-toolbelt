"""Tests for the tool-pack hook.

A fake pack and a fake `litellm.acompletion` exercise the loop in isolation:
what a pack does with a tool call is the pack's business, this is about whether
the proxy asks it at the right moments and never leaks a tool call to a client.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fakes import (
    FakeChunk,
    FakeDelta,
    FakeDeltaToolCall,
    FakeMessage,
    FakePack,
    FakeResponse,
    FakeToolCall,
    as_stream,
    collect,
    failing_stream,
    stream_content,
)

from litellm_toolbelt import ToolPack, ToolPackHook


@pytest.fixture
def pack() -> FakePack:
    return FakePack()


@pytest.fixture
def hook(pack: FakePack) -> ToolPackHook:
    return ToolPackHook(pack, max_iterations=3)


def _data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "openrouter/foo",
    }
    data.update(overrides)
    return data


def test_a_pack_satisfies_the_protocol(pack: FakePack) -> None:
    assert isinstance(pack, ToolPack)


# ----------------------------------------------------------------------------
# Pre-call hook
# ----------------------------------------------------------------------------


async def test_pre_call_injects_tools_and_system_marker(hook: ToolPackHook) -> None:
    out = await hook.async_pre_call_hook(None, None, _data(), "acompletion")

    assert out is not None
    names = [t["function"]["name"] for t in out["tools"]]
    assert names == ["fake_search", "fake_list"]
    system = [m for m in out["messages"] if m["role"] == "system"]
    assert system and hook.marker in system[0]["content"]
    assert "Use the fake tools." in system[0]["content"]


async def test_pre_call_merges_with_client_tools(hook: ToolPackHook) -> None:
    client_tool = {"type": "function", "function": {"name": "web_search"}}
    out = await hook.async_pre_call_hook(None, None, _data(tools=[client_tool]), "acompletion")

    assert out is not None
    names = [t["function"]["name"] for t in out["tools"]]
    assert names == ["web_search", "fake_search", "fake_list"]


async def test_pre_call_does_not_shadow_a_client_tool_of_the_same_name(
    hook: ToolPackHook,
) -> None:
    client_tool = {"type": "function", "function": {"name": "fake_search"}}
    out = await hook.async_pre_call_hook(None, None, _data(tools=[client_tool]), "acompletion")

    assert out is not None
    assert [t["function"]["name"] for t in out["tools"]] == ["fake_search", "fake_list"]
    assert out["tools"][0] is client_tool


async def test_pre_call_appends_to_an_existing_system_message(hook: ToolPackHook) -> None:
    data = _data(messages=[{"role": "system", "content": "You are terse."}])
    out = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert out is not None
    assert out["messages"][0]["content"].startswith("You are terse.")
    assert hook.marker in out["messages"][0]["content"]


async def test_pre_call_does_not_double_the_marker(hook: ToolPackHook) -> None:
    data = _data()
    first = await hook.async_pre_call_hook(None, None, data, "acompletion")
    assert first is not None
    second = await hook.async_pre_call_hook(None, None, first, "acompletion")

    assert second is not None
    assert "".join(m.get("content") or "" for m in second["messages"]).count(hook.marker) == 1


async def test_pre_call_skips_a_pack_without_a_prompt(pack: FakePack) -> None:
    pack.prompt = ""
    hook = ToolPackHook(pack)
    out = await hook.async_pre_call_hook(None, None, _data(), "acompletion")

    assert out is not None
    assert [m["role"] for m in out["messages"]] == ["user"]
    assert out["tools"]


@pytest.mark.parametrize("fmt", [{"type": "json_schema"}, {"type": "json_object"}])
async def test_pre_call_skips_structured_output(hook: ToolPackHook, fmt: dict[str, str]) -> None:
    data = _data(response_format=fmt)
    assert await hook.async_pre_call_hook(None, None, data, "acompletion") is None
    assert "tools" not in data


@pytest.mark.parametrize(
    "choice",
    ["none", "required", {"type": "function", "function": {"name": "x"}}],
)
async def test_pre_call_skips_pinned_tool_choice(hook: ToolPackHook, choice: Any) -> None:
    data = _data(tool_choice=choice)
    assert await hook.async_pre_call_hook(None, None, data, "acompletion") is None
    assert "tools" not in data


async def test_pre_call_still_injects_on_auto_tool_choice(hook: ToolPackHook) -> None:
    out = await hook.async_pre_call_hook(None, None, _data(tool_choice="auto"), "acompletion")
    assert out is not None and out["tools"]


async def test_pre_call_skips_non_completion_calls(hook: ToolPackHook) -> None:
    data = _data()
    assert await hook.async_pre_call_hook(None, None, data, "embeddings") is None
    assert "tools" not in data


# ----------------------------------------------------------------------------
# Post-call hook (non-streaming)
# ----------------------------------------------------------------------------


async def test_post_call_returns_unchanged_when_no_tool_calls(hook: ToolPackHook) -> None:
    response = FakeResponse(FakeMessage(content="just text"))
    assert await hook.async_post_call_success_hook(_data(), None, response) is response


async def test_post_call_dispatches_the_packs_tool_calls(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook, pack: FakePack
) -> None:
    """See the call, run it through the pack, loop back with a tool message."""
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(FakeMessage(content="Found nothing."))

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    response = FakeResponse(
        FakeMessage(tool_calls=[FakeToolCall("call_1", "fake_search", {"q": "vllm"})])
    )
    out = await hook.async_post_call_success_hook(_data(), None, response)

    assert pack.calls == [("fake_search", {"q": "vllm"}, None)]
    assert out.choices[0].message.content == "Found nothing."
    assert len(calls) == 1
    tool_msgs = [m for m in calls[0]["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps({"ok": True}),
    }


async def test_post_call_leaves_foreign_tool_calls_to_the_client(
    hook: ToolPackHook, pack: FakePack
) -> None:
    response = FakeResponse(
        FakeMessage(tool_calls=[FakeToolCall("call_1", "web_search", {"q": "x"})])
    )
    assert await hook.async_post_call_success_hook(_data(), None, response) is response
    assert pack.calls == []


async def test_post_call_leaves_a_mixed_turn_to_the_client(
    hook: ToolPackHook, pack: FakePack
) -> None:
    response = FakeResponse(
        FakeMessage(
            tool_calls=[
                FakeToolCall("c1", "fake_search", {"q": "x"}),
                FakeToolCall("c2", "web_search", {"q": "y"}),
            ]
        )
    )
    assert await hook.async_post_call_success_hook(_data(), None, response) is response
    assert pack.calls == []


async def test_post_call_respects_max_iterations(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook, pack: FakePack
) -> None:
    """A model that keeps calling tools forever must still terminate."""

    async def always_tool_call(**kwargs: Any) -> FakeResponse:
        return FakeResponse(FakeMessage(tool_calls=[FakeToolCall("c", "fake_list", {})]))

    import litellm

    monkeypatch.setattr(litellm, "acompletion", always_tool_call)

    response = FakeResponse(FakeMessage(tool_calls=[FakeToolCall("c0", "fake_list", {})]))
    out = await hook.async_post_call_success_hook(_data(), None, response)

    assert out is not None
    assert len(pack.calls) == hook.max_iterations


async def test_post_call_survives_a_pack_that_raises(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook, pack: FakePack
) -> None:
    async def boom(name: str, arguments: dict[str, Any]) -> str:
        raise RuntimeError("pack is down")

    pack.on_execute = boom
    response = FakeResponse(FakeMessage(tool_calls=[FakeToolCall("c", "fake_list", {})]))

    assert await hook.async_post_call_success_hook(_data(), None, response) is response


async def test_post_call_names_the_caller(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook, pack: FakePack
) -> None:
    class Key:
        key_alias = "lobehub"

    async def fake_acompletion(**kwargs: Any) -> FakeResponse:
        return FakeResponse(FakeMessage(content="done"))

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    response = FakeResponse(FakeMessage(tool_calls=[FakeToolCall("c", "fake_list", {})]))
    await hook.async_post_call_success_hook(_data(), Key(), response)

    assert pack.calls[0][2] == "lobehub"


# ----------------------------------------------------------------------------
# Streaming hook
# ----------------------------------------------------------------------------


async def test_stream_passes_plain_content_untouched(hook: ToolPackHook) -> None:
    chunks = [FakeChunk(FakeDelta(content="Hel")), FakeChunk(FakeDelta(content="lo"))]
    out = await collect(hook.async_post_call_streaming_iterator_hook(None, as_stream(chunks), {}))
    assert out == chunks


async def test_stream_eats_the_packs_tool_calls_and_streams_the_answer(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook, pack: FakePack
) -> None:
    """A streamed pack turn must emit no tool-call chunks; the follow-up stream
    continues in their place."""
    followups: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        followups.append(kwargs)
        return as_stream(
            [FakeChunk(FakeDelta(content="Found ")), FakeChunk(FakeDelta(content="nothing."))]
        )

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    client_chunks = [
        FakeChunk(FakeDelta(content="Let me check. ")),
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, id="call_1", name="fake_search")])),
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, arguments='{"q":')])),
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, arguments=' "vllm"}')])),
    ]
    out = await collect(
        hook.async_post_call_streaming_iterator_hook(None, as_stream(client_chunks), _data())
    )

    assert out[0] is client_chunks[0]
    assert stream_content(out) == "Let me check. Found nothing."
    assert not any(c.choices[0].delta.tool_calls for c in out)
    # Reassembled from three fragments.
    assert pack.calls == [("fake_search", {"q": "vllm"}, None)]
    assert followups[0]["stream"] is True
    tool_msgs = [m for m in followups[0]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_1"
    # The narration the model produced before calling the tool is kept.
    assistant = [m for m in followups[0]["messages"] if m.get("role") == "assistant"]
    assert assistant[0]["content"] == "Let me check. "


async def test_stream_mints_an_id_for_a_tool_call_that_arrives_without_one(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook
) -> None:
    followups: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        followups.append(kwargs)
        return as_stream([FakeChunk(FakeDelta(content="ok"))])

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    chunks = [
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, name="fake_list", arguments="{}")]))
    ]
    await collect(hook.async_post_call_streaming_iterator_hook(None, as_stream(chunks), _data()))

    tool_msgs = [m for m in followups[0]["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_fake_0"


async def test_stream_flushes_foreign_tool_calls(hook: ToolPackHook, pack: FakePack) -> None:
    """Client-side tool calls must reach the client untouched."""
    chunks = [
        FakeChunk(FakeDelta(content="Searching...")),
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, id="call_9", name="web_search")])),
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, arguments='{"q":"x"}')])),
    ]
    out = await collect(hook.async_post_call_streaming_iterator_hook(None, as_stream(chunks), {}))

    assert out == chunks
    assert pack.calls == []


async def test_stream_flushes_mixed_tool_calls(hook: ToolPackHook, pack: FakePack) -> None:
    """A turn mixing ours and the client's tools is the client's to resolve."""
    chunks = [
        FakeChunk(
            FakeDelta(
                tool_calls=[
                    FakeDeltaToolCall(0, id="c1", name="fake_search"),
                    FakeDeltaToolCall(1, id="c2", name="web_search"),
                ]
            )
        ),
        FakeChunk(
            FakeDelta(
                tool_calls=[
                    FakeDeltaToolCall(0, arguments='{"q":"x"}'),
                    FakeDeltaToolCall(1, arguments='{"q":"y"}'),
                ]
            )
        ),
    ]
    out = await collect(hook.async_post_call_streaming_iterator_hook(None, as_stream(chunks), {}))

    assert out == chunks
    assert pack.calls == []


async def test_stream_respects_max_iterations(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook
) -> None:
    """An endlessly tool-calling model must terminate the stream cleanly."""

    async def fake_acompletion(**kwargs: Any) -> Any:
        return as_stream(
            [
                FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, id="c", name="fake_list")])),
                FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, arguments="{}")])),
            ]
        )

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    client_chunks = [
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, id="c0", name="fake_list")])),
        FakeChunk(FakeDelta(tool_calls=[FakeDeltaToolCall(0, arguments="{}")])),
    ]
    out = await collect(
        hook.async_post_call_streaming_iterator_hook(None, as_stream(client_chunks), _data())
    )

    assert not any(c.choices[0].delta.tool_calls for c in out)


# ----------------------------------------------------------------------------
# Follow-up failures: retried while nothing reached the client, never silent
# ----------------------------------------------------------------------------


def _pack_turn() -> list[FakeChunk]:
    return [
        FakeChunk(FakeDelta(content="Checking. ")),
        FakeChunk(
            FakeDelta(
                tool_calls=[
                    FakeDeltaToolCall(0, id="call_1", name="fake_search", arguments='{"q": "x"}')
                ]
            )
        ),
    ]


@pytest.fixture
def quick_retries(monkeypatch: pytest.MonkeyPatch, pack: FakePack) -> ToolPackHook:
    import asyncio

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    return ToolPackHook(pack, max_iterations=3, followup_retries=2)


async def test_stream_retries_a_follow_up_that_failed_before_any_content(
    monkeypatch: pytest.MonkeyPatch, quick_retries: ToolPackHook
) -> None:
    attempts: list[int] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            return failing_stream(RuntimeError("provider overloaded"))
        return as_stream([FakeChunk(FakeDelta(content="Done."))])

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    out = await collect(
        quick_retries.async_post_call_streaming_iterator_hook(
            None, as_stream(_pack_turn()), _data()
        )
    )

    assert stream_content(out) == "Checking. Done."
    assert len(attempts) == 2


async def test_stream_ends_with_a_notice_when_a_follow_up_fails_mid_answer(
    monkeypatch: pytest.MonkeyPatch, quick_retries: ToolPackHook
) -> None:
    async def fake_acompletion(**kwargs: Any) -> Any:
        return failing_stream(RuntimeError("boom"), [FakeChunk(FakeDelta(content="Half"))])

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    out = await collect(
        quick_retries.async_post_call_streaming_iterator_hook(
            None, as_stream(_pack_turn()), _data()
        )
    )

    text = stream_content(out)
    assert text.startswith("Checking. Half")
    assert "The fake tool loop stopped" in text and "RuntimeError" in text


async def test_stream_gives_up_with_a_notice_when_a_follow_up_never_opens(
    monkeypatch: pytest.MonkeyPatch, quick_retries: ToolPackHook
) -> None:
    calls: list[int] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        calls.append(1)
        raise RuntimeError("503")

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    out = await collect(
        quick_retries.async_post_call_streaming_iterator_hook(
            None, as_stream(_pack_turn()), _data()
        )
    )

    assert "The fake tool loop stopped" in stream_content(out)
    assert len(calls) == 3  # the first try plus two retries


async def test_a_failing_client_stream_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, quick_retries: ToolPackHook
) -> None:
    """The client's own stream is not ours to reopen: it is already half sent."""
    calls: list[int] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        calls.append(1)
        return as_stream([FakeChunk(FakeDelta(content="never"))])

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    upstream = failing_stream(RuntimeError("upstream died"), [FakeChunk(FakeDelta(content="Hi"))])
    out = await collect(
        quick_retries.async_post_call_streaming_iterator_hook(None, upstream, _data())
    )

    assert "tool loop stopped" in stream_content(out)
    assert calls == []


async def test_follow_ups_use_the_proxy_router_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch, hook: ToolPackHook
) -> None:
    import sys
    import types

    seen: list[dict[str, Any]] = []

    class Router:
        async def acompletion(self, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return "routed"

    fake = types.ModuleType("litellm.proxy.proxy_server")
    fake.llm_router = Router()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake)

    assert await hook._complete({"model": "memory/extract", "messages": []}) == "routed"  # noqa: SLF001
    assert seen[0]["model"] == "memory/extract"
