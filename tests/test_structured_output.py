"""Tests for the structured-output repair hook."""

from __future__ import annotations

from typing import Any

import pytest
from fakes import FakeChunk, FakeDelta, FakeMessage, FakeResponse, as_stream, collect

from litellm_toolbelt.structured_output import StructuredOutputFix

JSON_REQUEST = {"response_format": {"type": "json_object"}}


@pytest.fixture
def hook() -> StructuredOutputFix:
    return StructuredOutputFix()


async def test_reasoning_becomes_the_answer(hook: StructuredOutputFix) -> None:
    response = FakeResponse(FakeMessage(content=None, reasoning_content='{"a": 1}'))
    out = await hook.async_post_call_success_hook(JSON_REQUEST, None, response)

    assert out.choices[0].message.content == '{"a": 1}'
    assert out.choices[0].message.reasoning_content is None


async def test_a_real_answer_is_left_alone(hook: StructuredOutputFix) -> None:
    response = FakeResponse(FakeMessage(content='{"a": 1}', reasoning_content="thinking..."))
    out = await hook.async_post_call_success_hook(JSON_REQUEST, None, response)

    assert out.choices[0].message.content == '{"a": 1}'
    assert out.choices[0].message.reasoning_content == "thinking..."


@pytest.mark.parametrize("data", [{}, {"response_format": {"type": "text"}}])
async def test_a_plain_chat_is_left_alone(hook: StructuredOutputFix, data: dict[str, Any]) -> None:
    response = FakeResponse(FakeMessage(content=None, reasoning_content="thinking..."))
    out = await hook.async_post_call_success_hook(data, None, response)

    assert out.choices[0].message.content is None


async def test_streamed_reasoning_becomes_content(hook: StructuredOutputFix) -> None:
    chunks = [
        FakeChunk(FakeDelta(reasoning_content='{"a":')),
        FakeChunk(FakeDelta(reasoning_content=" 1}")),
    ]
    out = await collect(
        hook.async_post_call_streaming_iterator_hook(None, as_stream(chunks), JSON_REQUEST)
    )

    assert [c.choices[0].delta.content for c in out] == ['{"a":', " 1}"]
    assert all(c.choices[0].delta.reasoning_content is None for c in out)


async def test_a_streamed_plain_chat_passes_through(hook: StructuredOutputFix) -> None:
    chunks = [FakeChunk(FakeDelta(reasoning_content="thinking..."))]
    out = await collect(hook.async_post_call_streaming_iterator_hook(None, as_stream(chunks), {}))

    assert out == chunks
    assert out[0].choices[0].delta.content is None
