"""Promote reasoning to content on structured-output calls.

Some reasoning parsers assume every completion opens inside a `<think>` block.
vLLM's `deepseek_r1` parser is one: an answer that never emits `</think>` —
which is exactly what guided JSON decoding produces — lands whole in
`reasoning_content` while `content` comes back empty. The caller then parses an
empty string, and a client that expected an object reports "expected object,
received undefined".

A caller that asked for JSON never wants an empty body, so when a request
carries a JSON `response_format` and the answer has no content but does have
reasoning, the reasoning is the answer. Everything else passes through.

    litellm_settings:
      callbacks:
        - litellm_toolbelt.structured_output.proxy_handler_instance
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from .wire import field, set_field

__all__ = ["StructuredOutputFix", "proxy_handler_instance"]


def _wants_json(data: dict[str, Any]) -> bool:
    fmt = data.get("response_format")
    return isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema")


def _promote(carrier: Any) -> None:
    """Move `reasoning_content` into `content` when content is empty."""
    if carrier is None or field(carrier, "content"):
        return
    reasoning = field(carrier, "reasoning_content")
    if reasoning:
        set_field(carrier, "content", reasoning)
        set_field(carrier, "reasoning_content", None)


class StructuredOutputFix(CustomLogger):
    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        if not _wants_json(data):
            return response
        for choice in field(response, "choices") or []:
            _promote(field(choice, "message"))
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        if not _wants_json(request_data):
            async for chunk in response:
                yield chunk
            return
        # Streamed JSON: a chunk that carries only reasoning is part of the
        # answer, so its text moves to content before the client sees it.
        async for chunk in response:
            for choice in field(chunk, "choices") or []:
                _promote(field(choice, "delta"))
            yield chunk


proxy_handler_instance = StructuredOutputFix()
