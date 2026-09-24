"""Server-side tool execution for the LiteLLM proxy.

A *tool pack* is a set of tools plus the code that runs them. Register the hook
below with a pack and every client of the proxy — a chat UI, Home Assistant,
n8n, curl — gets those tools without knowing they exist: the model calls them,
the proxy executes them, and the client sees one ordinary answer.

Three hooks carry that:

1. `async_pre_call_hook` runs on the way *into* the LLM call. It merges the
   pack's tool definitions into `data["tools"]` (tools the client passed are
   left alone) and appends the pack's system prompt once, marked so the loop's
   own follow-ups do not keep re-adding it.

2. `async_post_call_success_hook` runs after a NON-STREAMING response. Tool
   calls that belong to the pack are executed, appended to the conversation as
   `tool` messages, and the model is called again until it answers without
   asking for more tools (or `max_iterations` is reached).

3. `async_post_call_streaming_iterator_hook` is the STREAMING counterpart.
   LiteLLM streams chunks to the client as they arrive and only runs hook 2
   once the stream is over — too late to stop raw `tool_calls` from reaching a
   client that cannot resolve them. So this hook sits on the chunk stream
   itself: content flows through live, a tool-call region is held back, and the
   loop's follow-up stream continues in its place.

Register it in `config.yaml` as a module-level instance:

    litellm_settings:
      callbacks:
        - my_pack.hook.proxy_handler_instance

A pack that needs no loop at all (a logger, a rewriter) does not belong here —
subclass `litellm.integrations.custom_logger.CustomLogger` directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

from litellm.integrations.custom_logger import CustomLogger

from .wire import (
    accumulate_tool_calls,
    as_tool_call_message,
    assistant_message_from_response,
    chunk_delta,
    delta_content,
    delta_tool_calls,
    followup_kwargs,
    normalize_tool_calls,
    notice_chunk,
    tool_calls_of,
)

log = logging.getLogger("litellm_toolbelt.toolpack")

__all__ = ["ToolPack", "ToolPackHook", "caller_name"]


@runtime_checkable
class ToolPack(Protocol):
    """The tools a hook offers and the code that runs them."""

    #: Short identifier, used in log lines and in synthesized tool-call ids.
    name: str

    def tools(self) -> list[dict[str, Any]]:
        """The tool definitions, in OpenAI's list-of-functions format."""

    def owns(self, tool_name: str) -> bool:
        """True when `tool_name` is this pack's to execute."""

    async def system_prompt(self) -> str:
        """Instructions to append to the system message, or "" for none.

        Called once per client request (not per loop round), so a pack may do
        real work here — read a file, hit a cache — to build context.
        """

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> str:
        """Run one tool call and return the `tool` message content.

        The return value is fed straight back to the model, so errors belong in
        it (`{"error": "..."}`) rather than raised: the model can react to a
        failed tool, but a raised exception ends the turn. `actor` is the
        proxy caller's name, for packs that attribute their side effects.
        """


class ToolPackHook(CustomLogger):
    """LiteLLM callback that injects a pack's tools and executes their calls."""

    def __init__(
        self,
        pack: ToolPack,
        *,
        max_iterations: int = 5,
        followup_retries: int = 3,
        marker: str | None = None,
    ) -> None:
        super().__init__()
        self.pack = pack
        self.max_iterations = max_iterations
        # A follow-up that fails before any of it reached the client is tried
        # again: free endpoints report "provider overloaded" routinely.
        self.followup_retries = followup_retries
        # The marker we add to a single system message so subsequent iterations
        # of the loop don't keep prepending a fresh prompt.
        self.marker = marker or f"<!-- litellm-toolbelt:{pack.name} -->"

    # -- hooks ---------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any] | None:
        """Inject the pack's tools + system prompt before the LLM call."""
        if call_type != "acompletion":
            return None

        # Two kinds of caller must be left alone. One asks for a schema and
        # wants JSON back, not a tool call - handing it tools makes the model
        # answer with tool_calls and a null content, which a structured-output
        # caller reads as "expected object, received undefined". The other has
        # already pinned tool_choice, whether to "none" or to one function it
        # means to call; a structured answer produced through a forced tool is
        # the same request wearing a different hat, and extra tools are how it
        # picks the wrong one.
        if _wants_structured_output(data) or _tool_choice_is_pinned(data):
            return None

        data["tools"] = self._merged_tools(data.get("tools") or [])

        messages: list[dict[str, Any]] = list(data.get("messages") or [])
        if not any(self.marker in (m.get("content") or "") for m in messages):
            prompt = await self.pack.system_prompt()
            if prompt:
                fragment = f"{prompt}\n{self.marker}\n"
                self._append_system(messages, fragment)
        data["messages"] = messages

        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        """Execute the pack's tool calls and loop until the model answers.

        A turn whose tool calls are not *all* the pack's (none, or a mix of the
        pack's and the client's) is returned unchanged: the client executes its
        own tools, and a half-eaten turn cannot be run server-side without
        orphaning its tool results.
        """
        try:
            assistant_msg = assistant_message_from_response(response)
            calls = tool_calls_of(assistant_msg)
            if not assistant_msg or not calls or not self._all_ours(calls):
                return response

            return await self._run_agent_loop(
                data=data,
                assistant_msg=assistant_msg,
                original_response=response,
                actor=caller_name(user_api_key_dict),
            )
        except Exception:
            log.exception("%s tool loop failed after a non-streaming call", self.pack.name)
            return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        """Streaming counterpart of `async_post_call_success_hook`.

        * Content deltas are forwarded live — plain chat streams untouched.
        * The moment a tool-call delta appears, that chunk and everything after
          it is held back.
        * When the turn ends, a turn made purely of the pack's calls is executed
          here and the loop's follow-up stream continues where the held region
          left off. Raw tool chunks are never emitted.
        * A turn with no, or foreign, tool calls flushes the held chunks
          unchanged so client-side tool calling keeps working.

        The client sees the narration and the final answer of a tool-using turn
        as one ordinary streamed message, exactly as the non-streaming path
        returns one ordinary response.
        """
        actor = caller_name(user_api_key_dict)
        async for chunk in self._translate_stream(response, request_data, actor):
            yield chunk

    # -- the loop ------------------------------------------------------------

    async def _run_agent_loop(
        self,
        *,
        data: dict[str, Any],
        assistant_msg: dict[str, Any],
        original_response: Any,
        actor: str | None = None,
    ) -> Any:
        """Execute `assistant_msg`'s tool calls and loop until a final answer.

        Returns the first response that asks for no tools of ours (or the last
        one at the iteration cap).
        """
        messages: list[dict[str, Any]] = list(data.get("messages") or [])
        messages.append(assistant_msg)
        calls = tool_calls_of(assistant_msg)
        next_response = original_response

        for iteration in range(self.max_iterations):
            log.debug(
                "%s loop iteration %d, %d messages so far",
                self.pack.name,
                iteration,
                len(messages),
            )
            await self._execute_tool_calls(calls, messages, actor)
            next_response = await self._complete(followup_kwargs(data, messages))
            next_msg = assistant_message_from_response(next_response)
            calls = tool_calls_of(next_msg)
            if not next_msg or not calls or not self._all_ours(calls):
                # Model produced a final answer (or wants a client-side
                # tool — hand it back untouched either way).
                return next_response
            messages.append(next_msg)

        log.warning(
            "%s loop hit max_iterations=%d; returning last response",
            self.pack.name,
            self.max_iterations,
        )
        return next_response

    async def _translate_stream(
        self,
        response: Any,
        request_data: dict[str, Any],
        actor: str | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Rewrite a chunk stream so the pack's tool calls never reach the client.

        Each round of the loop gets its own upstream stream: the client's for
        the first round, a fresh completion for each follow-up.

        A follow-up that fails before any of it reached the client is retried.
        One that fails after that ends the answer with a visible note — never
        with silence, which a client reads as a successful, empty reply.
        """
        messages: list[dict[str, Any]] = list(request_data.get("messages") or [])
        upstream = response
        followup: dict[str, Any] | None = None  # None: the client's own stream
        last_chunk: Any = None

        for round_no in range(self.max_iterations + 1):
            for attempt in range(self.followup_retries + 1):
                content_parts: list[str] = []
                tool_acc: dict[int, dict[str, str]] = {}
                held: list[Any] = []
                forwarding = True
                delivered = False
                try:
                    async for chunk in upstream:
                        last_chunk = chunk
                        delta = chunk_delta(chunk)
                        calls = delta_tool_calls(delta)
                        if calls:
                            forwarding = False
                            accumulate_tool_calls(tool_acc, calls)
                        if forwarding:
                            text = delta_content(delta)
                            if text:
                                content_parts.append(text)
                                delivered = True
                            yield chunk
                        else:
                            held.append(chunk)
                    break
                except Exception as exc:
                    if followup is None or delivered or attempt >= self.followup_retries:
                        log.warning("%s stream loop stopped: %s", self.pack.name, exc)
                        notice = notice_chunk(last_chunk, self._stopped_notice(exc))
                        if notice is None:
                            raise
                        yield notice
                        return
                    log.info("follow-up failed (%s); retry %d", exc, attempt + 1)
                    await asyncio.sleep(min(2 ** (attempt + 1), 20))
                    upstream = await self._open(followup)

            turn_calls = normalize_tool_calls(tool_acc, id_prefix=f"call_{self.pack.name}")
            if not turn_calls:
                # No tool calls at all: nothing was held back (forwarding
                # never flipped), the stream is already fully delivered.
                return

            if not self._all_ours(turn_calls):
                # Foreign or mixed tool calls: not ours to eat. Flush the
                # held region so the client can resolve what it knows.
                for chunk in held:
                    yield chunk
                return

            if round_no >= self.max_iterations:
                log.warning(
                    "%s stream loop hit max_iterations=%d; dropping the trailing "
                    "tool-call turn",
                    self.pack.name,
                    self.max_iterations,
                )
                return

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [as_tool_call_message(c) for c in turn_calls],
            }
            turn_content = "".join(content_parts)
            if turn_content:
                assistant_msg["content"] = turn_content
            messages.append(assistant_msg)
            await self._execute_tool_calls(turn_calls, messages, actor)

            followup = followup_kwargs(request_data, messages)
            followup["stream"] = True
            upstream = await self._open(followup)

    async def _execute_tool_calls(
        self,
        calls: list[dict[str, str]],
        messages: list[dict[str, Any]],
        actor: str | None = None,
    ) -> None:
        """Run each tool call through the pack, appending `tool` messages."""
        for call in calls:
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await self.pack.execute(call["name"], args, actor=actor)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )

    # -- calling back into the proxy ----------------------------------------

    async def _open(self, kwargs: dict[str, Any]) -> Any:
        """Open a follow-up stream; a failure to open surfaces on first read,
        so opening and reading share one retry path."""
        try:
            return await self._complete(kwargs)
        except Exception as exc:
            return _failing_stream(exc)

    async def _complete(self, kwargs: dict[str, Any]) -> Any:
        """Run a follow-up completion the way the proxy ran the first one.

        Inside the proxy that means its router: the same model aliases,
        retries and fallbacks as the client's own call, where the bare SDK
        knows none of them and gives up on the first overloaded provider.
        """
        router = _proxy_router()
        if router is not None:
            return await router.acompletion(**kwargs)
        import litellm

        return await litellm.acompletion(**kwargs)

    # -- small parts --------------------------------------------------------

    def _merged_tools(self, existing: list[Any]) -> list[Any]:
        """The client's tools plus ours, without shadowing a name it sent."""
        client_names = {
            t.get("function", {}).get("name") for t in existing if isinstance(t, dict)
        }
        merged = list(existing)
        for tool in self.pack.tools():
            if tool["function"]["name"] not in client_names:
                merged.append(tool)
        return merged

    @staticmethod
    def _append_system(messages: list[dict[str, Any]], fragment: str) -> None:
        """Append to the first system message, or insert one if there is none."""
        for m in messages:
            if m.get("role") == "system" and isinstance(m.get("content") or "", str):
                m["content"] = (m.get("content") or "") + fragment
                return
        messages.insert(0, {"role": "system", "content": fragment})

    def _all_ours(self, calls: list[dict[str, str]]) -> bool:
        return all(bool(c.get("name")) and self.pack.owns(c["name"]) for c in calls)

    def _stopped_notice(self, exc: Exception) -> str:
        return (
            f"\n\n⚠️ The {self.pack.name} tool loop stopped: the model provider "
            f"failed mid-answer ({type(exc).__name__}). The steps above ran; "
            "nothing after them did."
        )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _proxy_router() -> Any:
    """The running proxy's router, when this module is loaded inside the proxy."""
    module = sys.modules.get("litellm.proxy.proxy_server")
    return getattr(module, "llm_router", None) if module else None


async def _failing_stream(exc: Exception) -> AsyncGenerator[Any, None]:
    """A stream that fails on first read."""
    raise exc
    yield  # pragma: no cover - makes this an async generator


# Identities LiteLLM gives callers that did not authenticate as anyone in
# particular; naming them in a pack's audit trail would say nothing.
_ANONYMOUS = {"default_user_id", "litellm-proxy-admin", "admin"}


def caller_name(user_api_key_dict: Any) -> str | None:
    """Who the proxy is serving: the key's alias, else its team or user.

    Give each client its own LiteLLM key with an alias ("lobehub",
    "home-assistant", "n8n") and a pack can attribute every side effect to it.
    """
    for attr in ("key_alias", "team_alias", "user_id"):
        value = getattr(user_api_key_dict, attr, None)
        if value and str(value) not in _ANONYMOUS:
            return str(value)
    return None


def _wants_structured_output(data: dict[str, Any]) -> bool:
    """True when the request asks the model for JSON matching a schema."""
    response_format = data.get("response_format")
    if not isinstance(response_format, dict):
        return False

    return response_format.get("type") in {"json_object", "json_schema"}


def _tool_choice_is_pinned(data: dict[str, Any]) -> bool:
    """True when the caller has already fixed which tools may be used."""
    choice = data.get("tool_choice")
    if choice is None:
        return False

    # "auto" is the default and leaves the model free to pick; everything else
    # ("none", "required", or a named function) is a decision we must not undo.
    if isinstance(choice, str):
        return choice != "auto"

    return True
