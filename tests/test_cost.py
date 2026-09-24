"""Tests for the cost passthrough hook."""

from __future__ import annotations

from typing import Any

import pytest
from fakes import FakeChunk, FakeDelta, FakeMessage, FakeResponse, FakeUsage

from litellm_toolbelt.cost import CostPassthrough, resolve_cost

PROVIDER_HEADER = "llm_provider-x-litellm-response-cost"


class FakeLogging:
    """Stands in for LiteLLM's logging object, the proxy's own cost fallback."""

    def __init__(self, stored: Any = None, computed: Any = None, raises: bool = False) -> None:
        self.model_call_details: dict[str, Any] = {}
        if stored is not None:
            self.model_call_details["response_cost"] = stored
        self.computed = computed
        self.raises = raises
        self.calculator_calls = 0

    def _response_cost_calculator(self, result: Any) -> Any:
        self.calculator_calls += 1
        if self.raises:
            raise RuntimeError("model not in the price map")
        return self.computed


@pytest.fixture
def hook() -> CostPassthrough:
    return CostPassthrough(enable_streaming=False)


def response(usage: FakeUsage | None = None, **hidden: Any) -> FakeResponse:
    return FakeResponse(
        FakeMessage(content="hi"),
        usage=FakeUsage() if usage is None else usage,
        hidden_params=hidden,
    )


async def test_cost_from_hidden_params(hook: CostPassthrough) -> None:
    out = await hook.async_post_call_success_hook({}, None, response(response_cost=0.0008902))
    assert out.usage.cost == 0.0008902


async def test_cost_from_the_logging_object(hook: CostPassthrough) -> None:
    data = {"litellm_logging_obj": FakeLogging(stored=0.25)}
    out = await hook.async_post_call_success_hook(data, None, response())
    assert out.usage.cost == 0.25


async def test_cost_computed_when_nothing_stored_one(hook: CostPassthrough) -> None:
    logging_obj = FakeLogging(computed=0.5)
    data = {"litellm_logging_obj": logging_obj}
    out = await hook.async_post_call_success_hook(data, None, response())
    assert out.usage.cost == 0.5
    assert logging_obj.calculator_calls == 1


async def test_an_unpriceable_model_leaves_the_field_absent(hook: CostPassthrough) -> None:
    data = {"litellm_logging_obj": FakeLogging(raises=True)}
    out = await hook.async_post_call_success_hook(data, None, response())
    assert getattr(out.usage, "cost", None) is None


async def test_the_providers_cost_rescues_a_model_litellm_cannot_price(
    hook: CostPassthrough,
) -> None:
    """A 0.0 from LiteLLM means "not in my price map", not "free"."""
    resp = response(response_cost=0.0, additional_headers={PROVIDER_HEADER: 0.0031})
    out = await hook.async_post_call_success_hook({}, None, resp)
    assert out.usage.cost == 0.0031


async def test_litellms_own_cost_wins_when_it_has_one(hook: CostPassthrough) -> None:
    resp = response(response_cost=0.002, additional_headers={PROVIDER_HEADER: 0.009})
    out = await hook.async_post_call_success_hook({}, None, resp)
    assert out.usage.cost == 0.002


async def test_a_genuinely_free_call_reports_zero(hook: CostPassthrough) -> None:
    out = await hook.async_post_call_success_hook({}, None, response(response_cost=0.0))
    assert out.usage.cost == 0.0


async def test_a_cost_the_provider_already_set_is_left_alone(hook: CostPassthrough) -> None:
    usage = FakeUsage()
    usage.cost = 0.0123  # type: ignore[attr-defined]
    resp = response(usage, response_cost=0.5)
    out = await hook.async_post_call_success_hook({}, None, resp)
    assert out.usage.cost == 0.0123


async def test_a_response_without_usage_is_returned_unchanged(hook: CostPassthrough) -> None:
    resp = FakeResponse(FakeMessage(content="hi"), usage=None, hidden_params={"response_cost": 1.0})
    assert await hook.async_post_call_success_hook({}, None, resp) is resp


async def test_a_streaming_chunk_is_not_this_hooks_business(hook: CostPassthrough) -> None:
    """LiteLLM's own include_cost_in_streaming_usage covers the stream."""
    chunk = FakeChunk(FakeDelta(content="hi"))
    assert await hook.async_post_call_success_hook({}, None, chunk) is chunk


@pytest.mark.parametrize("junk", [None, "0.5", True, float("nan"), -1.0])
async def test_junk_is_not_a_cost(hook: CostPassthrough, junk: Any) -> None:
    out = await hook.async_post_call_success_hook({}, None, response(response_cost=junk))
    assert getattr(out.usage, "cost", None) is None


def test_resolve_cost_is_usable_on_its_own() -> None:
    assert resolve_cost(response(response_cost=0.75), {}) == 0.75
    assert resolve_cost(response(), {}) is None


def test_constructing_the_hook_turns_on_streaming_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False, raising=False)
    CostPassthrough()
    assert litellm.include_cost_in_streaming_usage is True
