"""Put the cost LiteLLM already knows into the response the client reads.

The proxy prices every call — it has to, to charge keys and fill
`LiteLLM_SpendLogs` — and reports that number in the `x-litellm-response-cost`
response header. What it does not do is put it in the response *body*, and a
client that only parses JSON therefore cannot see it. Most UIs then fall back to
a price table of their own, which for a proxied model ("tensorx/z-ai/glm-5.3",
"openrouter/qwen/qwen3-235b-a22b", any alias) they do not have, so the call is
displayed as free.

This hook mirrors the cost into `usage.cost`, which is where OpenRouter puts it
and therefore the field a cost-aware client is most likely to already read:

    {"usage": {"prompt_tokens": 2959, "completion_tokens": 1,
               "total_tokens": 2960, "cost": 0.0008902}}

Streaming is LiteLLM's own `include_cost_in_streaming_usage`, which injects the
same field into the final usage chunk. Constructing this hook turns that setting
on, so one callback covers both response modes:

    litellm_settings:
      callbacks:
        - litellm_toolbelt.cost:proxy_handler_instance

Which number: a nonzero cost from LiteLLM wins, because that is what the key is
actually charged. Failing that, the cost the upstream provider reported for the
call is used — this is what rescues models missing from LiteLLM's price map,
where its own figure is a 0.0 that means "unpriced", not "free". Only if
neither exists does the field stay absent, leaving the client free to fall back
to its own estimate. A cost the provider itself put in `usage.cost` is left
exactly as it is.
"""

from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from .wire import field, set_field

log = logging.getLogger("litellm_toolbelt.cost")

__all__ = ["CostPassthrough", "proxy_handler_instance", "resolve_cost"]

# Header LiteLLM uses to carry a cost the upstream provider reported itself.
_PROVIDER_COST_HEADER = "llm_provider-x-litellm-response-cost"


def _as_cost(value: Any) -> float | None:
    """A usable cost, or None. Bools are not costs; NaN and negatives are not either."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    cost = float(value)
    if cost != cost or cost < 0:  # NaN, or nonsense
        return None
    return cost


def _hidden_params(response: Any) -> dict[str, Any]:
    params = getattr(response, "_hidden_params", None)
    return params if isinstance(params, dict) else {}


def _provider_reported(response: Any) -> float | None:
    headers = _hidden_params(response).get("additional_headers")
    if not isinstance(headers, dict):
        return None
    return _as_cost(headers.get(_PROVIDER_COST_HEADER))


def _litellm_cost(response: Any, data: dict[str, Any]) -> float | None:
    """LiteLLM's own cost for this call: from the response, else its logging object.

    A response that never recorded one (`_hidden_params` absent, or present but
    not yet populated at hook time) still has a logging object that either
    stored the cost or can compute it from the same calculator the proxy uses
    for its headers and spend logs.
    """
    cost = _as_cost(_hidden_params(response).get("response_cost"))
    if cost is not None:
        return cost

    logging_obj = data.get("litellm_logging_obj")
    details = getattr(logging_obj, "model_call_details", None)
    if isinstance(details, dict):
        cost = _as_cost(details.get("response_cost"))
        if cost is not None:
            return cost

    calculator = getattr(logging_obj, "_response_cost_calculator", None)
    if callable(calculator):
        try:
            return _as_cost(calculator(result=response))
        except Exception:  # noqa: BLE001 - an unpriceable model is not an error here
            log.debug("cost calculator declined to price the response", exc_info=True)
    return None


def resolve_cost(response: Any, data: dict[str, Any]) -> float | None:
    """The cost to report for `response`, or None when nobody knows one.

    See the module docstring for why a nonzero LiteLLM cost beats the
    provider's figure and why a zero one does not.
    """
    ours = _litellm_cost(response, data)
    if ours:
        return ours

    theirs = _provider_reported(response)
    if theirs is not None:
        return theirs

    return ours  # 0.0 when LiteLLM priced it as free, else None


class CostPassthrough(CustomLogger):
    """Mirror the proxy's cost for a call into `usage.cost`.

    `enable_streaming` flips LiteLLM's `include_cost_in_streaming_usage`, whose
    built-in injection covers the streaming half of the same job. Pass False to
    leave that setting to the config file.
    """

    def __init__(self, *, enable_streaming: bool = True) -> None:
        super().__init__()
        if enable_streaming:
            import litellm

            if not getattr(litellm, "include_cost_in_streaming_usage", False):
                litellm.include_cost_in_streaming_usage = True
                log.info(
                    "enabled litellm.include_cost_in_streaming_usage so streamed "
                    "responses carry usage.cost too"
                )

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        try:
            usage = field(response, "usage")
            if usage is None or _as_cost(field(usage, "cost")) is not None:
                # No usage to annotate, or the provider reported its own cost.
                return response

            cost = resolve_cost(response, data)
            if cost is None:
                return response

            set_field(usage, "cost", cost)
        except Exception:
            # A missing cost is a cosmetic problem; a raised hook is not.
            log.exception("failed to attach the response cost")
        return response


proxy_handler_instance = CostPassthrough()
