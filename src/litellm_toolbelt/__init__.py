"""litellm-toolbelt: reusable hooks for a LiteLLM proxy.

- `ToolPack` / `ToolPackHook` — run a set of tools server-side, so every client
  of the proxy gets them without implementing tool calling.
- `CostPassthrough` — put the cost the proxy already computed into `usage.cost`.
- `StructuredOutputFix` — rescue JSON answers a reasoning parser swallowed.

The two hook modules build the `proxy_handler_instance` LiteLLM's callback
contract asks for, and `CostPassthrough` flips a LiteLLM setting when it is
constructed. So they are resolved on attribute access rather than imported
here: reaching for `ToolPack` should not configure a proxy.
"""

from typing import TYPE_CHECKING, Any

from .toolpack import ToolPack, ToolPackHook, caller_name

if TYPE_CHECKING:
    from .cost import CostPassthrough
    from .structured_output import StructuredOutputFix

__version__ = "0.1.0"
__all__ = [
    "CostPassthrough",
    "StructuredOutputFix",
    "ToolPack",
    "ToolPackHook",
    "caller_name",
]

_LAZY = {
    "CostPassthrough": ".cost",
    "StructuredOutputFix": ".structured_output",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)
