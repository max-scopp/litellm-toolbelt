# litellm-toolbelt

Reusable hooks for a [LiteLLM proxy](https://github.com/BerriAI/litellm): things
that belong in the proxy because every client benefits from them, and that the
proxy does not ship.

| Hook | What it does |
|---|---|
| `ToolPackHook` | Runs a set of tools **server-side**, so every client of the proxy gets them without implementing tool calling — streaming included. |
| `CostPassthrough` | Puts the cost the proxy already computed into `usage.cost`, so a UI can show real spend for a proxied model. |
| `StructuredOutputFix` | Rescues JSON answers a reasoning parser swallowed into `reasoning_content`. |

Each hook is an ordinary LiteLLM `CustomLogger`; nothing here patches LiteLLM or
depends on the others.

## Install

```bash
pip install git+https://github.com/max-scopp/litellm-toolbelt
```

To use it inside the official proxy image, layer it on:

```dockerfile
FROM ghcr.io/berriai/litellm:main-stable
# The image ships its Python in /app/.venv but no pip, so bootstrap one.
RUN python -m ensurepip --upgrade \
 && python -m pip install --no-cache-dir git+https://github.com/max-scopp/litellm-toolbelt
```

## Tool packs

A *tool pack* is a set of tool definitions plus the code that runs them. The
hook injects the definitions into every chat completion, intercepts the calls
the model makes, executes them, and loops back into the model with the results —
so the client sees one ordinary answer and never a `tool_calls` it would have to
resolve itself.

That last part is the reason this exists. A client that does not know your tools
(Home Assistant, n8n, a cron job, a chat UI with its own plugin system) cannot
answer a tool call, and a streamed one reaches it before any post-call hook
could intervene.

```python
# my_pack/hook.py
from typing import Any

from litellm_toolbelt import ToolPackHook


class WeatherPack:
    name = "weather"

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "weather_now",
                    "description": "Current conditions for a place.",
                    "parameters": {
                        "type": "object",
                        "properties": {"place": {"type": "string"}},
                        "required": ["place"],
                    },
                },
            }
        ]

    def owns(self, tool_name: str) -> bool:
        return tool_name.startswith("weather_")

    async def system_prompt(self) -> str:
        return "Call weather_now instead of guessing the weather."

    async def execute(self, tool_name, arguments, *, actor=None) -> str:
        return await my_weather_api(arguments["place"])  # a JSON string


proxy_handler_instance = ToolPackHook(WeatherPack())
```

```yaml
# config.yaml
litellm_settings:
  callbacks:
    - my_pack.hook.proxy_handler_instance
```

The contract is `ToolPack` (a runtime-checkable `Protocol`):

| Member | Contract |
|---|---|
| `name` | Short identifier, used in log lines and synthesized tool-call ids. |
| `tools()` | Definitions in OpenAI's list-of-functions format. |
| `owns(tool_name)` | True when a call is yours to execute. |
| `system_prompt()` | Instructions appended to the system message once per request, or `""`. Async, so it may read a file or a cache to build context. |
| `execute(tool_name, arguments, *, actor)` | Runs one call, returns the `tool` message content. Put errors in the return value (`{"error": ...}`) — the model can react to a failed tool, but a raised exception ends the turn. |

What the hook handles for you:

- **Streaming.** Content deltas pass through live. From the first tool-call delta
  the stream is held back, the loop runs, and the follow-up stream continues in
  its place — the client sees the narration and the final answer as one message.
- **Mixed turns.** A turn whose calls are not *all* yours is handed to the client
  untouched, so its own tool calling keeps working. A half-executed turn would
  orphan its tool results.
- **Structured output.** A request with a JSON `response_format`, or a pinned
  `tool_choice`, gets no tools injected: a caller that wants an object back is
  not looking for a tool call, and extra tools are how it picks the wrong one.
- **Follow-ups through the proxy's router,** not the bare SDK — so aliases,
  retries and fallbacks from `config.yaml` apply to the loop's own calls.
- **Retries and a visible failure.** A follow-up that dies before anything
  reached the client is retried with backoff; one that dies mid-answer ends the
  stream with a readable notice rather than silence, which a client reads as a
  successful empty reply.
- **Attribution.** `caller_name()` resolves the LiteLLM key alias (then team,
  then user) and passes it as `actor`, so a pack can record who caused a write.

`ToolPackHook(pack, max_iterations=5, followup_retries=3, marker=None)` — the
loop stops after `max_iterations` rounds, and `marker` is the HTML comment that
keeps the prompt from being re-appended on each round.

A real pack: [obsidian-litellm-tools](https://github.com/max-scopp/obsidian-litellm-tools),
which gives every client of the proxy read/write access to an Obsidian vault
(with [obsidian-writer](https://github.com/max-scopp/obsidian-writer) behind it
and [obsidian-recall](https://github.com/max-scopp/obsidian-recall) for search
by meaning).

## Cost passthrough

The proxy prices every call — it has to, to charge keys and fill
`LiteLLM_SpendLogs` — and reports it in the `x-litellm-response-cost` response
header. It does not put it in the response *body*, so a client that only parses
JSON cannot see it, and most UIs fall back to a price table of their own. For a
proxied model (`tensorx/z-ai/glm-5.3`, `openrouter/qwen/qwen3-235b-a22b`, any
alias) they have no entry, so the call shows up as free.

```yaml
litellm_settings:
  callbacks:
    - litellm_toolbelt.cost.proxy_handler_instance
```

```jsonc
{"usage": {"prompt_tokens": 2959, "completion_tokens": 1,
           "total_tokens": 2960, "cost": 0.0008902}}
```

`usage.cost` is where OpenRouter puts it, and therefore the field a cost-aware
client is most likely to already read. Streaming is LiteLLM's own
`include_cost_in_streaming_usage`, which injects the same field into the final
usage chunk; constructing the hook turns that setting on, so one callback covers
both response modes (pass `CostPassthrough(enable_streaming=False)` to leave it
to the config file).

Which number, in order:

1. **A positive cost from LiteLLM** — what the key is actually charged, read from
   the response's `_hidden_params`, else its logging object.
2. **The cost the upstream provider reported** for the call, including an explicit
   zero. This is what rescues a model missing from LiteLLM's price map.
3. **Nothing.** The field stays absent and the client falls back to its own
   estimate.

A zero that only LiteLLM produced is deliberately not reported. Its calculator
returns `0.0` both for a model priced at zero and for one it cannot price at all,
and those are different claims: writing `0.0` would tell the client a call that
may have cost real money was free — the exact failure this hook exists to fix.
Absent means "nobody knows", which is true.

A cost the provider itself put in `usage.cost` is left exactly as it is; with
OpenRouter behind the proxy, that is already the case and this hook is a no-op.

## Structured-output repair

Some reasoning parsers assume every completion opens inside a `<think>` block.
vLLM's `deepseek_r1` parser is one: an answer that never emits `</think>` —
exactly what guided JSON decoding produces — lands whole in `reasoning_content`
while `content` comes back empty, and the caller parses an empty string.

```yaml
litellm_settings:
  callbacks:
    - litellm_toolbelt.structured_output.proxy_handler_instance
```

When a request carries a JSON `response_format` and the answer has no content but
does have reasoning, the reasoning *is* the answer. Streamed and non-streamed
both. Everything else passes through untouched.

## Development

```bash
make install   # editable install with dev extras
make test
make lint      # ruff + mypy
```

## License

MIT
