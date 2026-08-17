# Integrating LiteLLM with Dunetrace

## Quick Start

```bash
pip install dunetrace openai
```

```python
from openai import OpenAI
from dunetrace import Dunetrace

client = OpenAI(base_url="http://localhost:4000/v1", api_key="no-key")   # your LiteLLM Proxy

dt = Dunetrace()   # local dev, no API key needed
dt.init(agent_id="my-agent")
dt.auto_instrument(["openai"])

@dt.agent(model="gpt-4o-mini")
def answer(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or ""

print(answer("What is the capital of France?"))
```

Requires a running LiteLLM Proxy (`litellm --config litellm.yaml --port 4000`) and the Dunetrace backend (`docker compose up -d`).

## What this does

LiteLLM Proxy exposes an OpenAI-compatible endpoint. Dunetrace already patches the OpenAI Python client (via `dt.auto_instrument(["openai"])`), so any OpenAI client pointed at your LiteLLM Proxy gets auto-tracked with zero LiteLLM-specific code — model name, tokens, latency, and raw prompt/completion are all captured. Switching models is just a name change (`model="claude-haiku"`, etc.) — Dunetrace records whatever model string you pass.

If you call `litellm.completion()` directly instead of going through an OpenAI client, use `dt.run()` and emit `run.llm_called()`/`run.llm_responded()` manually — this auto-instrumentation path only covers the OpenAI-compatible HTTP interface.

## Verification

Run your agent once, then check the dashboard at `http://localhost:3000` — it should appear within ~15 seconds. Repeat the same tool call or return an empty response to confirm detectors fire.

---

## Advanced (optional)

### Troubleshooting

- **No runs appear** — confirm `dt.auto_instrument(["openai"])` runs before the first call, and the client's `base_url` points at your LiteLLM Proxy
- **LiteLLM returns 401 / provider errors** — check the proxy config and provider keys; Dunetrace doesn't manage LiteLLM credentials
- **Direct `litellm.completion()` calls aren't tracked** — use the OpenAI-compatible proxy path above, or instrument manually with `dt.run()`
- **Token counts missing** — come from the OpenAI-compatible response; if the upstream provider omits usage, detectors still run on step counts and latency
