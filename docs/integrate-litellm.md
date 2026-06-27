# Integrating LiteLLM with Dunetrace

LiteLLM can route OpenAI, Anthropic, Gemini, Cohere, Mistral, and 100+ other providers behind one OpenAI-compatible interface. When you call LiteLLM through the OpenAI-compatible proxy, Dunetrace can monitor your agent with the existing OpenAI auto-instrumentation patch.

---

## How It Works

Dunetrace patches the OpenAI Python client's `chat.completions.create()` method. LiteLLM Proxy exposes an OpenAI-compatible `/chat/completions` endpoint, so an OpenAI client pointed at LiteLLM emits Dunetrace events automatically inside a `@dt.agent()` or `dt.run()` context.

| LiteLLM call | Dunetrace event |
|---|---|
| Agent function starts | `RUN_STARTED` |
| `OpenAI(...).chat.completions.create(...)` against LiteLLM Proxy | `LLM_CALLED` → `LLM_RESPONDED` |
| Agent returns a final answer | `RUN_COMPLETED` |
| Agent raises an exception | `RUN_ERRORED` |

Dunetrace records model names, token counts, latency, finish reason, and output length. Raw prompts and completions are hashed in-process before any network call.

> **Note:** This zero-instrumentation path uses LiteLLM's OpenAI-compatible proxy. If you call the direct `litellm.completion()` Python function instead of the OpenAI client, wrap the agent with `dt.run()` and emit `run.llm_called()` / `run.llm_responded()` manually.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- LiteLLM Proxy running locally or remotely

> **Local dev — no API key needed.** The Dunetrace backend accepts requests without an API key when running locally. API keys are only required for production deployments.

---

## Step 1: Install Dependencies

```bash
pip install dunetrace openai
```

If you are running LiteLLM Proxy yourself:

```bash
pip install 'litellm[proxy]'
```

---

## Step 2: Start LiteLLM Proxy

Create a minimal LiteLLM config:

```yaml
# litellm.yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
```

Start the proxy:

```bash
litellm --config litellm.yaml --port 4000
```

The proxy now accepts OpenAI-compatible requests at `http://localhost:4000/v1`.

---

## Step 3: Add Dunetrace Auto-Instrumentation

Call `dt.init()` and `dt.auto_instrument()` once at startup, then wrap your agent entry point with `@dt.agent()`.

```python
import atexit
import os

from openai import OpenAI
from dunetrace import Dunetrace

client = OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY", "no-key""),
    base_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"),
)

dt = Dunetrace(endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"))
dt.init(agent_id="litellm-agent")
dt.auto_instrument(["openai"])
atexit.register(dt.shutdown)

@dt.agent(model="gpt-4o-mini")
def answer(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or ""

print(answer("What is the capital of France?"))
```

No manual event calls are needed. The OpenAI client call is patched before it reaches LiteLLM, and Dunetrace hashes the prompt/output metadata locally.

---

## Step 4: Multiple Providers Behind One Proxy

Keep the code unchanged and switch models through LiteLLM model names:

```python
@dt.agent(model="claude-haiku")
def summarize(text: str) -> str:
    response = client.chat.completions.create(
        model="claude-haiku",
        messages=[
            {"role": "system", "content": "Summarize in one sentence."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content or ""
```

Dunetrace records the model name passed to the OpenAI-compatible call, so dashboards and detectors can distinguish traffic routed to different providers.

---

## Step 5: Verify the Integration

Run your agent once, then check:

1. **Dashboard** (`http://localhost:3000`) — the agent should appear within 15 seconds.
2. **Runs API** — `GET http://localhost:8002/v1/runs?agent_id=litellm-agent`.
3. **Failure scenarios** — intentionally repeat the same tool call or return an empty response to confirm detectors fire.

For local debugging, enable verbose SDK logs:

```python
dt = Dunetrace(endpoint="http://localhost:8001", debug=True)
```

---

## Troubleshooting

**No runs appear**
- Confirm `dt.auto_instrument(["openai"])` runs before the first OpenAI client call.
- Confirm the OpenAI client is created with `base_url="http://localhost:4000/v1"` or your LiteLLM proxy URL.
- Confirm `dt.shutdown()` is called before the process exits, or register it with `atexit`.

**LiteLLM returns 401 or provider errors**
- Check the LiteLLM proxy config and provider API keys first. Dunetrace records the failed LLM response but does not manage LiteLLM credentials.

**Direct `litellm.completion()` calls are not shown as LLM events**
- Use the OpenAI-compatible proxy path above for zero-instrumentation monitoring.
- Or use `dt.run()` and emit manual LLM events around direct `litellm.completion()` calls.

**Token counts missing**
- Token usage comes from the OpenAI-compatible response metadata returned by LiteLLM. If the upstream provider omits usage, detectors still run on step counts, latency, and finish reasons.
