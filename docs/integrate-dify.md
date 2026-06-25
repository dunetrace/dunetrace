# Integrating a Dify Agent with Dunetrace

This guide covers adding Dunetrace monitoring to a [Dify](https://github.com/langgenius/dify) agent running in production. Since Dify agents run on a separate server (or cloud), you interact with them via the Dify REST API or Python SDK. You can monitor these interactions by wrapping your API calls with Dunetrace's `@dt.agent()` decorator or `dt.run()` context manager.

---

## How It Works

By wrapping your Dify API calls, Dunetrace automatically captures the input, output, and execution metadata of the interaction:

| Dify Event | Dunetrace Event |
|---|---|
| Client API Call | `RUN_STARTED` |
| Client receives response | `RUN_COMPLETED` (with latency and output length) |
| API throws error | `RUN_ERRORED` |

> **Note:** Because Dify manages its own internal tool executions on the server-side, standard client wrapping captures the entire Dify workflow as a single "black box" step.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- Dify API Key and API Base URL

> **Local dev — no API key needed.** The Dunetrace backend accepts requests without any API key when running locally. API keys are only required for production deployments — see [the main integration guide](./integrate-custom-python-agent.md#step-1-generate-an-api-key-production-only).

---

## Step 1: Install Dependencies

Install Dunetrace and the official typed Dify Python SDK:

```bash
pip install dunetrace dify-client-python
```

---

## Step 2: Wrap the Dify Call

You can use the `@dt.agent()` decorator on the function that interacts with your Dify agent.

```python
from dunetrace import Dunetrace
from dify_client import Client, models

# Initialize Dunetrace
dt = Dunetrace(endpoint="http://localhost:8001")
dt.init(agent_id="my-dify-agent")

# Initialize Dify Client
dify_client = Client(
    api_key="your-dify-api-key", 
    api_base="https://api.dify.ai/v1"
)

@dt.agent(model="dify-workflow", tools=["dify_internal_tools"])
def chat_with_dify(query: str, user_id: str) -> str:
    req = models.ChatRequest(
        query=query,
        inputs={},
        user=user_id,
        response_mode=models.ResponseMode.BLOCKING,
    )
    res = dify_client.chat_messages(req)
    return res.answer
```

---

## Complete Example

```python
import os
import uuid
import atexit
from dunetrace import Dunetrace
from dify_client import Client, models

# 1. Setup Dunetrace
dt = Dunetrace(
    endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"),
    api_key=os.environ.get("DUNETRACE_API_KEY", ""),  # omit for local dev
)
dt.init(agent_id="my-dify-agent")
atexit.register(dt.shutdown)

# 2. Setup Dify
dify_client = Client(
    api_key=os.environ.get("DIFY_API_KEY", "your-dify-api-key"),
    api_base=os.environ.get("DIFY_API_BASE", "https://api.dify.ai/v1")
)

# 3. Wrap the interaction
@dt.agent(model="dify-workflow", tools=["dify_internal"])
def run_dify_agent(query: str, user_id: str) -> str:
    req = models.ChatRequest(
        query=query,
        inputs={},
        user=user_id,
        response_mode=models.ResponseMode.BLOCKING,
    )
    res = dify_client.chat_messages(req, timeout=60.0)
    return res.answer

if __name__ == "__main__":
    user_session = str(uuid.uuid4())
    answer = run_dify_agent("What is the capital of France?", user_session)
    print("Dify response:", answer)
```

---

## What Is and Isn't Captured

**Captured automatically:**
- Overall agent latency
- Success/Failure status of the Dify API call
- The input/output length of the final response

**Not captured:**
- Internal Dify tool executions (unless you use streaming mode and manually parse the `agent_thoughts` or `tool_calls` into `run.tool_called()` using `dt.run()`)
- Token counts (unless manually extracted from Dify's API response metadata and fed into Dunetrace)

---

## Relevant Detectors for Dify

When tracking Dify agents as a single API call, the following Dunetrace detectors are most relevant:

| Detector | Relevance for Dify |
|---|---|
| `SLOW_STEP` | Fires if the Dify API takes too long to respond (e.g., due to an internal Dify tool loop or slow LLM). |
| `EMPTY_LLM_RESPONSE` | Detects if Dify returns an empty answer, which often indicates an internal workflow failure. |
| `FIRST_STEP_FAILURE` | Triggers if the Dify API request fails entirely (e.g., timeout, 500 error, or invalid credentials). |

If you manually instrument Dify's internal tool executions using `dt.run()` in streaming mode, detectors like `TOOL_LOOP`, `RETRY_STORM`, and `CASCADING_TOOL_FAILURE` become highly relevant for catching infinite tool loops on the Dify server.

---

## Verify the Integration

Run your Python script once, then check:

1. **Dashboard** (`http://your-dashboard:3000`) — the run should appear within 15 seconds
2. **Runs API** — `GET http://your-ingest:8002/v1/runs?agent_id=my-dify-agent`
