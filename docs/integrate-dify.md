# Integrating a Dify Agent with Dunetrace

## Quick Start

```bash
pip install dunetrace dify-client-python
```

```python
import time
from dunetrace import Dunetrace
from dify_client import Client, models

dt = Dunetrace()   # local dev, no API key needed
dify_client = Client(api_key="your-dify-api-key", api_base="https://api.dify.ai/v1")

def chat_with_dify(query: str, user_id: str) -> str:
    with dt.run("my-dify-agent", user_input=query, model="dify-workflow") as run:
        req = models.ChatRequest(query=query, inputs={}, user=user_id, response_mode=models.ResponseMode.BLOCKING)
        run.llm_called("dify-workflow")
        t0 = time.monotonic()
        res = dify_client.chat_messages(req, timeout=60.0)
        run.llm_responded(
            finish_reason="stop",
            output_length=len(res.answer or ""),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        run.final_answer()
        return res.answer or ""

chat_with_dify("What is the capital of France?", "user-1")
dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

Dify agents run on Dify's own server, so there's no local LLM client to auto-instrument — Dunetrace instead wraps the API call itself in `dt.run()` and treats the whole request/response round-trip as one LLM event. That's enough to catch latency spikes, empty responses, and first-step failures on the Dify side, even though Dify's internal tool executions aren't visible.

## Verification

Run your script once, then check the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds.

---

## Advanced (optional)

### What's captured

Overall latency, success/failure of the API call, response length, and token counts (from Dify's response metadata) — the whole Dify workflow as one step. Not captured: Dify's internal tool executions, unless you parse `agent_thoughts`/`tool_calls` from streaming mode yourself and emit `run.tool_called()`/`run.tool_responded()`.

### Most relevant detectors for this integration

`SLOW_STEP` (the Dify call itself takes too long), `EMPTY_LLM_RESPONSE` (often means an internal Dify workflow failure), `FIRST_STEP_FAILURE` (Dify returned an error or empty output early — wrap `dify_client.chat_messages()` in a try/except to catch network errors as `RUN_ERRORED` too). If you do manually instrument Dify's internal tools in streaming mode, `TOOL_LOOP`/`RETRY_STORM`/`CASCADING_TOOL_FAILURE` become relevant as well.
