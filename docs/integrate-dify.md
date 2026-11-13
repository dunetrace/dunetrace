# Integrating a Dify Agent with Dunetrace

<!--dunetrace:instrument
framework: dify
language: python
install: pip install dunetrace dify-client-python
primary_symbol: dunetrace.Dunetrace.run
mechanism: manual
opens_own_run: false
requires_run_context: true
target_file: the function wrapping your Dify API call
target_example: `app/dify_client.py`, `main.py`
target_hints: ["chat_messages(", "dify_client", "api.dify.ai"]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded]
verify_cmd: python main.py  # then check the dashboard
-->

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

## Where this goes

`dt.run()` goes around the Dify API call, in whichever function your
application uses to talk to Dify.

```bash
grep -rn "chat_messages(\|api.dify.ai" --include=*.py .
```

Dify executes on Dify's own servers, so there is no local client to
auto-instrument and no visibility into Dify's internal tool calls. The whole
round trip is recorded as one LLM event.

Consider [Langdock-style OTLP ingestion](./integrate-langdock.md) instead if
your Dify deployment can export OpenTelemetry.

## What this does

Dify agents run on Dify's own server, so there's no local LLM client to auto-instrument — Dunetrace instead wraps the API call itself in `dt.run()` and treats the whole request/response round-trip as one LLM event. That's enough to catch latency spikes, empty responses, and first-step failures on the Dify side, even though Dify's internal tool executions aren't visible.

## Verification

Run your script once, then check the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration attaches to a run that is already open. Outside one it emits nothing, silently. Confirm your top-level call is inside `dt.run(...)` (or a decorator that opens one).

2. **Did the process flush?** Events ship from a background thread. Call
   `dt.shutdown()` before the process exits, or the buffer dies with it.

3. **Is the backend reachable?** Both should return `{"status":"ok",...}`:

   ```bash
   curl -s localhost:8001/ready   # ingest
   curl -s localhost:8002/ready   # customer API
   ```

4. **Did anything land?** If your agent id is listed here, instrumentation is
   working and the problem is downstream:

   ```bash
   curl -s localhost:8002/v1/agents
   curl -s "localhost:8002/v1/agents/<your-agent-id>/runs?limit=5"
   ```

5. **Turn on debug logging.** `Dunetrace(debug=True)` logs every event as it is buffered and
   every batch as it ships.

**Runs appear but no signals?** That is usually correct, not a fault. The
detector polls every 5 seconds, and a healthy run produces no signals. Several
detectors also need cross-run baselines and stay dormant until the agent has
run history. Check `docker compose logs detector` if you expected one.

---

## Advanced (optional)

### What's captured

Overall latency, success/failure of the API call, response length, and token counts (from Dify's response metadata) — the whole Dify workflow as one step. Not captured: Dify's internal tool executions, unless you parse `agent_thoughts`/`tool_calls` from streaming mode yourself and emit `run.tool_called()`/`run.tool_responded()`.

### Most relevant detectors for this integration

`SLOW_STEP` (the Dify call itself takes too long), `EMPTY_LLM_RESPONSE` (often means an internal Dify workflow failure), `FIRST_STEP_FAILURE` (Dify returned an error or empty output early — wrap `dify_client.chat_messages()` in a try/except to catch network errors as `RUN_ERRORED` too). If you do manually instrument Dify's internal tools in streaming mode, `TOOL_LOOP`/`RETRY_STORM`/`CASCADING_TOOL_FAILURE` become relevant as well.
