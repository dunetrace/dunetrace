# Integrating a Custom Python Agent with Dunetrace

This guide covers how to instrument a Python agent already running in production so Dunetrace can monitor it for structural failures in real-time.

---

## What Dunetrace Captures

Dunetrace detects behavioral failures in AI agents — tool loops, reasoning stalls, context bloat, and 12 more patterns — within ~15 seconds of a run completing. It never transmits raw prompts or outputs: all user content, tool arguments, and completions are SHA-256 hashed in-process before any data leaves your agent.

What does transmit: model names, token counts, latencies, tool names, finish reasons, and step counts.

---

## Prerequisites

- Dunetrace backend running and accessible (self-hosted via Docker Compose, or cloud endpoint)
- Access to the Dunetrace Postgres instance to create an API key
- Python 3.11+

---

## Step 1: Generate an API Key

API keys are stored in the `api_keys` table in Postgres. There is no UI for this yet — insert a row directly.

Connect to your Dunetrace Postgres instance and run:

```sql
INSERT INTO api_keys (key, agent_id, customer_id)
VALUES ('dt_live_<your-random-string>', 'my-production-agent', 'my-company');
```

**Field notes:**

| Field | Description |
|---|---|
| `key` | The key string your agent sends on every request. Use the `dt_live_` prefix by convention. Generate the random suffix with the command below. |
| `agent_id` | Identifies which agent this key belongs to. Must match the `agent_id` you pass to `dt.init()`. |
| `customer_id` | Your organization identifier. Used for grouping in the dashboard. |
| `active` | Defaults to `TRUE`. Set to `FALSE` to revoke a key without deleting it. |

Generate a secure random suffix:

```bash
python3 -c "import secrets; print('dt_live_' + secrets.token_hex(16))"
# Example output: dt_live_3a9f2c1d8e4b7a6f0c5d2e1b9f8a3c7d
```

To revoke a key:

```sql
UPDATE api_keys SET active = FALSE WHERE key = 'dt_live_...';
```

> **Dev mode:** When running locally with `AUTH_MODE=dev`, any key prefixed `dt_dev_` is accepted without a database lookup. Use `dt_live_` keys only for production.

---

## Step 2: Install the SDK

```bash
pip install dunetrace
```

If you want OpenTelemetry export to Tempo or Datadog:

```bash
pip install 'dunetrace[otel]'
```

---

## Step 3: Choose an Integration Path

Pick the path that fits your agent's architecture:

| Path | Best for | Code change |
|---|---|---|
| `@dt.agent()` decorator | Single-function agents | Minimal |
| ASGI/WSGI middleware | FastAPI / Flask / Django | One line |
| `dt.run()` context manager | Full manual control | Moderate |
| LangChain callback | LangChain / LangGraph agents | One line |
| OTel receiver | Already instrumented with OpenLLMetry | Zero to agent |

---

## Path A: Decorator (Recommended for most agents)

Wrap your agent's entry point with `@dt.agent()`. Calls to OpenAI and Anthropic are captured automatically via `auto_instrument()`.

```python
from dunetrace import Dunetrace

dt = Dunetrace(
    endpoint="https://your-dunetrace-ingest",  # or http://localhost:8001 locally
    api_key="dt_live_...",
)
dt.init(agent_id="my-production-agent")
dt.auto_instrument()  # patches openai, anthropic, httpx, requests

@dt.agent(model="gpt-4o", tools=["web_search", "calculator"])
def run_agent(query: str) -> str:
    # OpenAI/Anthropic calls here are auto-tracked
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content

# Call normally — Dunetrace wraps it transparently
result = run_agent("What is the capital of France?")

# On process exit
dt.shutdown()
```

For async agents:

```python
@dt.agent(model="gpt-4o", tools=["web_search"])
async def run_agent_async(query: str) -> str:
    response = await async_openai_client.chat.completions.create(...)
    return response.choices[0].message.content
```

---

## Path B: FastAPI / ASGI Middleware

Add one middleware line. Each HTTP request becomes one agent run.

```python
from dunetrace import Dunetrace, DunetraceASGIMiddleware
from dunetrace.context import get_current_run
from fastapi import FastAPI

dt = Dunetrace(endpoint="https://your-dunetrace-ingest", api_key="dt_live_...")
dt.auto_instrument()

app = FastAPI()
app.add_middleware(
    DunetraceASGIMiddleware,
    dt=dt,
    agent_id="my-api-agent",
    model="gpt-4o",
)

@app.post("/chat")
async def chat(query: str):
    run = get_current_run()  # opened automatically by middleware

    # Instrument non-LLM steps manually
    run.tool_called("db_lookup", {"query": query})
    result = await db.get(query)
    run.tool_responded("db_lookup", success=True, output_length=len(str(result)))

    return result
```

For Flask / Django, use `DunetraceWSGIMiddleware` instead.

---

## Path C: Manual `dt.run()` Context Manager

Use this when you need full control over every event.

```python
from dunetrace import Dunetrace

dt = Dunetrace(endpoint="https://your-dunetrace-ingest", api_key="dt_live_...")
dt.init(agent_id="my-production-agent")

TOOLS = ["web_search", "calculator", "code_runner"]

with dt.run("my-production-agent", user_input=query, model="gpt-4o", tools=TOOLS) as run:

    # Before each LLM call
    run.llm_called("gpt-4o", prompt_tokens=150)

    response = call_llm(query)

    # After LLM responds
    run.llm_responded(
        completion_tokens=30,
        latency_ms=820,
        finish_reason="tool_calls",
        output_length=len(response),
    )

    # Before each tool call
    run.tool_called("web_search", {"query": query})
    result = web_search(query)

    # After tool responds
    run.tool_responded("web_search", success=True, output_length=len(result), latency_ms=300)

    # RAG/retrieval (if applicable)
    run.retrieval_called(index_name="product-docs", query_hash="abc123")
    docs = retrieve(query)
    run.retrieval_responded(index_name="product-docs", result_count=len(docs), top_score=0.91, latency_ms=45)

    # Mark run complete
    run.final_answer()

dt.shutdown()
```

### Full RunContext API reference

```python
# LLM events
run.llm_called(model, prompt_tokens)
run.llm_responded(completion_tokens, latency_ms, finish_reason, output_length)

# Tool events
run.tool_called(tool_name, args)           # args dict gets hashed in-process
run.tool_responded(tool_name, success, output_length, latency_ms, error)

# Retrieval/RAG events
run.retrieval_called(index_name, query_hash)
run.retrieval_responded(index_name, result_count, top_score, latency_ms)

# Infrastructure signals (no step counter increment)
run.external_signal("rate_limit", source="openai")
run.external_signal("cache_miss", source="redis", key_prefix="emb:")

# Terminal marker — always call this when agent produces final output
run.final_answer()
```

---

## Path D: LangChain / LangGraph

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace(endpoint="https://your-dunetrace-ingest", api_key="dt_live_...")
dt.init(agent_id="my-langchain-agent")

callback = DunetraceCallbackHandler(dt, agent_id="my-langchain-agent")

result = agent.invoke(
    {"messages": [("human", query)]},
    config={"callbacks": [callback]},
)
```

---

## Path E: OpenTelemetry (Already instrumented with OpenLLMetry)

If your agent already emits `gen_ai.*` OTel spans via OpenLLMetry, attach the receiver to your existing TracerProvider:

```python
from dunetrace import Dunetrace
from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver

dt = Dunetrace(endpoint="https://your-dunetrace-ingest", api_key="dt_live_...")
DunetraceOTelReceiver.attach(tracer_provider, dt, agent_id="my-agent")
```

No changes to agent code required.

---

## Step 4: Shutdown Gracefully

Always call `dt.shutdown()` before your process exits. This drains the background flush thread and ensures all pending events are sent.

```python
import atexit

dt = Dunetrace(...)
atexit.register(dt.shutdown)
```

Or with a timeout:

```python
dt.shutdown(timeout=5)  # waits up to 5 seconds
```

---

## Step 5: Configure Alerts

Set these environment variables on your Dunetrace server before restarting:

```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#agent-alerts
SLACK_MIN_SEVERITY=HIGH        # HIGH | MEDIUM | CRITICAL
DASHBOARD_URL=https://your-dashboard-url
```

Alerts fire within ~15 seconds of a run completing.

---

## Step 6: Tune Detectors (Optional)

Dunetrace ships 15 built-in detectors with defaults that work for most agents. If your agent legitimately calls the same tool many times (e.g., a search agent), loosen the thresholds in `detectors.yml` on the server:

```yaml
# detectors.yml
default:
  tool_loop:
    threshold: 3        # flag if same tool called ≥3x in 5-call window

my-production-agent:    # per-agent-id overrides
  tool_loop:
    threshold: 6        # search agents call tools more
  context_bloat:
    growth_factor: 4.0  # allow more token growth
  reasoning_stall:
    ratio_threshold: 6  # higher tolerance for LLM-heavy agents
```

After editing, restart the detector service:

```bash
docker compose restart detector
```

---

## Step 7: Verify the Integration

Run your agent once, then check:

1. **Dashboard** (`http://your-dashboard:3000`) — the run should appear within 15 seconds
2. **Runs API** — `GET http://your-ingest:8002/v1/runs?agent_id=my-production-agent`
3. **Alerts** — trigger a known failure pattern (e.g., call the same tool 4+ times) and confirm Slack fires

For local testing before pointing at production:

```python
dt = Dunetrace(endpoint="http://localhost:8001", api_key=None)  # dev mode, no key needed
```

---

## Detector Reference

These run automatically on every completed run. No configuration needed to enable them.

| Detector | Trigger | Severity |
|---|---|---|
| `TOOL_LOOP` | Same tool called ≥3x in a 5-call window | HIGH |
| `TOOL_THRASHING` | Agent alternates between exactly two tools | HIGH |
| `RETRY_STORM` | Same tool fails 3+ consecutive times | HIGH |
| `LLM_TRUNCATION_LOOP` | `finish_reason=length` ≥2 times | HIGH |
| `EMPTY_LLM_RESPONSE` | Zero-length output with `finish_reason=stop` | HIGH |
| `CASCADING_TOOL_FAILURE` | 3+ consecutive failures across 2+ tools | HIGH |
| `TOOL_AVOIDANCE` | Final answer without using available tools | MEDIUM |
| `GOAL_ABANDONMENT` | Tool use stops, then 4+ LLM-only calls | MEDIUM |
| `RAG_EMPTY_RETRIEVAL` | 0 retrieval results but agent answered | MEDIUM |
| `CONTEXT_BLOAT` | Prompt tokens grow 3× from first to last call | MEDIUM |
| `STEP_COUNT_INFLATION` | Run takes >2× baseline steps for this agent | MEDIUM |
| `FIRST_STEP_FAILURE` | Error or empty output at step ≤2 | MEDIUM |
| `REASONING_STALL` | LLM-to-tool-call ratio ≥4× | MEDIUM |
| `SLOW_STEP` | Tool >15s or LLM >30s | MEDIUM/HIGH |
| `PROMPT_INJECTION_SIGNAL` | Input matches injection/jailbreak patterns | CRITICAL |

---

## Privacy Summary

| Data | Transmitted? |
|---|---|
| User input text | No — SHA-256 hash only |
| LLM prompts and completions | No — SHA-256 hash only |
| Tool arguments | No — SHA-256 hash only |
| Tool outputs | No — SHA-256 hash only |
| Model names (`gpt-4o`, `claude-3-5-sonnet`) | Yes |
| Tool names (`web_search`, `calculator`) | Yes |
| Token counts | Yes |
| Latencies | Yes |
| Finish reasons | Yes |
| HTTP status codes | Yes |

Hashing happens in-process. Raw content never leaves your agent.

---

## Quick-Start Checklist

- [ ] Generate an API key via `INSERT INTO api_keys ...` in Postgres
- [ ] `pip install dunetrace`
- [ ] Instantiate `Dunetrace(endpoint=..., api_key=...)`
- [ ] Call `dt.init(agent_id="...")` and `dt.auto_instrument()`
- [ ] Wrap agent entry point (decorator, middleware, or `dt.run()`)
- [ ] Add manual `run.tool_called()` / `run.tool_responded()` for non-LLM steps
- [ ] Call `dt.shutdown()` on process exit (or register with `atexit`)
- [ ] Set `SLACK_WEBHOOK_URL` on the server
- [ ] Tune `detectors.yml` thresholds if needed
- [ ] Test locally against `http://localhost:8001` before production rollout

---

## What happens after a run completes

Within ~15 seconds of each run, Dunetrace:

1. **Detects** — 15 structural detectors run against the reconstructed run state
2. **Explains** — each signal produces a plain-English title, cause, and fix using deterministic templates (no LLM)
3. **Trends** — the dashboard Health Record panel shows failure rate per failure type over 30 days, a 7-day systemic pattern flag (≥10% of runs affected), and a sparkline showing rate over time
4. **Alerts** — Slack messages include a one-line rate context: "First occurrence", "5/20 runs affected (25%)", or "Systemic pattern — 8/12 runs affected (67%)"

The Health Record is available at `GET /v1/agents/{id}/insights` — it returns `failure_rates` (daily affected/total per failure type) and `systemic_patterns` (7-day rate + `is_systemic` flag).
