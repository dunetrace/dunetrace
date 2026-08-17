# Integrating a Python Agent with Dunetrace

> **Using TypeScript/Node.js?** See [integrate-typescript-agent.md](./integrate-typescript-agent.md).
> **Using LangChain, CrewAI, or AutoGen?** Those have dedicated guides with zero manual instrumentation — see [integrate-langchain-agent.md](./integrate-langchain-agent.md), [integrate-crewai-agent.md](./integrate-crewai-agent.md), [integrate-autogen-agent.md](./integrate-autogen-agent.md).

## Quick Start

```bash
pip install dunetrace
```

```python
from dunetrace import Dunetrace

dt = Dunetrace()  # local dev, no API key needed

@dt.tool
def web_search(query: str) -> list:
    return search_api(query)

@dt.trace
def my_agent(question: str) -> str:
    return web_search(question)[0]

my_agent("What is the capital of France?")
dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

Wrap your agent's entry point with `@dt.trace` and your tool functions with `@dt.tool`. Dunetrace then auto-traces every tool and LLM call made inside that function, ships the trace to the backend, and detects structural failures (tool loops, retry storms, cost spikes, and 26 more) within ~15 seconds — no other code changes.

## Recommended usage pattern

`@dt.trace` + `@dt.tool` decorators, as shown above. No SDK calls needed inside your function bodies. Works on sync and async functions identically, and is a no-op outside a Dunetrace run (your code still runs normally).

For a single-function agent that calls OpenAI/Anthropic directly, `@dt.agent()` plus auto-instrumentation is equally simple — see [Auto instrumentation](#auto-instrumentation) below.

## Initialization (optional)

```python
dt = Dunetrace(endpoint="http://localhost:8001")   # default — local dev, no key needed
dt.init(agent_id="my-production-agent")             # optional: fixed default agent ID
```

**Production** needs an API key:

```python
dt = Dunetrace(endpoint="https://your-ingest", api_key="dt_live_...")
```

Generate the first key directly in Postgres (there's no UI for this yet):

```sql
INSERT INTO organizations (id, name) VALUES ('my-company', 'My Company') ON CONFLICT (id) DO NOTHING;
INSERT INTO api_keys (key, org_id) VALUES ('dt_live_<random-string>', 'my-company');
```

```bash
python3 -c "import secrets; print('dt_live_' + secrets.token_hex(16))"
```

Events are shipped from a background thread, so anything still buffered when the
process exits needs flushing. The SDK registers an `atexit` hook that does this
for you, which is what makes short-lived scripts, CLIs and one-shot jobs work
without ceremony.

Still call `dt.shutdown()` explicitly where you can — it flushes at a point you
control, with a full timeout rather than the shorter at-exit one, and surfaces
delivery problems while your process is still alive to log them. Calling it
cancels the at-exit hook, so events are never sent twice.

Set `DUNETRACE_ATEXIT_TIMEOUT` to change how long the at-exit flush may block
interpreter shutdown (seconds, default `2`), or to `0` to disable it entirely.

## Auto instrumentation

`dt.auto_instrument()` patches `openai`, `anthropic`, `mistral`, `httpx`, and `requests` so every LLM/HTTP call inside a run is tracked with no manual event calls:

```python
dt.init(agent_id="my-production-agent")
dt.auto_instrument()

@dt.agent(model="gpt-4o", tools=["web_search"])
def run_agent(query: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content
```

Uninstalled frameworks are silently skipped. Restrict to specific clients with `dt.auto_instrument(["openai", "anthropic"])`.

## Verification

Run your agent once, then check:

1. **Dashboard** — `http://localhost:3000` — the run appears within ~15 seconds
2. **Runs API** — `GET http://localhost:8002/v1/runs?agent_id=<your-agent-id>`

To confirm detectors and alerts fire end-to-end:

```bash
SCENARIO=failures python examples/decorator_agent.py
```

This runs three agents that intentionally trigger `TOOL_LOOP`, `RETRY_STORM`, and `RAG_EMPTY_RETRIEVAL` — each should appear in the dashboard within ~15 seconds.

---

## Advanced (optional)

### FastAPI / ASGI middleware

One line — each HTTP request becomes one agent run.

```python
from dunetrace import Dunetrace, DunetraceASGIMiddleware
from fastapi import FastAPI

dt = Dunetrace()
dt.auto_instrument()
app = FastAPI()
app.add_middleware(DunetraceASGIMiddleware, dt=dt, agent_id="my-api-agent", model="gpt-4o")
```

Flask / Django: use `DunetraceWSGIMiddleware` the same way.

### Manual `dt.run()` context manager

Use this for full control over every event — useful when decorators/middleware don't fit your architecture:

```python
with dt.run("my-agent", user_input=query, model="gpt-4o", tools=["web_search"]) as run:
    run.llm_called("gpt-4o", prompt_tokens=150)
    response = call_llm(query)
    run.llm_responded(completion_tokens=30, latency_ms=820, finish_reason="stop")

    run.tool_called("web_search", {"query": query})
    result = web_search(query)
    run.tool_responded("web_search", success=True, output_length=len(result))

    run.final_answer()
```

Full `RunContext` API: `llm_called` / `llm_responded`, `tool_called` / `tool_responded`, `retrieval_called` / `retrieval_responded`, `external_signal`, `final_answer`.

### `get_current_run()`

Access the active run from any helper without threading it through your call stack:

```python
from dunetrace import get_current_run

def some_helper():
    run = get_current_run()
    if run:
        run.tool_called("cache_lookup")
```

### Already instrumented with OpenTelemetry / OpenLLMetry

```python
from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver
DunetraceOTelReceiver.attach(tracer_provider, dt, agent_id="my-agent")
```

No agent code changes required.

### Grafana / Loki (no HTTP ingest)

```python
dt = Dunetrace(emit_as_json=True)
```

Writes each event as an NDJSON line to stdout instead of (or alongside) HTTP ingest.

### Tuning detectors

Edit `detectors.yml` on the server, then `docker compose restart detector` — no code changes:

```yaml
default:
  tool_loop:
    threshold: 3
my-production-agent:       # per-agent-id override
  tool_loop:
    threshold: 6
```

### Configuring alerts

```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_MIN_SEVERITY=HIGH
```

### Data handling

User input, tool arguments, and completions are sent to the backend over TLS as-is — content-aware detectors need to see what the agent actually said and did. Self-host for an air-gapped deployment. Full detector list: [docs/detectors.md](detectors.md).
