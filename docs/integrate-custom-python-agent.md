# Integrating a Python Agent with Dunetrace

<!--dunetrace:instrument
framework: custom-python
language: python
install: pip install dunetrace
primary_symbol: dunetrace.Dunetrace
mechanism: decorator
opens_own_run: true
requires_run_context: false
target_file: the module defining your agent's entry function
target_example: `app/agent.py`, `src/agent.py`, or the function `main.py` calls
target_hints: ["chat.completions.create", "messages.create", "client.responses.create"]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded]
verify_cmd: SCENARIO=failures python packages/sdk-py/examples/decorator_agent.py
-->

> **Using TypeScript/Node.js?** See [integrate-typescript-agent.md](./integrate-typescript-agent.md).
> **Using LangChain, CrewAI, or AutoGen?** Those have dedicated guides with zero manual instrumentation — see [integrate-langchain-agent.md](./integrate-langchain-agent.md), [integrate-crewai-agent.md](./integrate-crewai-agent.md), [integrate-autogen-agent.md](./integrate-autogen-agent.md).

## Quick Start

```bash
pip install dunetrace
```

Save as `agent.py` and run it. No API key and no LLM account needed:

```python
# agent.py
from dunetrace import Dunetrace

dt = Dunetrace()  # local dev, no API key needed

@dt.tool
def web_search(query: str) -> list:
    return [f"result for {query}"]      # stand in for your real search

@dt.trace
def my_agent(question: str) -> str:
    return web_search(question)[0]

if __name__ == "__main__":
    print(my_agent("What is the capital of France?"))
    dt.shutdown()
```

```bash
python agent.py
```

Start the backend once, locally, before running this: `docker compose up -d`.

## Where this goes

Put `@dt.trace` on the **one function that represents a whole agent
turn**, and `@dt.tool` on each function the agent calls as a tool. Nothing else
moves.

Find the entry function by locating the LLM call and walking up to the nearest
function that owns a complete request:

```bash
grep -rn "chat.completions.create\|messages.create" --include=*.py .
```

In a typical layout that function lives in `app/agent.py` or `src/agent.py`. If
the LLM call sits directly inside a FastAPI or Flask route, do not decorate the
route: add the ASGI/WSGI middleware instead (see Advanced), which makes each
HTTP request one run.

If more than one function looks like an entry point, instrument the outermost
one. Two nested `dt.run()` contexts are recorded as a parent and child run, not
as one flat run, which is usually not what you want for a single agent.

## What this does

Wrap your agent's entry point with `@dt.trace` and your tool functions with `@dt.tool`. Dunetrace then auto-traces every tool and LLM call made inside that function, ships the trace to the backend, and detects structural failures (tool loops, retry storms, cost spikes, and 31 more) within ~15 seconds — no other code changes.

## Recommended usage pattern

`@dt.trace` + `@dt.tool` decorators, as shown above. No SDK calls needed inside
your function bodies. Both work on sync and async functions identically.

The two do different things, and the difference matters:

- **`@dt.trace` (and `@dt.agent`) opens the run.** It wraps the function in a
  `dt.run()` context, so it is what makes everything inside trackable. Put it on
  your agent's entry point.
- **`@dt.tool` attaches to a run that is already open.** Outside one it is a
  no-op: the function still runs, it just is not recorded.

That is the whole shape of instrumenting a Python agent. Something has to open
the run, and `@dt.trace` is the least invasive way to do it.

For a single-function agent that calls OpenAI/Anthropic directly, `@dt.agent()` plus auto-instrumentation is equally simple — see [Auto instrumentation](#auto-instrumentation) below.

## Initialization (optional)

```python
dt = Dunetrace(endpoint="http://localhost:8001")   # default — local dev, no key needed
dt.init(agent_id="my-production-agent")             # optional: fixed default agent ID
```

**Production** needs an API key.

A fresh self-hosted install has no keys, so the first one comes from the ingest
service's bootstrap endpoint. It is gated on `ADMIN_API_KEY` from your `.env`,
which is the operator's deployment secret rather than a tenant credential.
Omitting `scopes` mints an **admin** key, the one scope a fresh install cannot
obtain any other way:

```bash
curl -X POST http://localhost:8001/v1/keys \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "my-company", "org_name": "My Company", "admin_key": "'"$ADMIN_API_KEY"'"}'
```

The response carries `key` once. It is stored only as a SHA-256 hash and is
never logged, so save it now. Keep it for operators.

Then mint a narrower key for the agent itself, from the Customer API, using the
admin key you just created. An agent needs `ingest` and nothing else:

```bash
curl -X POST http://localhost:8002/v1/keys \
  -H "Authorization: Bearer $DUNETRACE_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"scopes": ["ingest"]}'
```

Give the agent that second key. An `ingest` key cannot mint keys, write
policies, or decide approvals, and `POST /v1/keys` returns 403 for any scope the
calling key does not itself hold, so an agent key cannot escalate to admin.

> **Do not INSERT into `api_keys` by hand.** Keys are verified with
> `WHERE key_hash = ... AND active = TRUE`. A row written with a plaintext
> `key` column and no `key_hash` never authenticates, and the schema migration
> marks any such row `active = FALSE`.

The Python SDK reads `DUNETRACE_API_KEY` and `DUNETRACE_API_URL` from the
environment, so the usual production wiring needs no key in code:

```python
dt = Dunetrace()   # endpoint + api_key from DUNETRACE_API_URL / DUNETRACE_API_KEY
```

Passing them explicitly also works:

```python
dt = Dunetrace(endpoint="https://your-ingest", api_key="dt_...")
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

`dt.auto_instrument()` patches `openai`, `anthropic`, `mistral`, `botocore`
(AWS Bedrock), `httpx`, `requests`, `langchain` (covers LangGraph) and `crewai`,
so every LLM/HTTP call **inside an open run** is tracked with no manual event
calls.

> **These patches only react to a run that is already open.** With the sole
> exception of `crewai`, which patches `Crew.kickoff`/`Agent.kickoff` and can
> open its own run, every target above is a silent no-op outside a run context:
> your code runs, the call succeeds, and **zero events are emitted, with no
> warning**. The run context is what `@dt.agent` / `@dt.trace` below (or an
> explicit `with dt.run(...)`, or the ASGI/WSGI middleware) provides. See
> [auto-instrumentation.md](./integrations/auto-instrumentation.md).

The decorator supplies the run, so this is tracked:

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
SCENARIO=failures python packages/sdk-py/examples/decorator_agent.py
```

This runs three agents that intentionally trigger `TOOL_LOOP`, `RETRY_STORM`, and `RAG_EMPTY_RETRIEVAL` — each should appear in the dashboard within ~15 seconds.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`Dunetrace`) actually ran, and ran **before** the first agent invocation.

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

Edit `detectors.yml` on the server, then restart **both** the detector and the ingest service — ingest serves the same file to SDKs over `GET /v1/detector-config`, so restarting only the detector leaves every agent's in-path pass on the old thresholds for the life of the ingest process:

```bash
docker compose restart detector ingest
```

No code changes:

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
