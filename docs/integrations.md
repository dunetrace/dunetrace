# Integrations

---

## TypeScript / Node.js

No SDK required — send events directly to the ingest HTTP endpoint from any TypeScript or JavaScript agent. Node 18+ (built-in `fetch`).

### Environment variables

```bash
DUNETRACE_ENDPOINT=http://localhost:8001   # or your hosted ingest URL
DUNETRACE_API_KEY=                         # empty for self-hosted, set for cloud
```

### Minimal client

```typescript
import { randomUUID } from "crypto";

const ENDPOINT = process.env.DUNETRACE_ENDPOINT ?? "http://localhost:8001";
const API_KEY  = process.env.DUNETRACE_API_KEY  ?? "";

type EventType =
  | "run.started" | "run.completed" | "run.errored"
  | "llm.called"  | "llm.responded"
  | "tool.called" | "tool.responded"
  | "retrieval.called" | "retrieval.responded"
  | "external.signal";

interface AgentEvent {
  event_type:    EventType;
  run_id:        string;
  agent_id:      string;
  agent_version: string;
  step_index:    number;
  timestamp:     number;             // Unix seconds (float)
  payload:       Record<string, unknown>;
  parent_run_id?: string | null;
}

class DunetraceRun {
  readonly runId = randomUUID();
  private step   = 0;
  private events: AgentEvent[] = [];

  constructor(private readonly agentId: string, private readonly version: string) {}

  private emit(type: EventType, payload: Record<string, unknown>) {
    this.step++;
    this.events.push({
      event_type: type, run_id: this.runId, agent_id: this.agentId,
      agent_version: this.version, step_index: this.step,
      timestamp: Date.now() / 1000, payload,
    });
  }

  llmCalled(model: string, promptTokens = 0) {
    this.emit("llm.called", { model, prompt_tokens: promptTokens });
  }
  llmResponded(opts: { completionTokens?: number; latencyMs?: number; finishReason?: string }) {
    this.emit("llm.responded", {
      completion_tokens: opts.completionTokens ?? 0,
      latency_ms: opts.latencyMs ?? 0,
      finish_reason: opts.finishReason ?? "stop",
    });
  }
  toolCalled(toolName: string, args: Record<string, unknown> = {}) {
    this.emit("tool.called", { tool_name: toolName, args_hash: JSON.stringify(args) });
  }
  toolResponded(toolName: string, success: boolean, outputLength = 0) {
    this.emit("tool.responded", { tool_name: toolName, success, output_length: outputLength });
  }
  finalAnswer() {
    this.emit("run.completed", { exit_reason: "final_answer", total_steps: this.step });
  }
  getEvents() { return this.events; }
}

class Dunetrace {
  async run(
    agentId: string,
    opts: { model?: string; tools?: string[] },
    fn: (run: DunetraceRun) => Promise<void>,
  ) {
    const version = opts.model ?? "unknown";
    const run     = new DunetraceRun(agentId, version);
    const start: AgentEvent = {
      event_type: "run.started", run_id: run.runId, agent_id: agentId,
      agent_version: version, step_index: 0, timestamp: Date.now() / 1000,
      payload: { model: opts.model ?? "unknown", tools: opts.tools ?? [] },
    };
    try {
      await fn(run);
    } catch (err) {
      await this.flush(agentId, [start, ...run.getEvents(), {
        event_type: "run.errored", run_id: run.runId, agent_id: agentId,
        agent_version: version, step_index: 999, timestamp: Date.now() / 1000,
        payload: { error_type: (err as Error).name },
      }]);
      throw err;
    }
    await this.flush(agentId, [start, ...run.getEvents()]);
  }

  private async flush(agentId: string, events: AgentEvent[]) {
    await fetch(`${ENDPOINT}/v1/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: API_KEY, agent_id: agentId, events }),
    });
  }
}
```

### Usage

```typescript
const dt = new Dunetrace();

await dt.run("my-ts-agent", { model: "gpt-4o", tools: ["web_search"] }, async (run) => {
  run.llmCalled("gpt-4o", 150);
  // ... call your LLM here ...
  run.llmResponded({ completionTokens: 30, latencyMs: 120, finishReason: "tool_calls" });

  run.toolCalled("web_search", { query: "latest AI news" });
  // ... call your tool here ...
  run.toolResponded("web_search", true, 512);

  run.llmCalled("gpt-4o", 400);
  run.llmResponded({ completionTokens: 120, latencyMs: 95, finishReason: "stop" });
  run.finalAnswer();
});
```

Runs appear in the dashboard alongside Python agents under the same `agent_id`.

For a full step-by-step guide including error handling, RAG, infrastructure signals, and a full API reference, see [integrate-typescript-agent.md](./integrate-typescript-agent.md).

---

## OpenLLMetry / OTel receiver

[OpenLLMetry](https://github.com/traceloop/openllmetry) instruments 40+ AI frameworks and emits standard OpenTelemetry spans with `gen_ai.*` semantic conventions. Dunetrace runs alongside it — add `DunetraceOTelReceiver` as a second exporter and get behavioral detection with zero changes to your agent code.

### Instrumentation paths

There are three ways to get data into Dunetrace. Which path you choose determines where raw content (prompts, completions, tool arguments) is hashed:

| Path | How it works | Privacy boundary | When to use |
|---|---|---|---|
| **Path 1 — Native SDK** | SHA-256 hash in-process; only hashes + metadata leave the process | Strongest: raw content never leaves the agent | Building a new agent; you control the stack |
| **Path 2 — OTel receiver, self-hosted** | Raw spans travel agent → self-hosted receiver over internal network; hashed at receiver boundary before persistence | Acceptable for most teams: raw content stays on internal network | Already instrumented with OpenLLMetry; self-hosted Dunetrace |
| **Path 3 — OTel receiver, managed cloud** *(future)* | Raw spans would travel to an external service | Requires Path 1 (native SDK) or a customer-side hash proxy before cloud ingestion | Managed cloud endpoint |

### Setup (Path 2)

```bash
pip install 'dunetrace[otel]'
```

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from dunetrace import Dunetrace
from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver

dt = Dunetrace(api_key="dt_live_...")

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))   # existing pipeline unchanged

# Add Dunetrace as a second exporter — one line
DunetraceOTelReceiver.attach(provider, dt, agent_id="my-agent")

# OpenLLMetry instruments everything automatically
from traceloop.sdk import Traceloop
Traceloop.init(app_name="my-agent", tracer_provider=provider)
```

Each OTel trace becomes one Dunetrace run. Spans with `gen_ai.request.model` are translated to `llm_called` / `llm_responded` events. Spans with `gen_ai.tool.name` become `tool_called` / `tool_responded`. The full detector suite runs at trace completion and fires Slack alerts as normal.

**Without OpenLLMetry** — any OTel pipeline emitting `gen_ai.*` spans works:

```python
from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver

receiver = DunetraceOTelReceiver(dt, agent_id="my-agent")
provider.add_span_processor(SimpleSpanProcessor(receiver))
```

### gen_ai.* attributes

| Attribute | Handling |
|---|---|
| `gen_ai.request.model` | Passed as-is (model name, not sensitive) |
| `gen_ai.usage.prompt_tokens` | Passed as-is (integer count) |
| `gen_ai.usage.completion_tokens` | Passed as-is (integer count) |
| `gen_ai.completion.0.finish_reason` | Passed as-is (short string, e.g. `"stop"`) |
| `gen_ai.tool.name` | Passed as-is (tool name) |
| `gen_ai.prompt` | SHA-256 hashed at receiver boundary |
| `gen_ai.completion` | SHA-256 hashed at receiver boundary |
| `gen_ai.prompt.0.content` | SHA-256 hashed at receiver boundary |
| `gen_ai.completion.0.content` | SHA-256 hashed at receiver boundary |

---

## LangChain / LangGraph

```bash
pip install 'dunetrace[langchain]' langchain-openai langgraph python-dotenv       # OpenAI
pip install 'dunetrace[langchain]' langchain-anthropic langgraph python-dotenv    # Anthropic
pip install 'dunetrace[langchain]' langchain-google-genai langgraph python-dotenv # Gemini
```

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for {query}"

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [web_search]
system_prompt = "You are a helpful assistant."
user_input = "What is the capital of France?"

dt = Dunetrace()
callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

agent = create_react_agent(llm, tools, prompt=system_prompt)
result = agent.invoke(
    {"messages": [("human", user_input)]},
    config={"callbacks": [callback]},
)
dt.shutdown()
```

---

## Langfuse

If you are already running Langfuse alongside Dunetrace, you can connect the two so that when Dunetrace detects a failure it pulls the full Langfuse trace and asks an LLM to explain the specific root cause.

**What you get:** when a `TOOL_LOOP`, `GOAL_ABANDONMENT`, or any other signal fires, the dashboard shows an **"Explain with Langfuse ↗"** button. Clicking it fetches the agent's execution trace from Langfuse, builds a prompt from the signal evidence + trace inputs/outputs, and returns a plain-English explanation with a specific fix.

### Prerequisites

- Langfuse account (cloud or self-hosted) with a project and API keys
- One LLM API key for the analysis call (`ANTHROPIC_API_KEY` preferred, `OPENAI_API_KEY` accepted as fallback)

### 1. Install

```bash
pip install 'dunetrace[langchain,langfuse]'
# The API backend also needs: httpx, anthropic (or openai) — included in services/api/requirements.txt
```

### 2. Add credentials to `.env`

```bash
# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # omit for cloud; set for self-hosted

# LLM for explain endpoint (Anthropic preferred, OpenAI fallback)
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

Restart the API container after editing `.env`:

```bash
docker compose up -d api
```

### 3. Run both callbacks together

The `DunetraceCallbackHandler.last_run_id` property exposes the Dunetrace run ID for the most recently completed invocation. Langfuse's `last_trace_id` gives you its corresponding trace ID. Pass `last_trace_id` to the explain endpoint so the two systems can join on the same run.

```python
import uuid as uuid_mod
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # v4+

dt = Dunetrace()
dt_cb = DunetraceCallbackHandler(dt, agent_id="my-agent", model="gpt-4o-mini", tools=["web_search"])
lf_cb = LangfuseCallbackHandler()  # reads LANGFUSE_* from env

result = agent.invoke(
    {"messages": [("human", query)]},
    config={"callbacks": [dt_cb, lf_cb]},
)

dt.shutdown(timeout=5)

import langfuse as lf_module
lf_module.get_client().flush()  # ensure trace is uploaded before querying

# IDs for the join:
dt_run_id   = dt_cb.last_run_id          # e.g. "b5ed23be-e4f0-43bc-..."
lf_trace_id = lf_cb.last_trace_id        # e.g. "b5ed23bee4f043bc..."  (same UUID, no dashes)
```

> **Langfuse v4 trace ID format:** Langfuse v4 uses OTel-style 32-character hex IDs (no dashes). The Dunetrace API strips dashes automatically when querying Langfuse, so `dt_run_id` and `lf_trace_id` represent the same run even though their formats differ.

### 4. Call the explain endpoint

```bash
POST /v1/signals/{signal_id}/explain
Content-Type: application/json
Authorization: Bearer <your-key>

{
  "langfuse_trace_id": "b5ed23bee4f043bc8625914223875508"
}
```

If `langfuse_trace_id` is omitted the endpoint falls back to the signal's own `run_id` (works when Dunetrace and Langfuse share the same trace ID).

Response:

```json
{
  "signal_id": 344,
  "source": "langfuse",
  "root_cause": "The agent re-issued the same search query because the system prompt contains no instruction to track previous queries...",
  "fix_content": "Do not repeat a search query you have already executed in this run.",
  "fix_type": "prompt_addition",
  "apply_blocked": false,
  "langfuse_prompt_name": "research-agent-prompt",
  "langfuse_prompt_version": 3
}
```

**`fix_type`** classifies the recommended fix:

| `fix_type` | Meaning | `apply_blocked` |
|---|---|---|
| `prompt_addition` | One sentence to append to the system prompt | `false` — apply button shown |
| `code_change` | Code or infrastructure fix needed (CONTEXT_BLOAT, SLOW_STEP, etc.) | `true` — apply button hidden |
| `no_auto_apply` | Security signal (PROMPT_INJECTION_SIGNAL) — never auto-apply | `true` — blocked at API level |

**`langfuse_prompt_name`** is `null` when the trace's system prompt was a hardcoded string rather than a Langfuse-managed prompt. The apply button only appears when this field is non-null and `apply_blocked` is false.

### 5. Apply a fix to a managed prompt

When `langfuse_prompt_name` is returned and `apply_blocked` is false, you can apply the fix directly to Langfuse:

```bash
POST /v1/signals/{signal_id}/apply-fix
Content-Type: application/json
Authorization: Bearer <your-key>

{
  "fix_content": "Do not repeat a search query you have already executed in this run.",
  "langfuse_prompt_name": "research-agent-prompt"
}
```

Response:

```json
{
  "fix_id": 12,
  "signal_id": 344,
  "new_version": 4,
  "prompt_url": "https://cloud.langfuse.com/prompts/research-agent-prompt",
  "old_text": "You are a research assistant...",
  "new_text": "You are a research assistant...\n\nDo not repeat a search query..."
}
```

The fix is appended to the current prompt text and published as a new version. The dashboard shows "Applied as v4 in Langfuse ↗" with a link.

### 6. Track fix effectiveness

After applying a fix, check whether the failure type has recurred:

```bash
GET /v1/signals/{signal_id}/fix-status
Authorization: Bearer <your-key>
```

Response:

```json
{
  "fix_applied": true,
  "applied_at": 1745000000.0,
  "applied_via": "langfuse",
  "langfuse_prompt_name": "research-agent-prompt",
  "langfuse_version": 4,
  "runs_after_fix": 23,
  "recurrences_after_fix": 0,
  "verdict": "verified"
}
```

Verdicts: `verified` (≥10 runs, 0 recurrences), `likely_fixed` (≥5 runs, 0 recurrences), `still_occurring`, `insufficient_data`.

### 7. Full working example

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-... \
LANGFUSE_SECRET_KEY=sk-lf-... \
ANTHROPIC_API_KEY=sk-ant-... \
SCENARIO=tool_loop python packages/sdk-py/examples/langfuse_agent.py
```

See [`packages/sdk-py/examples/langfuse_agent.py`](../packages/sdk-py/examples/langfuse_agent.py) for the complete runnable script.

### How the trace lookup works

1. Signal fires → `run_id` stored in Postgres
2. Dashboard calls `POST /v1/signals/{id}/explain` with optional `langfuse_trace_id`
3. API fetches `GET /api/public/traces/{traceId}` from Langfuse (retries up to 4× for ingestion lag; fetches full observation list separately when ≥10 observations are returned paginated)
4. Extracts system prompt from GENERATION observation `messages[]` arrays
5. Normalises step range from evidence using detector-specific field names (e.g. `first_truncation_step` for LLM_TRUNCATION_LOOP)
6. Builds a prompt: signal type + evidence + system prompt + relevant span inputs/outputs (600-char limit for failing steps, 150-char for others)
7. Calls Anthropic Haiku (or GPT-4o-mini fallback) — max 400 tokens — asking for `{"root_cause": "...", "fix_content": "..."}` JSON
8. Returns structured response with fix type classification

The Langfuse trace is never stored — fetched, analysed, discarded.

---

## `@dt.agent()` decorator

Wraps a function in a `dt.run()` context, sets it as the active run for `get_current_run()`, calls `run.final_answer()` on clean return, and emits `RUN_ERRORED` if an exception escapes. Works with both `def` and `async def`.

The recommended pattern is to call `dt.init()` once at startup and use `@dt.agent()` with no `agent_id` — it inherits the default set by `init()`:

```python
from dunetrace import Dunetrace

dt = Dunetrace()               # no api_key needed for local dev
# dt = Dunetrace(api_key="dt_live_...")  # production
dt.init(agent_id="my-agent")  # patches openai, anthropic, httpx, requests globally

@dt.agent()                    # agent_id="my-agent" inherited from init()
def run_agent(query: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": query}]
    )
    return resp.choices[0].message.content   # LLM call tracked automatically

run_agent("What is the capital of France?")
dt.shutdown()
```

**Explicit agent_id** — overrides the `dt.init()` default, useful for multi-agent apps:

```python
@dt.agent("retriever-agent", model="gpt-4o-mini")
def retrieve(query: str) -> list: ...

@dt.agent("answer-agent", model="gpt-4o")
def answer(context: str, query: str) -> str: ...
```

**Async** — identical API, no changes needed:

```python
@dt.agent(model="claude-3-5-sonnet-20241022")
async def run_agent(query: str) -> str:
    resp = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text
```

**`input_from`** — when the user query is not the first argument:

```python
@dt.agent(model="gpt-4o", input_from="question")
def rag(context: str, question: str) -> str: ...
```

**Manual tool instrumentation** — for non-LLM steps (DB, cache, search):

```python
from dunetrace import get_current_run

@dt.agent(model="gpt-4o", tools=["db_query", "web_search"])
def run_agent(query: str) -> str:
    run = get_current_run()

    run.tool_called("db_query", {"q": query})
    result = db.fetch(query)
    run.tool_responded("db_query", success=True, output_length=len(result))

    resp = openai_client.chat.completions.create(...)  # LLM auto-tracked
    return resp.choices[0].message.content
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `agent_id` | `dt.init()` default | Run label shown in the dashboard. Inherits from `dt.init()` if omitted. |
| `model` | `"unknown"` | Model name recorded on the run |
| `tools` | `[]` | Tool list used by `TOOL_AVOIDANCE` detector |
| `system_prompt` | `""` | Used for deterministic agent version hash |
| `input_from` | first arg | Parameter name to use as `user_input` |

---

## FastAPI / ASGI

`DunetraceASGIMiddleware` opens a run for every HTTP or WebSocket request and sets it as the active context so `get_current_run()` works anywhere downstream in the same async task.

```python
from fastapi import FastAPI
from dunetrace import Dunetrace, DunetraceASGIMiddleware, get_current_run

dt = Dunetrace()
dt.auto_instrument()

app = FastAPI()
app.add_middleware(
    DunetraceASGIMiddleware,
    dt=dt,
    agent_id="my-api",
    model="gpt-4o",
)

@app.post("/chat")
async def chat(query: str):
    run = get_current_run()                    # run opened by middleware

    run.tool_called("db_lookup")
    result = await db.get(query)
    run.tool_responded("db_lookup", success=True, output_length=len(result))

    resp = await openai_client.chat.completions.create(  # auto-tracked
        model="gpt-4o", messages=[{"role": "user", "content": query}]
    )
    return resp.choices[0].message.content
```

**Starlette** — identical API:

```python
from starlette.applications import Starlette
from starlette.middleware import Middleware
from dunetrace import Dunetrace, DunetraceASGIMiddleware

dt = Dunetrace()
app = Starlette(middleware=[
    Middleware(DunetraceASGIMiddleware, dt=dt, agent_id="my-api"),
])
```

The run is also available on `request.state.dunetrace_run` for Starlette/FastAPI `Request` objects.

**Middleware parameters:**

| Parameter | Default | Description |
|---|---|---|
| `dt` | required | `Dunetrace` client instance |
| `agent_id` | required | Run label |
| `model` | `"unknown"` | Model name recorded on the run |
| `tools` | `[]` | Tool list recorded on the run |

---

## Flask / WSGI

`DunetraceWSGIMiddleware` wraps any WSGI app. One run per request, cleaned up automatically after the response is sent.

```python
from flask import Flask, request as flask_request
from dunetrace import Dunetrace, DunetraceWSGIMiddleware, get_current_run

dt = Dunetrace()
dt.auto_instrument()

app = Flask(__name__)
app.wsgi_app = DunetraceWSGIMiddleware(app.wsgi_app, dt=dt, agent_id="my-api")

@app.post("/chat")
def chat():
    run = get_current_run()
    query = flask_request.json["query"]

    resp = openai_client.chat.completions.create(  # auto-tracked
        model="gpt-4o", messages=[{"role": "user", "content": query}]
    )
    return resp.choices[0].message.content
```

**Django:**

```python
# wsgi.py
from dunetrace import Dunetrace, DunetraceWSGIMiddleware

dt = Dunetrace()
dt.auto_instrument()

from django.core.wsgi import get_wsgi_application
application = DunetraceWSGIMiddleware(get_wsgi_application(), dt=dt, agent_id="django-api")
```

The run is also available in `environ["dunetrace.run"]` for direct WSGI environ access.

---

## Auto-instrumentation

`dt.auto_instrument()` patches supported AI framework clients at the class level so every LLM call made inside a `dt.run()` context (or inside a `@dt.agent()` function or middleware-wrapped request) is tracked automatically — no manual `run.llm_called()` / `run.llm_responded()` needed.

**Supported frameworks:** `openai`, `anthropic`, `httpx`, `requests`.  
Uninstalled frameworks are silently skipped. Calling `auto_instrument()` more than once is safe.

```python
dt.auto_instrument()                              # patch all installed frameworks
dt.auto_instrument(["openai", "anthropic"])       # LLM clients only
dt.auto_instrument(["httpx", "requests"])         # HTTP clients only
```

What gets captured automatically:

**LLM calls** (`openai`, `anthropic`):
- Model name, prompt + completion token counts, latency, finish reason
- Output length + SHA-256 hash (raw text never transmitted)

**HTTP calls** (`httpx`, `requests`):
- Hostname used as tool name (e.g. `serpapi.com`, `api.stripe.com`)
- Success / failure based on HTTP status code
- Response `content-length`, latency
- URL SHA-256 hash (raw URL never transmitted)

---

## `get_current_run()`

Returns the active `RunContext` for the current async task or thread, or `None` if no run is active. Use this inside helpers to access the run without passing it through your call stack.

```python
from dunetrace import get_current_run

def some_helper():
    run = get_current_run()
    if run:
        run.tool_called("cache_lookup")
        result = cache.get(key)
        run.tool_responded("cache_lookup", success=result is not None)
        return result
```

Works correctly with `@dt.agent()`, ASGI middleware, WSGI middleware, and `dt.run()` directly.

---

## Manual instrumentation

Use `dt.run()` directly when you need full control over every event, or when no other integration fits.

```python
from dunetrace import Dunetrace

dt = Dunetrace()
user_input = "What is the capital of France?"

with dt.run("my-agent", user_input=user_input, model="gpt-4o", tools=["search"]) as run:
    run.llm_called("gpt-4o", prompt_tokens=150)
    run.llm_responded(finish_reason="tool_calls", latency_ms=320)

    run.tool_called("search", {"query": user_input})
    run.tool_responded("search", success=True, output_length=512)

    run.llm_called("gpt-4o", prompt_tokens=480)
    run.llm_responded(finish_reason="stop", output_length=120)
    run.final_answer()

dt.shutdown()
```

**`run` method reference:**

| Method | When to call |
|---|---|
| `run.llm_called(model, prompt_tokens)` | Before each LLM API call |
| `run.llm_responded(completion_tokens, latency_ms, finish_reason, ...)` | After LLM response received |
| `run.tool_called(tool_name, args)` | Before each tool execution |
| `run.tool_responded(tool_name, success, output_length, latency_ms, error)` | After tool returns |
| `run.retrieval_called(index_name, query_hash)` | Before vector search |
| `run.retrieval_responded(index_name, result_count, top_score, latency_ms)` | After retrieval returns |
| `run.external_signal(signal_name, source, **meta)` | Rate limits, cache misses, upstream errors |
| `run.final_answer()` | When agent produces its final output |

---

## Policies

Runtime guardrails evaluated mid-run after every `tool_called`, `llm_responded`, and `tool_responded` event. Policies fire at most once per run (except `log` policies, which fire every time).

### Local policies (no backend required)

```python
from dunetrace import Dunetrace, PolicyViolation

dt = Dunetrace()

# Stop the run if tool calls exceed 5
dt.add_policy(
    name="cap tool calls",
    condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
    action={"type": "stop"},
)

# Downgrade model when estimated cost exceeds $0.50
dt.add_policy(
    name="cost cap",
    condition={"trigger": "cost_usd", "operator": "gt", "value": 0.50},
    action={"type": "switch_model", "params": {"model": "gpt-4o-mini"}},
)

# Inject a corrective prompt when a loop is detected mid-run
dt.add_policy(
    name="loop fix",
    condition={"trigger": "signal", "operator": "eq", "value": "TOOL_LOOP"},
    action={"type": "inject_prompt", "params": {"prompt": "Stop repeating tool calls. Summarise what you know and answer directly."}},
)

# Log without stopping (fires every time the condition is true)
dt.add_policy(
    name="warn slow",
    condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 10000},
    action={"type": "log"},
)
```

### Remote policies (dashboard-managed)

When `api_key` and `endpoint` are set, the SDK fetches policies from the backend at run start and caches them for 60 seconds per agent. Policies defined in the dashboard apply automatically — no code changes needed.

```python
dt = Dunetrace(api_key="dt_live_...", endpoint="https://ingest.dunetrace.com")
# Policies defined in the dashboard are pulled at run start.
```

Local policies (added via `add_policy`) take priority over remote ones at the same `priority` level and are never replaced by remote fetches.

### Condition reference

| Trigger | Type | What it measures |
|---|---|---|
| `tool_call_count` | int | Total tool calls in the run so far |
| `step_count` | int | Current step index |
| `cost_usd` | float | Accumulated LLM cost in USD (model-aware pricing) |
| `error_count` | int | Failed tool calls (`success=False`) |
| `finish_reason` | str | Latest LLM `finish_reason` (e.g. `"length"`, `"stop"`, `"tool_calls"`) |
| `llm_latency_ms` | int | Latest LLM call latency in milliseconds |
| `signal` | str | Detector signal name — runs the full detector suite lazily (e.g. `"TOOL_LOOP"`) |

Supported operators: `gt` `gte` `lt` `lte` `eq` `neq` `contains`

### Action reference

| Action type | Effect | Required params |
|---|---|---|
| `stop` | Raises `PolicyViolation`; run exits with `exit_reason="policy_violation"` | — |
| `switch_model` | Sets `run.model_override` (str) | `model` |
| `inject_prompt` | Appends to `run.prompt_additions` (list) | `prompt` |
| `log` | Emits `policy.triggered` event; no interruption; fires on every matching event | — |

**`stop`** — `PolicyViolation` propagates up through the agent code. `dt.run()` catches it, emits `run.errored` with `exit_reason="policy_violation"` and `policy_name`, then re-raises. Catch it if you want to handle the stop gracefully:

```python
from dunetrace import PolicyViolation

try:
    with dt.run("my-agent", user_input=query, tools=TOOLS) as run:
        for step in agent_loop():
            ...
except PolicyViolation as exc:
    print(f"Stopped by policy: {exc.policy_name}")
```

**`switch_model`** — the SDK sets `run.model_override` but does not intercept your LLM calls. You must read it between steps and switch the model yourself:

```python
with dt.run("my-agent", user_input=query) as run:
    for step in agent_loop():
        model = run.model_override or "gpt-4o"   # check after each step
        response = openai_client.chat.completions.create(model=model, ...)
```

**`inject_prompt`** — the SDK appends to `run.prompt_additions`. Read it with `run.pop_prompt_addition()` and prepend it to your next LLM messages array:

```python
with dt.run("my-agent", user_input=query) as run:
    messages = [{"role": "system", "content": system_prompt}]
    for step in agent_loop():
        addition = run.pop_prompt_addition()
        if addition:
            messages.insert(0, {"role": "system", "content": addition})
        response = openai_client.chat.completions.create(model="gpt-4o", messages=messages, ...)
```

### `add_policy` parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | required | Human-readable label shown in `policy.triggered` events |
| `condition` | required | `{trigger, operator, value}` dict |
| `action` | required | `{type, params?}` dict |
| `agent_id` | `"*"` | `"*"` applies to all agents; pass a specific agent_id to scope |
| `priority` | `100` | Lower numbers fire first |
| `enabled` | `True` | Set to `False` to disable without removing |

### Dashboard CRUD

Policies can be created, edited, toggled, and deleted from the **Policies** page in the dashboard at `http://localhost:3000`. Changes are fetched by the SDK within the 60-second TTL window.

The backend also exposes a REST API:

| Endpoint | Description |
|---|---|
| `GET /v1/policies` | List all policies |
| `POST /v1/policies` | Create a policy |
| `GET /v1/policies/{id}` | Get a single policy |
| `PUT /v1/policies/{id}` | Replace a policy |
| `DELETE /v1/policies/{id}` | Delete a policy |
| `PATCH /v1/policies/{id}/toggle` | Enable / disable |

The ingest endpoint also exposes `GET /v1/policies?agent_id=...&api_key=...` (used by the SDK for remote fetch).

---

## Grafana / Loki

```python
dt = Dunetrace(emit_as_json=True)
```

Writes every event to stdout as a Loki-compatible NDJSON line. Each line includes `ts`, `level`, `logger`, `event_type`, `agent_id`, `run_id`, `step_index`, and `payload`. Works alongside HTTP ingest, both can be active at the same time.

Minimal Promtail pipeline stage:

```yaml
pipeline_stages:
  - json:
      expressions: {ts: ts, event_type: event_type, agent_id: agent_id}
  - timestamp:
      source: ts
      format: RFC3339Nano
  - labels:
      agent_id:
      event_type:
```

---

## OpenTelemetry

```bash
pip install 'dunetrace[otel]' opentelemetry-exporter-otlp-proto-grpc
```

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from dunetrace.integrations.otel import DunetraceOTelExporter

resource = Resource.create({
    "service.name": "my-agent-service",
    "deployment.environment": "production",
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

dt = Dunetrace(otel_exporter=DunetraceOTelExporter(provider))
```

Each agent run produces a trace with a deterministic `trace_id` derived from `run_id`, so you can correlate Dunetrace signals with infra metrics in Tempo, Honeycomb, Datadog, or Jaeger:

```
Trace
└── Span: "agent_run"     [dunetrace.agent_id, dunetrace.model, dunetrace.tools, …]
    ├── Span: "llm_call"  [gen_ai.request.model, gen_ai.usage.input_tokens, …]
    ├── Span: "tool_call" [dunetrace.tool_name, dunetrace.success, dunetrace.latency_ms]
    └── Span: "retrieval" [dunetrace.index_name, dunetrace.result_count, dunetrace.top_score]
```

Failure signals detected at run end are written as indexed attributes on the root span (`dunetrace.signal.0.failure_type`, `.severity`, `.confidence`). HIGH/CRITICAL signals set `span.status = ERROR`.

Use `endpoint=None` to run OTel-only with no HTTP ingest:

```python
dt = Dunetrace(endpoint=None, otel_exporter=DunetraceOTelExporter(provider))
```

**With LangChain:** `DunetraceCallbackHandler` and `DunetraceOTelExporter` are independent and both active simultaneously without any extra configuration.

For deeper OTel internals (span structure, known gaps, parallel tool calls) see [architecture.md](architecture.md#otel-span-exporter).

---

## Examples

**LangChain agent** (real OpenAI calls, auto-instrumented via callback):

```bash
cd packages/sdk-py
pip install 'dunetrace[langchain]' langchain-openai langgraph python-dotenv
```

Add your key to the root `.env`:

```
OPENAI_API_KEY=sk-...
```

```bash
python examples/langchain_agent.py

# Force a tool-loop scenario:
SCENARIO=tool_loop python examples/langchain_agent.py
```

**Decorator + auto-instrument** (pure Python agents):

```bash
pip install dunetrace
python examples/decorator_agent.py

# Force failure scenarios (TOOL_LOOP, RETRY_STORM, RAG_EMPTY_RETRIEVAL):
SCENARIO=failures python examples/decorator_agent.py
```

**Basic agent** (manual instrumentation, detectors, prompt injection):

```bash
python examples/basic_agent.py
```

All examples send events to `http://localhost:8001` by default and require no API key in dev mode. Override the endpoint with `DUNETRACE_ENDPOINT=http://your-host:8001`.

---

## Tests

The SDK test suite runs entirely offline (no backend, no real API keys):

```bash
cd packages/sdk-py
python -m unittest discover -s tests -v
```

| Test file | What it covers |
|---|---|
| `tests/test_auto_instrument.py` | `auto_instrument()` for OpenAI, Anthropic, httpx, requests; `@dt.agent()` decorator (sync + async); ASGI + WSGI middleware; `get_current_run()` lifecycle |
| `tests/test_client.py` | `dt.run()` context manager, event emission, privacy (no raw content), prompt injection detection, shutdown |
| `tests/test_detectors.py` | All 15 structural detectors |
| `tests/test_detectors_evidence.py` | Enriched evidence fields on all detectors: `step_indices`, `args_identical`, `args_similar`, `success_rate` (ToolLoop); `stall_event_sequence` (GoalAbandonment); `token_counts_at_truncation`, `models` (LlmTruncationLoop); `token_growth_sequence` (ContextBloat); `error_hashes`, `reason_identical` (RetryStorm); `event_sequence` (ReasoningSpin); `coincident_signals` (SlowStep) |
| `tests/test_integrations/` | LangChain callback handler, OpenTelemetry exporter |
| `tests/test_policies.py` | `PolicyEngine` evaluation, deduplication, remote fetch TTL, cost computation, `stop`/`switch_model`/`inject_prompt`/`log` actions, `signal` trigger (lazy detector run), `PolicyViolation` propagation through `dt.run()` |

The explainer service has its own test suite in `services/explainer/tests/`:

| Test file | What it covers |
|---|---|
| `tests/test_explainer.py` | Core `explain()` templates for all 15 failure types |
| `tests/test_explainer_new.py` | Enriched evidence interpolation in explanation text (growth curves, stall sequences, token counts, model lists); `rate_context` attach/detach on `Explanation`; edge cases for count keys and singular/plural formatting |

The alerts service test suite is in `services/alerts/tests/`:

| Test file | What it covers |
|---|---|
| `tests/test_worker.py` | `poll_once()` integration: signal fetch, mark-alerted, skip-already-alerted, deliver |
| `tests/test_rate_context.py` | `_rate_context_text()` helper (systemic / first-occurrence / recurring branches, edge cases); `format_slack()` block ordering with rate context; `poll_once()` with mocked `fetch_signal_rate_context` |
| `tests/test_digest.py` | `should_send_digest()` day/hour guards; `format_digest_slack()` Block Kit structure (header, totals, failure types, systemic patterns, issue counts, dashboard button, colour, pct rounding, no-failures path); `send_weekly_digest()` all skip/send/fail paths |

The detector service test suite is in `services/detector/tests/`:

| Test file | What it covers |
|---|---|
| `tests/test_issues.py` | `upsert_fired_issues()` noop/pool-None/executemany correctness; `advance_clean_runs()` noop/params/empty-fired-passes-None/threshold constant; worker integration (upsert called when signals fire, advance called on clean run, tracking failure doesn't break `process_run`); `list_issues()` empty/fields/status-filter/no-filter |
