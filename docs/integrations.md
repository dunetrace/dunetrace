# Integrations

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

## `@dt.agent()` decorator

Wraps a function in a `dt.run()` context, sets it as the active run for `get_current_run()`, calls `run.final_answer()` on clean return, and emits `RUN_ERRORED` if an exception escapes. Works with both `def` and `async def`.

The recommended pattern is to call `dt.init()` once at startup and use `@dt.agent()` with no `agent_id` — it inherits the default set by `init()`:

```python
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_live_...")
dt.init(agent_id="my-agent")   # patches openai, anthropic, httpx, requests globally

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
```

**Basic agent** (manual instrumentation, detectors, prompt injection):

```bash
python examples/basic_agent.py
```

All examples send events to `http://localhost:8001` by default. Override with `DUNETRACE_ENDPOINT=http://your-host:8001`.

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
| `tests/test_integrations/` | LangChain callback handler, OpenTelemetry exporter |
