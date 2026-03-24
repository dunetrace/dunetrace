# Dunetrace SDK

Behavioral observability for AI agents at runtime. Zero-dependency Python SDK that detects tool loops, context bloat, prompt injection, and other failure patterns in real-time.

## Install

```bash
pip install dunetrace                    # core SDK, no dependencies
pip install 'dunetrace[langchain]'       # + LangChain callback handler
pip install 'dunetrace[otel]'            # + OpenTelemetry span exporter
pip install 'dunetrace[langchain,otel]'  # both
```

## Quickstart

```python
from dunetrace import Dunetrace

dt = Dunetrace()  # defaults to http://localhost:8001
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

## LangChain

```bash
pip install 'dunetrace[langchain]' langchain-openai langgraph python-dotenv
```

Add your key to the root `.env` file :

```
OPENAI_API_KEY=sk-...
DUNETRACE_ENDPOINT=http://localhost:8001
```

```python
from dotenv import load_dotenv
load_dotenv()

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

See `examples/langchain_agent.py` for a full working example.

## Output modes

Three independent output modes, combine freely:

| Mode | How to enable | Destination |
|---|---|---|
| HTTP ingest (default) | `endpoint="http://…"` | Dunetrace backend → detection, alerts, dashboard |
| Loki NDJSON | `emit_as_json=True` | stdout → Promtail/Alloy → Grafana |
| OpenTelemetry | `otel_exporter=DunetraceOTelExporter(provider)` | Any OTel collector (Tempo, Honeycomb, Datadog, Jaeger) |

Use `endpoint=None` to disable HTTP ingest entirely (OTel-only or Grafana-only mode):

```python
dt = Dunetrace(endpoint=None, otel_exporter=DunetraceOTelExporter(provider))
```

### OpenTelemetry

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

Each agent run produces a trace rooted at a deterministic `trace_id` derived from `run_id`:

```
Trace (trace_id = run_id as 128-bit int)
└── Span: "agent_run"         [dunetrace.agent_id, dunetrace.run_id, dunetrace.model, …]
    ├── Span: "llm_call"      [gen_ai.request.model, gen_ai.usage.input_tokens, …]
    ├── Span: "tool_call"     [dunetrace.tool_name, dunetrace.success, dunetrace.latency_ms]
    │   └── SpanEvent: "rate_limit"   (from run.external_signal("rate_limit", source="openai"))
    └── Span: "retrieval"     [dunetrace.index_name, dunetrace.result_count, dunetrace.top_score]
```

Failure signals detected at run end are written as indexed attributes on the root span:

```
dunetrace.signal.0.failure_type = "TOOL_LOOP"
dunetrace.signal.0.severity     = "HIGH"
dunetrace.signal.0.confidence   = 0.95
dunetrace.signal.0.evidence.*   = …
```

HIGH and CRITICAL signals also set `span.status = ERROR`.

**With LangChain:** pass `DunetraceOTelExporter` to `Dunetrace` alongside `DunetraceCallbackHandler`, no other changes needed. Both work simultaneously.

### Loki / Grafana

```python
dt = Dunetrace(emit_as_json=True)
```

Writes one NDJSON line per event to stdout. Compatible with Promtail and Grafana Alloy pipeline stages:

```json
{"ts":"2026-03-17T12:00:00Z","level":"info","logger":"dunetrace",
 "event_type":"tool.called","agent_id":"my-agent","run_id":"…","step_index":3,
 "payload":{…}}
```

## Infrastructure context

Annotate agent steps with external signals i.e. no step counter advance:

```python
run.tool_called("web_search", {"query": "..."})
run.external_signal("rate_limit", source="openai", retry_after=30)
run.tool_responded("web_search", success=True, output_length=800)
```

`SLOW_STEP` signals will include `coincident_signals` in evidence when an external signal fell within the step's time window.

## What it detects (15 detectors)

| Detector | What it catches | Severity |
|---|---|---|
| `TOOL_LOOP` | Same tool called 3+ times in a 5-call window | HIGH |
| `TOOL_THRASHING` | Agent alternates between exactly two tools | HIGH |
| `RETRY_STORM` | Same tool fails 3+ times in a row; evidence includes args/reason identity | HIGH |
| `LLM_TRUNCATION_LOOP` | `finish_reason=length` fires 2+ times | HIGH |
| `EMPTY_LLM_RESPONSE` | Zero-length output with `finish_reason=stop` | HIGH |
| `CASCADING_TOOL_FAILURE` | 3+ consecutive failures across 2+ distinct tools | HIGH |
| `SLOW_STEP` | Tool call >15s or LLM call >30s | MEDIUM/HIGH |
| `TOOL_AVOIDANCE` | Final answer without using available tools | MEDIUM |
| `GOAL_ABANDONMENT` | Tool use stops, then 4+ consecutive LLM calls with no exit | MEDIUM |
| `CONTEXT_BLOAT` | Prompt tokens grow 3× from first to last LLM call | MEDIUM |
| `STEP_COUNT_INFLATION` | Run used >2× the P75 step count for this agent | MEDIUM |
| `FIRST_STEP_FAILURE` | Error or empty output at step ≤2 | MEDIUM |
| `REASONING_STALL` | LLM:tool-call ratio ≥4× — reasoning without acting | MEDIUM |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned 0 results but agent answered anyway | MEDIUM |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns | CRITICAL |

## Self-hosted backend

The SDK ships events to the Dunetrace backend, which runs detection and sends alerts:

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace
cp .env.example .env
docker compose up -d
```

- Ingest: `http://localhost:8001`
- Dashboard: `http://localhost:3000`
- API docs: `http://localhost:8002/docs`

## Links

- [GitHub](https://github.com/dunetrace/dunetrace)
- [Issues](https://github.com/dunetrace/dunetrace/issues)
