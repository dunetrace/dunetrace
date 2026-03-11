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

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace()
callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

result = agent.invoke(
    {"messages": [("human", user_input)]},
    config={"callbacks": [callback]},
)
```

## Output modes

Three independent output modes, combine freely:

```python
# Default: HTTP POST to backend (detection, alerts, dashboard)
dt = Dunetrace(endpoint="http://localhost:8001")

# Loki/Grafana: NDJSON to stdout (works alongside HTTP)
dt = Dunetrace(emit_as_json=True)

# OpenTelemetry: spans to any OTel collector
from opentelemetry.sdk.trace import TracerProvider
from dunetrace.integrations.otel import DunetraceOTelExporter

provider = TracerProvider()
dt = Dunetrace(otel_exporter=DunetraceOTelExporter(provider))
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
