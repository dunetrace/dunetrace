# Dunetrace SDK

Runtime observability for AI agents. Detects tool loops, context bloat, prompt injection, and 12 other failure patterns in real-time — with a Slack alert while the run is still live.

Zero external dependencies.

## Install

```bash
pip install dunetrace                    # core SDK
pip install 'dunetrace[langchain]'       # + LangChain / LangGraph
pip install 'dunetrace[otel]'            # + OpenTelemetry exporter
```

## Quickstart

**LangChain / LangGraph**

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace()
callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

result = agent.invoke(input, config={"callbacks": [callback]})
dt.shutdown()
```

**Pure Python / custom agent**

```python
from dunetrace import Dunetrace

dt = Dunetrace()
dt.auto_instrument()   # patches openai, anthropic, httpx, requests

@dt.agent("my-agent", model="gpt-4o")
def run_agent(query: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content
```

**FastAPI / Flask** — one line each, see [docs/integrations.md](../../docs/integrations.md).

## What it detects


| Detector                  | What it catches                                            | Severity    |
| ------------------------- | ---------------------------------------------------------- | ----------- |
| `TOOL_LOOP`               | Same tool called 3+ times in a 5-call window               | HIGH        |
| `TOOL_THRASHING`          | Agent alternates between exactly two tools                 | HIGH        |
| `RETRY_STORM`             | Same tool fails 3+ times in a row                          | HIGH        |
| `LLM_TRUNCATION_LOOP`     | `finish_reason=length` fires 2+ times                      | HIGH        |
| `EMPTY_LLM_RESPONSE`      | Zero-length output with `finish_reason=stop`               | HIGH        |
| `CASCADING_TOOL_FAILURE`  | 3+ consecutive failures across 2+ distinct tools           | HIGH        |
| `SLOW_STEP`               | Tool call >15s or LLM call >30s                            | MEDIUM/HIGH |
| `TOOL_AVOIDANCE`          | Final answer without using available tools                 | MEDIUM      |
| `GOAL_ABANDONMENT`        | Tool use stops, then 4+ consecutive LLM calls with no exit | MEDIUM      |
| `CONTEXT_BLOAT`           | Prompt tokens grow 3× from first to last LLM call          | MEDIUM      |
| `STEP_COUNT_INFLATION`    | Run used >2× the P75 step count for this agent             | MEDIUM      |
| `FIRST_STEP_FAILURE`      | Error or empty output at step ≤2                           | MEDIUM      |
| `REASONING_STALL`         | LLM:tool-call ratio ≥4× — reasoning without acting         | MEDIUM      |
| `RAG_EMPTY_RETRIEVAL`     | Retrieval returned 0 results but agent answered anyway     | MEDIUM      |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns         | CRITICAL    |


## Output modes


| Mode                  | How to enable                                   | Destination                                      |
| --------------------- | ----------------------------------------------- | ------------------------------------------------ |
| HTTP ingest (default) | `endpoint="http://…"`                           | Dunetrace backend → detection, alerts, dashboard |
| Loki NDJSON           | `emit_as_json=True`                             | stdout → Promtail / Grafana Alloy                |
| OpenTelemetry         | `otel_exporter=DunetraceOTelExporter(provider)` | Tempo, Honeycomb, Datadog, Jaeger                |


## Backend

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env && docker compose up -d
```

Dashboard → `http://localhost:3000` · Ingest → `http://localhost:8001`

## Tests

```bash
python -m unittest discover -s tests -v
```

52 tests, no network required.

## Links

- [Full integration docs](../../docs/integrations.md)
- [GitHub](https://github.com/dunetrace/dunetrace)
- [Issues](https://github.com/dunetrace/dunetrace/issues)

