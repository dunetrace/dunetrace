# Dunetradece

![Dunetrace](dunetrace.png)

### Runtime observability for AI agents

**Detect structural failures automatically. Alert before your users do.**

[![PyPI version](https://img.shields.io/pypi/v/dunetrace.svg)](https://pypi.org/project/dunetrace/)
[![Python versions](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/dunetrace/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dunetrace?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dunetrace)
[![GitHub Stars](https://img.shields.io/github/stars/dunetrace/dunetrace?style=flat&logo=github)](https://github.com/dunetrace/dunetrace)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/NEtdVukx)

---

## The problem

AI agents fail in ways that traditional monitoring can't see.

Your API returns 200. Your logs show no exceptions. But the agent called the same tool 12 times in a row, burned $X in tokens, and gave the user a wrong answer or no answer at all.

These are structural failures: tool loops, context bloat, reasoning stalls etc. They happen inside the agent's decision loop, between the lines your existing monitoring covers. By the time a user reports a problem, the damage is already done.

**The result:** you're flying blind. You find out about agent failures from users, not dashboards.

## Why existing tools don't catch this

LangSmith/Langfuse are excellent but they answer "what happened?" after you already know something broke. You open the trace, investigate, find the problem.

OpenLLMetry and traceAI give you OTel spans across 40+ frameworks — great instrumentation coverage, no behavioral detection.

Dunetrace answers a different question: **"is something breaking right now?"**

It watches the structural pattern of every run automatically and fires a Slack alert while the run is still in progress or within 15 seconds of completion.

**Already using OpenLLMetry?** Dunetrace runs alongside it. Point your OTel spans at the Dunetrace receiver and get behavioral detection with no additional instrumentation. → [OpenLLMetry / OTel receiver](docs/integrations.md#openllmetry--otel-receiver)

## The solution

Dunetrace instruments the agent loop itself. Every tool call, LLM call, and retrieval is observed at runtime. A set of structural detectors runs against each completed run, flags failure patterns the moment they occur, and sends a Slack alert in <15s with a plain-English explanation and a concrete fix.

## Privacy architecture

No raw content ever leaves your agent process.

Every prompt, tool argument, and model output is SHA-256 hashed before transmission. The ingest API receives hashes, token counts, latency values, and call sequences - never plaintext.

The database has no raw content column. This matters for enterprise teams with data governance requirements (GDPR, HIPAA, internal data classification).

```json
// What the SDK actually transmits
{
  "event":      "tool_called",
  "tool_name":  "web_search",
  "args_hash":  "a3f7c9d2...",    // SHA(args), not args
  "step":       3,
  "timestamp":  1741772467.312
}
```

---

## Quick Start

### 1. Start the backend

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env && docker compose build && docker compose up -d
```

### 2. Install the SDK

```bash
pip install dunetrace                    # any framework
pip install 'dunetrace[langchain]'      # LangChain / LangGraph
```

### 3. Instrument your agent

```python
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_live_...")
dt.init(agent_id="my-agent")   # patches openai, anthropic, httpx, requests globally

@dt.agent()                    # agent_id inherited from init()
def run_agent(query: str) -> str:
    ...                        # LLM + HTTP calls tracked automatically
```

→ [Full integration options](docs/integrations.md) — LangChain, FastAPI, Flask, OpenLLMetry, manual

Then open the dashboard: **[http://localhost:3000](http://localhost:3000)**


| Endpoint     | URL                                                      |
| ------------ | -------------------------------------------------------- |
| Dashboard    | [http://localhost:3000](http://localhost:3000)           |
| API + docs   | [http://localhost:8002/docs](http://localhost:8002/docs) |
| Ingest (SDK) | [http://localhost:8001](http://localhost:8001)           |


---

## Dashboard

![Dashboard overview](dashboard.png)

The Mission Control dashboard is a live, API-driven single-page app served at **[http://localhost:3000](http://localhost:3000)**. It auto-refreshes every 15 seconds and requires no build step — static HTML fetching from the Customer API.

### Pages

| Page | What it shows |
|---|---|
| **Overview** | Four stat cards (Critical / High / Signals / Runs) with configurable trend deltas (↑↓ vs last hour / yesterday / last week). Risk Trend 24h bar chart (hourly signal count, colour-coded by intensity). Top failure patterns with ↑↓ trend arrows. Live run feed. |
| **All Runs** | Full run table — agent, signals, severity, failure types, duration, step count. Click any row to open run detail. |
| **Alerts** | Signals grouped by failure type, expandable per group. Shadow signals rendered below with dashed border + SHADOW badge. |
| **Analytics** | Cross-agent totals, top failure patterns, per-agent breakdown. |
| **Risk Heatmap** | Failure type × agent intensity grid. |
| **Agents** | Per-agent health cards — failure rate %, dominant pattern, run / critical / high counts, last seen, ungraduated shadow signal count. |
| **Compare Runs** | Side-by-side run comparison. Select any two runs from dropdowns — metrics, signals, and max confidence shown in both panels with a colour-coded delta table (new / resolved failure types highlighted). |
| **Detectors** | Threshold sliders and alert level selector. |

### Run detail

Click any run to open the detail panel with three tabs:

- **Analysis** — execution timeline (one node per step, loop detection), signal score cards with confidence bars, plain-English explanation + suggested fix
- **Run graph** — SVG node graph: green = LLM call, orange = tool call (ok), red = looping tool call, blue = start/end. Loop clusters highlighted with a dashed red outline.
- **Event log** — every event in chronological order, expandable to show full payload. Content fields are shown as SHA-256 hashes — no raw text stored.

### Stat card info buttons

Each stat card has an `ⓘ` button explaining the threshold for that severity level:

| Card | Threshold |
|---|---|
| Critical | conf ≥ 0.85, or prompt injection / cascading failure regardless of confidence |
| High | conf ≥ 0.70 — tool loops, retry storms, context bloat |
| Signals | All four levels: CRITICAL ≥ 0.85 · HIGH ≥ 0.70 · MEDIUM ≥ 0.50 · LOW < 0.50 |
| Total runs | Processed runs (clean + signal-bearing) counted within one 5s detector poll |

---

## What it detects


| Detector                  | What it catches                                                    | Severity    |
| ------------------------- | ------------------------------------------------------------------ | ----------- |
| `TOOL_LOOP`               | Same tool called ≥3× in a 5-tool-call window                       | HIGH        |
| `TOOL_THRASHING`          | Agent alternates between exactly two tools                         | HIGH        |
| `LLM_TRUNCATION_LOOP`     | `finish_reason=length` fires ≥2 times                              | HIGH        |
| `RETRY_STORM`             | Same tool fails 3+ times in a row                                  | HIGH        |
| `EMPTY_LLM_RESPONSE`      | Model returned zero-length output with `finish_reason=stop`        | HIGH        |
| `CASCADING_TOOL_FAILURE`  | 3+ consecutive failures across 2+ distinct tools                   | HIGH        |
| `SLOW_STEP`               | Tool call >15s or LLM call >30s                                    | MEDIUM/HIGH |
| `TOOL_AVOIDANCE`          | Final answer given without calling available tools                 | MEDIUM      |
| `GOAL_ABANDONMENT`        | Tool use stops, then ≥4 consecutive LLM calls with no exit         | MEDIUM      |
| `RAG_EMPTY_RETRIEVAL`     | Retrieval returned 0 results or relevance <0.3, but agent answered | MEDIUM      |
| `CONTEXT_BLOAT`           | Prompt tokens grow 3× from first to last LLM call                  | MEDIUM      |
| `STEP_COUNT_INFLATION`    | Run used >2× the P75 step count for this agent                     | MEDIUM      |
| `FIRST_STEP_FAILURE`      | Error or empty output at step ≤2                                   | MEDIUM      |
| `REASONING_STALL`         | LLM:tool-call ratio ≥4× — agent reasoning without acting           | MEDIUM      |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns                 | CRITICAL    |


Thresholds are configurable. → [docs/detectors.md](docs/detectors.md)

---

## Integrations

- [LangChain / LangGraph](docs/integrations.md#langchain--langgraph)
- [Python agents & web frameworks](docs/integrations.md) — decorator, FastAPI, Flask, manual
- [Grafana / Loki](docs/integrations.md#grafana--loki)
- [OpenTelemetry](docs/integrations.md#opentelemetry)

---

## Alerts

Slack and generic webhook (PagerDuty, Linear, custom). → [docs/alerts.md](docs/alerts.md)

![Slack alert](slack-alert.png)

---

## Architecture

```
Agent Code
  └─► Dunetrace SDK              (hashes content → Dunetrace event schema)
        │                        (mirrors to OTel span schema, content stripped)
        └─► Ingest API           (POST /v1/ingest -> Postgres)
                    ├─► Detector       (poll -> reconstruct RunState -> run detectors -> write signals)
                    ├─► Alerts         (poll -> explain -> Slack / webhook)
                    └─► Customer API   (query runs, signals, explanations)
        ├─► stdout NDJSON        (emit_as_json=True -> Loki / Grafana Alloy)
        └─► OTel exporter        (otel_exporter=… -> Tempo / Honeycomb / Datadog)
```

> OTel spans are derived from the same instrumentation as Dunetrace events. Content fields are SHA-256 hashed before either path emits. The OTel exporter runs a local detector pass (SDK default thresholds) and annotates the root span — independent of the server-side detector, which uses `detectors.yml` and is the source of truth for alerts and the dashboard. → [docs/architecture.md](docs/architecture.md#otel-span-exporter)

→ [docs/architecture.md](docs/architecture.md) for service internals, DB schema, and performance characteristics.

---

## Running tests

**SDK only** (no Docker required):

```bash
cd packages/sdk-py
python -m unittest discover -s tests -v
```

Covers: `get_current_run()`, `@dt.agent()` decorator (sync + async), `auto_instrument()` for OpenAI / Anthropic / httpx / requests, ASGI middleware, WSGI middleware, context var lifecycle, privacy (no raw content in events), prompt injection detection.

**Full suite** (requires Docker for backend services):

```bash
PYTHONPATH=packages/sdk-py:services/explainer:services/alerts:services/detector:services/api:services/ingest \
  python -m pytest packages/sdk-py/tests/ services/explainer/tests/ services/detector/tests/ \
    services/alerts/tests/ services/api/tests/ services/ingest/tests/ \
  --asyncio-mode=auto -v
```

---

## Requirements

- Python 3.11+
- Docker + Docker Compose
- PostgreSQL 16+ (included in Docker Compose)

## Contributing

1. Fork the repo and create a branch
2. Make your changes and add tests
3. Run the test suite (see above)
4. Open a pull request with a clear description of what and why

For larger changes (new detectors, architecture changes), open an issue first.

## Star us (⭐)

If Dunetrace looks useful, a GitHub star helps others find the project.

## Contact

[dunetrace@gmail.com](mailto:dunetrace@gmail.com)

## License

[Apache 2.0](LICENSE)