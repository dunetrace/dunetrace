# Dunetrace

![Dunetrace](dunetrace.png)

### Real time anomaly detection layer for AI agents.

**Detect structural failures automatically. Alert before your users do.**

[![PyPI version](https://img.shields.io/pypi/v/dunetrace.svg)](https://pypi.org/project/dunetrace/)
[![Python versions](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/dunetrace/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dunetrace?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dunetrace)
[![GitHub Stars](https://img.shields.io/github/stars/dunetrace/dunetrace?style=flat&logo=github)](https://github.com/dunetrace/dunetrace)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/yxFjATwHW4)

---

## The problem

AI agents fail in ways that traditional monitoring can't see.

Your API returns 200. Your logs show no exceptions. But the agent called the same tool 12 times in a row, burned $X in tokens, and gave the user a wrong answer or no answer at all.

LangSmith/Langfuse answer "what happened?" after you already know something broke. Dunetrace answers a different question: **"is something breaking right now?"** and if you're already running Langfuse, it can pull your trace context to explain *why* and suggest a fix to the root cause.

It watches the structural pattern of every run and fires a Slack alert within 15 seconds of completion.

---

## Quick Start

### 1. Start the backend

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env && docker compose build && docker compose up -d
```

### 2. Install the SDK

```bash
pip install dunetrace                              # any framework
pip install 'dunetrace[langchain]'                # LangChain / LangGraph
pip install 'dunetrace[langchain,langfuse]'       # + Langfuse deep analysis
```

### 3. Instrument your agent

**Auto-instrument with decorators** — zero SDK calls inside function bodies:

```python
from dunetrace import Dunetrace

dt = Dunetrace()

@dt.tool                                  # auto-emits tool.called / tool.responded
def web_search(query: str) -> list: ...

@dt.trace                                 # auto-emits run.started / run.completed
def my_agent(question: str) -> str:
    return web_search(question)[0]        # just call the tool — tracking is automatic
```

**Or with full control via `@dt.agent` + `dt.init()`:**

```python
dt.init(agent_id="my-agent")  # patches openai, anthropic, httpx, requests globally

@dt.agent()
def run_agent(query: str) -> str:
    ...                        # LLM + HTTP calls tracked automatically
```
## Examples

To verify signals fire end-to-end, run the examples with built-in failure scenarios:

```bash
cd packages/sdk-py

python examples/basic_agent.py # No LLM calls

SCENARIO=tool_loop python examples/langchain_agent.py  # TOOL_LOOP via LangChain callback

SCENARIO=failures python examples/decorator_agent.py   # TOOL_LOOP, RETRY_STORM, RAG_EMPTY_RETRIEVAL

SCENARIO=tool_loop python examples/langfuse_agent.py   # TOOL_LOOP + Langfuse explain
```

Then open the dashboard: **[http://localhost:3000](http://localhost:3000)**

| Endpoint     | URL                                                      |
| ------------ | -------------------------------------------------------- |
| Dashboard    | [http://localhost:3000](http://localhost:3000)           |
| API + docs   | [http://localhost:8002/docs](http://localhost:8002/docs) |
| Ingest (SDK) | [http://localhost:8001](http://localhost:8001)           |

---

## What it detects

15 structural detectors run automatically on every completed run i.e. tool loops, retry storms, context bloat, reasoning stalls, goal abandonment, prompt injection, and more. Each signal includes a plain-English explanation and a suggested fix. Alerts include rate context: whether this is a first occurrence, recurring pattern, or systemic issue affecting ≥10% of runs.

If you are running Langfuse alongside Dunetrace, click **Explain +** on any signal in the dashboard. Dunetrace fetches the full trace, extracts the system prompt in use, and asks an LLM for the specific root cause and fix. For behavioral failures (tool loops, goal abandonment, etc.) it also offers one-click **Apply via Langfuse** to create a new prompt version. Fix effectiveness is tracked automatically i.e. the dashboard shows whether recurrence dropped after the fix was applied.

→ [docs/detectors.md](docs/detectors.md): full detector reference, thresholds, shadow mode

---

## Deploy markers

Correlate failure spikes with releases. Call `mark_deploy()` from your CI/CD pipeline or at application startup:

```python
dt = Dunetrace(api_key="dt_live_...")

# In your deploy script or startup hook
dt.mark_deploy("my-agent", version="v1.4.2", commit="abc1234", env="production")
```

The dashboard overlays blue dashed vertical lines at each deploy boundary on the 30-day detector timeline, so you can immediately see whether a spike in `TOOL_LOOP` or any other failure type started before or after a release.

The call is fire-and-forget — it runs on a background thread and never blocks the caller. No pipeline access is required; it also works with `dt.mark_deploy()` at the top of `app.py`.

---

## Policies

Runtime guardrails that fire mid-run — before a failure propagates. Define a condition and an action; the SDK evaluates it after every tool call and LLM response.

```python
from dunetrace import Dunetrace, PolicyViolation

dt = Dunetrace()

dt.add_policy(
    name="cap tool calls",
    condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
    action={"type": "stop"},
)
dt.add_policy(
    name="cost cap",
    condition={"trigger": "cost_usd", "operator": "gt", "value": 0.50},
    action={"type": "switch_model", "params": {"model": "gpt-4o-mini"}},
)
dt.add_policy(
    name="loop fix",
    condition={"trigger": "signal", "operator": "eq", "value": "TOOL_LOOP"},
    action={"type": "inject_prompt", "params": {"prompt": "Stop repeating tool calls."}},
)
```

| Trigger | What it measures |
|---|---|
| `tool_call_count` | Total tool calls so far |
| `step_count` | Current step index |
| `cost_usd` | Accumulated LLM cost in USD |
| `error_count` | Failed tool calls |
| `finish_reason` | Latest LLM finish_reason |
| `llm_latency_ms` | Latest LLM latency (ms) |
| `signal` | Detector signal name e.g. `"TOOL_LOOP"` |

| Action | Effect |
|---|---|
| `stop` | Raises `PolicyViolation`; run exits with `exit_reason="policy_violation"` |
| `switch_model` | Sets `run.model_override`; read it between LLM calls |
| `inject_prompt` | Appends to `run.prompt_additions`; read with `run.pop_prompt_addition()` |
| `log` | Emits a `policy.triggered` event; no interruption |

Policies can also be created in the dashboard under **Policies** and are fetched automatically by the SDK at run start (60-second TTL per agent).

→ [docs/integrations.md#policies](docs/integrations.md#policies): full reference, operators, examples

---

## Dashboard

![Dashboard overview](dashboard.png)
![Analytics](analytics.png)
![Agent details](agent_detail.png)

Live dashboard at **[http://localhost:3000](http://localhost:3000)**. Auto-refreshes every 15s.

→ [docs/dashboard.md](docs/dashboard.md)

---

## Privacy

No raw content ever leaves your agent process. Every prompt, tool argument, and model output is SHA-256 hashed before transmission.

→ [docs/architecture.md](docs/architecture.md)

---

## Alerts

Slack and generic webhook (PagerDuty, Linear, custom).

![Slack alert](slack-alert.png)

→ [docs/alerts.md](docs/alerts.md)

---

## Integrations

- [Custom Python agent — `@dt.trace` / `@dt.tool` decorators, middleware, manual](docs/integrate-custom-python-agent.md)
- [LangChain / LangGraph](docs/integrate-langchain-agent.md)
- [TypeScript / JavaScript — raw HTTP, no npm package needed](docs/integrate-typescript-agent.md)
- [Langfuse — connect traces for deep root-cause analysis and one-click autofix](docs/integrations.md#langfuse)
- [FastAPI / Flask / ASGI / WSGI / OpenTelemetry / Loki](docs/integrations.md)

---

## Architecture

```
Agent Code
  └─► Dunetrace SDK        (hashes content → ingest events + OTel spans)
        └─► Ingest API      (POST /v1/ingest → Postgres)
                ├─► Detector       (poll → RunState → 15 detectors → signals)
                ├─► Alerts         (poll → explain → Slack / webhook)
                └─► Customer API   (runs, signals, explanations → dashboard)
        ├─► stdout NDJSON   (emit_as_json=True → Loki / Grafana Alloy)
        └─► OTel exporter   (otel_exporter=… → Tempo / Honeycomb / Datadog)
```

→ [docs/architecture.md](docs/architecture.md)

---

## Running tests

```bash
# SDK + services (no Docker required)
PYTHONPATH=packages/sdk-py:services/explainer:services/alerts:services/detector \
  python -m pytest packages/sdk-py/tests/ services/explainer/tests/ \
    services/detector/tests/ services/alerts/tests/ -q
```

---

## Requirements

- Python 3.11+
- Docker + Docker Compose
- PostgreSQL 16+ (included in Docker Compose)

## Contributing

1. Fork the repo and create a branch
2. Make your changes and add tests
3. Run the test suite
4. Open a pull request with a clear description of what and why

For larger changes (new detectors, architecture changes), open an issue first.

## Star us ⭐

If Dunetrace looks useful, a GitHub star helps others find the project.

[![Star History Chart](https://api.star-history.com/svg?repos=dunetrace/dunetrace&type=Date)](https://star-history.com/#dunetrace/dunetrace&Date)

## Contact

[dunetrace@gmail.com](mailto:dunetrace@gmail.com)

## License

[Apache 2.0](LICENSE)
