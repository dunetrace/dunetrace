# Dunetrace

![Dunetrace](dunetrace.png)

**Real-time monitoring for production AI agents. Catches failures before your users do.**

[![PyPI version](https://img.shields.io/pypi/v/dunetrace.svg)](https://pypi.org/project/dunetrace/)
[![Python versions](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/dunetrace/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dunetrace?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dunetrace)
[![GitHub Stars](https://img.shields.io/github/stars/dunetrace/dunetrace?style=flat&logo=github)](https://github.com/dunetrace/dunetrace)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/yxFjATwHW4)

---

## The problem

AI agents fail in ways that look fine from the outside.

Your API returns 200. Your logs are clean. But the agent just called the same tool 12 times in a row, burned $10 in tokens, and gave the user a wrong answer or no answer at all. The user didn't get a response. You didn't get an alert.

**Langfuse and similar tools answer "what happened?" after you already know something broke.**

Dunetrace answers a different question: **is something breaking right now?**  and fires a Slack alert within 15 seconds of the run completing.

---

## What it does

**Monitor** - watches every run as it completes: tool calls, LLM calls, latency, token usage, retrieval results.

**Detect** - 17 structural detectors run automatically. No LLM, no configuration. Catches tool loops, retry storms, context bloat, runaway cost, slow sessions, goal abandonment, and more.

**Diagnose** - each alert includes a plain-English explanation: what happened, why it matters, and a concrete fix. If you use Langfuse, click **Explain +** to get an LLM root-cause analysis against the actual trace.

**Fix** - one click to apply a prompt fix via Langfuse, or open a GitHub PR with a code change. Fix effectiveness is tracked automatically.

---

## Why it's different

| | Dunetrace | Langfuse / LangSmith |
|---|---|---|
| **When it fires** | Within 15s of run completion | You query it after you notice a problem |
| **What it watches** | Structural failure patterns | Raw trace data |
| **Alert channel** | Slack / webhook / Dashboard| Dashboard only |
| **Fix path** | One-click prompt apply or GitHub PR | Manual |

Dunetrace is not a replacement for tracing tools, it's the layer that tells you *when to look*.

---

## Quick Start

### 1. Start the backend

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env && docker compose build && docker compose up -d
```

### 2. Install the SDK

```bash
pip install dunetrace
pip install 'dunetrace[langchain]'          # LangChain / LangGraph
pip install 'dunetrace[langchain,langfuse]' # + Langfuse deep analysis
```

### 3. Instrument your agent

```python
from dunetrace import Dunetrace

dt = Dunetrace()

@dt.tool
def web_search(query: str) -> list: ...

@dt.trace
def my_agent(question: str) -> str:
    return web_search(question)[0]
```


### Examples

To verify signals fire end-to-end, run the built-in failure scenarios:

```bash
cd packages/sdk-py

python examples/basic_agent.py                          # No LLM calls
SCENARIO=tool_loop python examples/langchain_agent.py   # TOOL_LOOP via LangChain
SCENARIO=failures python examples/decorator_agent.py    # TOOL_LOOP, RETRY_STORM, RAG_EMPTY_RETRIEVAL
SCENARIO=tool_loop python examples/langfuse_agent.py    # TOOL_LOOP + Langfuse explain
```
Open the dashboard: **[http://localhost:3000](http://localhost:3000)**

| | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API + docs | http://localhost:8002/docs |
| Ingest (SDK) | http://localhost:8001 |

---

## Detectors

17 detectors run on every completed run, no configuration required.

| Signal | What it catches |
|---|---|
| `TOOL_LOOP` | Same tool called repeatedly with identical args |
| `COST_SPIKE` | Total token consumption unusually high vs per-agent baseline |
| `SESSION_LATENCY` | Wall-clock run time anomalously long vs per-agent baseline |
| `CONTEXT_BLOAT` | Prompt tokens growing unsustainably across LLM calls |
| `RETRY_STORM` | Tool failing, agent retrying it repeatedly |
| `CASCADING_TOOL_FAILURE` | Multiple different tools failing in sequence |
| `STEP_COUNT_INFLATION` | Far more steps than the agent's baseline |
| `SLOW_STEP` | Single step latency well above threshold |
| `GOAL_ABANDONMENT` | Agent stopped using tools before finishing |
| `REASONING_STALL` | Too many LLM calls per tool call — agent deliberating in circles |
| `LLM_TRUNCATION_LOOP` | Model output truncated repeatedly |
| `CONTEXT_BLOAT` | Prompt token count growing unsustainably |
| `TOOL_THRASHING` | Oscillating between two tools, unable to commit |
| `TOOL_AVOIDANCE` | Agent answered without using any tools |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned nothing, agent answered anyway |
| `EMPTY_LLM_RESPONSE` | Model returned an empty response |
| `FIRST_STEP_FAILURE` | Failed on the first step — config or setup issue |
| `PROMPT_INJECTION_SIGNAL` | Input matched adversarial injection patterns |

Each alert includes: what fired, why it matters, a concrete fix, and a rate context line (first occurrence / recurring / systemic).

→ [docs/detectors.md](docs/detectors.md)

---

## Dashboard

![Dashboard overview](dashboard.png)
![Analytics](analytics.png)
![Agent details](agent_detail.png)

Live at **[http://localhost:3000](http://localhost:3000)**. Auto-refreshes every 15s.

→ [docs/dashboard.md](docs/dashboard.md)

---

## Alerts

Slack and generic webhook (PagerDuty, Linear, custom). 

![Slack alert](slack-alert.png)

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_MIN_SEVERITY=LOW   # LOW | MEDIUM | HIGH | CRITICAL
```

A weekly digest (Monday 9am UTC) summarises top failure types, most-affected agents, and systemic patterns. Enable with `DIGEST_ENABLED=true`.

→ [docs/alerts.md](docs/alerts.md)

---

## Diagnose with Langfuse

Connect Langfuse to get LLM-powered root-cause analysis on any signal.

Click **Explain +** on any alert in the dashboard. Dunetrace fetches the full trace, extracts the system prompt, and asks an LLM for the specific root cause and fix.

- **Prompt fixes** (tool loops, goal abandonment, etc.)
— **Apply via Langfuse** creates a new prompt version in one click.
- **Code/infra fixes** (context bloat, slow steps, cost spikes, etc.) 
— **Open PR on GitHub** creates a draft PR with a LLM-generated unified diff.

Fix effectiveness is tracked: the dashboard shows whether recurrence dropped after a fix was applied.

---

## Policies

Runtime guardrails that fire mid-run — before a failure propagates.

```python
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
```

Policies can also be created in the dashboard and are fetched automatically by the SDK (60s TTL).

→ [docs/integrations.md#policies](docs/integrations.md#policies)

---

## Deploy markers

Correlate failure spikes with releases.

```python
dt.mark_deploy("my-agent", version="v1.4.2", commit="abc1234", env="production")
```

The dashboard overlays blue dashed lines at each deploy boundary so you can immediately see whether a spike started before or after a release.

---

## Privacy

No raw content ever leaves your agent process. Every prompt, tool argument, and model output is SHA-256 hashed before transmission.

→ [docs/architecture.md](docs/architecture.md)

---

## Architecture

```
Agent Code
  └─► Dunetrace SDK        (hashes content → ingest events)
        └─► Ingest API      (POST /v1/ingest → Postgres)
                ├─► Detector       (poll → 17 detectors → signals)
                ├─► Alerts         (poll → explain → Slack / webhook)
                └─► Customer API   (runs, signals, explanations → dashboard)
```

→ [docs/architecture.md](docs/architecture.md)

---

## Integrations

- [Custom Python agent](docs/integrate-custom-python-agent.md)
- [LangChain / LangGraph](docs/integrate-langchain-agent.md)
- [TypeScript / JavaScript](docs/integrate-typescript-agent.md)
- [Langfuse, FastAPI, Flask, OpenTelemetry, Loki](docs/integrations.md)

---

## Running tests

```bash
make test
```

---

## Requirements

- Python 3.11+
- Docker + Docker Compose

## Contributing

Fork, branch, change, test, PR. For larger changes (new detectors, architecture changes), open an issue first.

## Star us ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=dunetrace/dunetrace&type=Date)](https://star-history.com/#dunetrace/dunetrace&Date)

## Contact

[dunetrace@gmail.com](mailto:dunetrace@gmail.com)

## License

[Apache 2.0](LICENSE)
