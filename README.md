# Dunetrace

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

LangSmith/Langfuse answer "what happened?" after you already know something broke. Dunetrace answers a different question: **"is something breaking right now?"**

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
pip install dunetrace                    # any framework
pip install 'dunetrace[langchain]'      # LangChain / LangGraph
```

### 3. Instrument your agent

```python
from dunetrace import Dunetrace

dt = Dunetrace()               # no api_key needed for local dev
dt.init(agent_id="my-agent")  # patches openai, anthropic, httpx, requests globally

@dt.agent()
def run_agent(query: str) -> str:
    ...                        # LLM + HTTP calls tracked automatically
```

To verify signals fire end-to-end, run the examples with built-in failure scenarios:

```bash
cd packages/sdk-py
SCENARIO=failures python examples/decorator_agent.py   # TOOL_LOOP, RETRY_STORM, RAG_EMPTY_RETRIEVAL
SCENARIO=tool_loop python examples/langchain_agent.py  # TOOL_LOOP via LangChain callback
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

→ [docs/detectors.md](docs/detectors.md): full detector reference, thresholds, shadow mode

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

- [Custom Python agent - decorator, middleware, manual](docs/integrate-custom-python-agent.md)
- [LangChain / LangGraph](docs/integrate-langchain-agent.md)
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
