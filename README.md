# DuneTrace

Production observability for AI agents. DuneTrace detects, explains, and alerts on agent failures in real-time — without sending raw prompts or outputs anywhere.

## What it does

AI agents fail in ways that are invisible to traditional monitoring: tool loops, goal abandonment, prompt injection, RAG retrievals returning nothing, context bloat. DuneTrace instruments your agent, runs structural detectors on the event stream, and delivers actionable alerts with deterministic explanations and code fix suggestions.

- **Zero-overhead SDK** — ring buffer + background drain thread; your agent never blocks on I/O
- **Content hashing** — only SHA-256 hashes of prompts/outputs leave your process
- **Deterministic explanations** — no LLM calls in the pipeline; <1ms per explanation
- **Shadow mode** — validate detector precision on real traffic before waking anyone up

## Architecture

```
Agent Code
  └─► Dunetrace SDK          (instrument runs, emit hashed events)
        └─► Ingest API        (POST /v1/ingest → Postgres)
              └─► Detector    (poll → reconstruct RunState → run detectors)
                    └─► Alerts (poll → explain → format → Slack / webhook)
                          └─► Customer API  (query runs, signals, explanations)
```

**Services**

| Service | Port | Purpose |
|---|---|---|
| `services/ingest` | 8001 | Accept SDK events |
| `services/detector` | — | Tier 1 detection worker |
| `services/explainer` | — | Deterministic explanation templates |
| `services/alerts` | — | Slack / webhook delivery |
| `services/api` | 8002 | Customer REST API |

**SDK** lives in `packages/sdk-py`. The dashboard UI is in `packages/dashboard-ui`.

## Tier 1 Detectors

| Detector | What it catches |
|---|---|
| `TOOL_LOOP` | Same tool called ≥3× in a 5-step window |
| `TOOL_THRASHING` | Agent alternates between exactly two tools |
| `TOOL_AVOIDANCE` | Final answer given without calling available tools |
| `GOAL_ABANDONMENT` | Tool use stops, then ≥4 consecutive LLM calls |
| `PROMPT_INJECTION` | Pattern-matched injection signatures in inputs |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned 0 results but agent answered |
| `LLM_TRUNCATION_LOOP` | `finish_reason=length` fires ≥2 times |
| `CONTEXT_BLOAT` | Prompt tokens grow 3× from first to last LLM call |

## Quick Start (Docker)

```bash
cp .env.example .env
# Optionally add SLACK_WEBHOOK_URL to .env for real alerts
docker compose up -d
```

- Ingest API: http://localhost:8001
- Customer API: http://localhost:8002
- API docs: http://localhost:8002/docs
- Dashboard: open `packages/dashboard-ui/index.html` in a browser

## SDK Usage

```python
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_dev_local", endpoint="http://localhost:8001")

with dt.run(user_input="Find top 3 Python repos", agent_id="my-agent") as run:
    run.llm_called(model="gpt-4o", prompt="...", tools=["search"])
    run.tool_called("search", {"q": "python repos"})
    run.tool_responded("search", result="...", result_count=10)
    run.llm_responded(output="Here are the top 3...", finish_reason="stop", tokens=250)
    run.final_answer("Here are the top 3 Python repos...")
```

### LangChain

```python
from dunetrace.adapters.langchain import DunetraceCallback

callback = DunetraceCallback(
    client=dt,
    system_prompt=SYSTEM_PROMPT,
    model="gpt-4o",
    tools=["search", "calculator"],
)

agent.invoke({"input": "..."}, config={"callbacks": [callback]})
```

See `packages/sdk-py/examples/` for full demos.

## Local Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run services individually:

```bash
# Ingest API
cd services/ingest && uvicorn app.main:app --reload --port 8001

# Detector worker
cd services/detector && python -m app.worker

# Alerts worker
cd services/alerts && python -m alerts_svc.worker

# Customer API
cd services/api && uvicorn api_svc.main:app --reload --port 8002
```

## Tests

```bash
source .venv/bin/activate

# SDK detectors
PYTHONPATH=packages/sdk-py python -m unittest discover -s packages/sdk-py/tests -p "test_detectors.py" -v

# Explainer templates
PYTHONPATH=packages/sdk-py:services/explainer python -m unittest discover -s services/explainer/tests -p "test_explainer.py" -v

# Detector worker
PYTHONPATH=packages/sdk-py:services/detector python -m unittest discover -s services/detector/tests -p "test_worker.py" -v

# Alerts worker
PYTHONPATH=packages/sdk-py:services/explainer:services/alerts python -m unittest discover -s services/alerts/tests -p "test_alerts.py" -v

# Customer API
PYTHONPATH=packages/sdk-py:services/explainer:services/api python -m unittest discover -s services/api/tests -p "test_api.py" -v
```

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` to get started.

See [`docs/07-alerts.md`](docs/07-alerts.md) for Slack and webhook setup, and [`docs/shadow-mode.md`](docs/shadow-mode.md) for how to graduate detectors from shadow to live.

## Documentation

Full docs are in [`docs/`](docs/):

- [Introduction](docs/01-introduction.md)
- [Core Concepts](docs/02-core-concepts.md)
- [Architecture](docs/03-architecture.md)
- [SDK Reference](docs/04-sdk-reference.md)
- [Detectors](docs/05-detectors.md)
- [Tracing](docs/06-tracing.md)
- [Alerts](docs/07-alerts.md)
- [API Reference](docs/08-api-reference.md)
- [Dashboard](docs/09-dashboard.md)
- [Getting Started](docs/10-getting-started.md)
- [Shadow Mode](docs/shadow-mode.md)
- [FAQ](docs/11-faq.md)

## Requirements

- Python 3.12+
- PostgreSQL 16+
- Docker + Docker Compose (for the full stack)
