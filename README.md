# DuneTrace

Behavioral observability for AI agents. Detects tool loops, context bloat, prompt injection, and other failure patterns. Zero raw content transmitted.

## Quickstart

There are two parts:
- **Backend** (clone + Docker) : runs the ingest API, detector, alerts worker, and dashboard API on your machine
- **SDK** (pip install) : goes into your agent's Python environment to emit events to the backend

They can run on different machines i.e. the SDK just needs the ingest endpoint to be reachable.

**1. Start the backend**

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace
cp .env.example .env
docker compose build
docker compose up -d
```

- Ingest: http://localhost:8001 (SDK POST endpoint)
- API + docs: http://localhost:8002/docs
- Dashboard: http://localhost:3000

**2. Install the SDK** (in your agent's environment)

```bash
pip install dunetrace
```

**3. Instrument your agent**

```python
from dunetrace import Dunetrace

dt = Dunetrace()  # points to localhost:8001

# agent_id groups all runs from this agent in the dashboard
with dt.run("my-agent", user_input=user_input) as run:
    result = your_agent(user_input)
```

Runs appear in the dashboard immediately.

## LangChain

```bash
pip install dunetrace[langchain]
```

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace()

agent_executor = AgentExecutor(
    agent=agent, tools=tools,
    callbacks=[DunetraceCallbackHandler(dt, agent_id="my-agent")],
)
```
## Report events manually

```python
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
Manual reporting is the fallback until a native integration exists for your framework.


## Dashboard

```bash
python -m http.server 3000 -d dashboard
# then open http://localhost:3000
```

The dashboard fetches live data from the API at `http://localhost:8002` and auto-refreshes every 10 seconds.

## What it detects

| Detector | What it catches | Severity |
|---|---|---|
| `TOOL_LOOP` | Same tool called ≥3× in a 5-step window | HIGH |
| `TOOL_THRASHING` | Agent alternates between exactly two tools | HIGH |
| `TOOL_AVOIDANCE` | Final answer given without calling available tools | MEDIUM |
| `GOAL_ABANDONMENT` | Tool use stops, then ≥4 consecutive LLM calls | MEDIUM |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns | CRITICAL |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned 0 results but agent answered | MEDIUM |
| `LLM_TRUNCATION_LOOP` | `finish_reason=length` fires ≥2 times | HIGH |
| `CONTEXT_BLOAT` | Prompt tokens grow 3× from first to last LLM call | MEDIUM |
| `SLOW_STEP` | Tool call >15s or LLM call >30s | MEDIUM/HIGH |
| `RETRY_STORM` | Same tool fails 3+ times in a row | HIGH |
| `EMPTY_LLM_RESPONSE` | Model returned zero-length output | HIGH |
| `STEP_COUNT_INFLATION` | Run used >2× the P75 step count for this agent | MEDIUM |
| `CASCADING_TOOL_FAILURE` | 3+ consecutive failures across 2+ distinct tools | HIGH |
| `FIRST_STEP_FAILURE` | Error or empty output at step ≤2 | MEDIUM |

Detector thresholds are configurable. See `detectors.yml` in the repo root.

## What's supported now

**SDK**
- Python SDK :  zero external dependencies, <1ms overhead, in-process ring buffer
- LangChain callback handler (auto-instruments `AgentExecutor`)
- Manual instrumentation API (`llm_called`, `tool_called`, `retrieval_called`, etc.)
- All content SHA-256 hashed before leaving the process i.e. no raw prompts or outputs transmitted

**Detection (14 detectors)**
- Tool behaviour: `TOOL_LOOP`, `TOOL_THRASHING`, `TOOL_AVOIDANCE`, `RETRY_STORM`, `CASCADING_TOOL_FAILURE`
- LLM behaviour: `LLM_TRUNCATION_LOOP`, `CONTEXT_BLOAT`, `EMPTY_LLM_RESPONSE`, `GOAL_ABANDONMENT`
- RAG: `RAG_EMPTY_RETRIEVAL`
- Security: `PROMPT_INJECTION_SIGNAL`
- Performance: `SLOW_STEP`
- Run health: `FIRST_STEP_FAILURE`, `STEP_COUNT_INFLATION`
- Configurable thresholds per agent category via `detectors.yml`

**Infrastructure**
- Self-hosted Docker Compose stack (Postgres + ingest + detector + alerts + API)
- Slack and webhook alerting
- REST API with deterministic natural-language explanations for every signal
- `AUTH_MODE=dev` (no key needed locally) and `AUTH_MODE=prod` for production

## What's coming

**SDK integrations**
- OpenAI Agents SDK
- CrewAI, AutoGen, LlamaIndex, Haystack

**Detection**
- Per-agent-category detector tuning
- Tier 2 detectors: semantic drift, hallucination signal, plan–action mismatch

**Platform**
- Dashboard: filter runs by severity/date range, search by input hash

## Architecture

```
Agent Code
  └─► Dunetrace SDK          (instrument runs, emit hashed events)
        └─► Ingest API        (POST /v1/ingest → Postgres)
              └─► Detector    (poll → reconstruct RunState → run detectors)
                    └─► Alerts (poll → explain → Slack / webhook)
                          └─► Customer API  (query runs, signals, explanations)
```

| Service | Port | Purpose |
|---|---|---|
| `services/ingest` | 8001 | Accept SDK events |
| `services/detector` | — | Detection worker |
| `services/explainer` | — | Deterministic explanation library |
| `services/alerts` | — | Slack / webhook delivery |
| `services/api` | 8002 | REST API |

## Slack alerts

Add these lines to your `.env` to enable Slack notifications:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
SLACK_CHANNEL=#agent-alerts
SLACK_MIN_SEVERITY=HIGH   # LOW | MEDIUM | HIGH | CRITICAL
```

Get a webhook URL from [api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks).

Restart the alerts worker to pick up the change:

```bash
docker compose restart alerts
```

The alerts worker sends a message to Slack for every detected signal at or above `SLACK_MIN_SEVERITY`. 

**Shadow mode:** each signal is stored with a `shadow` flag. The alerts worker only delivers signals where `shadow = false`. Whether a signal is shadow or live is decided at detection time based on `LIVE_DETECTORS` in `services/detector/detector_svc/db.py`.

All 14 built-in detectors are live (`shadow = false`) by default i.e. no action needed. If you write a custom detector, it starts in shadow mode (stored in the DB, visible in the API, but never alerted) until you add its name to `LIVE_DETECTORS`:

```python
# services/detector/detector_svc/db.py
LIVE_DETECTORS: set[str] = {
    "TOOL_LOOP",
    "YOUR_NEW_DETECTOR",   # add here once precision > 80%
    ...
}
```

After editing, rebuild and restart the detector:

```bash
docker compose build detector && docker compose restart detector
```

**Generic webhook** (PagerDuty, Linear, custom endpoints):

```bash
WEBHOOK_URL=https://your-endpoint.example.com/alerts
WEBHOOK_SECRET=your-hmac-secret   # optional — enables HMAC-SHA256 signature header
```

Both destinations can be active at the same time. Leave a variable blank to disable that destination.

## Tuning detectors

Edit `detectors.yml` in the repo root. No code change or rebuild needed:

```yaml
default:
  tool_loop:
    threshold: 2        # lower = catch loops sooner
  context_bloat:
    growth_factor: 4.0  # raise for agents that intentionally accumulate context
```

Restart the detector to apply:

```bash
docker compose restart detector
```

Per-agent-category overrides are supported i.e. a named section inherits from `default` and overrides only what you specify:

```yaml
web-research:
  tool_loop:
    threshold: 5    # search agents legitimately repeat queries across pages
```

All thresholds and their defaults are documented in `detectors.yml`.

## Running tests

```bash
# Explainer
PYTHONPATH=packages/sdk-py:services/explainer pytest services/explainer/tests/ -v

# Detector worker
PYTHONPATH=packages/sdk-py:services/detector pytest services/detector/tests/ -v

# Alerts worker
PYTHONPATH=packages/sdk-py:services/explainer:services/alerts pytest services/alerts/tests/ -v

# API
PYTHONPATH=packages/sdk-py:services/explainer:services/api pytest services/api/tests/ -v
```

## Requirements

- Python 3.11+
- PostgreSQL 16+ (included in Docker Compose)
- Docker + Docker Compose

## If this helps you

If DuneTrace saves you debugging time, a GitHub star (⭐) goes a long way and it helps others find the project.

## Contributing

Contributions are welcome. To get started:

1. Fork the repo and create a branch
2. Make your changes, e.g. add tests for new detectors or SDK changes
3. Run the relevant test suite (see [Running tests](#running-tests))
4. Open a pull request with a clear description of what and why

For larger changes (new detectors, architecture changes), open an issue first to discuss the approach.

## Contact

Questions, feedback, or just want to say hi — [dunetrace@gmail.com](mailto:dunetrace@gmail.com)

## License

MIT
