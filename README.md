# Dunetrace

[![PyPI version](https://img.shields.io/pypi/v/dunetrace.svg)](https://pypi.org/project/dunetrace/)
[![Python versions](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/dunetrace/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dunetrace?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dunetrace)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Privacy-safe observability for AI agents at runtime. Detects tool loops, context bloat, prompt injection, and other failure patterns and get alerts immediately. Zero raw content transmitted. All text is SHA-256 hashed before leaving the agent process.


![Dunetrace demo](dunetrace-demo.gif)

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
pip install 'dunetrace[langchain]' langchain-openai      # OpenAI
pip install 'dunetrace[langchain]' langchain-anthropic   # Anthropic
pip install 'dunetrace[langchain]' langchain-google-genai  # Gemini
```

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler
from langchain.agents import create_agent

dt = Dunetrace()
callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

agent = create_agent(llm, tools, system_prompt="You are a helpful assistant.")
result = agent.invoke(
    {"messages": [("human", user_input)]},
    config={"callbacks": [callback]},
)
```
## Running the examples

**Basic agent** (no framework, simulates tool loops, prompt injection, RAG failures):

```bash
cd packages/sdk-py
pip install dunetrace
python examples/basic_agent.py
```

**LangChain agent** (real OpenAI calls, auto-instrumented via callback):

```bash
cd packages/sdk-py
pip install 'dunetrace[langchain]' langchain-openai
OPENAI_API_KEY=sk-... python examples/langchain_agent.py

# Force a tool-loop scenario:
OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/langchain_agent.py
```

Both examples send events to `http://localhost:8001` by default. Start the backend first (`docker compose up -d`). Override the endpoint with `DUNETRACE_ENDPOINT=http://your-host:8001`.

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

## Grafana / Loki integration

Pass `emit_as_json=True` to write every event to stdout as a Loki-compatible NDJSON line. Use this when your infrastructure already has a log collector (Promtail, Grafana Alloy, Fluentd) pointed at your agent process i.e Dunetrace events flow into your existing stack without any additional forwarding setup.

```python
dt = Dunetrace(emit_as_json=True)
```

Each line is structured JSON with `ts` (RFC3339), `level`, `logger`, `event_type`, `agent_id`, `run_id`, `step_index`, and `payload`. `event_type` and `agent_id` are natural Loki stream labels; `run_id` lets you group all events from a single run in a Grafana query.

`emit_as_json` and HTTP ingest are independent — both can be active at the same time.

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


## Dashboard

The dashboard starts automatically with `docker compose up -d` and is served at `http://localhost:3000`. It fetches live data from the API at `http://localhost:8002` and auto-refreshes every 10 seconds.

![Dashboard overview](dashboard.png)

Click any run to open the detail panel and get metrics, detected signals with fix suggestions, and a step-by-step timeline.

![Run detail panel](agentRun_detail.png)

## What it detects

| Detector | What it catches | Severity |
|---|---|---|
| `SLOW_STEP` | Tool call >15s or LLM call >30s | MEDIUM/HIGH |
| `TOOL_AVOIDANCE` | Final answer given without calling available tools | MEDIUM |
| `GOAL_ABANDONMENT` | Tool use stops, then ≥4 consecutive LLM calls with no exit | MEDIUM |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned 0 results or relevance score <0.3, but agent answered | MEDIUM |
| `CONTEXT_BLOAT` | Prompt tokens grow 3× from first to last LLM call | MEDIUM |
| `STEP_COUNT_INFLATION` | Run used >2× the P75 step count for this agent | MEDIUM |
| `FIRST_STEP_FAILURE` | Error or empty output at step ≤2 | MEDIUM |
| `REASONING_STALL` | LLM:tool-call ratio ≥4× — agent reasoning without acting | MEDIUM |
| `TOOL_LOOP` | Same tool called ≥3× in a 5-tool-call window | HIGH |
| `TOOL_THRASHING` | Agent alternates between exactly two tools | HIGH |
| `LLM_TRUNCATION_LOOP` | `finish_reason=length` fires ≥2 times | HIGH |
| `RETRY_STORM` | Same tool fails 3+ times in a row; evidence includes whether retries used identical args and identical failure reason | HIGH |
| `EMPTY_LLM_RESPONSE` | Model returned zero-length output with `finish_reason=stop` | HIGH |
| `CASCADING_TOOL_FAILURE` | 3+ consecutive failures across 2+ distinct tools | HIGH |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns | CRITICAL |

Detector thresholds are configurable per-instance. See `packages/sdk-py/dunetrace/detectors.py`.

## What's supported now

**SDK**
- Python SDK: zero external dependencies, <1ms overhead, in-process ring buffer
- LangChain callback handler (auto-instruments `AgentExecutor`)
- Manual instrumentation API (`llm_called`, `tool_called`, `retrieval_called`, etc.)
- `emit_as_json=True` - stdout NDJSON emitter compatible with Grafana Loki / Promtail
- `otel_exporter=DunetraceOTelExporter(provider)` -  OTel span exporter for Tempo, Honeycomb, Datadog, Jaeger
- `run.external_signal("rate_limit", source="openai")` - annotate agent steps with infrastructure context; `SLOW_STEP` signals include coincident infrastructure events
- All content SHA-256 hashed before leaving the process - no raw prompts or outputs transmitted

**Detection (15 detectors)**
- Tool behaviour: `TOOL_LOOP`, `TOOL_THRASHING`, `TOOL_AVOIDANCE`, `RETRY_STORM`, `CASCADING_TOOL_FAILURE`
- LLM behaviour: `LLM_TRUNCATION_LOOP`, `CONTEXT_BLOAT`, `EMPTY_LLM_RESPONSE`, `GOAL_ABANDONMENT`, `REASONING_STALL`
- RAG: `RAG_EMPTY_RETRIEVAL`
- Security: `PROMPT_INJECTION_SIGNAL`
- Performance: `SLOW_STEP`
- Run health: `FIRST_STEP_FAILURE`, `STEP_COUNT_INFLATION`
- Configurable thresholds via keyword overrides on each detector class

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
  └─► Dunetrace SDK              (instrument runs, emit hashed events)
        ├─► Ingest API           (POST /v1/ingest -> Postgres)        [default]
        │     └─► Detector       (poll -> reconstruct RunState -> run detectors)
        │           └─► Alerts   (poll -> explain -> Slack / webhook)
        │                 └─► Customer API  (query runs, signals, explanations)
        ├─► stdout NDJSON        (emit_as_json=True -> Loki / Grafana Alloy)
        └─► OTel spans           (otel_exporter=… -> Tempo / Honeycomb / Datadog)
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

All 15 built-in detectors are live (`shadow = false`) by default i.e. no action needed. If you write a custom detector, it starts in shadow mode (stored in the DB, visible in the API, but never alerted) until you add its name to `LIVE_DETECTORS`:

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

## Star us (⭐)

If Dunetrace look useful, give a GitHub star (⭐) and it helps others find the project.

## Contributing

Contributions are welcome. To get started:

1. Fork the repo and create a branch
2. Make your changes, e.g. add tests for new detectors or SDK changes
3. Run the relevant test suite (see [Running tests](#running-tests))
4. Open a pull request with a clear description of what and why

For larger changes (new detectors, architecture changes), open an issue first to discuss the approach.

## Contact

Questions, feedback, or just want to say hi - [dunetrace@gmail.com](mailto:dunetrace@gmail.com)

## License

[MIT](LICENSE)