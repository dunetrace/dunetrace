# Architecture

Dunetrace is a pipeline of five independent services communicating through a shared Postgres database.

---

## System Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              Your Agent                                   │
│                                                                           │
│   with dt.run(user_input) as run:                                         │
│       run.tool_called("web_search", {...})                                │
│       run.tool_responded("web_search", ...)                               │
│       run.external_signal("rate_limit", source="openai")                  │
└──────┬──────────────────────────────┬──────────────────────────┬──────────┘
       │  HTTP POST /v1/ingest        │  stdout NDJSON           │  OTel spans
       │  (async, 202)                │  (emit_as_json=True)     │  (otel_exporter=…)
       ▼                              ▼                          ▼
┌─────────────────────┐  ┌────────────────────────┐  ┌──────────────────────┐
│  Ingest API  :8001  │  │  Loki / Grafana Alloy  │  │  OTel Collector      │
│                     │  │                        │  │  (Tempo / Honeycomb  │
│  FastAPI ·          │  │  Promtail pipeline     │  │   / Datadog / Jaeger)│
│  validates ·        │  │  → Grafana dashboards  │  │                      │
│  202 immediately    │  └────────────────────────┘  └──────────────────────┘
│  BackgroundTask     │
│  writes to Postgres │
└──────────┬──────────┘
           │  writes: events table
           ▼
┌──────────────────────────────────────────────────────────────┐
│                      Postgres                                │
│                                                             │
│   events           failure_signals   processed_runs         │
│   api_keys                                                  │
└────────┬───────────────────────────────────────┬────────────┘
         │  polls every 5s                        │  polls every 10s
         ▼                                        ▼
┌─────────────────────┐              ┌─────────────────────────┐
│   Detector Worker   │              │    Alerts Worker        │
│                     │              │                         │
│  Reconstructs       │              │  Fetches unalerted      │
│  RunState from      │              │  shadow=FALSE signals   │
│  events             │              │  → explain()            │
│  Runs 15 detectors  │              │  → format Slack/webhook │
│  Writes signals     │              │  → HTTP POST with retry │
└─────────────────────┘              └─────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  Customer API  :8002                         │
│                                                             │
│   GET /v1/agents                  GET /v1/runs/{id}         │
│   GET /v1/agents/{id}/runs        GET /v1/runs/{id}/events  │
│   GET /v1/agents/{id}/signals     GET /v1/agents/{id}/insights│
│   Read-only · bearer token auth · explains signals inline   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  Dashboard  :3000                            │
│                                                             │
│   Static HTML served by nginx (Docker)                      │
│   Fetches live data from Customer API · auto-refreshes 15s  │
│   Pages: Overview · Runs · Alerts · Analytics · Heatmap     │
│          Agents · Compare runs · Detectors                  │
└──────────────────────────────────────────────────────────────┘
```

---

## SDK Output Modes

The SDK supports three independent output modes that can be combined:

| Mode | How to enable | Destination | Use case |
|---|---|---|---|
| HTTP ingest (default) | `endpoint="http://…"` (default); set `endpoint=None` to disable | Ingest API → Postgres → Detector | Full pipeline: detection, alerts, dashboard |
| Loki NDJSON | `emit_as_json=True` | stdout → Promtail/Alloy → Loki | Existing Grafana stack integration |
| OTel spans | `otel_exporter=DunetraceOTelExporter(provider)` | OTel collector → Tempo / Honeycomb / Datadog | Infra metric correlation |

All three modes can be active simultaneously. OTel and NDJSON are zero-cost when disabled. Pass `endpoint=None` to use OTel or NDJSON without any HTTP ingest (useful for local testing or pure-OTel deployments).

### emit_as_json=True

Writes one Loki-compatible NDJSON line to stdout per event. Fields match Promtail pipeline stages:

```
{"ts":"2026-03-01T12:00:00.123456Z","level":"info","logger":"dunetrace",
 "event_type":"tool.called","agent_id":"my-agent","run_id":"…","step_index":3,
 "payload":{…}}
```

Each line is written atomically under a lock i.e. no interleaving even when the agent is multi-threaded.

### OTel span exporter

`DunetraceOTelExporter` translates `AgentEvent` objects into OpenTelemetry spans in real time:

```
Trace (trace_id = run_id as 128-bit int)
└── Span: "agent_run"         [dunetrace.agent_id, dunetrace.model, …]
    ├── Span: "llm_call"      [gen_ai.request.model, gen_ai.usage.*, …]
    ├── Span: "tool_call"     [dunetrace.tool_name, dunetrace.success, …]
    │   └── SpanEvent: "rate_limit"   (from run.external_signal())
    └── Span: "retrieval"     [dunetrace.index_name, dunetrace.result_count]
```

At run end, Tier 1 detectors run on the completed `RunState`. Each signal is written as indexed attributes on the root span (`dunetrace.signal.0.failure_type`, `.severity`, `.confidence`, `.evidence.*`). HIGH/CRITICAL signals set `span.status = ERROR`.

**Two independent detector passes — thresholds may differ:**

| | OTel path | Server-side path |
|---|---|---|
| Where | Client-side, in the SDK process | Detector worker, server-side |
| When | At run end, before span export | After ingestion, on next poll cycle |
| Thresholds | SDK defaults (`TIER1_DETECTORS` hardcoded in `detectors.py`) | `detectors.yml` (configurable, loaded at worker startup) |
| Output | Root span attributes | `failure_signals` table → alerts → dashboard |
| Works without server | Yes | No |

If you tune thresholds in `detectors.yml`, the OTel span annotations and the dashboard signals can disagree. The server-side path is the source of truth for alerts and the dashboard. The OTel path is useful for SDK-only deployments or correlating signals with infra metrics in Tempo/Honeycomb without running the full server stack.

Orphaned child spans (a `tool_called` with no matching `tool_responded`, e.g. when an exception fires mid-tool) are force-closed with `status = ERROR` so backends visually flag the broken step.

**Resource attributes** are not set by the exporter — they are the caller's responsibility. Pass a `Resource` to the `TracerProvider` before constructing `DunetraceOTelExporter`:

```python
from opentelemetry.sdk.resources import Resource

resource = Resource.create({
    "service.name":           "my-agent-service",
    "service.version":        "0.3.1",
    "deployment.environment": "production",
})
provider = TracerProvider(resource=resource)
```

Without resource attributes, spans appear as an anonymous service in Datadog, Honeycomb, and similar backends. This makes it impossible to filter by agent service in a multi-service environment.

**Known gaps:**

| Gap | Detail |
|---|---|
| Streaming LLM | `llm_called` / `llm_responded` fire once each. Streaming runs produce a single `llm_call` span covering the full stream duration with no per-chunk events. No fix planned — the SDK's privacy model hashes all content, so chunk payloads can't be attached as span events anyway. |
| Parallel tool calls | The exporter tracks one open child span per run (`rs.child_span`). If two tools fire concurrently, the first span is orphan-closed (marked ERROR) when the second `tool_called` arrives. Both spans end up in the trace but the first loses response-time precision. Parallel agents (LangGraph `Send`) will see this. |
| Multi-agent W3C trace linking | `parent_run_id` is stored as a span attribute (`dunetrace.parent_run_id`) but child agent runs start a fresh trace with their own `trace_id`. Traces from parent and child agents appear as disconnected in Tempo/Honeycomb. Full W3C `traceparent` propagation is not implemented — callers who need linked traces must propagate the parent span context manually via the standard OTel API. |

### external_signal event type

`run.external_signal("rate_limit", source="openai")` emits an `external.signal` event that does **not** advance the step counter. It records infrastructure context alongside the agent step it coincided with. `SlowStepDetector` checks for coincident external signals within the step's time window and includes them in evidence (`coincident_signals`).

---

## Service Responsibilities

### Ingest API (port 8001)

The entry point for all SDK traffic. Its only job is to accept events as fast as possible and not lose them.

- Validates the event schema (Pydantic)
- Authenticates via `api_keys` table
- Returns `202 Accepted` before touching the database
- Writes events to Postgres in a `BackgroundTask` (after the 202)
- Never does any detection logic

**Why the 202 before writing?** Your agent is waiting. The round-trip to the agent should be as short as possible. Validation is synchronous; persistence is async.

---

### Detector Worker

A background polling loop that runs every 5 seconds. It is the only process that runs detection logic.

1. Fetches runs completed since last poll (terminal events `run.completed` or `run.errored`) plus any runs that have stalled (no new events for `STALL_TIMEOUT_SECS`, default 90s)
2. Checks `processed_runs` to skip already-processed runs
3. Reconstructs `RunState` by fetching and replaying all events for each run
4. Runs 14 Tier 1 detectors against the `RunState`. `PROMPT_INJECTION_SIGNAL` is handled separately — the SDK detects injection on raw input before hashing and embeds evidence in the `run.started` payload; the worker extracts it from there rather than running the detector on `RunState`
5. Writes any `FailureSignal` rows to Postgres
6. Marks the run as processed

Signals are written with `shadow=TRUE` unless the detector is in `LIVE_DETECTORS`.

**Why polling instead of streaming?** Simplicity and reliability. A polling worker requires no message broker, survives restarts gracefully, and is trivial to reason about. At current scale (sub-100 runs/sec), 5-second polling latency is acceptable. ClickHouse and Kafka are future considerations.

---

### Explain Layer (library, not a service)

Not a separate process i.e. imported as a library by both the alerts worker and the customer API.

Takes a `FailureSignal` and returns an `Explanation` in under 1ms. Uses deterministic string templates, not LLM calls. The template for each failure type interpolates actual evidence values (tool names, counts, patterns) into pre-written text.

Why no LLM? Three reasons: latency (templates are instant), cost (zero per-signal API cost), and consistency (same signal → same explanation, makes testing and debugging predictable).

---

### Alerts Worker

A background polling loop that runs every 10 seconds. It is the only process that sends external notifications.

1. Fetches unalerted signals (`shadow=FALSE AND alerted=FALSE`)
2. Calls `explain()` on each signal
3. Formats for Slack (Block Kit) or webhook (signed JSON)
4. Posts via HTTP with exponential backoff retry (up to 3 attempts)
5. Marks signals as `alerted=TRUE` only after at least one destination succeeds

**At-least-once delivery:** If the worker crashes between sending and marking, the signal will be re-sent on restart. Receivers should treat `(run_id, failure_type, detected_at)` as the idempotency key.

---

### Customer API (port 8002)

A read-only FastAPI service. Powers the dashboard and any customer integrations.

- All endpoints require `Authorization: Bearer <api_key>`
- In `AUTH_MODE=dev`, auth is skipped entirely i.e. no token required
- All signal responses include the full explanation (title, what, why, fixes)
- Pagination via `offset` / `limit` query params

**Endpoint reference:**

| Endpoint | Description |
|---|---|
| `GET /v1/agents` | List all agents with run counts, signal counts, critical/high counts, and failure type breakdown |
| `GET /v1/agents/{id}/runs` | Paginated run list for an agent — summary fields only (no events) |
| `GET /v1/agents/{id}/signals` | Paginated signal list with full explanations. Accepts `severity`, `failure_type`, `include_shadow` filters |
| `GET /v1/agents/{id}/insights` | Aggregated analytics: input hash patterns, signal trends by day, version stats, time-to-first-tool percentiles, hourly signal distribution |
| `GET /v1/runs/{id}` | Full run detail — metadata, all events, all signals with explanations |
| `GET /health` | Service health check — returns `{"status":"ok","db":"ok"}` |

**Signal endpoints** accept an optional `include_shadow` query parameter:

```
GET /v1/agents/{id}/signals?include_shadow=true
```

When `include_shadow=false` (default), only live signals (`shadow = FALSE`) are returned — the same set that triggers alerts. When `true`, shadow signals are included and each signal object contains a `shadow: bool` field. The dashboard fetches with `include_shadow=true` and renders shadow signals separately with a dashed border + SHADOW badge in the Alerts page.

---

### Dashboard (port 3000)

A single-page HTML app served by nginx (Docker). No build step — plain HTML/CSS/JS fetching from the Customer API with `Authorization: Bearer <api_key>`.

Auto-refreshes every 15 seconds. All data is computed client-side from the API responses (no server-side rendering).

**Pages:**

| Page | Data sources | Key behaviour |
|---|---|---|
| Overview | `/v1/agents` + per-agent runs + signals | Stat cards with configurable trend deltas (1h / 24h / 7d). Risk Trend bar chart (24 hourly buckets from `detected_at`). Top failure patterns with ↑↓ vs prior period. |
| All Runs | Per-agent `/runs` | Sortable table; click any row to open run detail |
| Alerts | Per-agent `/signals?include_shadow=true` | Live signals grouped by failure type. Shadow signals in a separate section with dashed border + SHADOW badge |
| Analytics | `/v1/agents` | Cross-agent totals, top failure patterns, per-agent breakdown |
| Risk Heatmap | `/v1/agents` (failure_types field) | Failure type × agent intensity grid |
| Agents | `/v1/agents` + per-agent runs | Health cards: failure rate %, dominant pattern, run/signal counts, shadow signal count |
| Compare Runs | Per-agent runs + signals | Side-by-side panel for any two runs — metrics diff, signal diff, new/resolved failure types |
| Detectors | Static | Threshold sliders (UI only; edit `detectors.yml` to apply) |

**Run detail panel** (opened from any run row) fetches `/v1/runs/{id}` and per-agent signals, then renders three tabs:

- **Analysis** — step timeline, signal score cards, plain-English explanation + suggested fix from the explain layer
- **Run graph** — SVG node graph built from raw events: green = LLM, orange = tool (ok), red = looping tool call, blue = start/end. Loop clusters highlighted with dashed outline.
- **Event log** — all events in order, expandable to full payload. Content fields shown as SHA-256 hashes.

**Shadow signal rendering** — the dashboard fetches all signals with `?include_shadow=true` and splits them client-side: `shadow=false` signals feed the normal alert groups; `shadow=true` signals feed the shadow section in Alerts and the shadow count badge on agent health cards.

---

## Database Schema

```sql
-- All agent events, raw
CREATE TABLE events (
    id             BIGSERIAL PRIMARY KEY,
    batch_id       TEXT             NOT NULL,
    event_type     TEXT             NOT NULL,
    run_id         TEXT             NOT NULL,
    agent_id       TEXT             NOT NULL,
    agent_version  TEXT             NOT NULL,
    step_index     INTEGER          NOT NULL,
    timestamp      DOUBLE PRECISION NOT NULL,   -- unix epoch, from SDK
    payload        JSONB            NOT NULL,
    parent_run_id  TEXT,
    received_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- Detected failures
CREATE TABLE failure_signals (
    id             BIGSERIAL PRIMARY KEY,
    failure_type   TEXT        NOT NULL,
    severity       TEXT        NOT NULL,
    run_id         TEXT        NOT NULL,
    agent_id       TEXT        NOT NULL,
    agent_version  TEXT        NOT NULL,
    step_index     INTEGER     NOT NULL,
    confidence     REAL        NOT NULL,
    evidence       JSONB       NOT NULL,
    detected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    shadow         BOOLEAN     NOT NULL DEFAULT TRUE,   -- TRUE = stored-only (not alerted); FALSE = live
    alerted        BOOLEAN     NOT NULL DEFAULT FALSE   -- set to TRUE after alerts worker delivers
);

-- Prevents detector from reprocessing completed runs
CREATE TABLE processed_runs (
    run_id         TEXT PRIMARY KEY,
    agent_id       TEXT        NOT NULL,
    agent_version  TEXT        NOT NULL,
    processed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signal_count   INTEGER     NOT NULL DEFAULT 0,
    trigger        TEXT        NOT NULL   -- "completed" | "errored" | "stalled"
);

-- API key → customer mapping
CREATE TABLE api_keys (
    key            TEXT PRIMARY KEY,
    agent_id       TEXT        NOT NULL,
    customer_id    TEXT        NOT NULL,
    active         BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## Performance Characteristics

| Component | Latency | Throughput |
|---|---|---|
| SDK `_emit()` | <1μs (deque append, default config) | Millions/sec |
| SDK drain thread | 200ms idle poll; continuous under load | 100 events/batch |
| Ingest API (202) | ~5ms | ~1,000 req/sec (single instance) |
| Ingest DB write | ~20ms | Background, non-blocking |
| Detector poll cycle | 5s | ~100 runs/cycle |
| Explain layer | <1ms | Synchronous |
| Alerts poll cycle | 10s | 50 signals/cycle |
| Customer API | ~10ms | ~500 req/sec |

**Agent overhead:** The SDK adds less than 500μs to any agent run when using the default HTTP ingest path (`emit_as_json=False`, no OTel exporter). With `emit_as_json=True` or an OTel exporter, `_emit()` also serialises to JSON or creates spans synchronously — overhead increases accordingly. The drain thread is entirely background. Even under backpressure (ingest API down), the ring buffer drops the oldest events rather than blocking the agent.

---

## Failure Modes

**Ingest API down:** The drain thread drains events from the buffer before shipping. If `_ship()` fails, those events are dropped — there is no retry. New events continue to buffer (up to 10,000) and will be shipped once the API recovers. The agent is never blocked.

**Detector worker down:** Runs queue up in the `events` table. When the worker restarts, it processes all unprocessed runs. Signals are delayed but not lost.

**Postgres down:** Ingest returns 503. SDK logs a warning and continues buffering. Events during the outage are lost (the buffer eventually overwrites). This is acceptable i.e. observability data loss during a DB outage is not a catastrophic failure.

**Alerts worker down:** Signals accumulate as `alerted=FALSE`. When the worker restarts, it picks up where it left off. Alerts are delayed but not lost (at-least-once delivery).
