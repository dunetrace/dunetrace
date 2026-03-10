# Architecture

Dunetrace is a pipeline of five independent services communicating through a shared Postgres database.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Agent                           │
│                                                             │
│   with dt.run(user_input) as run:                           │
│       run.tool_called("web_search", {...})                  │
│       run.tool_responded("web_search", ...)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP POST /v1/ingest (async, 202)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Ingest API  :8001                          │
│                                                             │
│   FastAPI · validates events · returns 202 immediately      │
│   BackgroundTask writes to Postgres                         │
└──────────────────────┬──────────────────────────────────────┘
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
│  Reconstructs       │  writes      │  Fetches unalerted      │
│  RunState from      │ ──────────▶  │  shadow=FALSE signals   │
│  events             │  signals     │  → explain()            │
│  Runs 14 detectors  │              │  → format Slack/webhook │
│  Writes signals     │              │  → HTTP POST with retry │
└─────────────────────┘              └─────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  Customer API  :8002                         │
│                                                             │
│   GET /v1/agents         GET /v1/runs/{id}                  │
│   GET /v1/agents/{id}/runs    GET /v1/agents/{id}/signals   │
│   Read-only · bearer token auth · explains signals inline   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  Dashboard  :3000                            │
│                                                             │
│   Static HTML/JSX served by nginx (Docker)                  │
│   Fetches live data from Customer API · auto-refreshes 10s  │
└──────────────────────────────────────────────────────────────┘
```

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

1. Fetches runs completed since last poll (terminal events `run.completed` or `run.errored`) plus any runs that have stalled (no new events for `STALL_TIMEOUT_SECS`)
2. Checks `processed_runs` to skip already-processed runs
3. Reconstructs `RunState` by fetching and replaying all events for each run
4. Runs all 15 Tier 1 detectors against the `RunState`
5. Writes any `FailureSignal` rows to Postgres
6. Marks the run as processed

Signals are written with `shadow=TRUE` unless the detector is in `LIVE_DETECTORS`.

**Why polling instead of streaming?** Simplicity and reliability. A polling worker requires no message broker, survives restarts gracefully, and is trivial to reason about. At current scale (sub-100 runs/sec), 5-second polling latency is acceptable. ClickHouse and Kafka are future considerations.

---

### Explain Layer (library, not a service)

Not a separate process — imported as a library by both the alerts worker and the customer API.

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
- In `AUTH_MODE=dev`, any non-empty token is accepted
- All signal responses include the full explanation (title, what, why, fixes)
- Pagination via `offset` / `limit` query params

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
    shadow         BOOLEAN     NOT NULL DEFAULT TRUE,
    alerted        BOOLEAN     NOT NULL DEFAULT FALSE
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
| SDK `_emit()` | <1μs (deque append) | Millions/sec |
| SDK drain thread | 200ms batch interval | 100 events/batch |
| Ingest API (202) | ~5ms | ~1,000 req/sec (single instance) |
| Ingest DB write | ~20ms | Background, non-blocking |
| Detector poll cycle | 5s | ~50 runs/cycle |
| Explain layer | <1ms | Synchronous |
| Alerts poll cycle | 10s | 50 signals/cycle |
| Customer API | ~10ms | ~500 req/sec |

**Agent overhead:** The SDK adds less than 500μs to any agent run in the common case. The drain thread is entirely background. Even under backpressure (ingest API down), the ring buffer drops the oldest events rather than blocking the agent.

---

## Failure Modes

**Ingest API down:** SDK background thread retries failed batches. Events buffer in memory for up to ~33 minutes at 200ms intervals with a 10,000-event buffer. Agent is never affected.

**Detector worker down:** Runs queue up in the `events` table. When the worker restarts, it processes all unprocessed runs. Signals are delayed but not lost.

**Postgres down:** Ingest returns 503. SDK logs a warning and continues buffering. Events during the outage are lost (the buffer eventually overwrites). This is acceptable — observability data loss during a DB outage is not a catastrophic failure.

**Alerts worker down:** Signals accumulate as `alerted=FALSE`. When the worker restarts, it picks up where it left off. Alerts are delayed but not lost (at-least-once delivery).
