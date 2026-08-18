# Architecture

Dunetrace is a pipeline of independent services communicating through a shared Postgres database — no message broker anywhere. The core pipeline (ingest, detector, alerts, customer API, dashboard) runs structural detection, always on. Two additional workers are optional and disabled by default: `semantic_worker` (LLM-based Tier 2 evaluation, see [Semantic Evaluation Layer](#semantic-evaluation-layer)) and `integrations_worker` (pulls evaluation results from Langfuse/LangSmith/Braintrust, see [External Integration Mode](#external-integration-mode)). Both write into the same `failure_signals` table the core pipeline already uses, so nothing downstream (dashboard, alerts, policies) needs to know which produced a given signal — except policies, which structurally cannot see anything either optional worker writes; see [Detection: Two Independent Paths](#detection-two-independent-paths) and [docs/policies.md](policies.md#structural-signals-only).

---

## System Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              Your Agent                                   │
│                                                                           │
│   Python SDK                        TypeScript / Node.js SDK              │
│   ──────────────────────────────    ─────────────────────────────────     │
│   with dt.run(input) as run:        await dt.run(id, opts, async run => { │
│       run.tool_called(name, {})         run.toolCalled(name, {})          │
│       run.tool_responded(name, …)       run.toolResponded(name, …)        │
│       run.llm_called(model, n)          run.llmCalled(model, n)           │
│       run.llm_responded(…)              run.llmResponded(…)               │
│       run.final_answer()                run.finalAnswer()                 │
│                                     })                                    │
│                                                                           │
│   Framework integrations: LangChain · CrewAI · AutoGen · OTel             │
└──────┬──────────────────────────────┬──────────────────────────┬──────────┘
       │  HTTP POST /v1/ingest        │  stdout NDJSON           │  OTel spans
       │  (async, 202)                │  (emit_as_json=True)     │  (otel_exporter=…)
       │  (Python + TypeScript)       │  (Python + TypeScript)   │  (Python only)
       ▼                              ▼                          ▼
┌─────────────────────┐  ┌────────────────────────┐  ┌──────────────────────┐
│  Ingest API  :8001  │  │  Loki / Grafana Alloy  │  │  OTel Collector      │
│                     │  │                        │  │  (Tempo / Honeycomb  │
│  POST /v1/ingest    │  │  Promtail pipeline     │  │   / Datadog / Jaeger)│
│  POST /v1/otlp/     │  │  → Grafana dashboards  │  │                      │
│    traces  ◄────────┼──┼──────────────────────── ──┼── Langdock · Dify   │
│  (OTLP/HTTP)        │  └────────────────────────┘  │  OpenLLMetry · any  │
│                     │                               │  OTLP exporter      │
│  validates ·        │                               └──────────────────────┘
│  202 immediately    │
│  BackgroundTask     │
│  writes to Postgres │
└──────────┬──────────┘
           │  writes: events table
           ▼
┌──────────────────────────────────────────────────────────────┐
│                      Postgres                                │
│                                                             │
│   events           failure_signals   processed_runs         │
│   api_keys         issues            digest_log             │
│   conversations     signal_groups     agent_semantic_config  │
└──┬──────────────────┬─────────────────────────┬───────────┬──┘
   │ polls 5s          │ polls 10s                │ polls 5s   │ polls (interval)
   ▼                   ▼                          ▼            ▼
┌───────────┐  ┌───────────────┐  ┌────────────────────┐  ┌─────────────────────┐
│ Detector  │  │ Alerts Worker │  │  Semantic Worker    │  │ Integrations Worker │
│ Worker    │  │               │  │  (optional, off     │  │  (optional, off     │
│           │  │ Fetches       │  │  by default)        │  │  by default)        │
│ Reconstr. │  │ unalerted     │  │                      │  │                     │
│ RunState  │  │ shadow=FALSE  │  │  Adaptive sampling   │  │  Pulls Langfuse /   │
│ Runs 29   │  │ signals       │  │  → DeepEval          │  │  LangSmith /        │
│ detectors │  │ → explain()   │  │  evaluators →        │  │  Braintrust results │
│ Writes    │  │ → Slack /     │  │  failure_signals     │  │  → failure_signals  │
│ signals   │  │ webhook       │  │  (source=semantic)   │  │  (source=provider)  │
└───────────┘  └───────────────┘  └────────────────────┘  └─────────────────────┘

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

## Detection: Two Independent Paths

Dunetrace runs Tier 1 detectors in two different places, and they are not the same
detection run seen twice — they are two independent evaluations that can disagree.
Understanding which path produced a given signal matters: only one of them is the
source of truth for alerts and the dashboard.

```
              ┌────────────────────────────────────────┐
              │            Your Agent Process           │
              │                                         │
              │   SDK builds RunState in memory          │
              │   as events occur                        │
              └───────────────────┬─────────────────────┘
                                  │  at run end, before span export
                                  ▼
              ┌────────────────────────────────────────┐
PATH 1        │  Client-side (OTel) pass                 │
client-side,  │  Tier 1 detectors run                     │
in-process    │  TIER1_DETECTORS thresholds                │
              │  (hardcoded in detectors.py)                │
              │  → root span attributes (dunetrace.signal.N.*)
              └────────────────────────────────────────┘
                (only exists if an OTel exporter is configured;
                 works with no backend at all)


              ┌────────────────────────────────────────┐
              │                Ingest API                │
              │  writes raw events to Postgres            │
              └───────────────────┬─────────────────────┘
                                  │  detector worker polls every 5s
                                  ▼
              ┌────────────────────────────────────────┐
PATH 2        │  Server-side pass                        │
server-side,  │  Tier 1 + custom detectors                │
after         │  detectors.yml thresholds (configurable,   │
ingestion     │  hot-reloadable without redeploying the SDK)│
              │  → failure_signals table                    │
              │    → alerts worker → Slack/webhook           │
              │    → dashboard                                │
              └────────────────────────────────────────┘
                (requires the full backend stack running)
```

| | Path 1 — client-side (OTel) | Path 2 — server-side |
|---|---|---|
| Where it runs | In the SDK process, at run end | Detector worker, after ingestion |
| When | Before span export, synchronously | On the next 5s poll cycle |
| Thresholds | `TIER1_DETECTORS` hardcoded in `detectors.py` | `detectors.yml`, loaded at worker startup |
| Output | Root span attributes (`dunetrace.signal.N.*`) | `failure_signals` table |
| Feeds alerts / dashboard? | No | Yes — this is the only path that does |
| Works without the backend running? | Yes | No |
| Custom detectors | Not evaluated | Evaluated (Tier 1 always runs first) |

**Why two paths instead of one:** the OTel path exists for SDK-only deployments — teams
piping spans straight into Tempo/Honeycomb/Datadog without running `ingest`/`detector`/
`alerts` at all, or wanting agent failures correlated with infra spans in one place. The
server-side path exists because it's the only one with access to `detectors.yml`
overrides, custom detectors, cross-run baselines (P75 step count, token growth, etc.),
and the issue-tracking/alerting/digest machinery — none of which a stateless SDK process
can compute for itself.

**They can disagree.** If you tune a threshold in `detectors.yml`, the OTel span
annotations (Path 1) keep using the SDK's hardcoded defaults — they will not pick up the
override. A signal that fires in your Tempo trace may not fire in the dashboard, or vice
versa. **The server-side path (Path 2) is the source of truth** for anything user-facing
— alerts, the dashboard, issue tracking. Treat Path 1 purely as a convenience for
SDK-only / infra-correlation use cases, not as a second copy of the same detection result.

---

## Semantic Evaluation Layer

A third detection path, architecturally distinct from both paths above:
LLM-based judgment for failure modes no regex/arithmetic check can catch
(hallucination, task completion, cross-turn user frustration). Full detail
in [docs/semantic-evaluation.md](semantic-evaluation.md) — this section
covers only how it fits into the system.

```
┌──────────────────────────────────────────────────────────────┐
│                      Postgres                                │
│   events   failure_signals (source='semantic')  ...          │
└────────┬───────────────────────────────────────────────────────┘
         │  polls every 5s (SEMANTIC_WORKER_ENABLED, default off)
         ▼
┌─────────────────────────────────────────────────────────────┐
│   Semantic Worker                                            │
│                                                               │
│   Adaptive sampling (structural-signal runs 100%,            │
│   semantic_critical agents 100%, retrieval runs 20%,         │
│   baseline 5%) → DeepEval evaluators (Hallucination,         │
│   Task Completion) → grouping/dedup → feedback-adjusted      │
│   confidence → optional second opinion (different model)     │
│   → org/agent quota check → failure_signals                  │
└─────────────────────────────────────────────────────────────┘
```

Runs entirely after a run completes — never in the SDK's request path, never
adding latency to your agent. This is *why* [policies](policies.md) (runtime
prevention) can only ever trigger on structural signals: by the time
semantic evaluation sees a run, there's nothing left to prevent.

---

## External Integration Mode

If you already run Langfuse, LangSmith, or Braintrust, `integrations_worker`
pulls their evaluation results in and correlates them to Dunetrace runs via
`trace_id`, writing them into the same `failure_signals` table tagged with
the provider's name as `source`. A synchronous generic push endpoint
(`POST /v1/semantic-signals/external`) covers evaluation tools without a
dedicated poller. Full detail, including per-provider auth and failure
handling, in [docs/integrations/external-evaluation.md](integrations/external-evaluation.md).

---

## Conversations

`runs` and `conversations` tables (Phase 3.1) group multiple runs under one
multi-turn conversation via an optional `dt.run(conversation_id=...)`
parameter — fully backward compatible, omitting it behaves exactly as
before. This unlocks conversation-level semantic evaluation
(`USER_FRUSTRATION`, which reads the last N runs in a conversation rather
than a single run) and the dashboard's conversation-detail view
([docs/dashboard.md](dashboard.md)). See
[docs/semantic-evaluation.md#conversation-level-evaluation](semantic-evaluation.md#conversation-level-evaluation)
for the sampling/quota details.

---

## SDKs

Two first-class SDKs send events to the same ingest API — runs from either appear together in the dashboard under the same `agent_id`.

| SDK | Install | Entry point |
|---|---|---|
| Python (`dunetrace`) | `pip install dunetrace` | `from dunetrace import Dunetrace` |
| TypeScript / Node.js (`dunetrace`) | `npm install dunetrace` | `import { Dunetrace } from "dunetrace"` |

The Python SDK supports all three output modes (HTTP ingest, Loki NDJSON, OTel spans) and ships framework integrations for LangChain, CrewAI, and AutoGen. The TypeScript SDK supports HTTP ingest and Loki NDJSON (`emitAsJson`); OTel spans are Python-only.

**Framework integrations (Python SDK):**

| Framework | Class | Install |
|---|---|---|
| LangChain / LangGraph | `DunetraceCallbackHandler` | `pip install 'dunetrace[langchain]'` |
| CrewAI 1.x | `DunetraceCrewCallback` | `pip install dunetrace crewai` |
| AutoGen (autogen-agentchat ≥ 0.4) | `DunetraceAutoGenObserver` | `pip install dunetrace autogen-agentchat autogen-ext` |
| OpenLLMetry / OTel receiver | `DunetraceOTelReceiver` | `pip install 'dunetrace[otel]'` |

See the [docs/](.) directory for per-framework integration guides (LangChain, CrewAI, AutoGen, TypeScript, Langdock).

---

## SDK Output Modes

The Python SDK supports three independent output modes that can be combined:

| Mode | How to enable | Destination | Use case |
|---|---|---|---|
| HTTP ingest (default) | `endpoint="http://…"` (default: `http://localhost:8001`) | Ingest API → Postgres → Detector | Full pipeline: detection, alerts, dashboard |
| Loki NDJSON | `emit_as_json=True` | stdout → Promtail/Alloy → Loki | Existing Grafana stack integration |
| OTel spans | `otel_exporter=DunetraceOTelExporter(provider)` | OTel collector → Tempo / Honeycomb / Datadog | Infra metric correlation |

All three modes can be active simultaneously. OTel and NDJSON are zero-cost when disabled.

HTTP ingest is shipped through a pluggable `BatchingEmitter` (see
[dunetrace/emitters.py](../packages/sdk-py/dunetrace/emitters.py)) — the drain
thread hands each batch to whichever emitter is configured. The default is
`HttpBatchingEmitter`, which is what "HTTP ingest" above refers to. To use OTel
or NDJSON without any HTTP ingest (local testing, pure-OTel deployments), pass
`emitter=NoopBatchingEmitter()` explicitly:

```python
from dunetrace import Dunetrace, NoopBatchingEmitter

dt = Dunetrace(emitter=NoopBatchingEmitter(), otel_exporter=my_exporter)
```

This is the one supported way to disable HTTP shipping — there is no
`endpoint=None` special case; an explicit `endpoint` value (including `""`) is
always taken literally rather than silently replaced by a default. Other
built-in emitters:

- `ConsoleBatchingEmitter` — prints each batch as JSON to stdout, once per
  drain cycle. Distinct from `emit_as_json`'s per-event NDJSON stream.
- `FileBatchingEmitter(path)` — appends each batch to a local file. For
  offline/audit trails.
- `DurableRetryEmitter(inner, ...)` — wraps any of the above (most commonly
  `HttpBatchingEmitter`) to survive a backend outage across process restarts.
  See [Failure Modes](#failure-modes) below for the full behavior.

Implement `BatchingEmitter.ship(batch) -> bool` directly for anything else.

The TypeScript SDK supports HTTP ingest and `emitAsJson` (Loki NDJSON). `otel_exporter` is Python-only.

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

This is Path 1 of the two independent detection paths — see [Detection: Two Independent Paths](#detection-two-independent-paths) above for the full comparison against the server-side path, and why the two can disagree.

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
| Streaming LLM | `llm_called` / `llm_responded` fire once each. Streaming runs produce a single `llm_call` span covering the full stream duration with no per-chunk events — the integration hooks only fire once per completed response, not per chunk. No fix planned. |
| Parallel tool calls | The exporter tracks one open child span per run (`rs.child_span`). If two tools fire concurrently, the first span is orphan-closed (marked ERROR) when the second `tool_called` arrives. Both spans end up in the trace but the first loses response-time precision. Parallel agents (LangGraph `Send`) will see this. |
| Multi-agent W3C trace linking | `parent_run_id` is stored as a span attribute (`dunetrace.parent_run_id`) but child agent runs start a fresh trace with their own `trace_id`. Traces from parent and child agents appear as disconnected in Tempo/Honeycomb. Full W3C `traceparent` propagation is not implemented — callers who need linked traces must propagate the parent span context manually via the standard OTel API. |

### external_signal event type

`run.external_signal("rate_limit", source="openai")` emits an `external.signal` event that does **not** advance the step counter. It records infrastructure context alongside the agent step it coincided with. `SlowStepDetector` checks for coincident external signals within the step's time window and includes them in evidence (`coincident_signals`).

---

## Runtime Policy Evaluation

Policies are evaluated **in the SDK process**, synchronously, during a run — the
only detection/prevention path that runs before a failure completes (see
[docs/policies.md](policies.md)). The engine (`dunetrace.policies`) is split into
three parts:

- **`PolicyEngine` / `Policy`** (`policies/__init__.py`) — loading, HMAC signature
  verification, priority ordering, and the flat `{trigger, operator, value}`
  match (metric/signal/tool-name conditions).
- **Expression conditions** (`policies/expressions.py`, `policies/evaluator.py`) —
  an optional `condition.match` block adds structured comparisons on tool
  arguments (`args.*`), run metadata (`run.*`), and event fields (`event.*`), with
  a 14-operator whitelist, `AND`/`OR` composition (max depth 3), and deterministic
  type coercion. The parser builds an immutable expression tree at load time
  (rejecting unknown operators/paths up front); the evaluator is `eval`-free —
  explicit dict lookups and a whitelist operator table — and runs in well under
  100µs per policy. `matches` uses a ReDoS-safe `regex` engine with a per-call
  timeout. `match` and the flat trigger **AND** together. Full reference:
  [docs/policies/condition-expressions.md](policies/condition-expressions.md).
- **Observability** (`policies/observability.py`) — each evaluation can produce a
  structured record (the policy, whether the trigger matched, each condition's
  compared-vs-expected value, the result). Surfaced as DEBUG logs on the
  `dunetrace.policies.evaluation` logger, and — when
  `policy_evaluation_reporting` is enabled — shipped (rate-limited to 100/policy/
  min, sampled beyond) as `policy.evaluated` events through the normal transport.
  Ingest routes these into the `policy_evaluations` table (not `events`), which
  `GET /v1/policies/{id}/evaluations` reads for the "why did/didn't my policy
  fire?" dashboard view.

Raw tool `args` are only available at the `before_tool_call` approval gate
(evaluated before the tool runs); the after-event metric path sees `run.*` /
`event.*` but not `args.*`. Signatures cover the whole `condition` including
`match`; policies using `match` sign under canonical-form version 2 while legacy
policies stay byte-identical at version 1.

---

## Service Responsibilities

### Ingest API (port 8001)

The entry point for all SDK traffic. Its only job is to accept events as fast as possible and not lose them.

- Validates the event schema (Pydantic)
- Authenticates via `api_keys` table and resolves the caller's `org_id`; every event is tagged with it before being written — see [Database Schema](#database-schema)
- Returns `202 Accepted` before touching the database
- Writes events to Postgres in a `BackgroundTask` (after the 202)
- Never does any detection logic
- `POST /v1/deploy` — accepts deploy markers from `dt.mark_deploy()` and writes to the `deploy_events` table synchronously (no background task; deploy markers are rare and low-volume)

**Why the 202 before writing?** Your agent is waiting. The round-trip to the agent should be as short as possible. Validation is synchronous; persistence is async.

**EventStore abstraction** (`services/ingest/ingest_svc/db/event_store.py`) — the write path (`insert_events`) and retention (`prune_old_events`) sit behind an `EventStore` interface, swapped via `get_event_store()`/`set_event_store()`. `PostgresEventStore` (the default) delegates to the same partition-aware functions in `db/postgres.py` described below; `InMemoryEventStore` is a fully in-process fake for tests that want to assert on what actually got "written" rather than mocking a free function. This covers only ingest_svc's own write path — the four read-side services (detector, alerts, api, explainer) each query Postgres directly and are intentionally left alone; introducing a shared cross-service storage interface would cross a boundary this codebase keeps deliberately separate (see System Overview: "communicate only through a shared Postgres database", no shared business logic beyond the SDK/schemas packages).

Partition creation (`_ensure_event_partitions`) is *not* part of the abstract `EventStore` contract — it's a Postgres-specific detail of how `PostgresEventStore` stays ready to receive writes, not a concept every backend would even have.

Retention enforcement runs on a daily background loop (`main.py::_prune_loop`, `EVENT_RETENTION_DAYS` env var, default 90) that calls `get_event_store().prune_old_events(...)` — this loop is what actually invokes the partition-dropping logic described under [Partitioning & Retention](#partitioning--retention); before it existed, `prune_old_events()` was implemented and tested but never called from anywhere, so partitions accumulated indefinitely.

**OTLP/HTTP receiver** (`POST /v1/otlp/traces`, `routers/otlp.py` + `otel.py`) — lets any OTel-instrumented agent (an OTel Collector, or any `OTLPSpanExporter`) send traces directly, no Dunetrace SDK required. Accepts both `application/x-protobuf` (the default for most real-world OTLP senders) and `application/json`, either gzip-compressed or not:

- `protobuf_to_resource_spans()` parses an `ExportTraceServiceRequest` via `google.protobuf.json_format.MessageToDict()`, which already produces the exact proto3 JSON mapping — the same `resourceSpans`/`scopeSpans`/`spans` shape a JSON request body has — so `otlp_to_events()` needs no protobuf-specific code path. The one correction needed: `MessageToDict` base64-encodes `bytes` fields (`traceId`/`spanId`/`parentSpanId`), but the OTLP spec's own JSON convention (and the rest of this module) uses plain hex — `_fix_ids_to_hex()` corrects that in place after parsing.
- Gzip (`Content-Encoding: gzip`) is decompressed before format detection, for both protobuf and JSON bodies.
- Rate-limited the same as `/v1/ingest`/`/v1/deploy` (see [Rate Limiting](#rate-limiting)) — bucketed by the `Authorization: Bearer` header directly, since OTLP auth isn't a JSON body field the way ingest's is.
- Writes through the same `EventStore` abstraction as `/v1/ingest`.
- `otlp_to_events()` processes each resourceSpan and each trace inside its own `try/except` — a single malformed one is logged and skipped, so it can't cost every other valid trace in the same batch (an OTel Collector commonly batches spans from many concurrent runs into one export).

---

### Detector Worker

A background polling loop that runs every 5 seconds. It is the only process that runs detection logic.

1. Fetches runs completed since last poll (terminal events `run.completed` or `run.errored`) plus any runs that have stalled (no new events for `STALL_TIMEOUT_SECS`, default 90s)
2. Checks `processed_runs` to skip already-processed runs
3. Reconstructs `RunState` by fetching and replaying all events for each run
4. Runs all 31 structural detectors against the `RunState` (`_DETECTOR_CLASSES` in `detector_svc/detectors.py`), then any active custom detectors and any detectors belonging to a pack this org has enabled. Three of the 31 don't evaluate off `RunState` alone: `PROMPT_INJECTION_SIGNAL` is detected by the SDK on raw input at run-start and embedded in the `run.started` payload, so the worker extracts it from there; `HANDOFF_CONTEXT_LOSS` and `DELEGATION_LOOP` need a second run's data and are evaluated against the cross-run delegation graph (`_handoff_signal_from_parent` / `_delegation_signal_from_chain` in `worker.py`)
5. Writes any `FailureSignal` rows to Postgres
6. Updates the `issues` table: UPSERTs an issue row for each live signal fired (`upsert_fired_issues`) and increments the clean-run counter for any open issues that did not fire this run (`advance_clean_runs`). An issue auto-resolves after 5 consecutive clean runs. Issue tracking failures are caught and logged — they do not affect run processing.
7. Marks the run as processed

Signals are written with `shadow=TRUE` unless the detector is in `LIVE_DETECTORS`.

**Why polling instead of streaming?** Simplicity and reliability. A polling worker requires no message broker, survives restarts gracefully, and is trivial to reason about. At current scale (sub-100 runs/sec), 5-second polling latency is acceptable. ClickHouse and Kafka are future considerations.

**Horizontal scaling:** Set `SHARD_COUNT=N` and run N replicas each with a distinct `SHARD_INDEX`. Each replica polls only its `agent_id` hash bucket. See [Worker Sharding](#worker-sharding) below.

**Poll watermark.** Run discovery is bounded by a persisted per-shard watermark
(`detector_watermarks`). Without it the queries have no lower time bound, so each
5-second poll rescans every terminal event in the whole retention window and
anti-joins it against `processed_runs` — work proportional to *total history*
rather than to new runs. It also defeats the `events` partitioning: the partition
key is `received_at`, and a predicate that never mentions `received_at` can't
prune, so the hottest queries in the system touch every partition.

Two details make the bound safe:

- The window is expressed as *"runs touched by a recent event"*, not *"terminal
  events that are recent"*. Those differ for late-arriving events: a straggler on
  an old run has a new `received_at` but its run's terminal event is old, so
  filtering on the terminal event's timestamp would silently stop re-detecting
  those runs. Selecting the `run_id` set first keeps that path intact while still
  pruning.
- The watermark advances **only after a poll that drained its backlog** (neither
  query hit its `LIMIT`). A full batch means more work sits behind it, and moving
  the window forward would step over runs that were never processed. This is also
  what makes downtime safe: the watermark is persisted and stays put while the
  worker is down, so a restart sees the whole backlog rather than skipping it.

A `NULL` watermark means "never drained" and reproduces the original unbounded
scan — correct for a first poll, which may have arbitrarily old pending work, and
self-correcting after one drained cycle.

| Env var | Default | Description |
|---|---|---|
| `WATERMARK_GRACE_SECS` | `3600` | Re-scan overlap. Effective lookback is up to 2× this. Must be `>= STALL_TIMEOUT_SECS` or the worker refuses to start — a smaller grace could advance the watermark past a run before it qualifies as stalled |

---

### Semantic Worker (optional, disabled by default)

A background polling loop, same shape as Detector Worker (poll `events` every
5s, track handled runs in its own table) but for LLM-based Tier 2 evaluation.
See [Semantic Evaluation Layer](#semantic-evaluation-layer) above and
[docs/semantic-evaluation.md](semantic-evaluation.md) for full detail.

1. Fetches completed/errored runs not yet seen (`semantic_processed_runs`)
2. Adaptive sampling decides whether to evaluate this run at all — recorded
   even when skipped, so a skipped run isn't re-considered every poll
3. For sampled runs: runs configured evaluators, groups findings into
   recurring patterns, applies accumulated false-positive feedback, gets a
   second opinion for HIGH-severity findings where configured, enforces
   per-agent and org-wide monthly quotas
4. Writes findings to `failure_signals` tagged `source='semantic'`

Enable with `SEMANTIC_WORKER_ENABLED=true`. Off by default — every other
service behaves identically without it.

---

### Integrations Worker (optional, disabled by default)

A background polling loop that pulls evaluation results from connected
Langfuse/LangSmith/Braintrust integrations and correlates them to Dunetrace
runs via `trace_id`. See [External Integration Mode](#external-integration-mode)
above and [docs/integrations/external-evaluation.md](integrations/external-evaluation.md)
for full detail, including the generic push endpoint
(`POST /v1/semantic-signals/external`), which is served by the Customer API
directly rather than this worker, since it's a synchronous customer-facing
call, not a background poll.

1. For each connected, due-for-poll integration: fetches new evaluation
   results since the last successful poll
2. Correlates each result to a run via `trace_id`
3. Writes matched results to `failure_signals` tagged `source={provider}`
4. Tracks `consecutive_failures` per integration; a provider down past 30
   minutes writes an internal `EXTERNAL_INTEGRATION_DOWN` operational signal
   (shadow-only) rather than failing silently or affecting any other
   integration

---

### Explain Layer (library, not a service)

Not a separate process i.e. imported as a library by both the alerts worker and the customer API.

Takes a `FailureSignal` and returns an `Explanation` in under 1ms. Uses deterministic string templates, not LLM calls. The template for each failure type interpolates actual evidence values (tool names, counts, patterns) into pre-written text.

Why no LLM? Three reasons: latency (templates are instant), cost (zero per-signal API cost), and consistency (same signal → same explanation, makes testing and debugging predictable).

---

### Alerts Worker

A background polling loop that runs every 10 seconds. It is the only process that sends external notifications.

1. Fetches unalerted signals (`shadow=FALSE AND alerted=FALSE`)
2. Fetches rate context for each signal via `fetch_signal_rate_context(agent_id, failure_type)` — concurrent `asyncio.gather` call, one per signal
3. Calls `explain(signal, rate_context=...)` on each signal — rate context is attached to the `Explanation` and rendered as a one-line summary in the Slack alert
4. Formats for Slack (Block Kit) or webhook (signed JSON)
5. Posts via HTTP with exponential backoff retry (up to 3 attempts)
6. Marks signals as `alerted=TRUE` only after at least one destination succeeds
7. After each poll cycle, calls `send_weekly_digest()` — checks whether it's the configured day/hour, whether a digest was already sent within the last 6 days (via `digest_log`), and if so, fetches 7-day aggregate data and sends a Slack Block Kit summary. Digest errors are caught and logged — they do not affect signal delivery.

**At-least-once delivery:** If the worker crashes between sending and marking, the signal will be re-sent on restart. Receivers should treat `(run_id, failure_type, detected_at)` as the idempotency key.

---

### Customer API (port 8002)

A read-only FastAPI service. Powers the dashboard and any customer integrations.

- All endpoints require `Authorization: Bearer <api_key>`
- In `AUTH_MODE=dev`, auth is skipped entirely i.e. no token required. **`AUTH_MODE` defaults to `prod`** in both the ingest and customer API — dev mode disables authentication outright, so it has to be an explicit opt-in and a deployment that forgets to set the variable gets a locked-down API rather than an open one. Both compose files set `dev` for the local quickstart (`docker-compose.ghcr.yml` reads it as `${AUTH_MODE:-dev}`, so `AUTH_MODE=prod` in the environment is enough to lock it down without editing the file). Each service logs a `WARNING` at startup whenever dev mode is active
- All signal responses include the full explanation (title, what, why, fixes)
- Pagination via `offset` / `limit` query params

**Endpoint reference:**

| Endpoint | Description |
|---|---|
| `GET /v1/agents` | List all agents with run counts, signal counts, critical/high counts, and failure type breakdown |
| `GET /v1/agents/{id}/runs` | Paginated run list for an agent — summary fields only (no events) |
| `GET /v1/agents/{id}/signals` | Paginated signal list with full explanations. Accepts `severity`, `failure_type`, `include_shadow` filters |
| `GET /v1/agents/{id}/insights` | Aggregated analytics: input hash patterns, signal trends by day, version stats, time-to-first-tool percentiles, hourly signal distribution. Also returns `failure_rates` (daily affected/total per failure type), `systemic_patterns` (7-day rate + `is_systemic` flag), and `deploy_events` (last 90 days of deploy markers) — the data powering the Health Record and Deploy Timeline panels |
| `GET /v1/agents/{id}/issues` | Open/resolved issue list for an agent. Accepts optional `status` filter (`open`, `resolved`, `reopened`). Returns id, failure_type, status, first_seen, last_seen, resolved_at, affected_runs, clean_runs_since |
| `GET /v1/runs/{id}` | Full run detail — metadata, all events, all signals with explanations |
| `POST /v1/signals/{id}/explain` | Root-cause analysis, fully native — built from Dunetrace's own events, no external tracing system involved. Returns `fix_category` (`dunetrace_native`, with a deterministic `suggested_policy`; or `customer_code`, with LLM-generated `fix_content`/`fix_patch`), plus `root_cause` and `apply_blocked`. Requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in env |
| `POST /v1/signals/{id}/open-pr` | For `customer_code` / `code_change` fixes only: opens a draft GitHub PR. Auth resolves per-org GitHub App installation first, else the legacy global `GITHUB_TOKEN`/`GITHUB_REPO` — see [docs/integrations/github-app.md](integrations/github-app.md). When two-tier source mapping resolves a real file, the PR edits it directly (diff computed via `difflib` from real before/after content, not LLM-authored); otherwise falls back to a `dunetrace-fixes/signal-{id}.md` summary file. Records the fix in the `fixes` table. Blocked for `PROMPT_INJECTION_SIGNAL` (returns 403) and when GitHub isn't configured (503). `dunetrace_native` fixes (a `suggested_policy`) are applied via the existing `POST /v1/policies` instead, after user confirmation — no separate endpoint. `prompt_addition` fixes have no automated apply path — always a manual copy |
| `GET /v1/orgs/integrations/github/install-url`, `.../callback`, `POST`/`GET`/`DELETE /v1/orgs/integrations/github` | Per-org GitHub App install flow and repos/reviewers config — see [docs/integrations/github-app.md](integrations/github-app.md) |
| `POST`/`GET`/`DELETE /v1/agents/{agent_id}/source-config` | Tier-1 explicit source mapping (`repo`, optional `file_path`) for an agent — see [docs/integrations/github-app.md](integrations/github-app.md#source-mapping-which-repofile-does-a-signal-correspond-to) |
| `POST /v1/signals/{id}/record-copy` | Record a clipboard-path fix in the `fixes` table |
| `GET /v1/signals/{id}/fix-status` | Return fix history and recurrence verdict (`verified / likely_fixed / still_occurring / insufficient_data`) |
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
| Overview | `/v1/agents` + per-agent runs + signals | Stat cards with configurable trend deltas (1h / 24h / 7d). Risk Trend bar chart (24 hourly buckets). **Step Drift panel** — 24-bar sparkline of avg step count per hour vs 7-day baseline; dashed green baseline line; WARNING ZONE badge when 24h avg exceeds baseline by >20%; current / baseline / trigger stats. **Failure Posture gauge** — half-circle SVG gauge with gradient fill and needle at avg confidence; rows for daily signals, avg confidence, false positive rate. **Top Failure Drivers** — ranked by signal count, grouped by agent + failure type; shows wasted steps estimate, avg confidence, severity. **Agent Signal Drift** — horizontal bar per agent showing 24h signal rate vs 7-day baseline, red/amber/green by direction. Top failure patterns with ↑↓ vs prior period. Live run feed. |
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
- **Event log** — all events in order, expandable to full payload, including raw content fields (args, input, output).

**Shadow signal rendering** — the dashboard fetches all signals with `?include_shadow=true` and splits them client-side: `shadow=false` signals feed the normal alert groups; `shadow=true` signals feed the shadow section in Alerts and the shadow count badge on agent health cards.

---

## Database Schema

Every table below carries `org_id` —
the primary tenancy dimension. `agent_id` is secondary and org-scoped: one org can
have many agents, discovered dynamically as they send their first events. Every
query filters `org_id` first, `agent_id` second.

```sql
-- Tenant root. One org per self-hosted install by default ('default'),
-- or many orgs for a multi-tenant deployment built on this backend.
CREATE TABLE organizations (
    id          TEXT PRIMARY KEY,
    name        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- API key → org mapping. A key is NOT bound to a single agent_id — see the
-- "Security posture change" section of the multi-tenancy migration guide.
CREATE TABLE api_keys (
    key            TEXT PRIMARY KEY,
    org_id         TEXT        NOT NULL REFERENCES organizations(id),
    active         BOOLEAN     NOT NULL DEFAULT TRUE,
    rate_limit_rpm INTEGER     NOT NULL DEFAULT 600,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Deploy markers: one row per dt.mark_deploy() call
CREATE TABLE deploy_events (
    id           BIGSERIAL PRIMARY KEY,
    org_id       TEXT        NOT NULL,
    agent_id     TEXT        NOT NULL,
    version      TEXT        NOT NULL,
    deployed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta         JSONB       NOT NULL DEFAULT '{}'   -- arbitrary key/value (commit, env, …)
);

-- All agent events, raw.
-- Partitioned by received_at (monthly range partitions).
-- PRIMARY KEY is composite (id, received_at) as Postgres requires the partition
-- key in the PK.  No other table uses events.id as a FK — cross-table joins
-- use run_id (TEXT) — so the composite PK is safe.
CREATE TABLE events (
    id             BIGSERIAL        NOT NULL,
    batch_id       TEXT             NOT NULL,
    event_type     TEXT             NOT NULL,
    run_id         TEXT             NOT NULL,
    org_id         TEXT             NOT NULL,
    agent_id       TEXT             NOT NULL,
    agent_version  TEXT             NOT NULL,
    step_index     INTEGER          NOT NULL,
    timestamp      DOUBLE PRECISION NOT NULL,   -- unix epoch, from SDK
    payload        JSONB            NOT NULL,
    parent_run_id  TEXT,
    received_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, received_at)
) PARTITION BY RANGE (received_at);

-- Detected failures
CREATE TABLE failure_signals (
    id             BIGSERIAL PRIMARY KEY,
    failure_type   TEXT        NOT NULL,
    severity       TEXT        NOT NULL,
    run_id         TEXT        NOT NULL,
    org_id         TEXT        NOT NULL,
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
    org_id         TEXT        NOT NULL,
    agent_id       TEXT        NOT NULL,
    agent_version  TEXT        NOT NULL,
    processed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signal_count   INTEGER     NOT NULL DEFAULT 0,
    trigger        TEXT        NOT NULL   -- "completed" | "errored" | "stalled"
);

-- Cross-run issue tracker: one row per (org_id, agent_id, failure_type)
CREATE TABLE issues (
    id               BIGSERIAL    PRIMARY KEY,
    org_id           TEXT         NOT NULL,
    agent_id         TEXT         NOT NULL,
    failure_type     TEXT         NOT NULL,
    status           TEXT         NOT NULL DEFAULT 'open',   -- open | resolved | reopened
    first_seen       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_seen        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    resolved_at      TIMESTAMPTZ,
    affected_runs    INTEGER      NOT NULL DEFAULT 1,
    clean_runs_since INTEGER      NOT NULL DEFAULT 0,
    UNIQUE (org_id, agent_id, failure_type)
);

-- Prevents weekly digest from re-sending within a 6-day window, per org
CREATE TABLE digest_log (
    id          BIGSERIAL    PRIMARY KEY,
    org_id      TEXT         NOT NULL,
    digest_type TEXT         NOT NULL DEFAULT 'weekly',
    sent_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Records every autofix action (clipboard copy or GitHub PR)
CREATE TABLE fixes (
    id                    BIGSERIAL    PRIMARY KEY,
    run_id                TEXT         NOT NULL,
    signal_id             BIGINT       NOT NULL,
    org_id                TEXT         NOT NULL,
    fix_content           TEXT         NOT NULL,
    fix_type              TEXT         NOT NULL DEFAULT 'prompt_addition',
    applied_via           TEXT         NOT NULL,   -- 'github_pr' or 'clipboard'
    langfuse_prompt_name  TEXT,                    -- historical column name — unused since the Langfuse
                                                    -- apply-fix integration was removed; always null now
    langfuse_version      INTEGER,                 -- historical column name — reused to store the GitHub
                                                    -- PR number when applied_via='github_pr'
    applied_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- User-defined detectors (plain-English description → structured config via LLM)
CREATE TABLE custom_detectors (
    id                BIGSERIAL    PRIMARY KEY,
    org_id            TEXT         NOT NULL,
    agent_id          TEXT         NOT NULL DEFAULT '*',
    name              TEXT         NOT NULL,
    description       TEXT         NOT NULL,
    config_json       JSONB        NOT NULL,
    status            TEXT         NOT NULL DEFAULT 'shadow',   -- shadow | active | paused
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    total_runs        INTEGER      NOT NULL DEFAULT 0,
    shadow_fire_count INTEGER      NOT NULL DEFAULT 0
);

-- Per-run evaluation outcomes for custom detectors
CREATE TABLE custom_detector_results (
    id           BIGSERIAL    PRIMARY KEY,
    detector_id  BIGINT       NOT NULL REFERENCES custom_detectors(id) ON DELETE CASCADE,
    org_id       TEXT         NOT NULL,
    run_id       TEXT         NOT NULL,
    agent_id     TEXT         NOT NULL,
    fired        BOOLEAN      NOT NULL,
    evaluated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- False-positive suppression state per (org_id, agent_id, failure_type)
CREATE TABLE agent_detector_overrides (
    org_id           TEXT        NOT NULL,
    agent_id         TEXT        NOT NULL,
    failure_type     TEXT        NOT NULL,
    fp_count         INTEGER     NOT NULL DEFAULT 0,
    confidence_floor FLOAT       NOT NULL DEFAULT 0.0,
    silenced         BOOLEAN     NOT NULL DEFAULT FALSE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, agent_id, failure_type)
);

-- Alert dedup window state per (org_id, agent_id, failure_type)
CREATE TABLE alert_dedup (
    org_id           TEXT        NOT NULL,
    agent_id         TEXT        NOT NULL,
    failure_type     TEXT        NOT NULL,
    last_alerted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    suppressed_count INTEGER     NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, agent_id, failure_type)
);
```

---

## Partitioning & Retention

### events table

The `events` table uses PostgreSQL range partitioning on `received_at` with monthly child partitions (`events_202606`, `events_202607`, …). A `events_default` partition catches any rows that fall outside the defined monthly range (e.g. rows written during startup before the first monthly partition is created).

Partition creation runs automatically on every ingest service startup via `ensure_schema()`; retention runs on its own daily background loop:

- `_ensure_event_partitions()` creates child partitions for the current month through 3 months ahead. All `CREATE TABLE IF NOT EXISTS` — idempotent and safe to call on every restart. Runs at startup, not on a timer — a long-running process still gets its next month's partition the next time it restarts or redeploys, well before it's needed.
- `prune_old_events(retention_days=90)` drops monthly partitions whose entire window is older than the retention cutoff. Dropping a partition is an instant DDL operation — no row-by-row DELETE, no vacuum debt. Invoked once a day by `main.py::_prune_loop` (via the [EventStore abstraction](#ingest-api-port-8001), `EVENT_RETENTION_DAYS` env var, default 90) — not tied to startup, since retention needs to fire on a process that's been running for a while, not just on restart.

**Why partitioning matters at scale:** Without it, `events` grows unboundedly and the `fetch_run_events` and baseline queries (which scan by `run_id` and `agent_id`) degrade as the table crosses hundreds of millions of rows. Monthly partitions let Postgres prune non-matching partitions from baseline queries that filter by `received_at` range, and make retention instant.

**Naming convention:** `events_YYYYMM` (e.g. `events_202606` for June 2026). `prune_old_events` identifies droppable partitions by name — it only touches tables matching `^events_[0-9]{6}$` and never touches `events_default`.

**Existing deployments:** The `CREATE TABLE IF NOT EXISTS events … PARTITION BY RANGE …` is a no-op when a non-partitioned `events` table already exists. Existing stacks continue to work unchanged. To adopt partitioning on an existing deployment, an offline migration is required:

```sql
-- Offline migration sketch (run with the ingest service stopped)
ALTER TABLE events RENAME TO events_old;
CREATE TABLE events ( … PRIMARY KEY (id, received_at) ) PARTITION BY RANGE (received_at);
CREATE TABLE events_default PARTITION OF events DEFAULT;
INSERT INTO events SELECT * FROM events_old;
DROP TABLE events_old;
```

### Other tables

`failure_signals` and `processed_runs` grow proportionally to run volume and are not partitioned. At moderate scale (< 50M rows), the existing indexes on `(agent_id, detected_at DESC)` and the `run_id TEXT PRIMARY KEY` are sufficient. Partitioning follows the same pattern as `events` when needed.

**`processed_runs` is pruned**, unlike `failure_signals`. It's one row per run and
it's the anti-join target in the detector's run-discovery query, so unbounded growth
directly slows the hottest query in the service. `prune_processed_runs()` runs daily
from the detector worker (shard 0 only — the table isn't shard-partitioned, so extra
replicas would only contend).

The ordering constraint is the subtle part, and it runs opposite to what you'd
expect: a `processed_runs` row may only be deleted **once its run's events are
already gone**. Delete it while the events remain and the run reads as unprocessed
— every detector runs against it again and writes a duplicate set of signals. So
the delete carries a `NOT EXISTS` against `events` rather than trusting a retention
constant, which makes the invariant hold whatever `EVENT_RETENTION_DAYS` is actually
set to, including a value this service never sees.

| Env var | Default | Description |
|---|---|---|
| `PROCESSED_RUNS_RETENTION_DAYS` | `120` | Age bound on the prune scan. Sits beyond the 90-day event default so the anti-join only considers rows that can plausibly qualify. Not the safety mechanism — the `NOT EXISTS` is |
| `PRUNE_BATCH_SIZE` | `10000` | Rows per delete batch. A pass keeps going while batches come back full, so a neglected table catches up over one pass rather than one batch per day |

---

## Worker Sharding

Both the detector and the alerts worker are horizontally scalable by `agent_id`, using the same scheme and the same env var names. Set `SHARD_COUNT=N` and run N replicas, each with a different `SHARD_INDEX` (0 through N-1):

```
docker compose up --scale detector=4 -d   # not enough — each replica needs its own SHARD_INDEX

# Use separate compose service entries or K8s deployments with env-var overrides:
# detector-0: SHARD_COUNT=4, SHARD_INDEX=0
# detector-1: SHARD_COUNT=4, SHARD_INDEX=1
# detector-2: SHARD_COUNT=4, SHARD_INDEX=2
# detector-3: SHARD_COUNT=4, SHARD_INDEX=3
```

Each worker polls only runs whose `agent_id` hashes to its bucket:

```sql
AND ($n::int = 1 OR abs(hashtext(e.agent_id)) % $n = $m)
```

`abs()` is required because `hashtext()` returns signed int4 and Postgres modulo of a negative dividend is negative.

**`SHARD_COUNT=1` (default)** — the condition short-circuits to `TRUE` and all runs are processed. No configuration change needed for single-instance deployments.

**Invariants:**
- A run belongs to exactly one agent, so there is no cross-shard coordination. Sharding is by `agent_id` hash bucket only, not `org_id` — an org's agents can land on different shards.
- `processed_runs` deduplication still works — a run can only be claimed by the shard whose bucket matches its `agent_id`.
- Baseline queries (`fetch_step_count_baseline`, `fetch_token_growth_baseline`, etc.) are scoped by `(org_id, agent_id, agent_version)` — `org_id` is required, since `agent_id`/`agent_version` are not guaranteed unique across orgs.
- The customer API needs no sharding — each request is independent, so replicas don't coordinate.

**Validation:** Both workers raise `ValueError` at startup if `SHARD_COUNT < 1` or `SHARD_INDEX` is outside `[0, SHARD_COUNT)`, so misconfigured replicas crash immediately rather than silently claiming no work.

| Env var | Default | Description |
|---|---|---|
| `SHARD_COUNT` | `1` | Total number of replicas of this worker |
| `SHARD_INDEX` | `0` | This replica's bucket index (0-based) |

### Alerts worker: sharding plus claiming

The alerts worker is **not** a stateless reader, and replicating it needs more than
a shard filter. It writes (`alerted`, `alert_dedup`, `digest_log`) and it has an
external side effect — the Slack/webhook/Linear call. `alerted` is only set *after*
a successful send, so it can't double as a claim: two workers scanning the same
window both see `alerted = FALSE`, both deliver, and the customer gets duplicate
alerts. `alert_dedup` doesn't save it either — the window is read before either
worker writes it, so both pass the check.

Two mechanisms cover that, and both matter:

1. **Shard filter** on `agent_id`, as above. This is what makes the *content*
   correct: the worker groups signals by `(org_id, agent_id, failure_type)` and
   emits one alert per group, so a group must never be split across replicas.
   Hashing on `agent_id` keeps every group whole on one worker.

2. **Claiming.** `claim_unalerted_signals` sets `failure_signals.alert_claimed_at`
   and `alert_claimed_by` in the *same statement* that selects the rows, using
   `FOR UPDATE SKIP LOCKED`. This is the backstop for what sharding can't cover:
   two replicas accidentally given the same `SHARD_INDEX`, an overlapping rolling
   deploy, or a plain `--scale alerts=2` with the default `SHARD_COUNT=1`.

Claims expire after `CLAIM_TIMEOUT_SECS` so a worker that dies mid-delivery doesn't
strand its rows, and a poll that ends without delivering hands its claims back
immediately (`release_claims`) rather than waiting out the timeout. Delivery stays
**at-least-once** — receiver-side idempotency is still the receiver's job.

| Env var | Default | Description |
|---|---|---|
| `CLAIM_TIMEOUT_SECS` | `300` | How long a claim stays valid. Must exceed the worst-case delivery time for one batch, or a slow-but-alive worker's rows get stolen and double-delivered |
| `ALERTS_SHARD_COUNT` | falls back to `SHARD_COUNT` | Shard the alerts worker independently of the detector |
| `ALERTS_SHARD_INDEX` | falls back to `SHARD_INDEX` | As above |

**Scale the two workers independently with the `ALERTS_`-prefixed vars.** The
alerts worker reads the shared `SHARD_COUNT`/`SHARD_INDEX` when no prefixed value is
set, which is convenient when both are scaled together and a trap when they aren't:
the service also loads the repo-root `.env`, so a `SHARD_COUNT=4` intended for four
detector replicas would leave a lone alerts worker claiming only bucket 0 — three
quarters of signals would never alert, and nothing would look broken. The worker logs
a `WARNING` at startup whenever `SHARD_COUNT > 1` naming the indices that must be
running.

---

## Rate Limiting

The ingest API (`services/ingest/ingest_svc/rate_limiter.py`) enforces a per-key sliding-window request limit on `POST /v1/ingest` and `POST /v1/deploy`, applied by the `rate_limit_and_log` middleware in `main.py`.

**Bucketing:** a real (non-`dt_dev_`) API key is rate-limited independently by key — `rate_limit_rpm` (from `api_keys`, default `600`) governs it. A `dt_dev_*` key, or a request with no parseable `api_key` at all, is bucketed by client IP instead — dev keys are for local-only usage and aren't expected to carry a meaningful per-tenant identity. Requests from `is_trusted()` callers (the internal gateway path — see the Ingest API Endpoints section) skip rate limiting entirely; the gateway is expected to enforce its own limits before proxying here.

A denied request gets `429` with a `Retry-After` header (seconds until the oldest request in the window expires).

**Cross-process coordination:** the limiter is a per-process in-memory singleton, so a single instance enforces `rate_limit_rpm` exactly. Running multiple ingest workers or replicas would otherwise let each one enforce the *full* limit independently — N processes silently allowing N× the configured rate. `RateLimiter._heartbeat()` closes this gap approximately rather than exactly: every 10s, each process upserts a liveness row into `rate_limit_workers`, reaps rows older than 30s, and sets `self._active_workers` to the resulting count; `is_allowed()` then checks each request against `rate_limit_rpm // active_workers` instead of the raw configured value. With the default single-instance deployment, `active_workers` is always `1` and enforcement is exact, identical to before this mechanism existed.

This is a deliberate approximation, not a shortcut: an exact answer would mean a synchronous Postgres round-trip on every ingest request, which conflicts with the API's "return `202` before writing to DB" design and its throughput target. The tradeoff is a lag of up to one heartbeat interval (~10s) when a worker joins or leaves — during that window the aggregate limit across all workers can be briefly over- or under-enforced. A transient heartbeat failure (DB blip) leaves `active_workers` at its last known value rather than resetting to `1`, since resetting would itself cause every worker to briefly re-enforce the full limit — exactly the overshoot this mechanism exists to prevent.

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

**Agent overhead:** On the default HTTP ingest path (`emit_as_json=False`, no OTel exporter, no signal-trigger policy) the SDK adds roughly **10–25μs per event** — well under 500μs for a typical run. Two things change this:

- **Per-run, not per-event:** overhead scales with event count. A run emitting ~20 events stays under ~500μs total; a run with hundreds of events accumulates proportionally (still tens of µs each). The "sub-500μs" figure is a per-event/small-run guideline, not a per-run guarantee for arbitrarily large runs.
- **Signal-trigger policies run detectors in-path.** If you configure a policy with `trigger: "signal"`, the SDK evaluates the detector suite on every step — measured ~130μs/event (~2.9ms for a 20-event run), roughly 10× the baseline. Latency-sensitive agents should prefer non-signal triggers (`tool_call_count`, `llm_latency_ms`, etc.) or accept the added in-path cost.

With `emit_as_json=True` or an OTel exporter, `_emit()` also serialises to JSON or creates spans synchronously — overhead increases accordingly. The drain thread is entirely background. Even under backpressure (ingest API down), the ring buffer drops the oldest events rather than blocking the agent.

---

## Failure Modes

**Client lifecycle:** The drain thread holds only a *weak* reference to its client, so a client the caller drops is garbage-collected and its thread exits on the next poll — code that builds a client per tenant, per request, or per test doesn't accumulate one OS thread each. Anything still buffered when a client is dropped that way is lost; call `shutdown()` (or rely on the at-exit flush, which covers process exit) to ship it.

**Ingest API down:** The drain thread drains events from the buffer before shipping. With the default emitter, a failed batch is dropped — no retry, no persistence. New events continue to buffer (up to 10,000) and will be shipped once the API recovers. The agent is never blocked. To survive an outage that outlasts the buffer (or a process restart mid-outage), wrap the emitter with a durable retry queue — same design in both SDKs, backed by a local SQLite queue, retried roughly every 30s (±5s jitter) once the backend is reachable again, bounded (100k events / 100MB by default, oldest evicted first) with a rate-limited eviction warning so a long outage doesn't silently lose data without a trace in the logs:

- **Python**: `Dunetrace(emitter=DurableRetryEmitter(HttpBatchingEmitter(endpoint, api_key)))` — queue at `~/.dunetrace/queue.db` by default, or `DUNETRACE_QUEUE_PATH`. `sqlite3` is stdlib, no extra install.
- **TypeScript**: `new Dunetrace({ emitter: new DurableRetryEmitter(new HttpBatchEmitter(endpoint, apiKey)) })` — queue at `~/.dunetrace/queue-ts.db` by default (a different filename from Python's, so a mixed-language deployment sharing `~/.dunetrace/` can't have one SDK corrupt the other's queue), or `DUNETRACE_QUEUE_PATH`. Requires the optional peer dependency `better-sqlite3` (Node has no SQLite in its standard library) — see the [TypeScript SDK README](https://github.com/dunetrace/dunetrace/blob/main/packages/sdk-ts/README.md#durable-retry). Node.js only; there is no browser build of the TypeScript SDK.

**Detector worker down:** Runs queue up in the `events` table. When the worker restarts, it processes all unprocessed runs. Signals are delayed but not lost.

**Postgres down:** Ingest returns 503. SDK logs a warning and continues buffering. Events during the outage are lost (the buffer eventually overwrites). This is acceptable i.e. observability data loss during a DB outage is not a catastrophic failure.

**Alerts worker down:** Signals accumulate as `alerted=FALSE`. When the worker restarts, it picks up where it left off. Alerts are delayed but not lost (at-least-once delivery).
