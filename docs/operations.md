# Operations

Retention, storage growth, and the manual controls available for both. Covers `services/ingest` — the only service that writes to or prunes the `events` table.

---

## Event retention

`ingest_svc` runs a background retention pass that deletes events older than the
configured window. There are two paths, chosen automatically by whether the
`events` table is partitioned:

- **Partitioned table** (the intended form — monthly `events_YYYYMM` partitions):
  the pass drops whole partitions once all their rows are older than the window —
  an instant DDL `DROP TABLE`, no vacuum, no lock contention with ingest.
- **Non-partitioned table** (a deployment whose `events` table predates
  partitioning, i.e. it was created before the partitioning DDL and
  `CREATE TABLE IF NOT EXISTS` therefore never converted it): the pass falls back
  to a **batched `DELETE`** of old rows (10k per iteration). Retention still runs,
  but `DELETE` is heavier than a partition drop and leaves dead tuples for
  autovacuum to reclaim. On every pass the log carries a WARNING that the table
  isn't partitioned.

**Check which path you're on:**

```sql
SELECT relkind FROM pg_class WHERE relname = 'events';
-- 'p' = partitioned (partition-drop path) | 'r' = plain table (DELETE fallback)
```

**To move an existing plain table onto the efficient partition-drop path**, run
the opt-in offline migration (BACKUP + stop writers first; dry-run by default):

```bash
DATABASE_URL=postgres://... python scripts/migrate_events_to_partitioned.py        # dry run
DATABASE_URL=postgres://... python scripts/migrate_events_to_partitioned.py --yes  # apply
```

> Historical note: earlier versions silently no-opped retention on a
> non-partitioned table (it appeared configured but nothing was ever deleted).
> The DELETE fallback above ensures retention now runs everywhere.

### Configuration

```bash
# .env
EVENT_RETENTION_DAYS=90   # default; set to 0 to effectively disable pruning (nothing is ever old enough)
```

Restart `ingest` to pick up a change:

```bash
docker compose up -d --force-recreate ingest
```

### Schedule

The retention pass runs once immediately at startup (covers a service that was down long enough for pruning to matter right away), then every 24 hours, as a background task inside `ingest_svc`'s process lifetime — not a separate cron job or service. It shares the connection pool with the rest of `ingest_svc` and is deliberately co-located with partition creation (`_ensure_event_partitions()`, same file) rather than split into a separate maintenance service — both operate on the same table, the same pool, the same schema-ownership boundary.

A failed pass (e.g. a transient DB error) is logged and retried on the next scheduled tick — a single bad tick never kills the loop or the service.

### Monitoring

Every pass that actually drops something logs at `INFO`:

```
Pruned event partition events_202501 (data before 2025-02-01, ~184023 rows)
Retention pass complete: 1 partition(s) dropped, ~184023 rows freed
Retention pass took 0.14s, dropped 1 partition(s)
```

Row counts are Postgres's own planner estimate (`pg_class.reltuples`, refreshed by autovacuum/`ANALYZE`) rather than an exact `COUNT(*)` — exact counts would mean a full scan of each partition just for a log line.

**Startup staleness check**: on every boot, `ingest_svc` checks whether any partition already exceeds `EVENT_RETENTION_DAYS` *before* the scheduled pass runs. If so, it logs a `WARNING`:

```
Retention check: a partition already exceeds EVENT_RETENTION_DAYS=90 at startup — either
this is the first startup after enabling retention on older data (harmless, the prune loop
below will catch up momentarily), or the retention loop has been silently failing across
restarts. Watch for 'Retention pass' log lines after startup to confirm it catches up.
```

There's no persisted "last successful prune" timestamp anywhere — in-memory state wouldn't survive a restart, which is exactly the failure mode this exists to catch. This check is a DB-state proxy instead: if a partition this old still exists, pruning hasn't kept up, whether that's because it's never run, is broken, or this is a legitimate one-time catch-up. Both cases look identical from here; if the "Retention pass" log line doesn't show up shortly after this warning, something is actually wrong.

### Manual invocation

`POST /admin/prune-events` runs an out-of-band retention pass immediately, without waiting for the next scheduled tick — useful right after the startup staleness warning, or to reclaim space on demand. Admin-only, same pattern as `POST /v1/keys`:

```bash
curl -s -X POST "http://localhost:8001/admin/prune-events" \
  -H "Content-Type: application/json" \
  -d '{"admin_key": "<ADMIN_API_KEY>"}'

# Response:
{"partitions_dropped": 1, "signals_scrubbed": 412, "retention_days": 90}
```

Pass `retention_days` explicitly to override the configured default for this one invocation:

```bash
curl -s -X POST "http://localhost:8001/admin/prune-events" \
  -H "Content-Type: application/json" \
  -d '{"admin_key": "<ADMIN_API_KEY>", "retention_days": 30}'
```

`ADMIN_API_KEY` must be set in the environment — an unset or empty value rejects every request (closed by default), same as the key-creation endpoint.

Unlike the daily loop, a scrub failure here is **not** swallowed — the call returns a 500. The loop can afford to log and retry on the next tick; a manual invocation has no next tick, and reporting `signals_scrubbed: 0` for a pass that actually errored is indistinguishable from "there was nothing to scrub".

---

## Instrumentation health

`INSTRUMENTATION_DEGRADED` (see [detectors.md](detectors.md)) answers *"was this
run measurable?"*. This query answers *"is this agent's telemetry broken?"*,
which is only visible in aggregate: one blank LLM call is unremarkable, the same
call on 100% of an agent's traffic is a broken pipeline.

The fingerprint is a call that measurably took time and measurably produced
nothing:

```
output_length = 0 AND finish_reason = 'stop' AND completion_tokens = 0 AND latency_ms > 0
```

`latency_ms > 0` is what separates a real round-trip from a call that never
happened. A genuinely empty model response has this shape too — that is the
point. One such call is a finding; **above ~30% of an agent's calls it is not a
model answering nothing 30% of the time, it is an extractor reading the wrong
object.**

The canonical SQL lives in `services/api/api_svc/instrumentation_health.py` as a
single template rendered for both Postgres and SQLite, so the query documented
here and the one the test exercises cannot drift. Render it with
`blank_response_rate_sql("postgres")`.

`provider` comes from `llm.called`, not `llm.responded`, so the query joins the
two on `(org_id, run_id, step_index)` — `llm.responded` is emitted with
`advance=False` and therefore shares its `llm.called`'s `step_index`, which makes
that key work even for events predating `call_id`.

Both `finish_reason = 'stop'` and `finish_reason IS NULL` count. A current SDK
omits `finish_reason` when it could not read one, but events already stored — and
every agent still running a pre-provenance SDK — carry the fabricated `'stop'`.

**Worked example.** The incident this comes from: `langchain_openai` calls
`client.with_raw_response.create()`, so `Completions.create` returned a
`LegacyAPIResponse` rather than a `ChatCompletion`. The extractors hit their
fallback branches, substituted `("", "stop")`, and produced this fingerprint on
100% of calls — firing `EMPTY_LLM_RESPONSE` on every run including the control.
This query would have shown `blank_fraction = 1.0` for that agent on day one.

---

## Signal evidence scrub

`failure_signals` has no retention policy and deliberately keeps its rows: the dashboard compares the last 30 days of signals against a days-30-to-90 baseline to decide whether a fix worked (`verified` / `likely_fixed` / `still_occurring`), so deleting aged signals would permanently truncate the baseline arm of that comparison. The table is also not partitioned, so expiry would mean a batched `DELETE` with vacuum pressure rather than an instant partition drop, and five tables carry a bare `signal_id` with no foreign key (`fixes`, `signal_feedback`, `signal_group_members`, `linear_issue_signals`) — deleting would silently orphan them.

But a signal's `evidence` dict embeds excerpts of the same raw agent content the event retention pass exists to expire — in two cases untruncated:

| Evidence key | Detector | Content |
|---|---|---|
| `args` | `TOOL_LOOP` | Full raw arguments of **every** call in the loop |
| `taint_source` | `UNGROUNDED_DESTINATION` | Objects carrying untruncated `input_text` / tool output / retrieval content / memory values |
| `destination`, `destination_host` | `UNGROUNDED_DESTINATION` | An email address or URL |
| `tool_error` | `PREMATURE_TERMINATION`, `UNREAD_TOOL_ERROR` | Raw tool error text |
| `output_snippet` | `PREMATURE_TERMINATION` | LLM output excerpt |
| `args_snippet` | `TOOL_ARGUMENT_FABRICATION`, `UNGROUNDED_DESTINATION` | Tool argument excerpt |
| `content_snippet` | `RETRIEVED_CONTENT_INJECTION` | Retrieved text excerpt |
| `value_snippet` | `MEMORY_POISONING` | Memory value excerpt |
| `fabricated_entity` | `TOOL_ARGUMENT_FABRICATION` | The fabricated value itself |
| `missing_entities` | `HANDOFF_CONTEXT_LOSS` | Entities lifted from the parent's context |
| `memory_key` | `MEMORY_POISONING`, `UNGROUNDED_DESTINATION` | Caller-chosen memory key |

So the rows stay and those keys are stripped, on the same `EVENT_RETENTION_DAYS` window as the event prune — **one content horizon**, so raw content leaves `events` and `failure_signals.evidence` at the same moment rather than on two schedules that can drift apart.

This costs no analytics. Every SQL consumer of `evidence` reads metadata (`tool`, `count`, `consecutive_fails`, `args_identical`, `growth_factor`, `conversation_id`); none reads a key in the table above. Detector-owned labels are deliberately kept — `matched_marker` and `matched_patterns` name which of Dunetrace's own constants matched, `grounded_surfaces` names surfaces (`"input_text"`), `failure_source` is a `"declared"`/`"output_text"` literal, and every `*_length` field is an integer. The explainer's display templates already null-guard the content keys, so a scrubbed signal renders without them rather than erroring.

The scrub runs as a second pass in the same daily loop (`main.py::_run_scrub_once`), kept separate from the prune so a partition-drop error can't skip it. It is idempotent: a `WHERE evidence ?| ARRAY[...]` guard means an already-scrubbed row is never rewritten, so repeat passes cost a scan and no writes. Work is batched by `ctid` (10,000 rows) so a large backlog doesn't hold one long transaction.

```
Evidence scrub complete: 412 signal(s) stripped of content keys (detected before 2026-05-21)
Evidence scrub took 0.31s, scrubbed 412 signal(s)
```

**Adding a detector**: if it puts a content excerpt in `evidence`, add the key to `CONTENT_EVIDENCE_KEYS` in `services/ingest/ingest_svc/db/postgres.py`. `TestContentEvidenceKeyCoverage` in `services/ingest/tests/test_ingest.py` walks the detector source and fails the build if a content-derived key isn't listed — it is what catches the "new detector quietly stores content that outlives the horizon forever" case, which a count assertion cannot.

---

## processed_runs retention

`processed_runs` is the detector's idempotency ledger — one row per run, recording
that the run has been analysed. It's also the anti-join target in run discovery, so
letting it grow forever slows the detector's hottest query.

The detector worker prunes it daily (shard 0 only — the table isn't
shard-partitioned, so extra replicas would only contend on the same rows). A pass
keeps deleting while batches come back full, so a long-neglected table catches up in
one pass rather than one batch per day.

**The ordering constraint runs opposite to intuition.** A `processed_runs` row may
only be deleted *after* its run's events are gone. Delete it while the events remain
and the run reads as unprocessed: the detector re-analyses it and writes a second,
duplicate set of signals — which then alert. The delete therefore carries a
`NOT EXISTS` against `events` rather than trusting a retention constant, so the
invariant holds no matter what `EVENT_RETENTION_DAYS` is set to, including values
the detector never sees.

| Env var | Default | Description |
|---|---|---|
| `PROCESSED_RUNS_RETENTION_DAYS` | `120` | Age bound on the scan. Keep it beyond `EVENT_RETENTION_DAYS` (90) — it's a scan bound for cheapness, not the safety mechanism |
| `PRUNE_BATCH_SIZE` | `10000` | Rows per delete batch |

Lowering `PROCESSED_RUNS_RETENTION_DAYS` below `EVENT_RETENTION_DAYS` is not
dangerous — the `NOT EXISTS` still refuses to delete rows whose events survive — it
just wastes work re-examining rows that can't yet qualify.

There's no manual trigger endpoint; the loop runs on worker startup and every 24h.
To verify it's working, watch for `Pruned N processed_runs row(s)` in the detector
logs, or compare `SELECT count(*) FROM processed_runs` against the retained run
count over time.

---

## Rate limiting

`ingest_svc` enforces a per-API-key sliding-window rate limit (60s window, `rate_limit_rpm` from the `api_keys` table) on `POST /v1/ingest`, `POST /v1/deploy`, and `POST /v1/otlp/traces`. Keys are org-scoped, not agent-scoped — one key can carry traffic for many agents. Without any further limiting, one runaway agent under a shared key can consume the entire key's budget and starve its siblings.

### Per-agent sub-limits

Within a key's overall budget, each distinct `agent_id` gets its own sliding-window sub-limit — by default, 20% of the key's effective rpm. An agent hitting its own sub-limit gets a 429; other agents under the same key are unaffected. The key-level limit is still checked first and always applies regardless of agent — many agents each within their own sub-limit can still collectively exhaust the key.

Agent identity for sub-limiting comes from the request body's `agent_id` field for `/v1/ingest` and `/v1/deploy` (already parsed for auth), and from the `X-Dunetrace-Agent-Id` header for `/v1/otlp/traces` (cheap to read; the `service.name` resource-attribute fallback would require decoding the OTLP body, which the rate-limit middleware deliberately avoids — an OTLP trace relying on `service.name` alone only gets key-level limiting, not a per-agent sub-limit).

**Important if a key genuinely has only one real agent**: the 20% default still applies unless overridden — it does not detect "this key only has one agent" and skip sub-limiting. A single-agent-per-key deployment relying on the full key rpm for that one agent should set an explicit override (see below) closer to 1.0, or the effective throughput for that agent will be capped at 20% of what the key alone would otherwise allow.

### Adjusting a per-agent quota

```bash
# View current quota (default or override) for an agent under a key
curl -s "http://localhost:8001/admin/keys/{key_id}/agents/{agent_id}/quota?admin_key=<ADMIN_API_KEY>"

# Response:
{"key_id": 42, "agent_id": "worker-1", "quota_pct": 0.20, "is_override": false}

# Set an override — e.g. give this agent 50% of the key's rpm
curl -s -X PUT "http://localhost:8001/admin/keys/{key_id}/agents/{agent_id}/quota" \
  -H "Content-Type: application/json" \
  -d '{"admin_key": "<ADMIN_API_KEY>", "quota_pct": 0.5}'
```

`key_id` is the numeric id from the key-creation response or `GET /v1/keys` (api_svc) — never the raw secret key string, which shouldn't appear in a URL path. Quota changes take effect within a few minutes (cached the same way `rate_limit_rpm` is, not read fresh on every request) — not instantly.

### 429 response headers

```
HTTP/1.1 429 Too Many Requests
Retry-After: 12
X-RateLimit-Key-Remaining: 0
X-RateLimit-Agent-Remaining: 0
```

`X-RateLimit-Agent-Remaining` is only present when an `agent_id` was resolvable for the request (see above) — its absence means only the key-level limit was checked, not that the agent has unlimited quota.
