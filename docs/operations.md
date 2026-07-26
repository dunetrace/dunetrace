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
{"partitions_dropped": 1, "retention_days": 90}
```

Pass `retention_days` explicitly to override the configured default for this one invocation:

```bash
curl -s -X POST "http://localhost:8001/admin/prune-events" \
  -H "Content-Type: application/json" \
  -d '{"admin_key": "<ADMIN_API_KEY>", "retention_days": 30}'
```

`ADMIN_API_KEY` must be set in the environment — an unset or empty value rejects every request (closed by default), same as the key-creation endpoint.

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
