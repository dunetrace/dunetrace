# Operations

Retention, storage growth, and the manual controls available for both. Covers `services/ingest` — the only service that writes to or prunes the `events` table.

---

## Event retention

The `events` table is partitioned monthly (`events_YYYYMM`, see `CLAUDE.md`'s Database Schema section). `ingest_svc` runs a background retention pass that drops whole partitions once every one of their rows is older than the configured retention window — an instant DDL `DROP TABLE`, not a row-by-row `DELETE`, so it doesn't need a vacuum and doesn't compete with ingest traffic for locks.

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
