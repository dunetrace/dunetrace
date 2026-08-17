"""
Async Postgres connection pool via asyncpg. Created at startup and shared via FastAPI lifespan.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

try:
    import asyncpg
except ImportError:  # pragma: no cover - allows tests without db driver
    asyncpg = None  # type: ignore

from ingest_svc.config import settings

logger = logging.getLogger("dunetrace.ingest.db")

_pool: Optional[asyncpg.Pool] = None  # type: ignore[attr-defined]


# ── Pool lifecycle ─────────────────────────────────────────────────────────────


def get_pool():
    return _pool


async def init_pool() -> None:
    global _pool
    if asyncpg is None:
        raise RuntimeError("asyncpg is not installed")
    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=10,
        # DATABASE_URL points at Supabase's transaction-mode PgBouncer pooler
        # (port 6543). PgBouncer in that mode doesn't preserve server-side
        # prepared statements across pooled connections, so asyncpg's default
        # client-side statement cache goes stale mid-connection and every
        # query fails with "prepared statement ... does not exist". Disabling
        # it forces the extended query protocol without server-side PREPARE,
        # which is what PgBouncer transaction pooling actually supports.
        statement_cache_size=0,
    )
    logger.info("DB pool ready")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def check_db() -> str:
    if not _pool:
        return "no_pool"
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return "ok"
    except Exception as exc:
        return str(exc)


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
-- Partitioned by received_at (monthly range partitions).
-- PRIMARY KEY includes received_at because Postgres requires the partition key
-- in the PK of a partitioned table.  No other table references events.id as a
-- FK — cross-table joins use run_id (TEXT) — so the composite PK is safe.
-- NOTE: this CREATE is a no-op on existing deployments that already have a
-- non-partitioned events table; those require an explicit offline migration.
CREATE TABLE IF NOT EXISTS events (
    id             BIGSERIAL        NOT NULL,
    batch_id       TEXT             NOT NULL,
    event_type     TEXT             NOT NULL,
    run_id         TEXT             NOT NULL,
    agent_id       TEXT             NOT NULL,
    agent_version  TEXT             NOT NULL,
    step_index     INTEGER          NOT NULL,
    timestamp      DOUBLE PRECISION NOT NULL,
    payload        JSONB            NOT NULL DEFAULT '{}',
    parent_run_id  TEXT,
    received_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    event_id       TEXT,
    PRIMARY KEY (id, received_at)
) PARTITION BY RANGE (received_at);

-- audit Finding 14: client-generated dedup id. A global UNIQUE isn't possible on
-- a table partitioned by received_at (a unique index must include the partition
-- key, but received_at differs per retry) — so ingest dedups at the application
-- layer using this id (see insert_events). ALTER covers pre-existing tables.
ALTER TABLE events ADD COLUMN IF NOT EXISTS event_id TEXT;
CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_run_id  ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_agent   ON events(agent_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent_run ON events(agent_id, run_id);

CREATE TABLE IF NOT EXISTS failure_signals (
    id             BIGSERIAL PRIMARY KEY,
    failure_type   TEXT        NOT NULL,
    severity       TEXT        NOT NULL,
    run_id         TEXT        NOT NULL,
    agent_id       TEXT        NOT NULL,
    agent_version  TEXT        NOT NULL,
    step_index     INTEGER     NOT NULL,
    confidence     REAL        NOT NULL,
    evidence       JSONB       NOT NULL DEFAULT '{}',
    detected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alerted        BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_signals_agent     ON failure_signals(agent_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_unalerted ON failure_signals(alerted) WHERE alerted = FALSE;

ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS co_signal_count INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS companies (
    id          TEXT PRIMARY KEY,
    name        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    key         TEXT PRIMARY KEY,
    agent_id    TEXT        NOT NULL,
    customer_id TEXT        NOT NULL,
    active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    company_id  TEXT        REFERENCES companies(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS fixes (
    id                    BIGSERIAL PRIMARY KEY,
    run_id                TEXT        NOT NULL,
    signal_id             BIGINT      NOT NULL,
    fix_content           TEXT        NOT NULL,
    fix_type              TEXT        NOT NULL DEFAULT 'prompt_addition',
    applied_via           TEXT        NOT NULL,
    langfuse_prompt_name  TEXT,
    langfuse_version      INTEGER,
    applied_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fixes_signal_id ON fixes(signal_id);
CREATE INDEX IF NOT EXISTS idx_fixes_run_id    ON fixes(run_id, applied_at DESC);

CREATE TABLE IF NOT EXISTS policies (
    id          BIGSERIAL PRIMARY KEY,
    agent_id    TEXT        NOT NULL DEFAULT '*',
    name        TEXT        NOT NULL,
    condition   JSONB       NOT NULL,
    action      JSONB       NOT NULL,
    enabled     BOOLEAN     NOT NULL DEFAULT TRUE,
    priority    INT         NOT NULL DEFAULT 100,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_policies_agent ON policies(agent_id, enabled);

-- Policy evaluation observability (Phase 5). One row per shipped policy.evaluated
-- record (rate-limited SDK-side). `trigger_name` avoids the SQL reserved word
-- `trigger`. Read by the customer API's GET /v1/policies/{id}/evaluations.
CREATE TABLE IF NOT EXISTS policy_evaluations (
    id              BIGSERIAL PRIMARY KEY,
    org_id          TEXT,
    policy_id       BIGINT,
    policy_name     TEXT        NOT NULL DEFAULT '',
    agent_id        TEXT        NOT NULL DEFAULT '',
    run_id          TEXT,
    trigger_name    TEXT,
    trigger_matched BOOLEAN,
    fired           BOOLEAN,
    sampled         BOOLEAN     NOT NULL DEFAULT FALSE,
    reason          TEXT,
    conditions      JSONB,
    evaluated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_policy_evals_policy
    ON policy_evaluations(org_id, policy_id, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS idx_policy_evals_agent
    ON policy_evaluations(org_id, agent_id, evaluated_at DESC);

CREATE TABLE IF NOT EXISTS deploy_events (
    id           BIGSERIAL PRIMARY KEY,
    agent_id     TEXT        NOT NULL,
    version      TEXT        NOT NULL,
    deployed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta         JSONB       NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_deploys_agent ON deploy_events(agent_id, deployed_at DESC);

-- Cross-process rate-limit coordination: each ingest worker process upserts
-- its own row on a heartbeat (see rate_limiter.py::RateLimiter._heartbeat).
-- Rows older than the heartbeat's own staleness window are deleted by the
-- same heartbeat, so this table is self-cleaning — no separate reaper needed.
CREATE TABLE IF NOT EXISTS rate_limit_workers (
    worker_id  TEXT             PRIMARY KEY,
    last_seen  DOUBLE PRECISION NOT NULL
);

-- Per-agent rate-limit sub-quotas within a key's overall budget (see
-- rate_limiter.py's module docstring for the "one runaway agent starves its
-- siblings" problem this solves). key_id references api_keys.id, added by
-- api_svc's own migration (services/api/api_svc/db/queries.py's _KEYS_DDL) —
-- not enforced as a DB foreign key here, since that column may not exist yet
-- if ingest_svc starts before api_svc ever has (same cross-service ordering
-- rate_limiter.py's rate_limit_rpm lookup already tolerates). Referential
-- integrity is checked at the admin endpoint's write time instead.
CREATE TABLE IF NOT EXISTS agent_rate_quotas (
    key_id     BIGINT      NOT NULL,
    agent_id   TEXT        NOT NULL,
    quota_pct  REAL        NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (key_id, agent_id)
);

-- OTel receiver observability (Phase 5). One row per (org, hour): what the OTLP
-- receiver did, drained here from ingest's in-memory counters. org_id='_system'
-- holds events that predate org attribution (auth failures, malformed bodies).
CREATE TABLE IF NOT EXISTS otel_receiver_stats (
    org_id            TEXT        NOT NULL,
    hour_bucket       TIMESTAMPTZ NOT NULL,
    batches_received  BIGINT      NOT NULL DEFAULT 0,
    spans_received    BIGINT      NOT NULL DEFAULT 0,
    events_translated BIGINT      NOT NULL DEFAULT 0,
    spans_rejected    BIGINT      NOT NULL DEFAULT 0,
    auth_failures     BIGINT      NOT NULL DEFAULT 0,
    rate_limit_hits   BIGINT      NOT NULL DEFAULT 0,
    rejections        JSONB       NOT NULL DEFAULT '{}',
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, hour_bucket)
);
CREATE INDEX IF NOT EXISTS idx_otel_receiver_stats_org_hour
    ON otel_receiver_stats(org_id, hour_bucket DESC);
"""

# ── Multi-tenancy unification (v0.5.0) ──────────────────────────────────────────
#
# `organizations` replaces `companies`; `org_id` replaces `api_keys.customer_id`.
# Renamed (not recreated) so existing data survives. `company_id` is dropped —
# it was always redundant with customer_id (create_api_key wrote the same value
# to both).
#
# Every other org_id-bearing table below is nullable until _backfill_org_id()
# populates it from the (now-renamed) api_keys.org_id, then NOT NULL is applied.
# This runs after _SCHEMA so the tables it touches already exist.
_MULTI_TENANCY_DDL = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'companies')
       AND NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'organizations')
    THEN
        ALTER TABLE companies RENAME TO organizations;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS organizations (
    id          TEXT PRIMARY KEY,
    name        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO organizations (id, name) VALUES ('default', 'Default Organization')
ON CONFLICT (id) DO NOTHING;

-- Semantic feedback loop opt-in (Phase 1.4.3) — owned by api_svc's own
-- migration (services/api/api_svc/db/queries.py's _SEMANTIC_FEEDBACK_DDL),
-- added here defensively too since this service creates `organizations`
-- first on a fresh install and semantic_svc reads these columns without
-- necessarily waiting on api_svc to have started.
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_feedback_enabled BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_feedback_auto_suppress BOOLEAN NOT NULL DEFAULT FALSE;

-- OTel ingestion opt-out per org (Phase 2). Defaults TRUE so every org that
-- accepts OTLP today keeps working; an admin flips it FALSE to stop accepting a
-- specific org's OTLP traffic (a per-org kill switch on top of rate limiting).
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS otel_ingestion_enabled BOOLEAN NOT NULL DEFAULT TRUE;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'api_keys' AND column_name = 'customer_id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'api_keys' AND column_name = 'org_id'
    ) THEN
        ALTER TABLE api_keys RENAME COLUMN customer_id TO org_id;
    END IF;
END $$;

ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE api_keys DROP COLUMN IF EXISTS company_id;

-- Installs where org_id was added by some path other than the rename above (so the
-- rename's "org_id does not exist yet" guard never fired) are left with both columns
-- permanently: customer_id stranded with its real value, org_id NULL. Recover it here,
-- before customer_id is dropped, for any row where org_id never got a value at all.
-- Guarded on column existence — a fresh install past the rename never had customer_id
-- at all, and referencing a nonexistent column fails to parse, not just fails at runtime.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'api_keys' AND column_name = 'customer_id'
    ) THEN
        -- organizations rows must exist first — org_id gets an FK to organizations(id)
        -- below, and a bare customer_id value (e.g. 'acme-corp') was never registered as
        -- one. Scoped to rows actually being backfilled (org_id IS NULL) — an install
        -- whose org_id is already resolved (however it got there) shouldn't gain orphan
        -- organizations rows for customer_id values nothing will ever point at.
        INSERT INTO organizations (id, name)
            SELECT DISTINCT customer_id, customer_id FROM api_keys
            WHERE customer_id IS NOT NULL AND org_id IS NULL
        ON CONFLICT (id) DO NOTHING;

        UPDATE api_keys SET org_id = customer_id
            WHERE org_id IS NULL AND customer_id IS NOT NULL;

        ALTER TABLE api_keys DROP COLUMN customer_id;
    END IF;
END $$;

-- api_keys.agent_id is NOT dropped here — _backfill_org_id() below still needs it
-- to join events/signals/etc to the org that issued the key. It's dropped by
-- _backfill_org_id() itself, after the join is done. See that function's docstring.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'api_keys' AND constraint_name = 'api_keys_org_id_fkey'
    ) THEN
        ALTER TABLE api_keys
            ADD CONSTRAINT api_keys_org_id_fkey FOREIGN KEY (org_id) REFERENCES organizations(id);
    END IF;
END $$;

ALTER TABLE events          ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE fixes           ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE deploy_events   ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE policies        ADD COLUMN IF NOT EXISTS org_id TEXT;
-- HMAC canonical-form version (see api_svc _sign_policy). Added defensively here
-- too so fetch_policies' SELECT never fails on a missing column regardless of
-- which service ran its schema first.
ALTER TABLE policies        ADD COLUMN IF NOT EXISTS signature   TEXT NOT NULL DEFAULT '';
ALTER TABLE policies        ADD COLUMN IF NOT EXISTS sig_version INT NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_events_org_agent  ON events(org_id, agent_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_org_agent ON failure_signals(org_id, agent_id, detected_at DESC);

-- External evaluation integration correlation key (Langfuse/LangSmith/
-- Braintrust — Phase 2). Optional/instrumentation-dependent, same as
-- system_prompt: populated when the customer's dt.run(trace_id=...) sets it,
-- or automatically for OTLP-ingested runs (the raw OTel traceId, before
-- otel.py's lossy _trace_to_uuid() conversion into run_id). A genuine column
-- (not payload-only) because it's looked up in the REVERSE direction from
-- every other per-run field here — "given an external trace_id, find the
-- run" — which a JSONB payload scan can't do efficiently.
ALTER TABLE events ADD COLUMN IF NOT EXISTS trace_id TEXT;
CREATE INDEX IF NOT EXISTS idx_events_trace_id ON events(trace_id) WHERE trace_id IS NOT NULL;

-- Conversation modeling (Phase 3.1). Same column-not-payload rationale as
-- trace_id — detector_svc reads this back off every event in a processed
-- run to upsert the conversations/runs registry (services/detector/detector_svc/db.py),
-- which is a reverse-direction lookup a JSONB payload scan can't do
-- efficiently either. Optional: dt.run(conversation_id=...) sets it, threaded
-- onto every event in the run the same way trace_id is; omitted entirely for
-- single-turn agents and any run predating this field.
ALTER TABLE events ADD COLUMN IF NOT EXISTS conversation_id TEXT;
CREATE INDEX IF NOT EXISTS idx_events_conversation_id ON events(conversation_id) WHERE conversation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fixes_org         ON fixes(org_id);
CREATE INDEX IF NOT EXISTS idx_deploys_org       ON deploy_events(org_id);
CREATE INDEX IF NOT EXISTS idx_policies_org      ON policies(org_id, enabled);
"""

_ORG_BACKFILL_TABLES = ("events", "failure_signals", "fixes", "deploy_events", "policies")


async def _backfill_org_id(conn) -> None:
    """Populate org_id on pre-v0.5.0 rows via the agent_id -> api_keys.org_id mapping,
    then enforce NOT NULL. Idempotent — a second run is a fast no-op since org_id IS NULL
    matches nothing once backfilled.

    agent_ids that map to more than one distinct org_id in api_keys (only possible if
    a self-hosted install issued keys for the same agent_id under different customer_ids
    pre-migration) can't be safely auto-resolved. They're logged and fall back to
    'default'.
    """
    agent_id_column_exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'api_keys' AND column_name = 'agent_id'
        )
        """
    )
    if not agent_id_column_exists:
        # api_keys.agent_id is dropped as the final step of a previous successful
        # run of this function — its absence means the backfill already completed.
        return

    ambiguous = await conn.fetch(
        """
        SELECT agent_id, COUNT(DISTINCT org_id) AS org_count
        FROM api_keys
        WHERE org_id IS NOT NULL
        GROUP BY agent_id
        HAVING COUNT(DISTINCT org_id) > 1
        """
    )
    if ambiguous:
        logger.warning(
            "Multi-tenancy backfill: %d agent_id(s) map to multiple orgs in api_keys and "
            "cannot be auto-assigned — defaulting to org_id='default': %s.",
            len(ambiguous),
            [r["agent_id"] for r in ambiguous],
        )

    for table in _ORG_BACKFILL_TABLES:
        if table == "fixes":
            # fixes has no agent_id column of its own — it only carries signal_id/run_id.
            # Backfill via the failure_signals row the fix was recorded against instead.
            await conn.execute(
                """
                UPDATE fixes f
                SET org_id = fs.org_id
                FROM failure_signals fs
                WHERE f.signal_id = fs.id AND f.org_id IS NULL AND fs.org_id IS NOT NULL
                """
            )
        else:
            await conn.execute(
                f"""
                UPDATE {table} t
                SET org_id = ak.org_id
                FROM (
                    SELECT agent_id, MIN(org_id) AS org_id
                    FROM api_keys
                    WHERE org_id IS NOT NULL
                    GROUP BY agent_id
                    HAVING COUNT(DISTINCT org_id) = 1
                ) ak
                WHERE t.agent_id = ak.agent_id AND t.org_id IS NULL
                """
            )
        defaulted = await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE org_id IS NULL")
        if defaulted:
            logger.warning(
                "Multi-tenancy backfill: %d row(s) in %s had no resolvable org_id "
                "(no matching api_keys.agent_id) — defaulting to org_id='default'.",
                defaulted,
                table,
            )
        await conn.execute(f"UPDATE {table} SET org_id = 'default' WHERE org_id IS NULL")
        await conn.execute(f"ALTER TABLE {table} ALTER COLUMN org_id SET NOT NULL")

    defaulted_keys = await conn.fetchval("SELECT COUNT(*) FROM api_keys WHERE org_id IS NULL")
    if defaulted_keys:
        logger.warning(
            "Multi-tenancy backfill: %d row(s) in api_keys had no resolvable org_id "
            "(customer_id present but no matching organizations row, or neither "
            "column populated) — defaulting to org_id='default'.",
            defaulted_keys,
        )
    await conn.execute("UPDATE api_keys SET org_id = 'default' WHERE org_id IS NULL")
    await conn.execute("ALTER TABLE api_keys ALTER COLUMN org_id SET NOT NULL")

    # Safe to drop now — every row that needed it for the join above has been read.
    await conn.execute("ALTER TABLE api_keys DROP COLUMN IF EXISTS agent_id")


async def _ensure_event_partitions(conn, months_ahead: int = 3) -> None:
    """Create monthly child partitions for the events table from the current month
    through months_ahead months from now.  Safe to call on every startup.

    Only runs when events is actually a partitioned table (relkind='p').
    This is a no-op on existing non-partitioned deployments.
    """
    from datetime import date

    is_partitioned = await conn.fetchval(
        "SELECT COUNT(*) FROM pg_class WHERE relname = 'events' AND relkind = 'p'"
    )
    if not is_partitioned:
        return

    # Default partition must exist before any named partitions so rows inserted
    # before the first monthly partition is created don't fail.
    await conn.execute("CREATE TABLE IF NOT EXISTS events_default PARTITION OF events DEFAULT")

    today = date.today()
    for delta in range(months_ahead + 1):
        offset = today.month - 1 + delta
        year, month = today.year + offset // 12, offset % 12 + 1
        end_year = year + (1 if month == 12 else 0)
        end_month = month % 12 + 1
        start, end = date(year, month, 1), date(end_year, end_month, 1)
        name = f"events_{start.strftime('%Y%m')}"
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS {name} "
            f"PARTITION OF events FOR VALUES FROM ('{start}') TO ('{end}')"
        )
        logger.debug("Ensured partition %s (%s–%s)", name, start, end)


async def retention_looks_stale(retention_days: int = 90) -> bool:
    """True if any events_YYYYMM partition already exceeds retention_days.

    There's no persisted "last successful prune" timestamp anywhere — in-memory
    state wouldn't survive a restart, which is exactly the failure mode this
    exists to catch (the retention asyncio task silently dying and nobody
    noticing across a redeploy). This checks the same partition-age condition
    prune_old_events() acts on, read-only, as a DB-state-derived proxy: if a
    partition this old still exists, pruning hasn't kept up, whether that's
    because it's never run, is broken, or this is the very first startup after
    enabling retention on old data (a one-time, non-alarming catch-up case
    that looks identical from here — the caller decides how to react).
    """
    from datetime import date, timedelta

    if not _pool:
        return False

    cutoff = date.today() - timedelta(days=retention_days)
    async with _pool.acquire() as conn:
        is_partitioned = await conn.fetchval(
            "SELECT COUNT(*) FROM pg_class WHERE relname = 'events' AND relkind = 'p'"
        )
        if not is_partitioned:
            return False

        rows = await conn.fetch("""
            SELECT c.relname
            FROM pg_class c
            JOIN pg_inherits i ON i.inhrelid = c.oid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'events'
              AND c.relname ~ '^events_[0-9]{6}$'
            """)
        for row in rows:
            name = row["relname"]
            try:
                year, month = int(name[7:11]), int(name[11:13])
                end_year = year + (1 if month == 12 else 0)
                end_month = month % 12 + 1
                partition_end = date(end_year, end_month, 1)
                if partition_end <= cutoff:
                    return True
            except (ValueError, IndexError):
                pass
        return False


async def prune_old_events(retention_days: int = 90) -> int:
    """Drop monthly event partitions whose data is entirely older than retention_days.

    Uses the partition name (events_YYYYMM) to determine the end of each partition
    window without parsing pg_partbound.  Only touches partitions matching that
    naming convention; the default partition (events_default) is never dropped.

    Returns the number of partitions dropped.  Safe to call at any time.
    """
    from datetime import date, timedelta

    if not _pool:
        return 0

    cutoff = date.today() - timedelta(days=retention_days)
    async with _pool.acquire() as conn:
        is_partitioned = await conn.fetchval(
            "SELECT COUNT(*) FROM pg_class WHERE relname = 'events' AND relkind = 'p'"
        )
        if not is_partitioned:
            # audit Finding 13: the intended monthly partitions don't exist here
            # (this table predates partitioning or was never migrated). Instead of
            # silently no-opping — which let events grow forever while docs claimed
            # 90-day retention — fall back to a batched DELETE so retention ACTUALLY
            # runs, and warn loudly so an operator can migrate for the far cheaper
            # partition-drop path (scripts/migrate_events_to_partitioned.py).
            logger.warning(
                "events table is NOT partitioned — retention is running via a "
                "batched DELETE (slower + needs vacuum, unlike an instant "
                "partition drop). Migrate to a partitioned events table for "
                "efficient retention: scripts/migrate_events_to_partitioned.py."
            )
            deleted_total = 0
            while True:
                result = await conn.execute(
                    "DELETE FROM events WHERE ctid IN "
                    "(SELECT ctid FROM events WHERE received_at < $1 LIMIT 10000)",
                    cutoff,
                )
                n = int(result.split()[-1]) if result.startswith("DELETE") else 0
                deleted_total += n
                if n < 10000:
                    break
            if deleted_total:
                logger.info(
                    "Retention pass (DELETE fallback): %d event(s) deleted (before %s)",
                    deleted_total,
                    cutoff,
                )
            return deleted_total

        rows = await conn.fetch("""
            SELECT c.relname, c.reltuples
            FROM pg_class c
            JOIN pg_inherits i ON i.inhrelid = c.oid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'events'
              AND c.relname ~ '^events_[0-9]{6}$'
            """)

        dropped = 0
        approx_rows_freed = 0
        for row in rows:
            name = row["relname"]
            try:
                year, month = int(name[7:11]), int(name[11:13])
                end_year = year + (1 if month == 12 else 0)
                end_month = month % 12 + 1
                partition_end = date(end_year, end_month, 1)
                if partition_end <= cutoff:
                    # reltuples is a planner estimate (updated by autovacuum/analyze),
                    # not an exact count — a full COUNT(*) before every drop would
                    # scan the whole partition just for a log line. Good enough for
                    # "how much did this actually free" at operator-log granularity.
                    partition_rows = max(0, int(row["reltuples"] or 0))
                    await conn.execute(f'DROP TABLE IF EXISTS "{name}"')
                    logger.info(
                        "Pruned event partition %s (data before %s, ~%d rows)",
                        name,
                        partition_end,
                        partition_rows,
                    )
                    dropped += 1
                    approx_rows_freed += partition_rows
            except (ValueError, IndexError):
                pass

        if dropped:
            logger.info(
                "Retention pass complete: %d partition(s) dropped, ~%d rows freed",
                dropped,
                approx_rows_freed,
            )

        return dropped


async def ensure_schema() -> None:
    """Create this service's base tables, then bring the SHARED schema up to
    date. Idempotent — safe to call on every startup.

    Migrations run last, after the base DDL, because they reshape tables this
    block creates. They own every definition more than one service touches (see
    dunetrace_schemas.migrations); anything above is single-owner.
    """
    from dunetrace_schemas.migrations import apply_migrations

    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_SCHEMA)
        await conn.execute(_MULTI_TENANCY_DDL)
        await _backfill_org_id(conn)
        await _ensure_event_partitions(conn)
        version = await apply_migrations(conn)
    logger.info("Schema ready (shared schema at version %d)", version)


# ── Queries ────────────────────────────────────────────────────────────────────


async def insert_events(events: list, batch_id: str, org_id: str) -> int:
    """Bulk insert IngestEvent objects. Called from a BackgroundTask after the response is already sent.

    org_id is resolved once per request (from the API key) at the router, not carried
    per-event — a batch always belongs to one org.
    """
    if not _pool:
        logger.error("insert_events: pool not available, dropping %d events", len(events))
        return 0

    try:
        async with _pool.acquire() as conn:
            # audit Finding 14: drop events whose client-generated event_id was
            # already ingested (an at-least-once SDK retry re-sends the exact same
            # events with the same ids). Events with no event_id (older clients)
            # are never deduped. This filter is partition-safe (no unique index on
            # the received_at-partitioned table); sequential retries from one SDK
            # don't race, which is the case this protects.
            ids = [e.event_id for e in events if getattr(e, "event_id", None)]
            already: set = set()
            if ids:
                existing = await conn.fetch(
                    "SELECT event_id FROM events WHERE event_id = ANY($1::text[])", ids
                )
                already = {r["event_id"] for r in existing}

            rows = [
                (
                    batch_id,
                    e.event_type,
                    e.run_id,
                    e.agent_id,
                    e.agent_version,
                    e.step_index,
                    e.timestamp,
                    json.dumps(e.payload),
                    e.parent_run_id,
                    org_id,
                    e.trace_id,
                    e.conversation_id,
                    getattr(e, "event_id", None),
                )
                for e in events
                if not (getattr(e, "event_id", None) and e.event_id in already)
            ]
            if not rows:
                return 0
            await conn.executemany(
                """
                INSERT INTO events
                    (batch_id, event_type, run_id, agent_id, agent_version,
                     step_index, timestamp, payload, parent_run_id, org_id, trace_id,
                     conversation_id, event_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13)
                """,
                rows,
            )
        return len(rows)
    except Exception as exc:
        logger.error("insert_events failed: %s", exc)
        return 0


async def insert_policy_evaluations(events: list, batch_id: str, org_id: str) -> int:
    """Persist policy.evaluated observability records into policy_evaluations.

    Each event's payload is a PolicyEvaluationRecord dict (policy_name/id, trigger,
    trigger_matched, fired, conditions, reason, sampled, ts). Best-effort — a
    failure here never affects the main event-ingest path (they're inserted
    separately by the router). Returns rows written.
    """
    if not _pool:
        logger.error(
            "insert_policy_evaluations: pool not available, dropping %d records", len(events)
        )
        return 0
    try:
        rows = []
        for e in events:
            p = getattr(e, "payload", None) or {}
            rows.append(
                (
                    org_id,
                    p.get("policy_id"),
                    p.get("policy_name") or "",
                    p.get("agent_id") or getattr(e, "agent_id", "") or "",
                    p.get("run_id") or getattr(e, "run_id", None),
                    p.get("trigger"),
                    p.get("trigger_matched"),
                    p.get("fired"),
                    bool(p.get("sampled", False)),
                    p.get("reason"),
                    json.dumps(p.get("conditions") or []),
                    float(p.get("ts") or getattr(e, "timestamp", 0.0) or 0.0),
                )
            )
        if not rows:
            return 0
        async with _pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO policy_evaluations
                    (org_id, policy_id, policy_name, agent_id, run_id, trigger_name,
                     trigger_matched, fired, sampled, reason, conditions, evaluated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb,
                        to_timestamp($12))
                """,
                rows,
            )
        return len(rows)
    except Exception as exc:
        logger.error("insert_policy_evaluations failed: %s", exc)
        return 0


async def insert_deploy_event(agent_id: str, version: str, meta: dict, org_id: str) -> int:
    """Insert a deploy marker. Returns the new row id, or 0 on failure."""
    if not _pool:
        return 0
    import json as _json

    try:
        async with _pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO deploy_events (agent_id, version, meta, org_id)
                VALUES ($1, $2, $3::jsonb, $4)
                RETURNING id
                """,
                agent_id,
                version,
                _json.dumps(meta),
                org_id,
            )
        return int(row_id)
    except Exception as exc:
        logger.error("insert_deploy_event failed: %s", exc)
        return 0


async def fetch_policies(agent_id: str, org_id: str) -> list:
    """
    Return enabled policies for this org matching agent_id or '*'.
    Called by the ingest service's policy fetch endpoint (SDK-facing).

    org_id is filtered first: a wildcard agent_id policy only applies within
    the org that created it, never across orgs.
    """
    if not _pool:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, agent_id, name, condition, action, enabled, priority,
                       signature, sig_version
                FROM policies
                WHERE org_id = $1
                  AND enabled = TRUE
                  AND (agent_id = $2 OR agent_id = '*')
                ORDER BY priority ASC, id ASC
                """,
                org_id,
                agent_id,
            )

        def _j(v):
            if not v:
                return {}
            if isinstance(v, str):
                return json.loads(v)
            return dict(v)

        return [
            {
                "id": r["id"],
                "agent_id": r["agent_id"],
                "name": r["name"],
                "condition": _j(r["condition"]),
                "action": _j(r["action"]),
                "enabled": r["enabled"],
                "priority": r["priority"],
                "signature": r["signature"] or "",
                "sig_version": r["sig_version"] or 1,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("fetch_policies failed: %s", exc)
        return []


async def create_api_key(
    org_id: str, org_name: str | None = None, rate_limit_rpm: int = 600
) -> str:
    """Generate a new API key for an org, upsert the organization, store the key, and return it.

    Keys are org-scoped, not agent-scoped: this key can submit events for any
    agent_id under org_id, discovered on first ingest.
    """
    from dunetrace_schemas.keys import generate_api_key, hash_api_key, key_prefix

    key = generate_api_key()
    if not _pool:
        raise RuntimeError("DB pool not ready")
    name = org_name or org_id
    async with _pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO organizations (id, name) VALUES ($1, $2) "
                "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                org_id,
                name,
            )
            # Only the hash is persisted. `key` is still written because the
            # column is NOT NULL from the pre-hash schema, but it holds the
            # hash too — nothing anywhere reads it as a secret any more, and
            # the plaintext leaves this function only as the return value,
            # shown to the caller once.
            await conn.execute(
                "INSERT INTO api_keys (key, key_hash, key_prefix, org_id, rate_limit_rpm) "
                "VALUES ($1, $1, $2, $3, $4)",
                hash_api_key(key),
                key_prefix(key),
                org_id,
                rate_limit_rpm,
            )
    return key


async def get_agent_quota_by_key(api_key: str, agent_id: str) -> Optional[float]:
    """Look up a per-agent rate-limit quota override (fraction of the key's
    sustained rpm) by the raw key string — the rate limiter's hot path only
    has this, never the numeric key_id. Returns None if no override is set
    (caller applies its own default)."""
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT q.quota_pct FROM agent_rate_quotas q
                JOIN api_keys k ON k.id = q.key_id
                WHERE k.key = $1 AND q.agent_id = $2
                """,
                api_key,
                agent_id,
            )
        return float(row["quota_pct"]) if row else None
    except Exception as exc:
        logger.debug("agent quota lookup failed: %s", type(exc).__name__)
        return None


async def get_agent_quota(key_id: int, agent_id: str) -> Optional[float]:
    """Admin-facing lookup by key_id (the id the admin endpoint deals in —
    never the raw secret key itself)."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT quota_pct FROM agent_rate_quotas WHERE key_id = $1 AND agent_id = $2",
            key_id,
            agent_id,
        )
    return float(row["quota_pct"]) if row else None


async def set_agent_quota(key_id: int, agent_id: str, quota_pct: float) -> None:
    """Upsert a per-agent quota override. Raises RuntimeError if key_id
    doesn't reference a real api_keys row — checked here rather than via a DB
    foreign key, since api_keys.id is added by a separate service's migration
    (see the agent_rate_quotas table comment in _SCHEMA) and may not exist yet
    on a fresh ingest_svc-only deployment."""
    if not _pool:
        raise RuntimeError("DB pool not ready")
    async with _pool.acquire() as conn:
        key_exists = await conn.fetchval("SELECT 1 FROM api_keys WHERE id = $1", key_id)
        if not key_exists:
            raise RuntimeError(f"No api_keys row with id={key_id}")
        await conn.execute(
            """
            INSERT INTO agent_rate_quotas (key_id, agent_id, quota_pct, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (key_id, agent_id) DO UPDATE
              SET quota_pct = EXCLUDED.quota_pct, updated_at = NOW()
            """,
            key_id,
            agent_id,
            quota_pct,
        )


async def verify_api_key(api_key: str) -> Optional[str]:
    """Returns org_id if the key is valid, None otherwise. In dev mode, any dt_dev_* key
    (or no key at all) resolves to the 'default' org — a real row created by the
    multi-tenancy migration, not a sentinel."""
    if settings.is_dev and (not api_key or api_key.startswith("dt_dev_")):
        return "default"

    if not _pool:
        return None

    from dunetrace_schemas.keys import hash_api_key

    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                # Matched on the hash: the database never holds the secret.
                "SELECT org_id FROM api_keys WHERE key_hash = $1 AND active = TRUE",
                hash_api_key(api_key),
            )
        return row["org_id"] if row else None
    except Exception as exc:
        # Log the exception type only, never the exception object itself —
        # some DB drivers embed bound parameter values (here, the raw
        # api_key) in error message strings, which would otherwise put a
        # live credential in plaintext logs.
        logger.error("verify_api_key failed: %s", type(exc).__name__)
        return None


async def fetch_otel_ingestion_enabled(org_id: str) -> bool:
    """Whether OTel ingestion is enabled for org_id.

    Fail-open: returns True on any error, a missing row, or no configured pool.
    A transient DB problem must never silently drop a customer's OTLP traffic,
    so only a row that explicitly reads FALSE disables ingestion.
    """
    if not _pool:
        return True
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT otel_ingestion_enabled FROM organizations WHERE id = $1", org_id
            )
        return bool(row["otel_ingestion_enabled"]) if row is not None else True
    except Exception as exc:
        logger.debug("otel_ingestion_enabled lookup failed: %s", type(exc).__name__)
        return True


async def set_otel_ingestion_enabled(org_id: str, enabled: bool) -> None:
    """Enable or disable OTel ingestion for org_id. Raises RuntimeError if the
    org does not exist (so the admin endpoint can 404 rather than silently no-op)."""
    if not _pool:
        raise RuntimeError("no database configured")
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE organizations SET otel_ingestion_enabled = $2 WHERE id = $1",
            org_id,
            enabled,
        )
    if result.rsplit(" ", 1)[-1] == "0":
        raise RuntimeError(f"unknown org_id: {org_id}")


async def flush_otel_stats(buckets: dict) -> int:
    """Additively upsert drained OTLP-receiver counters into
    otel_receiver_stats. `buckets` is {(org_id, hour_epoch): counter_dict} from
    OtelStats.drain(). Returns the number of (org, hour) rows written; a no-op
    when nothing drained or no pool is configured. Best-effort telemetry — the
    caller (the maintenance loop) tolerates failure and retries next tick."""
    if not buckets or not _pool:
        return 0
    written = 0
    async with _pool.acquire() as conn:
        for (org_id, hour_epoch), b in buckets.items():
            await conn.execute(
                """
                INSERT INTO otel_receiver_stats (
                    org_id, hour_bucket, batches_received, spans_received,
                    events_translated, spans_rejected, auth_failures,
                    rate_limit_hits, rejections
                ) VALUES ($1, to_timestamp($2), $3, $4, $5, $6, $7, $8, $9::jsonb)
                ON CONFLICT (org_id, hour_bucket) DO UPDATE SET
                    batches_received  = otel_receiver_stats.batches_received  + EXCLUDED.batches_received,
                    spans_received    = otel_receiver_stats.spans_received    + EXCLUDED.spans_received,
                    events_translated = otel_receiver_stats.events_translated + EXCLUDED.events_translated,
                    spans_rejected    = otel_receiver_stats.spans_rejected    + EXCLUDED.spans_rejected,
                    auth_failures     = otel_receiver_stats.auth_failures     + EXCLUDED.auth_failures,
                    rate_limit_hits   = otel_receiver_stats.rate_limit_hits   + EXCLUDED.rate_limit_hits,
                    -- Sum the per-reason counts across the stored and incoming
                    -- rejection maps rather than overwriting keys.
                    rejections        = (
                        SELECT COALESCE(jsonb_object_agg(k, v), '{}'::jsonb)
                        FROM (
                            SELECT k, SUM(val::bigint) AS v
                            FROM (
                                SELECT k, val FROM jsonb_each_text(otel_receiver_stats.rejections) AS a(k, val)
                                UNION ALL
                                SELECT k, val FROM jsonb_each_text(EXCLUDED.rejections) AS b(k, val)
                            ) merged GROUP BY k
                        ) summed
                    ),
                    updated_at = NOW()
                """,
                org_id,
                hour_epoch,
                b["batches_received"],
                b["spans_received"],
                b["events_translated"],
                b["spans_rejected"],
                b["auth_failures"],
                b["rate_limit_hits"],
                json.dumps(b["rejections"]),
            )
            written += 1
    return written
