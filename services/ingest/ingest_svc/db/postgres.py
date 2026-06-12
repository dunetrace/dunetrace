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
    PRIMARY KEY (id, received_at)
) PARTITION BY RANGE (received_at);

CREATE INDEX IF NOT EXISTS idx_events_run_id  ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_agent   ON events(agent_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type);

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

CREATE TABLE IF NOT EXISTS deploy_events (
    id           BIGSERIAL PRIMARY KEY,
    agent_id     TEXT        NOT NULL,
    version      TEXT        NOT NULL,
    deployed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta         JSONB       NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_deploys_agent ON deploy_events(agent_id, deployed_at DESC);
"""


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
            return 0

        rows = await conn.fetch("""
            SELECT c.relname
            FROM pg_class c
            JOIN pg_inherits i ON i.inhrelid = c.oid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'events'
              AND c.relname ~ '^events_[0-9]{6}$'
            """)

        dropped = 0
        for row in rows:
            name = row["relname"]
            try:
                year, month = int(name[7:11]), int(name[11:13])
                end_year = year + (1 if month == 12 else 0)
                end_month = month % 12 + 1
                partition_end = date(end_year, end_month, 1)
                if partition_end <= cutoff:
                    await conn.execute(f"DROP TABLE IF EXISTS {name}")
                    logger.info(
                        "Pruned event partition %s (data before %s)",
                        name,
                        partition_end,
                    )
                    dropped += 1
            except (ValueError, IndexError):
                pass

        return dropped


async def ensure_schema() -> None:
    """Idempotent — safe to call on every startup."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_SCHEMA)
        await _ensure_event_partitions(conn)
    logger.info("Schema ready")


# ── Queries ────────────────────────────────────────────────────────────────────


async def insert_events(events: list, batch_id: str) -> int:
    """Bulk insert IngestEvent objects. Called from a BackgroundTask after the response is already sent."""
    if not _pool:
        logger.error("insert_events: pool not available, dropping %d events", len(events))
        return 0

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
        )
        for e in events
    ]

    try:
        async with _pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO events
                    (batch_id, event_type, run_id, agent_id, agent_version,
                     step_index, timestamp, payload, parent_run_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                """,
                rows,
            )
        return len(rows)
    except Exception as exc:
        logger.error("insert_events failed: %s", exc)
        return 0


async def insert_deploy_event(agent_id: str, version: str, meta: dict) -> int:
    """Insert a deploy marker. Returns the new row id, or 0 on failure."""
    if not _pool:
        return 0
    import json as _json

    try:
        async with _pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO deploy_events (agent_id, version, meta)
                VALUES ($1, $2, $3::jsonb)
                RETURNING id
                """,
                agent_id,
                version,
                _json.dumps(meta),
            )
        return int(row_id)
    except Exception as exc:
        logger.error("insert_deploy_event failed: %s", exc)
        return 0


async def fetch_policies(agent_id: str) -> list:
    """
    Return enabled policies matching agent_id or '*'.
    Called by the ingest service's policy fetch endpoint (SDK-facing).
    """
    if not _pool:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, agent_id, name, condition, action, enabled, priority, signature
                FROM policies
                WHERE enabled = TRUE
                  AND (agent_id = $1 OR agent_id = '*')
                ORDER BY priority ASC, id ASC
                """,
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
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("fetch_policies failed: %s", exc)
        return []


async def create_api_key(
    agent_id: str, customer_id: str, company_name: str | None = None, rate_limit_rpm: int = 600
) -> str:
    """Generate a new API key, upsert the company, store the key, and return it."""
    import secrets
    key = "dt_" + secrets.token_urlsafe(32)
    if not _pool:
        raise RuntimeError("DB pool not ready")
    name = company_name or customer_id
    async with _pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO companies (id, name) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                customer_id,
                name,
            )
            await conn.execute(
                "INSERT INTO api_keys (key, agent_id, customer_id, company_id, rate_limit_rpm) VALUES ($1, $2, $3, $3, $4)",
                key,
                agent_id,
                customer_id,
                rate_limit_rpm,
            )
    return key


async def verify_api_key(api_key: str) -> Optional[str]:
    """Returns agent_id if the key is valid, None otherwise. In dev mode, any dt_dev_* key is accepted."""
    if settings.is_dev and (not api_key or api_key.startswith("dt_dev_")):
        return "dev"

    if not _pool:
        return None

    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT agent_id FROM api_keys WHERE key = $1 AND active = TRUE",
                api_key,
            )
        return row["agent_id"] if row else None
    except Exception as exc:
        logger.error("verify_api_key failed: %s", exc)
        return None
