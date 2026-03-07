"""
services/ingest/ingest_svc/db/postgres.py

Async Postgres via asyncpg connection pool.
Pool is created once at startup and injected via FastAPI lifespan.
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
CREATE TABLE IF NOT EXISTS events (
    id             BIGSERIAL PRIMARY KEY,
    batch_id       TEXT             NOT NULL,
    event_type     TEXT             NOT NULL,
    run_id         TEXT             NOT NULL,
    agent_id       TEXT             NOT NULL,
    agent_version  TEXT             NOT NULL,
    step_index     INTEGER          NOT NULL,
    timestamp      DOUBLE PRECISION NOT NULL,
    payload        JSONB            NOT NULL DEFAULT '{}',
    parent_run_id  TEXT,
    received_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS api_keys (
    key         TEXT PRIMARY KEY,
    agent_id    TEXT        NOT NULL,
    customer_id TEXT        NOT NULL,
    active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def ensure_schema() -> None:
    """Idempotent — safe to call on every startup."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_SCHEMA)
    logger.info("Schema ready")


# ── Queries ────────────────────────────────────────────────────────────────────

async def insert_events(events: list, batch_id: str) -> int:
    """
    Bulk insert a list of IngestEvent objects.
    Called from a BackgroundTask — response is already sent.
    """
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


async def verify_api_key(api_key: str) -> Optional[str]:
    """
    Returns agent_id if valid, None if not.

    Dev mode:  any dt_dev_* key is accepted immediately.
    Prod mode: checks the api_keys table.
    """
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
