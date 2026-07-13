"""
Database layer for the integrations worker. Owns external_evaluation_integrations
and external_evaluation_processed outright (mirrors api_svc's own defensive
duplicate of the same DDL — whichever service starts first wins). Reads
events (owned by ingest_svc) for trace_id correlation and writes to the
shared failure_signals table, same conventions semantic_svc already
established for a service that isn't the schema's primary owner.
"""

from __future__ import annotations

import json
import logging

try:
    import asyncpg

    _ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG = False

from integrations_svc.config import settings

logger = logging.getLogger("dunetrace.integrations.db")

_pool = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────


async def init_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=1,
        max_size=5,
        command_timeout=15,
        # See ingest_svc/db/postgres.py::init_pool for why this is required —
        # DATABASE_URL is Supabase's transaction-mode PgBouncer pooler, which
        # is incompatible with asyncpg's default prepared-statement cache.
        statement_cache_size=0,
    )
    logger.info("DB pool ready")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── Schema additions ───────────────────────────────────────────────────────────

_INTEGRATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS external_evaluation_integrations (
    id                    BIGSERIAL    PRIMARY KEY,
    org_id                TEXT         NOT NULL,
    provider              TEXT         NOT NULL,
    endpoint_url          TEXT         NOT NULL,
    encrypted_credentials TEXT         NOT NULL,
    poll_interval_secs    INTEGER      NOT NULL DEFAULT 60,
    enabled               BOOLEAN      NOT NULL DEFAULT TRUE,
    last_polled_at        TIMESTAMPTZ,
    last_success_at       TIMESTAMPTZ,
    consecutive_failures  INTEGER      NOT NULL DEFAULT 0,
    first_failure_at      TIMESTAMPTZ,
    last_alerted_at       TIMESTAMPTZ,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (org_id, provider)
);
CREATE INDEX IF NOT EXISTS idx_ext_integrations_enabled
    ON external_evaluation_integrations(enabled) WHERE enabled = TRUE;

CREATE TABLE IF NOT EXISTS external_evaluation_processed (
    org_id       TEXT        NOT NULL,
    provider     TEXT        NOT NULL,
    external_id  TEXT        NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, provider, external_id)
);

-- Defensive: failure_signals.source is owned by semantic_svc's migration,
-- which may not have run (SEMANTIC_WORKER_ENABLED defaults to false).
ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'structural';
"""


async def ensure_integrations_schema() -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_INTEGRATIONS_SCHEMA)
    logger.info("Integrations schema ready")


# ── Reads ──────────────────────────────────────────────────────────────────────


async def fetch_due_integrations(provider: str) -> list[dict]:
    """Enabled integrations for this provider that are due for a poll — either
    never polled, or poll_interval_secs have elapsed since last_polled_at.
    Each org's own poll_interval_secs is respected independently even though
    this worker wakes on its own fixed WAKE_INTERVAL."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, org_id, endpoint_url, encrypted_credentials, poll_interval_secs,
                   last_success_at, first_failure_at, last_alerted_at, created_at
            FROM external_evaluation_integrations
            WHERE provider = $1
              AND enabled = TRUE
              AND (
                  last_polled_at IS NULL
                  OR last_polled_at <= NOW() - (poll_interval_secs || ' seconds')::INTERVAL
              )
            """,
            provider,
        )
    return [dict(r) for r in rows]


async def has_processed(org_id: str, provider: str, external_id: str) -> bool:
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM external_evaluation_processed "
            "WHERE org_id = $1 AND provider = $2 AND external_id = $3)",
            org_id,
            provider,
            external_id,
        )


async def fetch_run_by_trace_id(trace_id: str) -> dict | None:
    """Correlates an external evaluation back to the Dunetrace run it's
    about. Returns None if no event carries this trace_id — either the run
    predates trace_id support, wasn't instrumented with it, or genuinely
    isn't a Dunetrace run."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT run_id, agent_id, agent_version, org_id
            FROM events
            WHERE trace_id = $1
            LIMIT 1
            """,
            trace_id,
        )
    return dict(row) if row else None


# ── Writes ─────────────────────────────────────────────────────────────────────


async def mark_processed(org_id: str, provider: str, external_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO external_evaluation_processed (org_id, provider, external_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (org_id, provider, external_id) DO NOTHING
            """,
            org_id,
            provider,
            external_id,
        )


async def write_external_signal(
    org_id: str,
    agent_id: str,
    agent_version: str,
    run_id: str,
    provider: str,
    failure_type: str,
    confidence: float,
    evidence: dict,
) -> None:
    """Writes an externally-sourced evaluation as a semantic signal.
    shadow=FALSE — alert-eligible immediately (maintainer decision: the
    customer explicitly configured this integration and is trusted to have
    judged their own provider's evaluation quality; Dunetrace has no local
    false-positive management for third-party scores the way Phase 1.4 built
    for its own evaluators, but that's a documented gap, not a silent one)."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow, co_signal_count, org_id, source)
            VALUES ($1, 'MEDIUM', $2, $3, $4, 0, $5, $6::jsonb, FALSE, 0, $7, $8)
            """,
            failure_type,
            run_id,
            agent_id,
            agent_version,
            confidence,
            json.dumps(evidence),
            org_id,
            provider,
        )


async def record_poll_success(integration_id: int) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE external_evaluation_integrations
            SET last_polled_at       = NOW(),
                last_success_at      = NOW(),
                consecutive_failures = 0,
                first_failure_at     = NULL,
                updated_at           = NOW()
            WHERE id = $1
            """,
            integration_id,
        )


async def record_poll_failure(integration_id: int) -> dict:
    """Increments the failure streak and returns the updated
    (consecutive_failures, first_failure_at, last_alerted_at) so the caller
    can decide whether the >30min operator-alert threshold has been crossed."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE external_evaluation_integrations
            SET last_polled_at       = NOW(),
                consecutive_failures = consecutive_failures + 1,
                first_failure_at     = COALESCE(first_failure_at, NOW()),
                updated_at           = NOW()
            WHERE id = $1
            RETURNING consecutive_failures, first_failure_at, last_alerted_at
            """,
            integration_id,
        )
    return dict(row)


async def record_alert_sent(integration_id: int) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE external_evaluation_integrations SET last_alerted_at = NOW() WHERE id = $1",
            integration_id,
        )


async def write_integration_down_signal(org_id: str, provider: str, error_message: str) -> None:
    """Operational signal, not a customer-facing failure — same shadow=TRUE
    pattern as semantic_svc's SEMANTIC_QUOTA_EXCEEDED. No dedicated
    operator-alert channel exists anywhere in this codebase (confirmed
    during Phase 2.1 discovery); this is the only precedent to reuse.

    Not about any specific run — failure_signals' schema assumes one, so
    run_id/agent_id/agent_version get an obviously-synthetic placeholder
    rather than an empty string, in case any dashboard/ops code tries to
    build a "view this run" link from it (harmless 404 either way, but more
    debuggable than a blank identifier in a raw query).
    """
    placeholder = f"integration:{provider}:{org_id}"
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow, co_signal_count, org_id, source)
            VALUES ($1, 'LOW', $2, $2, $2, 0, 1.0, $3::jsonb, TRUE, 0, $4, $5)
            """,
            "EXTERNAL_INTEGRATION_DOWN",
            placeholder,
            json.dumps({"provider": provider, "error": error_message}),
            org_id,
            provider,
        )
