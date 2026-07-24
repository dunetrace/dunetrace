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
from typing import TYPE_CHECKING

try:
    import asyncpg

    _ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG = False

from integrations_svc.config import settings

if TYPE_CHECKING:
    from integrations_svc.providers.elevenlabs import ElevenLabsGeneration

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


# ── ElevenLabs schema (Phase 4.3) ──────────────────────────────────────────────
#
# elevenlabs_integrations is duplicated defensively (api_svc owns the config
# side; whichever starts first wins). elevenlabs_generations is owned here: the
# poller writes it, Phase 4.4 correlation updates correlated_to_event_id in
# place, and the customer API reads it. correlated_* columns stay NULL in this
# phase — correlation is Phase 4.4.
_ELEVENLABS_SCHEMA = """
CREATE TABLE IF NOT EXISTS elevenlabs_integrations (
    id                      BIGSERIAL        PRIMARY KEY,
    org_id                  TEXT             NOT NULL UNIQUE,
    encrypted_credentials   TEXT             NOT NULL,
    poll_interval_secs      INTEGER          NOT NULL DEFAULT 300,
    enabled                 BOOLEAN          NOT NULL DEFAULT TRUE,
    last_polled_at          TIMESTAMPTZ,
    last_success_at         TIMESTAMPTZ,
    last_seen_generation_at DOUBLE PRECISION,
    consecutive_failures    INTEGER          NOT NULL DEFAULT 0,
    first_failure_at        TIMESTAMPTZ,
    last_alerted_at         TIMESTAMPTZ,
    created_at              TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ       NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_elevenlabs_integrations_enabled
    ON elevenlabs_integrations(enabled) WHERE enabled = TRUE;

CREATE TABLE IF NOT EXISTS elevenlabs_generations (
    id                     BIGSERIAL        PRIMARY KEY,
    org_id                 TEXT             NOT NULL,
    generation_id          TEXT             NOT NULL,
    voice_id               TEXT,
    voice_name             TEXT,
    model                  TEXT,
    character_count        INTEGER          NOT NULL DEFAULT 0,
    -- ElevenLabs bills characters as credits on standard TTS plans, so credits
    -- == character_count. Stored as its own column so a future plan whose
    -- credit model differs can be represented without a migration.
    cost_credits           INTEGER,
    text                   TEXT,
    source                 TEXT,
    generated_at           DOUBLE PRECISION NOT NULL,
    -- Correlation (Phase 4.4). correlated_* set on a match; unmatched_reason set
    -- when the generation is old enough that a match should have arrived and
    -- none did (recorded as drift, no longer retried). A row with all four NULL
    -- is still pending and will be retried next cycle.
    correlated_to_event_id BIGINT,
    correlation_method     TEXT,
    correlation_confidence REAL,
    correlated_at          TIMESTAMPTZ,
    unmatched_reason       TEXT,
    -- Denormalized from the matched event at correlation time (Phase 5.1) so the
    -- run/call/filter read queries never need an events.id join (events is a big
    -- partitioned table with no standalone id index). NULL until correlated.
    run_id                 TEXT,
    agent_id               TEXT,
    fetched_at             TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    UNIQUE (org_id, generation_id)
);
-- Defensive for installs where an earlier build created the table without these.
ALTER TABLE elevenlabs_generations ADD COLUMN IF NOT EXISTS unmatched_reason TEXT;
ALTER TABLE elevenlabs_generations ADD COLUMN IF NOT EXISTS run_id TEXT;
ALTER TABLE elevenlabs_generations ADD COLUMN IF NOT EXISTS agent_id TEXT;
CREATE INDEX IF NOT EXISTS idx_elevenlabs_gen_org_time
    ON elevenlabs_generations(org_id, generated_at DESC);
-- Phase 4.4 correlation pass scans only generations still awaiting a decision.
CREATE INDEX IF NOT EXISTS idx_elevenlabs_gen_uncorrelated
    ON elevenlabs_generations(org_id) WHERE correlated_to_event_id IS NULL;
-- Phase 5.1 read paths: generations for a run, and the filter/high-cost listing.
CREATE INDEX IF NOT EXISTS idx_elevenlabs_gen_run
    ON elevenlabs_generations(org_id, run_id);

-- Correlation candidate lookup: tts.generated events for an org in a time
-- window. Partial index keeps it scoped to just the voice events, small even on
-- a large events table.
CREATE INDEX IF NOT EXISTS idx_events_tts_correlation
    ON events(org_id, timestamp) WHERE event_type = 'tts.generated';

-- Defensive, same reason as _INTEGRATIONS_SCHEMA: write_integration_down_signal
-- writes failure_signals.source, owned by semantic_svc's migration which may
-- not have run. This worker can run standalone, so ensure the column here too.
ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'structural';
"""


async def ensure_elevenlabs_schema() -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_ELEVENLABS_SCHEMA)
    logger.info("ElevenLabs schema ready")


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


# ── ElevenLabs reads/writes (Phase 4.3) ────────────────────────────────────────


async def fetch_due_elevenlabs_integrations() -> list[dict]:
    """Enabled ElevenLabs integrations due for a poll — never polled, or
    poll_interval_secs elapsed since last_polled_at. Each org's own interval is
    respected independently of this worker's WAKE_INTERVAL, same as
    fetch_due_integrations for the evaluation providers."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, org_id, encrypted_credentials, poll_interval_secs,
                   last_seen_generation_at, first_failure_at, last_alerted_at, created_at
            FROM elevenlabs_integrations
            WHERE enabled = TRUE
              AND (
                  last_polled_at IS NULL
                  OR last_polled_at <= NOW() - (poll_interval_secs || ' seconds')::INTERVAL
              )
            """
        )
    return [dict(r) for r in rows]


async def store_generation(org_id: str, gen: "ElevenLabsGeneration") -> bool:
    """Insert one generation, deduped on (org_id, generation_id). Returns True
    if it was newly stored, False if it was already present (an overlap re-fetch
    or a retry). correlated_* are left NULL for the Phase 4.4 correlation pass.
    cost_credits is set from character_count: ElevenLabs bills 1 credit/char on
    standard TTS plans."""
    async with _pool.acquire() as conn:
        result = await conn.execute(
            """
            INSERT INTO elevenlabs_generations
                (org_id, generation_id, voice_id, voice_name, model,
                 character_count, cost_credits, text, source, generated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $6, $7, $8, $9)
            ON CONFLICT (org_id, generation_id) DO NOTHING
            """,
            org_id,
            gen.generation_id,
            gen.voice_id,
            gen.voice_name,
            gen.model,
            gen.character_count,
            gen.text,
            gen.source,
            gen.generated_at,
        )
    # asyncpg returns "INSERT 0 1" on insert, "INSERT 0 0" when the conflict
    # skipped it.
    return result != "INSERT 0 0"


async def record_elevenlabs_poll_success(
    integration_id: int, last_seen_generation_at: float | None
) -> None:
    """Mark a successful poll and advance the generation high-water mark.
    last_seen_generation_at is None when the poll returned no generations, in
    which case COALESCE leaves the existing mark untouched. The worker only ever
    passes a value >= the current mark, so it never regresses."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE elevenlabs_integrations
            SET last_polled_at          = NOW(),
                last_success_at         = NOW(),
                last_seen_generation_at = COALESCE($2, last_seen_generation_at),
                consecutive_failures    = 0,
                first_failure_at        = NULL,
                updated_at              = NOW()
            WHERE id = $1
            """,
            integration_id,
            last_seen_generation_at,
        )


async def record_elevenlabs_poll_failure(integration_id: int) -> dict:
    """Increment the failure streak and return the updated state so the caller
    can decide whether the >30min operator-alert threshold has been crossed.
    Mirrors record_poll_failure for the evaluation providers."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE elevenlabs_integrations
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


async def record_elevenlabs_alert_sent(integration_id: int) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE elevenlabs_integrations SET last_alerted_at = NOW() WHERE id = $1",
            integration_id,
        )


# ── Correlation reads/writes (Phase 4.4) ───────────────────────────────────────


async def fetch_uncorrelated_generations(limit: int) -> list[dict]:
    """Generations still awaiting a correlation decision (neither matched nor
    given up), oldest first so the most-likely-to-decide ones are handled before
    newer arrivals. The LIMIT bounds each pass so correlation can never starve
    polling, even with a large backlog."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, org_id, generation_id, character_count, text, voice_id,
                   generated_at, fetched_at
            FROM elevenlabs_generations
            WHERE correlated_to_event_id IS NULL AND unmatched_reason IS NULL
            ORDER BY generated_at
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def fetch_candidate_tts_events(org_id: str, start: float, end: float) -> list[dict]:
    """tts.generated events for an org whose emit timestamp falls in
    [start, end]. Pulls the correlation-relevant payload fields directly so the
    matching logic stays pure (no JSON handling in the algorithm)."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id,
                   run_id,
                   agent_id,
                   timestamp,
                   payload->>'text'                   AS text,
                   payload->>'voice_id'               AS voice_id,
                   payload->>'provider_generation_id' AS provider_generation_id
            FROM events
            WHERE org_id = $1
              AND event_type = 'tts.generated'
              AND timestamp BETWEEN $2 AND $3
            """,
            org_id,
            start,
            end,
        )
    return [dict(r) for r in rows]


async def mark_generation_correlated(
    generation_row_id: int,
    event_id: int,
    run_id: str | None,
    agent_id: str | None,
    method: str,
    confidence: float,
) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE elevenlabs_generations
            SET correlated_to_event_id = $2,
                run_id                 = $3,
                agent_id               = $4,
                correlation_method     = $5,
                correlation_confidence = $6,
                correlated_at          = NOW(),
                unmatched_reason       = NULL
            WHERE id = $1
            """,
            generation_row_id,
            event_id,
            run_id,
            agent_id,
            method,
            confidence,
        )


async def mark_generation_unmatched(generation_row_id: int, reason: str) -> None:
    """Record an honest non-match: the generation is old enough that a matching
    event should have arrived and none did. Kept as drift for debugging, not
    retried again (the uncorrelated query excludes unmatched_reason IS NOT NULL)."""
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE elevenlabs_generations SET unmatched_reason = $2 WHERE id = $1",
            generation_row_id,
            reason,
        )


async def get_correlation_metrics(org_id: str) -> dict:
    """Correlation health for an org: counts, match rate over decided
    generations, average correlation latency, and the breakdown of unmatched
    reasons. match_rate is over correlated + unmatched (pending is not yet
    decided, so counting it would understate the true rate)."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                count(*)                                                            AS total,
                count(*) FILTER (WHERE correlated_to_event_id IS NOT NULL)          AS correlated,
                count(*) FILTER (WHERE correlated_to_event_id IS NULL
                                   AND unmatched_reason IS NULL)                    AS pending,
                count(*) FILTER (WHERE unmatched_reason IS NOT NULL)                AS unmatched,
                avg(EXTRACT(EPOCH FROM (correlated_at - fetched_at)))
                    FILTER (WHERE correlated_to_event_id IS NOT NULL)               AS avg_latency_secs
            FROM elevenlabs_generations
            WHERE org_id = $1
            """,
            org_id,
        )
        reason_rows = await conn.fetch(
            """
            SELECT unmatched_reason, count(*) AS n
            FROM elevenlabs_generations
            WHERE org_id = $1 AND unmatched_reason IS NOT NULL
            GROUP BY unmatched_reason
            """,
            org_id,
        )
    correlated = row["correlated"] or 0
    unmatched = row["unmatched"] or 0
    decided = correlated + unmatched
    return {
        "total": row["total"] or 0,
        "correlated": correlated,
        "pending": row["pending"] or 0,
        "unmatched": unmatched,
        "match_rate": (correlated / decided) if decided else None,
        "avg_latency_secs": float(row["avg_latency_secs"]) if row["avg_latency_secs"] else None,
        "unmatched_reasons": {r["unmatched_reason"]: r["n"] for r in reason_rows},
    }
