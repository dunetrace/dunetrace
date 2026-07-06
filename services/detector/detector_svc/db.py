"""
Database layer for the detector worker. Reads from events, writes to failure_signals,
and tracks processed_runs to avoid running the same run twice.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

try:
    import asyncpg

    _ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG = False

from detector_svc.config import settings

logger = logging.getLogger("dunetrace.detector.db")

_pool = None  # asyncpg.Pool when running for real


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

# NOTE: The ingest service owns the core schema (events, failure_signals, api_keys).
# The detector service adds only what it needs on top.

_DETECTOR_SCHEMA = """
-- Tracks which runs the detector has already processed.
-- Prevents double-processing if the worker restarts.
CREATE TABLE IF NOT EXISTS processed_runs (
    run_id        TEXT PRIMARY KEY,
    agent_id      TEXT        NOT NULL,
    agent_version TEXT        NOT NULL,
    processed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signal_count  INTEGER     NOT NULL DEFAULT 0,
    trigger       TEXT        NOT NULL   -- "completed" | "errored" | "stalled"
);

-- Add shadow column to failure_signals if it doesn't exist.
-- Shadow signals are stored but never sent to customers.
-- A detector graduates out of shadow mode manually once precision > 80%.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'failure_signals' AND column_name = 'shadow'
    ) THEN
        ALTER TABLE failure_signals ADD COLUMN shadow BOOLEAN NOT NULL DEFAULT TRUE;
    END IF;
END $$;

-- co_signal_count: how many signals fired on the same run (>=2 means boosted).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'failure_signals' AND column_name = 'co_signal_count'
    ) THEN
        ALTER TABLE failure_signals ADD COLUMN co_signal_count INTEGER NOT NULL DEFAULT 0;
    END IF;
END $$;

-- Persistent issue tracking: one row per (org_id, agent_id, failure_type) triple.
-- status: open | resolved | reopened
-- clean_runs_since: consecutive runs with no signal of this type (reset to 0 on each hit).
-- Resolved when clean_runs_since reaches CLEAN_RUNS_THRESHOLD (default 5).
CREATE TABLE IF NOT EXISTS issues (
    id               BIGSERIAL    PRIMARY KEY,
    agent_id         TEXT         NOT NULL,
    failure_type     TEXT         NOT NULL,
    status           TEXT         NOT NULL DEFAULT 'open',
    first_seen       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_seen        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    resolved_at      TIMESTAMPTZ,
    affected_runs    INTEGER      NOT NULL DEFAULT 1,
    clean_runs_since INTEGER      NOT NULL DEFAULT 0,
    UNIQUE (agent_id, failure_type)
);

-- Custom detector tables (also created by the API service; duplicated here so the
-- detector worker can start before the API without failing on missing tables).
CREATE TABLE IF NOT EXISTS custom_detectors (
    id                BIGSERIAL    PRIMARY KEY,
    agent_id          TEXT         NOT NULL DEFAULT '*',
    name              TEXT         NOT NULL,
    description       TEXT         NOT NULL,
    config_json       JSONB        NOT NULL,
    status            TEXT         NOT NULL DEFAULT 'shadow',
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    total_runs        INTEGER      NOT NULL DEFAULT 0,
    shadow_fire_count INTEGER      NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_custom_detectors_agent ON custom_detectors(agent_id, status);

CREATE TABLE IF NOT EXISTS custom_detector_results (
    id           BIGSERIAL    PRIMARY KEY,
    detector_id  BIGINT       NOT NULL REFERENCES custom_detectors(id) ON DELETE CASCADE,
    run_id       TEXT         NOT NULL,
    agent_id     TEXT         NOT NULL,
    fired        BOOLEAN      NOT NULL,
    evaluated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cdr_detector ON custom_detector_results(detector_id, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS idx_cdr_run      ON custom_detector_results(run_id);

-- Performance indexes for baseline queries (processed_runs) and alert worker queries (failure_signals).
CREATE INDEX IF NOT EXISTS idx_processed_runs_agent_version
    ON processed_runs(agent_id, agent_version, processed_at DESC);

CREATE INDEX IF NOT EXISTS idx_failure_signals_agent_shadow_alerted
    ON failure_signals(agent_id, shadow, alerted, detected_at DESC);
"""

# ── Multi-tenancy unification (v0.5.0) ──────────────────────────────────────────
#
# processed_runs/issues/custom_detectors/custom_detector_results all gain org_id.
# Unlike ingest_svc, this service doesn't own api_keys — org_id is backfilled by
# joining through events.org_id (agent_id -> most-recent org_id seen for that
# agent), which ingest_svc's own migration guarantees is populated and NOT NULL.
#
# Startup-order hazard: docker-compose starts detector only after ingest has
# *started* (service_started, not "migration complete"), so it's possible for
# this service's first startup to race ingest's. If events.org_id doesn't exist
# yet, the backfill/NOT NULL step is skipped this run and retried on the next
# restart — every ensure_*_schema() call is idempotent, so this converges
# without operator action once ingest's migration has actually run.
_MULTI_TENANCY_DDL = """
ALTER TABLE processed_runs        ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE issues                ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE custom_detectors      ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE custom_detector_results ADD COLUMN IF NOT EXISTS org_id TEXT;
"""

_ORG_BACKFILL_TABLES_BY_AGENT = ("issues", "custom_detectors", "custom_detector_results")


async def _backfill_org_id(conn) -> None:
    """See module docstring above. No-ops (leaves org_id nullable) until
    events.org_id exists and is populated."""
    events_ready = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'events' AND column_name = 'org_id'
        )
        """
    )
    if not events_ready:
        logger.warning(
            "Multi-tenancy backfill deferred: events.org_id doesn't exist yet "
            "(ingest_svc migration hasn't run). Will retry on next restart."
        )
        return

    # processed_runs has run_id, which maps 1:1 to events.run_id — exact join,
    # no ambiguity possible (every event in a run shares one org_id by construction).
    await conn.execute(
        """
        UPDATE processed_runs pr
        SET org_id = e.org_id
        FROM (SELECT DISTINCT run_id, org_id FROM events) e
        WHERE pr.run_id = e.run_id AND pr.org_id IS NULL
        """
    )
    await conn.execute("UPDATE processed_runs SET org_id = 'default' WHERE org_id IS NULL")
    await conn.execute("ALTER TABLE processed_runs ALTER COLUMN org_id SET NOT NULL")

    # issues/custom_detectors/custom_detector_results have no run_id — backfill by
    # agent_id's most-recently-seen org in events. Ambiguous agent_ids (same
    # agent_id used by >1 org — only possible pre-migration) fall back to 'default'.
    for table in _ORG_BACKFILL_TABLES_BY_AGENT:
        await conn.execute(
            f"""
            UPDATE {table} t
            SET org_id = e.org_id
            FROM (
                SELECT DISTINCT ON (agent_id) agent_id, org_id
                FROM events
                ORDER BY agent_id, received_at DESC
            ) e
            WHERE t.agent_id = e.agent_id AND t.org_id IS NULL
            """
        )
        await conn.execute(f"UPDATE {table} SET org_id = 'default' WHERE org_id IS NULL")
        await conn.execute(f"ALTER TABLE {table} ALTER COLUMN org_id SET NOT NULL")

    # issues' UNIQUE constraint must widen to include org_id, or two orgs with an
    # identically-named agent_id + failure_type would collide.
    await conn.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'issues_agent_id_failure_type_key'
            ) THEN
                ALTER TABLE issues DROP CONSTRAINT issues_agent_id_failure_type_key;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'issues_org_agent_failure_type_key'
            ) THEN
                ALTER TABLE issues
                    ADD CONSTRAINT issues_org_agent_failure_type_key
                    UNIQUE (org_id, agent_id, failure_type);
            END IF;
        END $$;
        """
    )

    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_processed_runs_org_agent_version "
        "ON processed_runs(org_id, agent_id, agent_version, processed_at DESC)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_issues_org_agent ON issues(org_id, agent_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_custom_detectors_org_agent "
        "ON custom_detectors(org_id, agent_id, status)"
    )


# Detectors that have graduated out of shadow mode.
# Add a detector name here ONLY after verifying precision > 80% on real data.
# LIVE_DETECTORS: set[str] = set()  # empty until we validate on real traffic
LIVE_DETECTORS: set[str] = {
    "PROMPT_INJECTION_SIGNAL",
    "TOOL_LOOP",
    "TOOL_THRASHING",
    "TOOL_AVOIDANCE",
    "GOAL_ABANDONMENT",
    "RAG_EMPTY_RETRIEVAL",
    "LLM_TRUNCATION_LOOP",
    "CONTEXT_BLOAT",
    "SLOW_STEP",
    "RETRY_STORM",
    "EMPTY_LLM_RESPONSE",
    "STEP_COUNT_INFLATION",
    "CASCADING_TOOL_FAILURE",
    "FIRST_STEP_FAILURE",
    "REASONING_STALL",
    "COST_SPIKE",
    "SESSION_LATENCY",
}


async def ensure_detector_schema() -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_DETECTOR_SCHEMA)
        await conn.execute(_MULTI_TENANCY_DDL)
        await _backfill_org_id(conn)
    logger.info("Detector schema ready")


# ── Reads ──────────────────────────────────────────────────────────────────────

# Minimum completed runs required before a P75 baseline is considered reliable.
# Below this threshold all baseline functions return None and detectors fall back
# to their static defaults.  P75 computed from fewer runs is too sensitive to
# single-run outliers to be useful — 20 gives a stable enough percentile estimate.
_MIN_BASELINE_RUNS = 20


async def fetch_step_count_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """
    P75 step count over the last `lookback` successfully completed runs, excluding the current run.
    Returns None if fewer than `min_runs` historical runs exist.

    Errored runs are excluded — early-exit errors pull P75 down and cause STEP_COUNT_INFLATION
    to fire on runs that took a perfectly normal number of steps for a complex task.

    org_id is required: agent_id/agent_version are not guaranteed unique across
    orgs, so without it a baseline could mix in another org's identically-named
    agent's history.
    """
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH recent AS (
                -- Only include runs that completed with a final_answer,
                -- not runs that errored out at step 1-2.
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.org_id        = $1
                  AND pr.agent_id      = $2
                  AND pr.agent_version = $3
                  AND pr.run_id       != $4
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $5
            ),
            step_counts AS (
                SELECT MAX(e.step_index) AS step_count
                FROM recent r
                JOIN events e ON e.run_id = r.run_id
                GROUP BY r.run_id
            )
            SELECT
                COUNT(*)                                                      AS sample_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY step_count)     AS p75
            FROM step_counts
            """,
            org_id,
            agent_id,
            agent_version,
            exclude_run_id,
            lookback,
        )

    if not row or (row["sample_size"] or 0) < min_runs:
        return None
    return float(row["p75"]) if row["p75"] is not None else None


async def fetch_latency_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    event_type: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """
    P75 wall-clock gap (ms) that follows each event of ``event_type`` across the last
    ``lookback`` successfully completed runs.  Mirrors the step_durations_ms computation
    in run_builder: gap = timestamp of the next event minus timestamp of this event.

    Used by SlowStepDetector to replace the hard-coded 15s/30s thresholds with a
    per-agent learned baseline.  Returns None when fewer than ``min_runs`` runs have
    at least one matching event. org_id required — see fetch_step_count_baseline.
    """
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH recent AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.org_id        = $1
                  AND pr.agent_id      = $2
                  AND pr.agent_version = $3
                  AND pr.run_id       != $4
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $5
            ),
            event_gaps AS (
                SELECT
                    e.run_id,
                    (LEAD(e.timestamp) OVER (
                        PARTITION BY e.run_id
                        ORDER BY e.step_index, e.timestamp
                    ) - e.timestamp) * 1000.0 AS gap_ms
                FROM events e
                WHERE e.run_id IN (SELECT run_id FROM recent)
                  AND e.event_type = $6
            )
            SELECT
                COUNT(DISTINCT run_id)                                              AS sample_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY gap_ms)               AS p75
            FROM event_gaps
            WHERE gap_ms >= 0
              AND gap_ms IS NOT NULL
            """,
            org_id,
            agent_id,
            agent_version,
            exclude_run_id,
            lookback,
            event_type,
        )

    if not row or (row["sample_size"] or 0) < min_runs:
        return None
    return float(row["p75"]) if row["p75"] is not None else None


async def fetch_token_growth_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """
    P75 context growth factor (last / first prompt_tokens) across the last ``lookback``
    successfully completed runs.  Excludes short runs (<3 LLM calls), tiny contexts
    (<10 first tokens), and runs whose final context never exceeded 2000 tokens —
    matching the ContextBloatDetector guard conditions.

    Returns None when fewer than ``min_runs`` qualifying runs exist.
    """
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH recent AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.org_id        = $1
                  AND pr.agent_id      = $2
                  AND pr.agent_version = $3
                  AND pr.run_id       != $4
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $5
            ),
            token_data AS (
                SELECT
                    e.run_id,
                    (e.payload->>'prompt_tokens')::integer AS prompt_tokens
                FROM events e
                WHERE e.run_id IN (SELECT run_id FROM recent)
                  AND e.event_type = 'llm.called'
                  AND e.payload->>'prompt_tokens' IS NOT NULL
                  AND (e.payload->>'prompt_tokens')::integer > 0
            ),
            run_growth AS (
                SELECT
                    run_id,
                    MIN(prompt_tokens) AS first_tokens,
                    MAX(prompt_tokens) AS last_tokens,
                    COUNT(*)           AS call_count
                FROM token_data
                GROUP BY run_id
                HAVING COUNT(*)           >= 3
                   AND MIN(prompt_tokens) >= 10
                   AND MAX(prompt_tokens) >= 2000
            )
            SELECT
                COUNT(*)                                                                    AS sample_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (
                    ORDER BY last_tokens::float / first_tokens::float
                )                                                                           AS p75
            FROM run_growth
            """,
            org_id,
            agent_id,
            agent_version,
            exclude_run_id,
            lookback,
        )

    if not row or (row["sample_size"] or 0) < min_runs:
        return None
    return float(row["p75"]) if row["p75"] is not None else None


async def fetch_llm_tool_ratio_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """
    P75 LLM:tool call ratio (llm_count / max(tool_count, 1)) across the last ``lookback``
    successfully completed runs.  Restricted to runs with at least 5 LLM calls, matching
    the ReasoningSpinDetector guard.

    Returns None when fewer than ``min_runs`` qualifying runs exist.
    """
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH recent AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.org_id        = $1
                  AND pr.agent_id      = $2
                  AND pr.agent_version = $3
                  AND pr.run_id       != $4
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $5
            ),
            run_counts AS (
                SELECT
                    e.run_id,
                    COUNT(*) FILTER (WHERE e.event_type = 'llm.called')  AS llm_count,
                    COUNT(*) FILTER (WHERE e.event_type = 'tool.called') AS tool_count
                FROM events e
                WHERE e.run_id IN (SELECT run_id FROM recent)
                GROUP BY e.run_id
                HAVING COUNT(*) FILTER (WHERE e.event_type = 'llm.called') >= 5
            )
            SELECT
                COUNT(*)                                                                        AS sample_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (
                    ORDER BY llm_count::float / GREATEST(tool_count, 1)
                )                                                                               AS p75
            FROM run_counts
            """,
            org_id,
            agent_id,
            agent_version,
            exclude_run_id,
            lookback,
        )

    if not row or (row["sample_size"] or 0) < min_runs:
        return None
    return float(row["p75"]) if row["p75"] is not None else None


async def fetch_total_tokens_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """
    P75 total token consumption (prompt + completion) per run across the last ``lookback``
    successfully completed runs.  Restricted to runs with at least one LLM call that
    reported token counts.

    Used by CostSpikeDetector.  Returns None when fewer than ``min_runs`` qualifying runs exist.
    """
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH recent AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.org_id        = $1
                  AND pr.agent_id      = $2
                  AND pr.agent_version = $3
                  AND pr.run_id       != $4
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $5
            ),
            run_tokens AS (
                -- prompt_tokens may be in llm.called (direct SDK) or llm.responded (LangChain);
                -- completion_tokens are always in llm.responded. Sum both event types to cover all paths.
                SELECT
                    e.run_id,
                    SUM(
                        COALESCE((e.payload->>'prompt_tokens')::integer, 0) +
                        COALESCE((e.payload->>'completion_tokens')::integer, 0)
                    ) AS total_tokens
                FROM events e
                WHERE e.run_id IN (SELECT run_id FROM recent)
                  AND e.event_type IN ('llm.called', 'llm.responded')
                GROUP BY e.run_id
                HAVING SUM(COALESCE((e.payload->>'prompt_tokens')::integer, 0)) > 0
            )
            SELECT
                COUNT(*)                                                          AS sample_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_tokens)       AS p75
            FROM run_tokens
            """,
            org_id,
            agent_id,
            agent_version,
            exclude_run_id,
            lookback,
        )

    if not row or (row["sample_size"] or 0) < min_runs:
        return None
    return float(row["p75"]) if row["p75"] is not None else None


async def fetch_duration_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """
    P75 wall-clock run duration in seconds (MAX(timestamp) − MIN(timestamp)) across
    the last ``lookback`` successfully completed runs.

    Used by SessionLatencyDetector.  Returns None when fewer than ``min_runs``
    qualifying runs exist or when all durations are zero.
    """
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH recent AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.org_id        = $1
                  AND pr.agent_id      = $2
                  AND pr.agent_version = $3
                  AND pr.run_id       != $4
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $5
            ),
            run_durations AS (
                SELECT
                    e.run_id,
                    (MAX(e.timestamp) - MIN(e.timestamp)) AS duration_s
                FROM events e
                WHERE e.run_id IN (SELECT run_id FROM recent)
                GROUP BY e.run_id
                HAVING (MAX(e.timestamp) - MIN(e.timestamp)) > 0
            )
            SELECT
                COUNT(*)                                                          AS sample_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY duration_s)         AS p75
            FROM run_durations
            """,
            org_id,
            agent_id,
            agent_version,
            exclude_run_id,
            lookback,
        )

    if not row or (row["sample_size"] or 0) < min_runs:
        return None
    return float(row["p75"]) if row["p75"] is not None else None


async def fetch_completed_runs(
    limit: int,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[dict]:
    """Runs with a terminal event (run.completed or run.errored) that haven't been processed yet.

    When shard_count > 1 each worker instance claims only the runs whose agent_id
    hashes to its bucket: abs(hashtext(agent_id)) % shard_count = shard_index.
    shard_count=1 (default) bypasses the filter entirely.

    Intentionally not org_id-scoped: this worker processes every org's runs in one
    poll loop (sharding is by agent_id hash bucket, not by org). org_id is returned
    per-row so the caller can propagate it into processed_runs/failure_signals/issues.
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (e.run_id)
                e.run_id,
                e.agent_id,
                e.agent_version,
                e.org_id,
                e.event_type AS trigger
            FROM events e
            WHERE e.event_type IN ('run.completed', 'run.errored')
              AND NOT EXISTS (
                  SELECT 1 FROM processed_runs p WHERE p.run_id = e.run_id
              )
              AND ($2::int = 1 OR abs(hashtext(e.agent_id)) % $2 = $3)
            ORDER BY e.run_id, e.received_at ASC
            LIMIT $1
            """,
            limit,
            shard_count,
            shard_index,
        )
    return [dict(r) for r in rows]


async def fetch_stalled_runs(
    stall_timeout_secs: float,
    limit: int,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[dict]:
    """Runs that started, never completed, and haven't received a new event in stall_timeout_secs — likely agents stuck mid-run.

    When shard_count > 1 only the runs whose agent_id hashes to shard_index are returned.
    Not org_id-scoped — see fetch_completed_runs.
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                e.run_id,
                e.agent_id,
                e.agent_version,
                e.org_id,
                'stalled' AS trigger
            FROM events e
            WHERE e.event_type = 'run.started'
              AND NOT EXISTS (
                  SELECT 1 FROM events t
                  WHERE t.run_id = e.run_id
                    AND t.event_type IN ('run.completed', 'run.errored')
              )
              AND NOT EXISTS (
                  SELECT 1 FROM processed_runs p WHERE p.run_id = e.run_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM events recent
                  WHERE recent.run_id = e.run_id
                    AND recent.received_at > NOW() - ($1 || ' seconds')::INTERVAL
              )
              AND ($3::int = 1 OR abs(hashtext(e.agent_id)) % $3 = $4)
            LIMIT $2
            """,
            str(stall_timeout_secs),
            limit,
            shard_count,
            shard_index,
        )
    return [dict(r) for r in rows]


async def fetch_run_events(run_id: str) -> list[dict]:
    """All events for a run, ordered by step_index then timestamp."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                event_type, run_id, agent_id, agent_version,
                step_index, timestamp, payload, parent_run_id
            FROM events
            WHERE run_id = $1
            ORDER BY step_index ASC, timestamp ASC
            """,
            run_id,
        )
    return [
        {
            **dict(r),
            "payload": (
                json.loads(r["payload"]) if isinstance(r["payload"], str) else dict(r["payload"])
            ),
        }
        for r in rows
    ]


# ── Writes ─────────────────────────────────────────────────────────────────────


async def write_signals(signals: list, shadow: bool, org_id: str) -> int:
    """Write FailureSignal objects to failure_signals. shadow=True stores them without alerting. Returns row count written.

    org_id is a parameter, not a field on FailureSignal — that dataclass is shared
    with the SDK (packages/sdk-py/dunetrace/models.py), which has no concept of
    org_id. All signals in one call come from processing a single run, which
    belongs to exactly one org.
    """
    if not signals:
        return 0

    rows = [
        (
            s.failure_type.value,
            s.severity.value,
            s.run_id,
            s.agent_id,
            s.agent_version,
            s.step_index,
            s.confidence,
            json.dumps(s.evidence),
            shadow,
            s.co_signal_count,
            org_id,
        )
        for s in signals
    ]

    async with _pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow, co_signal_count, org_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11)
            """,
            rows,
        )
    return len(rows)


async def mark_run_processed(
    run_id: str, agent_id: str, agent_version: str, trigger: str, signal_count: int, org_id: str
) -> None:
    """Record that this run has been processed. Prevents double-processing."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO processed_runs
                (run_id, agent_id, agent_version, trigger, signal_count, org_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (run_id) DO NOTHING
            """,
            run_id,
            agent_id,
            agent_version,
            trigger,
            signal_count,
            org_id,
        )


# ── Issue persistence ──────────────────────────────────────────────────────────

CLEAN_RUNS_THRESHOLD = 5  # consecutive clean runs before an issue is marked resolved


async def upsert_fired_issues(org_id: str, agent_id: str, fired_types: list[str]) -> None:
    """For each failure type that fired this run: open a new issue or reopen/update an existing one."""
    if not fired_types or not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO issues (org_id, agent_id, failure_type, status, affected_runs, clean_runs_since)
            VALUES ($1, $2, $3, 'open', 1, 0)
            ON CONFLICT (org_id, agent_id, failure_type) DO UPDATE
                SET last_seen        = NOW(),
                    affected_runs    = issues.affected_runs + 1,
                    clean_runs_since = 0,
                    status           = CASE
                        WHEN issues.status = 'resolved' THEN 'reopened'
                        ELSE issues.status
                    END,
                    resolved_at      = CASE
                        WHEN issues.status = 'resolved' THEN NULL
                        ELSE issues.resolved_at
                    END
            """,
            [(org_id, agent_id, ft) for ft in fired_types],
        )


async def advance_clean_runs(org_id: str, agent_id: str, fired_types: list[str]) -> None:
    """For open/reopened issues that did NOT fire this run: increment clean counter, resolve if threshold reached."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE issues
            SET clean_runs_since = clean_runs_since + 1,
                status      = CASE
                    WHEN clean_runs_since + 1 >= $4 THEN 'resolved'
                    ELSE status
                END,
                resolved_at = CASE
                    WHEN clean_runs_since + 1 >= $4 THEN NOW()
                    ELSE resolved_at
                END
            WHERE org_id      = $1
              AND agent_id    = $2
              AND status      IN ('open', 'reopened')
              AND ($3::text[] IS NULL OR failure_type != ALL($3::text[]))
            """,
            org_id,
            agent_id,
            fired_types or None,
            CLEAN_RUNS_THRESHOLD,
        )


async def fetch_custom_detectors(org_id: str, agent_id: str) -> list[dict]:
    """Load active and shadow custom detectors for the given org_id + agent_id (plus '*' wildcards).

    A '*' wildcard applies to all agents within org_id — never across orgs.
    """
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, config_json, status
            FROM custom_detectors
            WHERE org_id = $1
              AND (agent_id = $2 OR agent_id = '*')
              AND status IN ('shadow', 'active')
            ORDER BY id
            """,
            org_id,
            agent_id,
        )
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "config": json.loads(r["config_json"])
            if isinstance(r["config_json"], str)
            else dict(r["config_json"]),
            "shadow": r["status"] == "shadow",
        }
        for r in rows
    ]


async def record_custom_detector_results(
    results: list[dict],
    org_id: str,
) -> None:
    """Bulk-insert custom detector evaluation results for shadow analytics.

    Each entry: {detector_id, run_id, agent_id, fired}. org_id is a single param,
    not per-entry — all results in one call come from evaluating one run.
    """
    if not results or not _pool:
        return
    fired_ids = [r["detector_id"] for r in results if r["fired"]]
    all_ids = [r["detector_id"] for r in results]
    async with _pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO custom_detector_results (detector_id, run_id, agent_id, fired, org_id)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (r["detector_id"], r["run_id"], r["agent_id"], r["fired"], org_id)
                    for r in results
                ],
            )
            if all_ids:
                await conn.execute(
                    "UPDATE custom_detectors SET total_runs = total_runs + 1 WHERE id = ANY($1::bigint[])",
                    all_ids,
                )
            if fired_ids:
                # Only increment shadow_fire_count for detectors still in shadow mode;
                # active detectors that fire should not skew the shadow analytics.
                await conn.execute(
                    """
                    UPDATE custom_detectors
                    SET shadow_fire_count = shadow_fire_count + 1
                    WHERE id = ANY($1::bigint[]) AND status = 'shadow'
                    """,
                    fired_ids,
                )


async def write_custom_signal(
    failure_type: str,
    severity: str,
    run_id: str,
    agent_id: str,
    agent_version: str,
    step_index: int,
    confidence: float,
    evidence: dict,
    shadow: bool,
    org_id: str,
) -> None:
    """Write a custom detector signal directly as TEXT failure_type (no enum required)."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow, co_signal_count, org_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, 0, $10)
            """,
            failure_type,
            severity,
            run_id,
            agent_id,
            agent_version,
            step_index,
            confidence,
            json.dumps(evidence),
            shadow,
            org_id,
        )
