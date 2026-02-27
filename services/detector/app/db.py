"""
services/detector/app/db.py

All database I/O for the detector worker.

Reads:   events table (written by ingest service)
Writes:  failure_signals table
Tracks:  processed_runs table (prevents double-processing)
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

from app.config import settings

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
"""

# Detectors that have graduated out of shadow mode.
# Add a detector name here ONLY after verifying precision > 80% on real data.
#LIVE_DETECTORS: set[str] = set()  # empty until we validate on real traffic
LIVE_DETECTORS: set[str] = {
    "TOOL_LOOP",
    "TOOL_THRASHING",
    "TOOL_AVOIDANCE",
    "GOAL_ABANDONMENT",
    "RAG_EMPTY_RETRIEVAL",
    "LLM_TRUNCATION_LOOP",
    "CONTEXT_BLOAT",
    "SLOW_STEP",
}


async def ensure_detector_schema() -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_DETECTOR_SCHEMA)
    logger.info("Detector schema ready")


# ── Reads ──────────────────────────────────────────────────────────────────────

async def fetch_completed_runs(limit: int) -> list[dict]:
    """
    Find runs that have a terminal event (run.completed or run.errored)
    and have NOT been processed yet.
    Returns list of {run_id, agent_id, agent_version, trigger}.
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (e.run_id)
                e.run_id,
                e.agent_id,
                e.agent_version,
                e.event_type AS trigger
            FROM events e
            WHERE e.event_type IN ('run.completed', 'run.errored')
              AND NOT EXISTS (
                  SELECT 1 FROM processed_runs p WHERE p.run_id = e.run_id
              )
            ORDER BY e.run_id, e.received_at ASC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def fetch_stalled_runs(stall_timeout_secs: float, limit: int) -> list[dict]:
    """
    Find runs that haven't had a new event for stall_timeout_secs
    and haven't completed or been processed yet.
    These may be agents that are stuck mid-run.
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                e.run_id,
                e.agent_id,
                e.agent_version,
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
            LIMIT $2
            """,
            str(stall_timeout_secs), limit,
        )
    return [dict(r) for r in rows]


async def fetch_run_events(run_id: str) -> list[dict]:
    """
    Fetch all events for a run, ordered by step_index.
    Returns list of raw event dicts.
    """
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
            "payload": json.loads(r["payload"]) if isinstance(r["payload"], str) else dict(r["payload"]),
        }
        for r in rows
    ]


# ── Writes ─────────────────────────────────────────────────────────────────────

async def write_signals(signals: list, shadow: bool) -> int:
    """
    Write FailureSignal objects to failure_signals table.
    shadow=True means stored but not alerted.
    Returns number of rows written.
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
        )
        for s in signals
    ]

    async with _pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
            """,
            rows,
        )
    return len(rows)


async def mark_run_processed(run_id: str, agent_id: str, agent_version: str,
                              trigger: str, signal_count: int) -> None:
    """Record that this run has been processed. Prevents double-processing."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO processed_runs
                (run_id, agent_id, agent_version, trigger, signal_count)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (run_id) DO NOTHING
            """,
            run_id, agent_id, agent_version, trigger, signal_count,
        )