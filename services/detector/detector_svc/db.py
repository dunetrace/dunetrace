"""
Database layer for the detector worker. Reads from events, writes to failure_signals,
and tracks processed_runs to avoid running the same run twice.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
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
-- audit Finding 15: remember how many events a run had when we processed it, so
-- late-arriving events (event count grew) trigger a re-detection instead of being
-- silently ignored. Defaults to 0 for pre-existing rows (they'll reprocess once
-- if any new event arrives, which is harmless — writes are deduped by run_id+type).
ALTER TABLE processed_runs ADD COLUMN IF NOT EXISTS event_count INTEGER NOT NULL DEFAULT 0;

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

-- Conversation modeling (Phase 3.1). A "run" has no other dedicated table
-- anywhere in this codebase — it's just a run_id value shared across events/
-- failure_signals rows — so this is genuinely new infrastructure, not a
-- column bolted onto an existing table. conversation_id is nullable: every
-- processed run gets a runs row regardless of whether the SDK's
-- dt.run(conversation_id=...) was ever set, old runs and single-turn agents
-- simply never get a conversations row or FK.
CREATE TABLE IF NOT EXISTS conversations (
    id            BIGSERIAL   PRIMARY KEY,
    org_id        TEXT        NOT NULL,
    agent_id      TEXT        NOT NULL,
    user_id       TEXT,                     -- nullable, unpopulated for now — no SDK source exists yet
    external_id   TEXT        NOT NULL,     -- the SDK's own conversation_id string
    first_run_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_count     INTEGER     NOT NULL DEFAULT 0,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (org_id, agent_id, external_id)
);
CREATE INDEX IF NOT EXISTS idx_conversations_org_agent ON conversations(org_id, agent_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT        PRIMARY KEY,
    org_id          TEXT        NOT NULL,
    agent_id        TEXT        NOT NULL,
    agent_version   TEXT        NOT NULL,
    conversation_id BIGINT      REFERENCES conversations(id),
    started_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id) WHERE conversation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_runs_org_agent    ON runs(org_id, agent_id, started_at DESC);

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

# ── Pack activation (Phase 1.0) ─────────────────────────────────────────────────
#
# packs is a small, Dunetrace-owned lookup table (which detector-pack classes
# exist) — org_enabled_packs is the per-org activation side table, following
# the same org_id-TEXT-no-FK convention every other org-scoped table in this
# codebase uses (org_github_integrations, org_alert_integrations, etc.) rather
# than a FK to organizations(id), which is itself TEXT (e.g. the literal
# string 'default' in dev mode), not the uuid the original spec assumed.
# pack_name DOES FK to packs(name) — that's Dunetrace's own small, rarely-
# changing registry, not customer-controlled data, so the same "don't FK
# across service/tenant boundaries" reasoning doesn't apply to it.
_PACKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS packs (
    name           TEXT         PRIMARY KEY,
    description    TEXT         NOT NULL,
    detector_names TEXT[]       NOT NULL DEFAULT '{}',
    added_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS org_enabled_packs (
    org_id      TEXT         NOT NULL,
    pack_name   TEXT         NOT NULL REFERENCES packs(name),
    enabled_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    enabled_by  TEXT,
    PRIMARY KEY (org_id, pack_name)
);
CREATE INDEX IF NOT EXISTS idx_org_enabled_packs_org ON org_enabled_packs(org_id);
"""

# Per-run state metrics (Capability 3, Phase 3.3). One row per (run, state);
# api_svc reads these to build cross-run state analytics. run_started_at is when
# the run happened (from run.started), so trends bucket by run time, not compute
# time. org_id TEXT NOT NULL, no FK — same convention as every org-scoped table.
_RUN_STATE_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_state_metrics (
    run_id         TEXT         NOT NULL,
    org_id         TEXT         NOT NULL,
    agent_id       TEXT         NOT NULL,
    state          TEXT         NOT NULL,
    total_ms       BIGINT       NOT NULL,
    segment_count  INT          NOT NULL,
    run_started_at TIMESTAMPTZ,
    computed_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, state)
);
CREATE INDEX IF NOT EXISTS idx_rsm_agent
    ON run_state_metrics(org_id, agent_id, run_started_at);
"""

# Per-shard poll watermark. Without it the run-discovery queries have no lower
# time bound, so every 5s poll rescans every terminal event in the whole
# retention window and anti-joins it against processed_runs — work proportional
# to total history rather than to new runs. It also makes `events`' monthly
# partitioning useless for the hottest queries in the system: the partition key
# is received_at, and a predicate that never mentions received_at can't prune.
#
# NULL watermark means "never drained" and reproduces the original unbounded
# scan, so a fresh install (or one migrating in) picks up all pending history on
# its first poll and only then starts bounding. One row per shard because shards
# progress independently.
#
# Keyed by (shard_count, shard_index), NOT shard_index alone. Which agents a
# shard owns is a function of BOTH — `hashtext(agent_id) % shard_count` — so a
# resize reshuffles ownership. Keyed by index alone, a shard that inherited new
# agents on a resize kept the watermark it earned under the old topology and
# skipped every one of their runs older than that bound, permanently and
# silently. Under the composite key a resize simply presents as "never drained"
# for the new topology, which reproduces the safe unbounded first scan.
# Bounded retry for a run whose processing raised.
#
# The failure path used to write processed_runs(signal_count=0) — a permanent
# "clean run" verdict that nothing can revisit, because a completed run never
# gains events and so never re-enters the poll. process_run's guarded block is
# not pure computation (seven baseline queries, a detector lookup, a parent
# fetch, a lineage walk, all against a 5-connection pool), so one Postgres blip
# silently discarded detection for every run in flight while the dashboard
# showed them green.
#
# Leaving the run unmarked instead means it is retried on the next poll. This
# table bounds that: a run that fails MAX_PROCESSING_ATTEMPTS times is marked
# processed with the error recorded, so a deterministically-bad run cannot spin
# forever and is still visible as failed rather than clean.
_RUN_FAILURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_processing_failures (
    run_id           TEXT        PRIMARY KEY,
    attempts         INT         NOT NULL DEFAULT 1,
    last_error       TEXT,
    last_attempt_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE processed_runs ADD COLUMN IF NOT EXISTS processing_error TEXT;
"""

# Learned per-agent destination set for UNGROUNDED_DESTINATION's opt-in novelty
# mode. Detector-owned (nothing else reads it), so it lives here rather than in
# migrations — same rule run_state_metrics follows.
#
# It exists because every other baseline in this service is a scalar P75 that a
# bounded query can compute on demand; a destination baseline is a SET, and
# deriving it at detection time would mean re-parsing every tool call's args
# across 50 historical runs, per run. Materialising it turns that into one
# indexed lookup.
#
# Only GROUNDED destinations are ever written here (see the worker). Seeding it
# with ungrounded ones would teach the baseline that the first successful
# exfiltration is normal — the detector would disarm itself on exactly the
# traffic it exists to catch.
_DESTINATION_BASELINE_SCHEMA = """
-- Durable per-run scalars for the P75 baselines. See baseline_metrics.py for
-- why these are stored rather than re-derived: `events` is pruned at
-- EVENT_RETENTION_DAYS, so a low-traffic agent could never mature a baseline
-- from raw events, and the SQL that re-derived them had drifted from the
-- detectors it fed. One narrow row per run outlives the events it came from.
--
-- Deliberately NOT folded into processed_runs, which is pruned as soon as a
-- run's events are gone (that ordering is a correctness invariant, see
-- prune_processed_runs) and is the anti-join target in the hottest query in
-- the service — growing it is exactly what its own prune exists to prevent.
--
-- NULL in a metric column means "this run does not qualify for that baseline"
-- (it failed the corresponding detector's guard), and the aggregate skips it —
-- so one run can feed the token baseline while sitting out the growth one.
CREATE TABLE IF NOT EXISTS run_baseline_metrics (
    org_id          TEXT             NOT NULL,
    run_id          TEXT             NOT NULL,
    agent_id        TEXT             NOT NULL,
    agent_version   TEXT             NOT NULL,
    clean           BOOLEAN          NOT NULL,
    step_count      DOUBLE PRECISION,
    gap_p75_tool_ms DOUBLE PRECISION,
    gap_p75_llm_ms  DOUBLE PRECISION,
    token_growth    DOUBLE PRECISION,
    llm_tool_ratio  DOUBLE PRECISION,
    total_tokens    DOUBLE PRECISION,
    duration_s      DOUBLE PRECISION,
    recorded_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, run_id)
);
-- Partial on `clean`: every baseline read filters to clean runs, so the dirty
-- ones are dead weight in the index.
CREATE INDEX IF NOT EXISTS idx_rbm_agent_clean
    ON run_baseline_metrics(org_id, agent_id, agent_version, recorded_at DESC)
    WHERE clean;

CREATE TABLE IF NOT EXISTS agent_destination_baseline (
    org_id            TEXT        NOT NULL,
    agent_id          TEXT        NOT NULL,
    destination       TEXT        NOT NULL,
    destination_type  TEXT        NOT NULL,
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_count         INTEGER     NOT NULL DEFAULT 1,
    PRIMARY KEY (org_id, agent_id, destination)
);
CREATE INDEX IF NOT EXISTS idx_adb_agent_seen
    ON agent_destination_baseline(org_id, agent_id, last_seen_at DESC);
"""

_WATERMARK_SCHEMA = """
CREATE TABLE IF NOT EXISTS detector_watermarks (
    shard_index  INT         NOT NULL,
    shard_count  INT         NOT NULL DEFAULT 1,
    watermark    TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (shard_count, shard_index)
);
ALTER TABLE detector_watermarks ADD COLUMN IF NOT EXISTS shard_count INT NOT NULL DEFAULT 1;

-- Repair the PRIMARY KEY on a table created before the key became composite.
-- CREATE TABLE IF NOT EXISTS is a no-op on an existing table and the ALTER above
-- only adds the COLUMN, so such a deployment kept a single-column pkey on
-- shard_index while advance_watermark() upserts ON CONFLICT (shard_count,
-- shard_index) — which raises InvalidColumnReferenceError on every poll tick and
-- crash-loops the worker. Guarded on the current key being single-column, so it
-- no-ops on a fresh install. Safe to run: with the old key, shard_index is
-- already unique and shard_count defaults to 1, so the composite key cannot
-- collide on existing rows.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        WHERE c.relname = 'detector_watermarks'
          AND i.indisprimary
          AND i.indnatts = 1
    ) THEN
        ALTER TABLE detector_watermarks DROP CONSTRAINT detector_watermarks_pkey;
        ALTER TABLE detector_watermarks
            ADD CONSTRAINT detector_watermarks_pkey PRIMARY KEY (shard_count, shard_index);
    END IF;
END $$;
"""


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
    "PREMATURE_TERMINATION",
    "UNREAD_TOOL_ERROR",
    "TOOL_ARGUMENT_FABRICATION",
    "RETRIEVED_CONTENT_INJECTION",
    "HANDOFF_CONTEXT_LOSS",
    "RUNAWAY_ITERATION",
    "EXCESSIVE_RETRIEVAL",
    "SILENT_TRUNCATION",
    "AGENT_HANDOFF_FAILURE",
    "MODEL_FALLBACK_DRIFT",
    "MEMORY_POISONING",
    "DELEGATION_LOOP",
    "UNGROUNDED_DESTINATION",
}


async def ensure_detector_schema() -> None:
    """Bring the SHARED schema up to date first, then this service's own tables.

    Migrations own every definition more than one service touches, so this runs
    before the local DDL below — booting detector against an empty database used
    to crash on a table another service happened to create first.
    """
    from dunetrace_schemas.migrations import apply_migrations

    if _pool:
        async with _pool.acquire() as _c:
            await apply_migrations(_c)

    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_DETECTOR_SCHEMA)
        await conn.execute(_MULTI_TENANCY_DDL)
        await _backfill_org_id(conn)
        await conn.execute(_PACKS_SCHEMA)
        await _seed_packs(conn)
        await conn.execute(_RUN_STATE_METRICS_SCHEMA)
        await conn.execute(_DESTINATION_BASELINE_SCHEMA)
        await conn.execute(_WATERMARK_SCHEMA)
        await conn.execute(_RUN_FAILURE_SCHEMA)
    logger.info("Detector schema ready")


async def _seed_packs(conn) -> None:
    """Registers every currently-imported DetectorPack into the packs table.
    Idempotent (ON CONFLICT DO NOTHING on name) — safe to call on every
    startup, including ones where a pack's description or detector list
    changed: those get picked up via the explicit UPDATE below rather than
    silently staying stale, since a pack is Dunetrace-owned code, not
    customer data."""
    from dunetrace.packs import PACK_REGISTRY  # import here: avoids a hard

    # dependency from db.py's module-load time on the packs subpackage having
    # finished registering everything yet (packs register on import, and
    # detector_svc/detectors.py is what actually imports dunetrace.packs).
    for pack in PACK_REGISTRY.values():
        await conn.execute(
            """
            INSERT INTO packs (name, description, detector_names)
            VALUES ($1, $2, $3)
            ON CONFLICT (name) DO UPDATE
                SET description = EXCLUDED.description,
                    detector_names = EXCLUDED.detector_names
            """,
            pack.name,
            pack.description,
            [d.__name__ for d in pack.detectors],
        )


# ── Reads ──────────────────────────────────────────────────────────────────────

# Minimum completed runs required before a P75 baseline is considered reliable.
# Below this threshold all baseline functions return None and detectors fall back
# to their static defaults.  P75 computed from fewer runs is too sensitive to
# single-run outliers to be useful — 20 gives a stable enough percentile estimate.
_MIN_BASELINE_RUNS = 20

# The run-selection CTE every P75 baseline shares. Kept in one place because it
# was copy-pasted into six queries, and a predicate that has to be fixed six
# times gets fixed five times.
#
# Placeholders are positional and fixed by contract: $1 org_id, $2 agent_id,
# $3 agent_version, $4 run_id to exclude, $5 lookback. A query needing more
# binds them from $6 up (see fetch_latency_baseline).
#
# Two filters decide what counts as a reference for "normal":
#
#   completed  — errored runs are excluded because an early exit at step 1-2
#                drags P75 down and makes STEP_COUNT_INFLATION fire on runs
#                that took a perfectly ordinary number of steps.
#
# Both probes into `events` carry org_id, and so does every downstream join in
# the queries that use this CTE. run_id is caller-supplied (the SDK exposes
# run_id=, the OTLP path derives it from a caller-supplied trace id), so a join
# on run_id alone reads across tenants — the same class of bug that made
# (org_id, run_id) the composite key on `runs` and `processed_runs`. Unscoped,
# another tenant's run.completed could mark THIS tenant's errored run as
# completed, and their events could contribute steps and tokens to this
# tenant's baseline. An event with a NULL org_id is unattributable and belongs
# to no tenant, so excluding it is correct rather than a lost sample.
#
#   clean      — a run that fired a LIVE signal records the agent misbehaving,
#                so it cannot also define what normal looks like. Without this
#                a chronically looping agent raises its own step and token
#                baselines until the loop reads as typical and the detector
#                goes quiet — the failure mode learns its way out of detection.
#
# Shadow signals are deliberately NOT disqualifying. A shadow detector is
# unvalidated by definition (see LIVE_DETECTORS), and letting one shrink the
# baseline population would mean enabling a noisy detector for evaluation
# silently retuned six unrelated live ones.
_RECENT_CLEAN_RUNS_CTE = """
            WITH recent AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.org_id        = $1
                  AND pr.agent_id      = $2
                  AND pr.agent_version = $3
                  AND pr.run_id       != $4
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.org_id = pr.org_id
                        AND e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM failure_signals fs
                      WHERE fs.org_id = pr.org_id
                        AND fs.run_id = pr.run_id
                        AND fs.shadow = FALSE
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $5
            )"""


# Columns fetch_metric_baseline may aggregate. A closed set, not a formatting
# convenience: the column name is interpolated into SQL (asyncpg cannot bind an
# identifier), so anything outside this tuple must never reach that f-string.
_BASELINE_COLUMNS = (
    "step_count",
    "gap_p75_tool_ms",
    "gap_p75_llm_ms",
    "token_growth",
    "llm_tool_ratio",
    "total_tokens",
    "duration_s",
)


async def write_baseline_metrics(
    org_id: str,
    run_id: str,
    agent_id: str,
    agent_version: str,
    clean: bool,
    metrics: dict,
) -> None:
    """Record this run's baseline scalars. Idempotent on (org_id, run_id).

    ON CONFLICT UPDATE rather than DO NOTHING because a re-detection (late
    events arriving, see audit Finding 15) produces a longer run whose metrics
    supersede the earlier partial ones — and can flip `clean` to FALSE once the
    new events reveal a signal.
    """
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO run_baseline_metrics
                (org_id, run_id, agent_id, agent_version, clean, step_count,
                 gap_p75_tool_ms, gap_p75_llm_ms, token_growth, llm_tool_ratio,
                 total_tokens, duration_s)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (org_id, run_id) DO UPDATE SET
                clean           = EXCLUDED.clean,
                step_count      = EXCLUDED.step_count,
                gap_p75_tool_ms = EXCLUDED.gap_p75_tool_ms,
                gap_p75_llm_ms  = EXCLUDED.gap_p75_llm_ms,
                token_growth    = EXCLUDED.token_growth,
                llm_tool_ratio  = EXCLUDED.llm_tool_ratio,
                total_tokens    = EXCLUDED.total_tokens,
                duration_s      = EXCLUDED.duration_s,
                recorded_at     = NOW()
            """,
            org_id,
            run_id,
            agent_id,
            agent_version,
            clean,
            metrics.get("step_count"),
            metrics.get("gap_p75_tool_ms"),
            metrics.get("gap_p75_llm_ms"),
            metrics.get("token_growth"),
            metrics.get("llm_tool_ratio"),
            metrics.get("total_tokens"),
            metrics.get("duration_s"),
        )


async def fetch_metric_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    column: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """P75 of `column` over the last `lookback` clean runs that carry it.

    Reads run_baseline_metrics, which outlives `events` — this is what lets a
    low-traffic agent accumulate a baseline across more than one retention
    window. Returns None below `min_runs`, which is the signal for the caller
    to fall back to the legacy events-derived query.

    Note the LIMIT applies to runs that HAVE the metric, where the legacy query
    took the last 50 runs and then discarded the ones that didn't qualify. This
    yields up to `lookback` usable samples instead of however many of the last
    `lookback` runs happened to qualify.
    """
    if not _pool or column not in _BASELINE_COLUMNS:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            WITH recent AS (
                SELECT {column} AS v
                FROM run_baseline_metrics
                WHERE org_id        = $1
                  AND agent_id      = $2
                  AND agent_version = $3
                  AND run_id       != $4
                  AND clean
                  AND {column} IS NOT NULL
                ORDER BY recorded_at DESC
                LIMIT $5
            )
            SELECT COUNT(*)                                          AS sample_size,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY v)   AS p75
            FROM recent
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


async def _legacy_step_count_baseline(
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
            f"""
{_RECENT_CLEAN_RUNS_CTE},
            step_counts AS (
                SELECT MAX(e.step_index) AS step_count
                FROM recent r
                JOIN events e ON e.run_id = r.run_id AND e.org_id = $1
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


async def _legacy_latency_baseline(
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
            f"""
{_RECENT_CLEAN_RUNS_CTE},
            event_gaps AS (
                SELECT
                    e.run_id,
                    (LEAD(e.timestamp) OVER (
                        PARTITION BY e.run_id
                        ORDER BY e.step_index, e.timestamp
                    ) - e.timestamp) * 1000.0 AS gap_ms
                FROM events e
                WHERE e.org_id = $1
                  AND e.run_id IN (SELECT run_id FROM recent)
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

async def fetch_per_tool_latency_baselines(
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[dict[str, float]]":
    """
    P75 wall-clock gap (ms) that follows each 'tool.called' event, grouped by tool name.
    """
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH recent AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE pr.agent_id      = $1
                  AND pr.agent_version = $2
                  AND pr.run_id       != $3
                  AND EXISTS (
                      SELECT 1 FROM events e
                      WHERE e.run_id = pr.run_id
                        AND e.event_type = 'run.completed'
                  )
                ORDER BY pr.processed_at DESC
                LIMIT $4
            ),
            event_sequence AS (
                SELECT
                    e.run_id,
                    e.event_type,
                    e.payload->>'tool_name' as tool_name,
                    (LEAD(e.timestamp) OVER (
                        PARTITION BY e.run_id
                        ORDER BY e.step_index, e.timestamp
                    ) - e.timestamp) * 1000.0 AS gap_ms
                FROM events e
                WHERE e.run_id IN (SELECT run_id FROM recent)
            ),
            event_gaps AS (
                SELECT run_id, tool_name, gap_ms
                FROM event_sequence
                WHERE event_type = 'tool.called'
            ),
            stats AS (
                SELECT
                    tool_name,
                    COUNT(DISTINCT run_id) AS sample_size,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY gap_ms) AS p75
                FROM event_gaps
                WHERE gap_ms >= 0
                  AND gap_ms IS NOT NULL
                  AND tool_name IS NOT NULL
                GROUP BY tool_name
            )
            SELECT tool_name, p75
            FROM stats
            WHERE sample_size >= $5
            """,
            agent_id,
            agent_version,
            exclude_run_id,
            lookback,
            min_runs,
        )

    if not rows:
        return None
    return {r["tool_name"]: float(r["p75"]) for r in rows if r["p75"] is not None}


async def _legacy_token_growth_baseline(
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
            f"""
{_RECENT_CLEAN_RUNS_CTE},
            -- One row per LLM call, carrying the SAME prompt_tokens value the
            -- detector sees. Two things this has to match, both of which it
            -- previously got wrong:
            --
            --   1. run_builder takes prompt_tokens from llm.called and then
            --      OVERRIDES it from llm.responded when that carries one (a
            --      streamed call reports an estimate at call time and the exact
            --      count on completion; LangChain only ever reports it on
            --      responded). Reading llm.called alone meant the baseline was
            --      built from estimates while the detector compared exacts.
            --      DISTINCT ON keeps one row per call, preferring responded.
            --
            --   2. llm.responded is emitted with advance=False, so it shares its
            --      llm.called's step_index — ordering by step_index therefore
            --      reproduces the detector's call order exactly.
            token_data AS (
                SELECT DISTINCT ON (e.run_id, COALESCE((e.payload->>'call_id')::int, e.step_index))
                    e.run_id,
                    e.step_index,
                    (e.payload->>'prompt_tokens')::integer AS prompt_tokens
                FROM events e
                WHERE e.org_id = $1
                  AND e.run_id IN (SELECT run_id FROM recent)
                  AND e.event_type IN ('llm.called', 'llm.responded')
                  AND e.payload->>'prompt_tokens' IS NOT NULL
                  AND (e.payload->>'prompt_tokens')::integer > 0
                ORDER BY
                    e.run_id,
                    COALESCE((e.payload->>'call_id')::int, e.step_index),
                    (e.event_type = 'llm.responded') DESC
            ),
            -- FIRST and LAST by call order, not MIN and MAX. The detector
            -- computes calls_with_tokens[-1] / calls_with_tokens[0]; MIN/MAX
            -- only agrees with that when context grows monotonically, and is
            -- strictly larger whenever a run compacts or summarises mid-way.
            -- That biased the baseline high and made CONTEXT_BLOAT quieter than
            -- its configured threshold on exactly the runs that manage context.
            run_growth AS (
                SELECT
                    run_id,
                    (ARRAY_AGG(prompt_tokens ORDER BY step_index))[1]      AS first_tokens,
                    (ARRAY_AGG(prompt_tokens ORDER BY step_index DESC))[1] AS last_tokens,
                    COUNT(*)                                               AS call_count
                FROM token_data
                GROUP BY run_id
                HAVING COUNT(*) >= 3
            )
            SELECT
                COUNT(*)                                                                    AS sample_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (
                    ORDER BY last_tokens::float / first_tokens::float
                )                                                                           AS p75
            FROM run_growth
            -- Applied after first/last are resolved, mirroring the detector's
            -- own guards (first_tokens < 10 and last_tokens < MIN_LAST_TOKENS
            -- both return None there).
            WHERE first_tokens >= 10
              AND last_tokens  >= 2000
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


async def _legacy_llm_tool_ratio_baseline(
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
            f"""
{_RECENT_CLEAN_RUNS_CTE},
            run_counts AS (
                SELECT
                    e.run_id,
                    COUNT(*) FILTER (WHERE e.event_type = 'llm.called')  AS llm_count,
                    COUNT(*) FILTER (WHERE e.event_type = 'tool.called') AS tool_count
                FROM events e
                WHERE e.org_id = $1
                  AND e.run_id IN (SELECT run_id FROM recent)
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


async def _legacy_total_tokens_baseline(
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
            f"""
{_RECENT_CLEAN_RUNS_CTE},
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
                WHERE e.org_id = $1
                  AND e.run_id IN (SELECT run_id FROM recent)
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


async def _legacy_duration_baseline(
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
            f"""
{_RECENT_CLEAN_RUNS_CTE},
            run_durations AS (
                SELECT
                    e.run_id,
                    (MAX(e.timestamp) - MIN(e.timestamp)) AS duration_s
                FROM events e
                WHERE e.org_id = $1
                  AND e.run_id IN (SELECT run_id FROM recent)
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


# ── Public baseline API ───────────────────────────────────────────────────────
#
# Each reads run_baseline_metrics first and falls back to the legacy
# events-derived query when that table has fewer than `min_runs` samples for the
# agent. The fallback is transitional: it exists so an existing deployment keeps
# its baselines on the deploy that introduces the table, instead of dropping
# every agent back to static thresholds until 20 fresh runs accumulate. Once a
# deployment has been running longer than EVENT_RETENTION_DAYS the legacy path
# can no longer contribute anything the table doesn't already have, and the
# _legacy_* functions can be deleted.


async def fetch_step_count_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """P75 step count over recent clean runs. See fetch_metric_baseline."""
    stored = await fetch_metric_baseline(
        org_id, agent_id, agent_version, exclude_run_id, "step_count", min_runs, lookback
    )
    if stored is not None:
        return stored
    return await _legacy_step_count_baseline(
        org_id, agent_id, agent_version, exclude_run_id, min_runs, lookback
    )


async def fetch_latency_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    event_type: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """P75 step latency (ms) for `event_type` over recent clean runs.

    NOTE — this one changes statistic when served from the stored table. The
    legacy query pooled every step gap from 50 runs into a single percentile;
    a one-row-per-run store cannot preserve that, so the stored path takes the
    P75 of each run's own P75 gap. The stored form is more robust (a single
    500-step run can no longer dominate the sample) but it is NOT the same
    number, and SLOW_STEP's effective threshold shifts accordingly on agents
    with high within-run latency variance.
    """
    column = "gap_p75_tool_ms" if event_type == "tool.called" else "gap_p75_llm_ms"
    stored = await fetch_metric_baseline(
        org_id, agent_id, agent_version, exclude_run_id, column, min_runs, lookback
    )
    if stored is not None:
        return stored
    return await _legacy_latency_baseline(
        org_id, agent_id, agent_version, exclude_run_id, event_type, min_runs, lookback
    )


async def fetch_token_growth_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """P75 context growth factor (last/first prompt tokens) over recent clean runs."""
    stored = await fetch_metric_baseline(
        org_id, agent_id, agent_version, exclude_run_id, "token_growth", min_runs, lookback
    )
    if stored is not None:
        return stored
    return await _legacy_token_growth_baseline(
        org_id, agent_id, agent_version, exclude_run_id, min_runs, lookback
    )


async def fetch_llm_tool_ratio_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """P75 LLM:tool call ratio over recent clean runs."""
    stored = await fetch_metric_baseline(
        org_id, agent_id, agent_version, exclude_run_id, "llm_tool_ratio", min_runs, lookback
    )
    if stored is not None:
        return stored
    return await _legacy_llm_tool_ratio_baseline(
        org_id, agent_id, agent_version, exclude_run_id, min_runs, lookback
    )


async def fetch_total_tokens_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """P75 total tokens (prompt+completion+reasoning) over recent clean runs."""
    stored = await fetch_metric_baseline(
        org_id, agent_id, agent_version, exclude_run_id, "total_tokens", min_runs, lookback
    )
    if stored is not None:
        return stored
    return await _legacy_total_tokens_baseline(
        org_id, agent_id, agent_version, exclude_run_id, min_runs, lookback
    )


async def fetch_duration_baseline(
    org_id: str,
    agent_id: str,
    agent_version: str,
    exclude_run_id: str,
    min_runs: int = _MIN_BASELINE_RUNS,
    lookback: int = 50,
) -> "Optional[float]":
    """P75 wall-clock run duration (seconds) over recent clean runs."""
    stored = await fetch_metric_baseline(
        org_id, agent_id, agent_version, exclude_run_id, "duration_s", min_runs, lookback
    )
    if stored is not None:
        return stored
    return await _legacy_duration_baseline(
        org_id, agent_id, agent_version, exclude_run_id, min_runs, lookback
    )


# How many rows per (org, agent, agent_version) the metrics table keeps. The
# baselines only ever read the newest `lookback` (50), so this is that with a
# wide margin — enough that raising `lookback` doesn't immediately starve, and
# small enough that the table is bounded by agent count rather than by traffic.
_BASELINE_METRICS_KEEP_PER_AGENT = 200


async def prune_baseline_metrics(keep_per_agent: int = _BASELINE_METRICS_KEEP_PER_AGENT) -> int:
    """Trim run_baseline_metrics to the newest `keep_per_agent` rows per
    (org_id, agent_id, agent_version). Returns rows deleted.

    Bounded by rank, not by age. An age bound would reintroduce exactly the
    problem this table was built to solve: a low-traffic agent's history would
    expire before it accumulated enough runs to form a baseline. Rank keeps the
    newest N however long they took to arrive, so an agent doing four runs a
    month still reaches the minimum — it just takes months, which is correct.

    This table is deliberately NOT tied to EVENT_RETENTION_DAYS. Its whole
    purpose is to outlive the events its rows were derived from.
    """
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM run_baseline_metrics rbm
             USING (
                 SELECT ctid,
                        ROW_NUMBER() OVER (
                            PARTITION BY org_id, agent_id, agent_version
                            ORDER BY recorded_at DESC
                        ) AS rn
                   FROM run_baseline_metrics
             ) ranked
             WHERE ranked.ctid = rbm.ctid
               AND ranked.rn   > $1
            """,
            keep_per_agent,
        )
    return int(result.split()[-1]) if result.startswith("DELETE") else 0


async def prune_processed_runs(batch_size: int = 10_000) -> int:
    """Delete processed_runs rows whose events are already gone. Returns the count.

    `processed_runs` is one row per run, forever — nothing pruned it, while
    `events` drops partitions at EVENT_RETENTION_DAYS. It's also the anti-join
    target in fetch_completed_runs, so unbounded growth directly slows the
    hottest query in the service.

    The ordering constraint is the subtle part: a processed_runs row may only be
    deleted once its run's events are gone. Delete it while the events remain and
    the run looks unprocessed again — the detector re-runs every detector against
    it and writes a second full set of duplicate signals.

    Rather than couple this to ingest_svc's EVENT_RETENTION_DAYS (a value this
    service doesn't own and could disagree with), the NOT EXISTS makes the
    invariant hold by construction whatever retention is actually configured.

    Candidates are chosen by that same absence-of-events test rather than by a
    `processed_at` age bound. `processed_at` records when a run was last
    *analysed*, not when it happened, and it is refreshed every time late events
    trigger a re-detection — so an age bound over it never expires rows for runs
    that were re-processed recently but whose events aged out long ago. That left
    314 permanently unprunable rows here, each rendering as a hollow entry in the
    run list. Selecting on event absence directly also guarantees every pass makes
    progress; an age-bounded pass could return a full batch of rows that all still
    have events and delete nothing.
    """
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        deleted = await conn.fetchval(
            """
            WITH doomed AS (
                SELECT pr.run_id
                FROM processed_runs pr
                WHERE NOT EXISTS (
                    SELECT 1 FROM events e WHERE e.run_id = pr.run_id
                )
                LIMIT $1
            ),
            removed AS (
                DELETE FROM processed_runs p
                USING doomed d
                WHERE p.run_id = d.run_id
                  AND NOT EXISTS (
                      SELECT 1 FROM events e WHERE e.run_id = p.run_id
                  )
                RETURNING 1
            )
            SELECT count(*) FROM removed
            """,
            batch_size,
        )
    return int(deleted or 0)


async def get_watermark(shard_index: int, shard_count: int = 1) -> Optional[datetime]:
    """This shard's poll watermark, or None if it has never fully drained.

    None deliberately reproduces the unbounded scan — see _WATERMARK_SCHEMA. It's
    the correct behaviour for a first poll (there may be arbitrarily old pending
    work) and it self-corrects after one drained cycle.
    """
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT watermark FROM detector_watermarks WHERE shard_index = $1 AND shard_count = $2",
            shard_index,
            shard_count,
        )


async def advance_watermark(shard_index: int, grace_secs: float, shard_count: int = 1) -> None:
    """Move this shard's watermark up to NOW() - grace_secs.

    `grace_secs` is the re-scan overlap: events that landed in the last
    grace_secs stay inside the next poll's window, which covers a run whose
    terminal event is written a moment after its earlier events and, more
    importantly, keeps the late-event re-detection path (see
    fetch_completed_runs) working for anything arriving within the window.

    GREATEST means the watermark only ever moves forward, so a concurrent poll
    or a clock adjustment can't drag it backwards into a wider scan.

    Call this ONLY after a poll that drained its backlog. Advancing while runs
    are still queued would move the window past unprocessed work — those runs
    would never be detected.
    """
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO detector_watermarks (shard_index, shard_count, watermark, updated_at)
            VALUES ($1, $3, NOW() - ($2 || ' seconds')::INTERVAL, NOW())
            ON CONFLICT (shard_count, shard_index) DO UPDATE
            SET watermark  = GREATEST(
                    detector_watermarks.watermark,
                    NOW() - ($2 || ' seconds')::INTERVAL
                ),
                updated_at = NOW()
            """,
            shard_index,
            str(grace_secs),
            shard_count,
        )


async def fetch_completed_runs(
    limit: int,
    shard_count: int = 1,
    shard_index: int = 0,
    watermark: Optional[datetime] = None,
) -> list[dict]:
    """Runs with a terminal event (run.completed or run.errored) that haven't been processed yet.

    When shard_count > 1 each worker instance claims only the runs whose agent_id
    hashes to its bucket: abs(hashtext(agent_id)) % shard_count = shard_index.
    shard_count=1 (default) bypasses the filter entirely.

    `watermark` bounds the scan to runs touched by an event newer than it, which
    is what lets Postgres prune `events` partitions instead of walking the whole
    retention window on every poll. None = unbounded (first poll after install).

    The bound is deliberately expressed as "runs with a recent event" rather than
    "terminal events that are recent". Those differ for the late-event
    re-detection path: a straggler event on an old run arrives with a *new*
    received_at but its run's terminal event is old, so filtering on the terminal
    event's timestamp would silently stop re-detecting those runs. Selecting the
    run_id set first keeps that path intact while still pruning.

    Intentionally not org_id-scoped: this worker processes every org's runs in one
    poll loop (sharding is by agent_id hash bucket, not by org). org_id is returned
    per-row so the caller can propagate it into processed_runs/failure_signals/issues.
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH touched AS (
                SELECT DISTINCT run_id
                FROM events
                WHERE ($4::timestamptz IS NULL OR received_at > $4::timestamptz)
                  AND ($2::int = 1 OR abs(hashtext(agent_id)) % $2 = $3)
            )
            SELECT DISTINCT ON (e.run_id)
                e.run_id,
                e.agent_id,
                e.agent_version,
                e.org_id,
                e.event_type AS trigger
            FROM events e
            JOIN touched t ON t.run_id = e.run_id
            WHERE e.event_type IN ('run.completed', 'run.errored')
              AND (
                  -- not yet processed …
                  NOT EXISTS (SELECT 1 FROM processed_runs p WHERE p.run_id = e.run_id)
                  -- … OR processed, but new events arrived since (audit Finding 15:
                  -- late events must trigger a re-detection, not be dropped).
                  OR (SELECT count(*) FROM events e2 WHERE e2.run_id = e.run_id)
                     > (SELECT p.event_count FROM processed_runs p WHERE p.run_id = e.run_id)
              )
            ORDER BY e.run_id, e.received_at ASC
            LIMIT $1
            """,
            limit,
            shard_count,
            shard_index,
            watermark,
        )
    return [dict(r) for r in rows]


async def fetch_stalled_runs(
    stall_timeout_secs: float,
    limit: int,
    shard_count: int = 1,
    shard_index: int = 0,
    watermark: Optional[datetime] = None,
) -> list[dict]:
    """Runs that started, never completed, and haven't received a new event in stall_timeout_secs — likely agents stuck mid-run.

    When shard_count > 1 only the runs whose agent_id hashes to shard_index are returned.
    Not org_id-scoped — see fetch_completed_runs.

    `watermark` bounds the driving `run.started` scan the same way
    fetch_completed_runs is bounded; None = unbounded. Bounding is safe here even
    though a stalled run's `run.started` ages out of the window: a run becomes
    stall-eligible stall_timeout_secs after its last event (90s by default), which
    is orders of magnitude inside the grace window, so it is always caught while
    still visible. The one case that could outrun the window — the worker being
    down longer than the grace period — is covered because the watermark is
    persisted and only advances on a drained poll, so downtime leaves it behind
    rather than skipping past the backlog.
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
              AND ($5::timestamptz IS NULL OR e.received_at > $5::timestamptz)
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
            watermark,
        )
    return [dict(r) for r in rows]


async def fetch_run_lineage(run_id: str) -> Optional[dict]:
    """One lightweight lineage row for a run — its identity and immediate parent,
    read from `run.started` — without pulling the run's full event list.

    Used to walk a delegation graph's parent_run_id chain one hop at a time
    (see run_graph.py / the DELEGATION_LOOP detector), where fetching every
    event of every ancestor would be wasteful — only agent_id + parent_run_id
    are needed per hop. Returns None if the run has no `run.started` on record.
    """
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT run_id, agent_id, agent_version, parent_run_id
            FROM events
            WHERE run_id = $1 AND event_type = 'run.started'
            ORDER BY step_index ASC, timestamp ASC
            LIMIT 1
            """,
            run_id,
        )
    return dict(row) if row else None


async def fetch_memory_writes(
    org_id: str,
    agent_id: str,
    keys: list[str],
    exclude_run_id: str,
    lookback_days: int = 30,
    limit: int = 100,
) -> list[dict]:
    """Prior `memory.written` events for this agent, restricted to `keys`.

    Feeds UNGROUNDED_DESTINATION's cross-run taint check (T6). A destination
    planted in run N's memory and read back in run N+2 is ungrounded in N+2 but
    has no in-run taint source: `memory.read` carries only a key, because the SDK
    never sees the value that comes back (run_context.py::memory_read). Without
    this the flagship scenario — poisoned memory, innocent later run, send to the
    planted address — tops out at HIGH.

    Deliberately reads EVENTS, not signals. Ancestor *signals* are derived and
    may not exist yet when a sibling run is processed, which would make severity
    depend on poll ordering; an ingested event row is already durable. Reading
    another run's events is also the precedent _handoff_signal_from_events set.

    Bounded three ways: the keys this run actually read, a lookback window (which
    also prunes `events` partitions), and LIMIT. Rides
    idx_events_org_agent(org_id, agent_id, received_at DESC).
    """
    if not _pool or not keys:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, step_index, timestamp, payload
            FROM events
            WHERE org_id      = $1
              AND agent_id    = $2
              AND event_type  = 'memory.written'
              AND run_id     <> $3
              AND received_at >= NOW() - ($4 || ' days')::interval
              AND payload->>'key' = ANY($5::text[])
            ORDER BY received_at DESC
            LIMIT $6
            """,
            org_id,
            agent_id,
            exclude_run_id,
            str(int(lookback_days)),
            list(keys),
            int(limit),
        )

    out: list[dict] = []
    for r in rows:
        payload = r["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        out.append(
            {
                "run_id": r["run_id"],
                "step_index": r["step_index"],
                "key": payload.get("key"),
                "value": payload.get("value"),
                "source": payload.get("source"),
            }
        )
    return out


async def fetch_destination_baseline(
    org_id: str, agent_id: str, destinations: list[str]
) -> set[str]:
    """Which of `destinations` this agent has already been seen sending to.

    Probing only the run's own candidates keeps this an index lookup over a
    handful of keys rather than a full baseline load.
    """
    if not _pool or not destinations:
        return set()
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT destination
            FROM agent_destination_baseline
            WHERE org_id = $1 AND agent_id = $2 AND destination = ANY($3::text[])
            """,
            org_id,
            agent_id,
            list(destinations),
        )
    return {r["destination"] for r in rows}


async def count_agent_runs_capped(org_id: str, agent_id: str, cap: int) -> int:
    """Processed-run count for this agent, counted no further than `cap`.

    Novelty mode only needs to know whether the baseline has matured past
    MIN_BASELINE_RUNS, so the count stops at the threshold. An uncapped
    COUNT(*) over processed_runs would grow without bound as an agent ages.
    """
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        return int(
            await conn.fetchval(
                """
                SELECT COUNT(*) FROM (
                    SELECT 1 FROM processed_runs
                    WHERE org_id = $1 AND agent_id = $2
                    LIMIT $3
                ) t
                """,
                org_id,
                agent_id,
                int(cap),
            )
            or 0
        )


async def upsert_destination_baseline(
    org_id: str, agent_id: str, destinations: list[tuple[str, str]]
) -> None:
    """Record GROUNDED destinations as normal for this agent.

    Callers must pass only destinations that passed the grounding check — see
    the table's schema comment for why ungrounded ones must never land here.
    """
    if not _pool or not destinations:
        return
    async with _pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO agent_destination_baseline
                (org_id, agent_id, destination, destination_type)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (org_id, agent_id, destination) DO UPDATE
               SET last_seen_at = NOW(),
                   run_count    = agent_destination_baseline.run_count + 1
            """,
            [(org_id, agent_id, d, t) for d, t in destinations],
        )


async def fetch_run_events(run_id: str) -> list[dict]:
    """All events for a run, ordered by step_index then timestamp."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                event_type, run_id, agent_id, agent_version,
                step_index, timestamp, payload, parent_run_id, conversation_id
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


# A run gets this many detection attempts before it is written off. Three
# covers a transient pool timeout or a restart mid-run without letting a
# genuinely undetectable run be retried on every poll forever.
MAX_PROCESSING_ATTEMPTS = 3


async def record_processing_failure(run_id: str, error: str) -> int:
    """Count this failed attempt and return the running total."""
    if not _pool:
        return MAX_PROCESSING_ATTEMPTS
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO run_processing_failures (run_id, attempts, last_error)
            VALUES ($1, 1, $2)
            ON CONFLICT (run_id) DO UPDATE
                SET attempts        = run_processing_failures.attempts + 1,
                    last_error      = EXCLUDED.last_error,
                    last_attempt_at = NOW()
            RETURNING attempts
            """,
            run_id,
            error[:2000],
        )


async def clear_processing_failures(run_id: str) -> None:
    """Called after a run finally processes, so a transient blip leaves nothing
    behind and the next failure starts its budget fresh."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("DELETE FROM run_processing_failures WHERE run_id = $1", run_id)


async def mark_run_processed(
    run_id: str,
    agent_id: str,
    agent_version: str,
    trigger: str,
    signal_count: int,
    org_id: str,
    event_count: int = 0,
    processing_error: str = "",
) -> None:
    """Record that this run has been processed. On a reprocess (audit Finding 15,
    late events grew the count) UPDATE the stored event_count/signal_count so the
    run isn't re-fetched forever."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO processed_runs
                (run_id, agent_id, agent_version, trigger, signal_count, org_id,
                 event_count, processing_error)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NULLIF($8, ''))
            -- (org_id, run_id), not run_id: migration 2 (composite_run_key) made the
            -- key composite because run_id is caller-supplied and collides across
            -- tenants. A bare run_id target has no matching unique constraint, so
            -- this raised on EVERY call and the run was never marked processed —
            -- leaving it to be re-detected on every poll, writing duplicate
            -- signals and re-alerting them.
            ON CONFLICT (org_id, run_id) DO UPDATE
                SET signal_count     = processed_runs.signal_count + EXCLUDED.signal_count,
                    event_count      = EXCLUDED.event_count,
                    processing_error = EXCLUDED.processing_error,
                    processed_at     = NOW()
            """,
            run_id,
            agent_id,
            agent_version,
            trigger,
            signal_count,
            org_id,
            event_count,
            processing_error,
        )


async def fetch_run_state(run_id: str) -> Optional[dict]:
    """Return {'event_count', 'signal_types'} for a run's prior processing, or
    None if never processed. `signal_types` is the set of failure_type values
    already stored for the run — used to make re-detection additive (Finding 15):
    we only write failure types not already recorded, so a reprocess never
    duplicates a signal or re-alerts an existing one."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        prow = await conn.fetchrow(
            "SELECT event_count FROM processed_runs WHERE run_id = $1", run_id
        )
        if prow is None:
            return None
        types = await conn.fetch(
            "SELECT DISTINCT failure_type FROM failure_signals WHERE run_id = $1", run_id
        )
    return {
        "event_count": prow["event_count"],
        "signal_types": {t["failure_type"] for t in types},
    }


# ── Conversation modeling (Phase 3.1) ────────────────────────────────────────────


async def upsert_run_and_conversation(
    run_id: str,
    org_id: str,
    agent_id: str,
    agent_version: str,
    started_at: float,
    conversation_external_id: Optional[str],
) -> None:
    """Registers this run in the runs registry — called once per processed
    run, regardless of whether conversation_id was ever set. When it was,
    also upserts the owning conversation (creating it on first sight, else
    bumping run_count/last_run_at) and links the run to its internal id.
    ON CONFLICT DO NOTHING on runs mirrors mark_run_processed's own
    idempotency guarantee — a run is only ever processed once under normal
    operation."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        conversation_pk = None
        if conversation_external_id:
            row = await conn.fetchrow(
                """
                INSERT INTO conversations (org_id, agent_id, external_id, run_count)
                VALUES ($1, $2, $3, 1)
                ON CONFLICT (org_id, agent_id, external_id) DO UPDATE
                    SET last_run_at = NOW(),
                        run_count   = conversations.run_count + 1
                RETURNING id
                """,
                org_id,
                agent_id,
                conversation_external_id,
            )
            conversation_pk = row["id"]

        await conn.execute(
            """
            INSERT INTO runs (run_id, org_id, agent_id, agent_version, conversation_id, started_at)
            VALUES ($1, $2, $3, $4, $5, to_timestamp($6))
            -- (org_id, run_id) — see mark_run_processed for why. A bare run_id
            -- target raised here too, so the runs registry stopped being written
            -- and GET /v1/runs/{id} reported "no signals" for runs that had them.
            ON CONFLICT (org_id, run_id) DO NOTHING
            """,
            run_id,
            org_id,
            agent_id,
            agent_version,
            conversation_pk,
            started_at,
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


# ── Pack activation reads ───────────────────────────────────────────────────────


async def fetch_org_enabled_packs(org_id: str) -> list[str]:
    """Pack names currently activated for this org. Empty list if none —
    never raises for an org with no row, since that's the default state."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch("SELECT pack_name FROM org_enabled_packs WHERE org_id = $1", org_id)
    return [r["pack_name"] for r in rows]


async def write_run_state_metrics(
    run_id: str,
    org_id: str,
    agent_id: str,
    run_started_ts: Optional[float],
    states: dict,
) -> None:
    """Upsert one row per state for a run (Capability 3, Phase 3.3). Idempotent
    on (run_id, state) — reprocessing a run overwrites its metrics rather than
    duplicating them. `states` is summarize_states(events)["states"]."""
    if not _pool or not states:
        return
    from datetime import datetime, timezone

    started_at = datetime.fromtimestamp(run_started_ts, tz=timezone.utc) if run_started_ts else None
    async with _pool.acquire() as conn:
        for state, agg in states.items():
            await conn.execute(
                """
                INSERT INTO run_state_metrics
                    (run_id, org_id, agent_id, state, total_ms, segment_count, run_started_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (run_id, state) DO UPDATE
                    SET total_ms = EXCLUDED.total_ms,
                        segment_count = EXCLUDED.segment_count,
                        run_started_at = EXCLUDED.run_started_at,
                        computed_at = NOW()
                """,
                run_id,
                org_id,
                agent_id,
                state,
                int(agg["total_ms"]),
                int(agg["count"]),
                started_at,
            )
