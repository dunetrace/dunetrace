"""
Database layer for the semantic worker. Reads completed/errored runs from the
shared `events` table (same source detector_svc polls) and records evaluation
outcomes in its own tracking table.

Schema ownership: this service owns `semantic_processed_runs` outright (no other
service reads or writes it). It also adds a `source` column to the shared
`failure_signals` table, the same way detector_svc already adds `shadow` and
`co_signal_count` to a table ingest_svc owns — additive, IF NOT EXISTS, safe
regardless of which service starts first.
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

from semantic_svc.config import settings

logger = logging.getLogger("dunetrace.semantic.db")

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

# NOTE: ingest_svc owns the core schema (events, failure_signals). This service
# adds only what it needs on top, same convention as detector_svc/alerts_svc.

_SEMANTIC_SCHEMA = """
-- Tracks which runs the semantic worker has already made a sampling decision
-- for. Recorded even when a run is skipped (sampled=FALSE) so a skipped run
-- isn't re-evaluated every poll cycle forever.
CREATE TABLE IF NOT EXISTS semantic_processed_runs (
    run_id        TEXT PRIMARY KEY,
    agent_id      TEXT        NOT NULL,
    agent_version TEXT        NOT NULL,
    org_id        TEXT        NOT NULL,
    sampled       BOOLEAN     NOT NULL,
    sample_reason TEXT,
    processed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_semantic_processed_runs_agent
    ON semantic_processed_runs(agent_id, processed_at DESC);

-- Distinguishes structural detector signals from semantic evaluator signals in
-- the shared failure_signals table. Existing rows (all structural, written
-- before this column existed) default to 'structural'.
ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'structural';

-- Per-agent adaptive sampling overrides (Phase 1.2). Mirrors the existing
-- agent_rate_quotas / agent_detector_overrides convention — a side table
-- keyed by (org_id, agent_id), not a column on a central "agents" table,
-- because no such table exists anywhere in this codebase.
-- sample_rate:    overrides retrieval_rate/baseline_rate for this agent's
--                 general population. Does NOT override the structural-signal
--                 or semantic_critical 100% rules — see sampling.py.
-- budget_monthly: cap on semantic evaluations/month for this agent. NULL means
--                 unlimited (org-level billing quotas are Phase 1.5's concern).
-- evaluators:     which evaluators to run when sampled in (Phase 1.3 reads this).
CREATE TABLE IF NOT EXISTS agent_semantic_config (
    org_id            TEXT        NOT NULL,
    agent_id          TEXT        NOT NULL,
    sample_rate       REAL,
    budget_monthly    INTEGER,
    evaluators        JSONB       NOT NULL DEFAULT '[]',
    semantic_critical BOOLEAN     NOT NULL DEFAULT FALSE,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, agent_id)
);

-- Monthly per-agent evaluation counter backing budget_monthly enforcement.
-- month is a 'YYYY-MM' UTC bucket — a simple calendar-month reset, not a real
-- billing-cycle anchor (Phase 1.5's org-level quota owns billing-cycle
-- semantics; this is just the per-agent spending cap from 1.2's own spec).
CREATE TABLE IF NOT EXISTS semantic_evaluation_usage (
    org_id     TEXT    NOT NULL,
    agent_id   TEXT    NOT NULL,
    month      TEXT    NOT NULL,
    eval_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, agent_id, month)
);

-- Signal grouping/dedup (Phase 1.4.2). One row per distinct recurring
-- pattern — (org_id, agent_id, evaluator, root_cause_hash). root_cause_hash
-- is a crude hash of the evaluator's reasoning text (see grouping.py); this
-- is a DIFFERENT concept from detector_svc's `issues` table (keyed on the
-- closed FailureType enum, not applicable here — semantic evaluators have no
-- fixed type enum, only free-text reasoning to group on).
-- root_cause_sample is a human-readable (normalized, truncated) copy of the
-- first reasoning seen for this group, for dashboard display — the hash
-- itself isn't reversible/readable.
CREATE TABLE IF NOT EXISTS signal_groups (
    id                BIGSERIAL    PRIMARY KEY,
    org_id            TEXT         NOT NULL,
    agent_id          TEXT         NOT NULL,
    evaluator         TEXT         NOT NULL,
    root_cause_hash   TEXT         NOT NULL,
    root_cause_sample TEXT         NOT NULL,
    first_seen        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_seen         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    signal_count      INTEGER      NOT NULL DEFAULT 0,
    UNIQUE (org_id, agent_id, evaluator, root_cause_hash)
);
CREATE INDEX IF NOT EXISTS idx_signal_groups_org_agent ON signal_groups(org_id, agent_id);

-- Membership: which failure_signals rows belong to which group.
-- "affected_run_ids" (per the brief) is derived by joining this table to
-- failure_signals via signal_id, rather than a JSON array column — queryable
-- and consistent with how the rest of this schema prefers relational tables
-- over JSON arrays for anything that needs indexing (e.g. custom_detector_results).
-- signal_id has no DB foreign key to failure_signals(id) — that table is
-- owned by ingest_svc, and nothing in this codebase ever deletes a
-- failure_signals row, so referential integrity is a non-issue in practice;
-- same "not enforced as a DB constraint" call already made for
-- agent_rate_quotas.key_id (services/ingest/ingest_svc/db/postgres.py).
CREATE TABLE IF NOT EXISTS signal_group_members (
    id         BIGSERIAL   PRIMARY KEY,
    group_id   BIGINT      NOT NULL REFERENCES signal_groups(id) ON DELETE CASCADE,
    signal_id  BIGINT      NOT NULL,
    run_id     TEXT        NOT NULL,
    added_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signal_group_members_group ON signal_group_members(group_id, added_at DESC);
CREATE INDEX IF NOT EXISTS idx_signal_group_members_signal ON signal_group_members(signal_id);

-- Phase 1.4.3 — feedback loop. Primarily owned by api_svc's own migration
-- (services/api/api_svc/db/queries.py's _SEMANTIC_FEEDBACK_DDL); added here
-- defensively too since this worker reads both without depending on api_svc
-- (or even ingest_svc, which normally creates `organizations` first) having
-- started first — hence the CREATE TABLE IF NOT EXISTS below, not just the
-- ALTER, matching this schema's usual "whichever starts first wins" rule.
CREATE TABLE IF NOT EXISTS organizations (
    id          TEXT PRIMARY KEY,
    name        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_feedback_enabled BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_feedback_auto_suppress BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS signal_group_overrides (
    group_id   BIGINT      PRIMARY KEY REFERENCES signal_groups(id) ON DELETE CASCADE,
    fp_count   INTEGER     NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Phase 1.5 — org-level billing quota. Primarily owned by api_svc's own
-- migration (services/api/api_svc/db/queries.py's _SEMANTIC_QUOTA_DDL); added
-- here defensively too, same reasoning as the Phase 1.4.3 organizations
-- columns above — this worker reads/writes both without depending on api_svc
-- having started first.
-- semantic_evaluation_quota: monthly cap on evaluations across ALL of this
--   org's agents combined — the mandatory plan-tier ceiling. Distinct from
--   agent_semantic_config.budget_monthly (Phase 1.2), which is an optional,
--   per-agent, customer-configurable sub-limit; both are enforced
--   independently in worker.py.
-- allow_semantic_overage: FALSE (default) stops sampling once the org quota
--   is hit; TRUE keeps sampling (and counting, for billing) past it.
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_evaluation_quota INTEGER NOT NULL DEFAULT 1000;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS allow_semantic_overage BOOLEAN NOT NULL DEFAULT FALSE;

-- Org-level monthly usage counter. Deliberately NOT derived from
-- semantic_evaluation_usage (Phase 1.2) — that table is only incremented for
-- agents with a configured budget_monthly, so summing it would silently
-- undercount usage from every agent without one (the common case). This
-- counter increments on every sampled run, unconditionally, regardless of
-- per-agent config. Same 'YYYY-MM' UTC calendar-month bucket as
-- semantic_evaluation_usage — no per-org billing-cycle anchor date exists
-- anywhere in this codebase to hook into instead.
CREATE TABLE IF NOT EXISTS org_semantic_evaluation_usage (
    org_id     TEXT    NOT NULL,
    month      TEXT    NOT NULL,
    eval_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, month)
);

-- Per-evaluation cost/token log — every evaluate() call, fired or not.
-- failure_signals.evidence only ever captures cost for evaluations that
-- FIRED (evaluate()'s EvalResult.cost_usd is discarded otherwise in
-- worker.py's _run_evaluators) — most evaluations on healthy runs don't
-- fire, so billing math built only from failure_signals would badly
-- undercount real spend. This table is the honest source for cost/usage
-- reporting; failure_signals stays a table of actual findings only.
CREATE TABLE IF NOT EXISTS semantic_evaluation_log (
    id                BIGSERIAL   PRIMARY KEY,
    org_id            TEXT        NOT NULL,
    agent_id          TEXT        NOT NULL,
    evaluator         TEXT        NOT NULL,
    fired             BOOLEAN     NOT NULL,
    prompt_tokens     INTEGER     NOT NULL,
    completion_tokens INTEGER     NOT NULL,
    cost_usd          REAL        NOT NULL,
    evaluated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_semantic_evaluation_log_org_time ON semantic_evaluation_log(org_id, evaluated_at);

-- Phase 3.2 — conversation-level evaluation (UserFrustrationEvaluator) has
-- its own quota, separate from semantic_evaluation_quota/org_semantic_
-- evaluation_usage above, since it's deliberately more expensive per call
-- (up to MIN_CONVERSATION_RUNS runs' worth of context, not one run's) and
-- must not silently eat the per-run budget or vice versa. Same
-- insert-then-conditional-UPDATE race-safe pattern as
-- consume_org_semantic_quota; same 'YYYY-MM' UTC calendar-month bucket.
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS conversation_evaluation_quota INTEGER NOT NULL DEFAULT 200;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS allow_conversation_overage BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS org_conversation_evaluation_usage (
    org_id     TEXT    NOT NULL,
    month      TEXT    NOT NULL,
    eval_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, month)
);
"""


async def ensure_semantic_schema() -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_SEMANTIC_SCHEMA)
    logger.info("Semantic schema ready")


# ── Reads ──────────────────────────────────────────────────────────────────────


async def fetch_unevaluated_runs(limit: int) -> list[dict]:
    """Completed or errored runs that the semantic worker hasn't made a sampling
    decision for yet.

    Mirrors detector_svc.db.fetch_completed_runs's shape (same terminal events,
    same "not yet in my tracking table" filter) but reads independently — this
    service must not depend on detector_svc's processed_runs, since a run can
    be structurally processed and semantically unprocessed at different times.
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (e.run_id)
                e.run_id,
                e.agent_id,
                e.agent_version,
                e.org_id
            FROM events e
            WHERE e.event_type IN ('run.completed', 'run.errored')
              AND NOT EXISTS (
                  SELECT 1 FROM semantic_processed_runs p WHERE p.run_id = e.run_id
              )
            ORDER BY e.run_id, e.received_at ASC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def has_structural_signal(run_id: str) -> bool:
    """Whether detector_svc already wrote a structural failure_signals row for
    this run. Used by the Phase 1.2 sampling engine's "100% of runs where
    structural signals fired" rule."""
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM failure_signals WHERE run_id = $1 AND source = 'structural')",
            run_id,
        )


async def has_retrieval_event(run_id: str) -> bool:
    """Whether this run has any retrieval.called/retrieval.responded event.
    Used by the "20% of runs where retrieval events occurred" sampling rule —
    RAG is the highest hallucination-risk case in the general population."""
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM events
                WHERE run_id = $1 AND event_type IN ('retrieval.called', 'retrieval.responded')
            )
            """,
            run_id,
        )


async def fetch_agent_semantic_config(org_id: str, agent_id: str) -> dict | None:
    """Per-agent sampling override, if one has been configured. None means
    "no override — use the tiered defaults" (see sampling.decide_sampling)."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT sample_rate, budget_monthly, evaluators, semantic_critical
            FROM agent_semantic_config
            WHERE org_id = $1 AND agent_id = $2
            """,
            org_id,
            agent_id,
        )
    if row is None:
        return None
    return {
        "sample_rate": row["sample_rate"],
        "budget_monthly": row["budget_monthly"],
        "evaluators": json.loads(row["evaluators"])
        if isinstance(row["evaluators"], str)
        else list(row["evaluators"]),
        "semantic_critical": row["semantic_critical"],
    }


async def fetch_signal_group_fp_count(
    org_id: str, agent_id: str, evaluator: str, root_cause_hash: str
) -> int:
    """False-positive count for the recurring pattern this (would-be) signal
    belongs to. 0 if the group doesn't exist yet (brand new pattern) or has
    no override row (no false_positive feedback yet) — see
    api_svc.db.queries.record_signal_feedback, the only writer of
    signal_group_overrides.
    """
    async with _pool.acquire() as conn:
        return (
            await conn.fetchval(
                """
                SELECT COALESCE(o.fp_count, 0)
                FROM signal_groups g
                LEFT JOIN signal_group_overrides o ON o.group_id = g.id
                WHERE g.org_id = $1 AND g.agent_id = $2 AND g.evaluator = $3 AND g.root_cause_hash = $4
                """,
                org_id,
                agent_id,
                evaluator,
                root_cause_hash,
            )
            or 0
        )


async def fetch_org_semantic_feedback_settings(org_id: str) -> dict:
    """Returns {enabled, auto_suppress}, defaulting both to False if the org
    row is somehow missing — the safe (never-suppress, never-adjust) default."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT semantic_feedback_enabled, semantic_feedback_auto_suppress "
            "FROM organizations WHERE id = $1",
            org_id,
        )
    if row is None:
        return {"enabled": False, "auto_suppress": False}
    return {
        "enabled": row["semantic_feedback_enabled"],
        "auto_suppress": row["semantic_feedback_auto_suppress"],
    }


async def fetch_org_semantic_quota_settings(org_id: str) -> dict:
    """Returns {quota, allow_overage}. Defaults (quota=1000, allow_overage=False)
    if the org row is somehow missing — matches the column defaults, so a
    missing row behaves identically to a freshly-created one."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT semantic_evaluation_quota, allow_semantic_overage FROM organizations WHERE id = $1",
            org_id,
        )
    if row is None:
        return {"quota": 1000, "allow_overage": False}
    return {
        "quota": row["semantic_evaluation_quota"],
        "allow_overage": row["allow_semantic_overage"],
    }


async def fetch_org_conversation_quota_settings(org_id: str) -> dict:
    """Returns {quota, allow_overage} for the Phase 3.2 conversation-level
    quota bucket. Defaults (quota=200, allow_overage=False) if the org row is
    somehow missing — matches the column defaults."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT conversation_evaluation_quota, allow_conversation_overage "
            "FROM organizations WHERE id = $1",
            org_id,
        )
    if row is None:
        return {"quota": 200, "allow_overage": False}
    return {
        "quota": row["conversation_evaluation_quota"],
        "allow_overage": row["allow_conversation_overage"],
    }


async def fetch_run_conversation_id(run_id: str) -> str | None:
    """Cheap single-column lookup — used to decide whether a run belongs to
    a conversation at all, without the cost of fetching its full event list
    (that only happens once conversation-level evaluation is actually going
    to run, in fetch_run_events per sibling run)."""
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT conversation_id FROM events WHERE run_id = $1 "
            "AND conversation_id IS NOT NULL LIMIT 1",
            run_id,
        )


async def fetch_conversation_run_ids(
    org_id: str, agent_id: str, conversation_external_id: str, limit: int
) -> list[str]:
    """The last `limit` run_ids sharing this conversation_id, oldest -> newest.

    Reads events.conversation_id directly rather than detector_svc's
    conversations/runs registry (Phase 3.1) — this worker and detector_svc
    poll independently, so that registry's row for the *current* run isn't
    guaranteed to exist yet when this runs. events is the one table both
    workers already read from directly, sidestepping the race entirely.
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, MIN(timestamp) AS started_at
            FROM events
            WHERE conversation_id = $1 AND org_id = $2 AND agent_id = $3
            GROUP BY run_id
            ORDER BY started_at DESC
            LIMIT $4
            """,
            conversation_external_id,
            org_id,
            agent_id,
            limit,
        )
    return [r["run_id"] for r in reversed(rows)]


# ── Writes ─────────────────────────────────────────────────────────────────────


async def consume_budget(org_id: str, agent_id: str, month: str, budget_monthly: int) -> bool:
    """Atomically increments this agent's (org_id, agent_id, month) usage
    counter if it's still under budget_monthly. Returns whether the increment
    happened — False means the budget is exhausted and the caller must not
    spend this evaluation slot.

    The bare UPDATE ... WHERE eval_count < $4 is what makes this race-safe:
    Postgres takes a row lock on the first concurrent writer and serializes
    the second, which re-evaluates the WHERE clause against the now-updated
    row — no lost updates even when process_run() runs many runs for the same
    agent concurrently via asyncio.gather.
    """
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO semantic_evaluation_usage (org_id, agent_id, month, eval_count)
            VALUES ($1, $2, $3, 0)
            ON CONFLICT (org_id, agent_id, month) DO NOTHING
            """,
            org_id,
            agent_id,
            month,
        )
        row = await conn.fetchrow(
            """
            UPDATE semantic_evaluation_usage
            SET eval_count = eval_count + 1
            WHERE org_id = $1 AND agent_id = $2 AND month = $3 AND eval_count < $4
            RETURNING eval_count
            """,
            org_id,
            agent_id,
            month,
            budget_monthly,
        )
    return row is not None


async def consume_org_semantic_quota(
    org_id: str, month: str, quota: int, allow_overage: bool
) -> bool:
    """Atomically increments this org's (org_id, month) usage counter and
    returns whether evaluation should proceed. Same race-safe
    insert-then-conditional-UPDATE pattern as consume_budget, generalized
    with an overage escape hatch: when allow_overage is True, the ceiling
    check is skipped entirely — the counter keeps incrementing past quota
    (for accurate billing) and every call returns True. When False, once
    eval_count reaches quota the UPDATE's WHERE clause stops matching, the
    counter freezes at exactly quota, and this returns False from then on
    for the rest of the month.
    """
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO org_semantic_evaluation_usage (org_id, month, eval_count)
            VALUES ($1, $2, 0)
            ON CONFLICT (org_id, month) DO NOTHING
            """,
            org_id,
            month,
        )
        if allow_overage:
            await conn.execute(
                """
                UPDATE org_semantic_evaluation_usage
                SET eval_count = eval_count + 1
                WHERE org_id = $1 AND month = $2
                """,
                org_id,
                month,
            )
            return True

        row = await conn.fetchrow(
            """
            UPDATE org_semantic_evaluation_usage
            SET eval_count = eval_count + 1
            WHERE org_id = $1 AND month = $2 AND eval_count < $3
            RETURNING eval_count
            """,
            org_id,
            month,
            quota,
        )
        return row is not None


async def consume_org_conversation_quota(
    org_id: str, month: str, quota: int, allow_overage: bool
) -> bool:
    """Same race-safe insert-then-conditional-UPDATE pattern as
    consume_org_semantic_quota, against the separate Phase 3.2 conversation
    quota bucket (org_conversation_evaluation_usage) — a conversation
    evaluation must never draw down the per-run quota's counter or vice
    versa."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO org_conversation_evaluation_usage (org_id, month, eval_count)
            VALUES ($1, $2, 0)
            ON CONFLICT (org_id, month) DO NOTHING
            """,
            org_id,
            month,
        )
        if allow_overage:
            await conn.execute(
                """
                UPDATE org_conversation_evaluation_usage
                SET eval_count = eval_count + 1
                WHERE org_id = $1 AND month = $2
                """,
                org_id,
                month,
            )
            return True

        row = await conn.fetchrow(
            """
            UPDATE org_conversation_evaluation_usage
            SET eval_count = eval_count + 1
            WHERE org_id = $1 AND month = $2 AND eval_count < $3
            RETURNING eval_count
            """,
            org_id,
            month,
            quota,
        )
        return row is not None


async def log_semantic_evaluation(
    org_id: str,
    agent_id: str,
    evaluator: str,
    fired: bool,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    """Records one evaluate() call's cost/tokens, fired or not — the honest
    source for usage/cost reporting (Phase 1.5). See _SEMANTIC_SCHEMA's
    semantic_evaluation_log comment for why this can't be derived from
    failure_signals alone."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO semantic_evaluation_log
                (org_id, agent_id, evaluator, fired, prompt_tokens, completion_tokens, cost_usd)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            org_id,
            agent_id,
            evaluator,
            fired,
            prompt_tokens,
            completion_tokens,
            cost_usd,
        )


async def write_quota_exceeded_signal(
    org_id: str,
    agent_id: str,
    agent_version: str,
    run_id: str,
    month: str,
    scope: str,
    limit: int,
) -> None:
    """Operational signal, not a customer-facing failure. Written with
    shadow=TRUE — alerts_svc.fetch_unalerted_signals only ever alerts on
    shadow=FALSE rows, so this can never reach Slack/webhook, only internal
    /shadow-inclusive views.

    scope: "agent_budget" (Phase 1.2's per-agent agent_semantic_config.budget_monthly)
    or "org_quota" (Phase 1.5's org-level organizations.semantic_evaluation_quota).
    Same failure_type either way — evidence.scope distinguishes which ceiling was hit.
    """
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow, co_signal_count, org_id, source)
            VALUES ($1, $2, $3, $4, $5, 0, 1.0, $6::jsonb, TRUE, 0, $7, 'semantic')
            """,
            "SEMANTIC_QUOTA_EXCEEDED",
            "LOW",
            run_id,
            agent_id,
            agent_version,
            json.dumps({"month": month, "scope": scope, "limit": limit}),
            org_id,
        )


async def mark_run_processed(
    run_id: str,
    agent_id: str,
    agent_version: str,
    org_id: str,
    sampled: bool,
    sample_reason: str | None,
) -> None:
    """Record the sampling decision for this run. Prevents re-deciding on every
    poll cycle, whether or not the run was actually evaluated."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO semantic_processed_runs
                (run_id, agent_id, agent_version, org_id, sampled, sample_reason)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (run_id) DO NOTHING
            """,
            run_id,
            agent_id,
            agent_version,
            org_id,
            sampled,
            sample_reason,
        )


async def fetch_run_events(run_id: str) -> list[dict]:
    """All events for a run, ordered by step_index then timestamp. Mirrors
    detector_svc.db.fetch_run_events's shape exactly (read independently —
    semantic_svc has no dependency on detector_svc)."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_type, run_id, agent_id, agent_version, step_index, timestamp,
                   payload, conversation_id
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


async def write_semantic_signal(
    evaluator: str,
    severity: str,
    run_id: str,
    agent_id: str,
    agent_version: str,
    confidence: float,
    evidence: dict,
    org_id: str,
) -> int:
    """Write a semantic evaluator's finding to the shared failure_signals
    table. failure_type is the evaluator's own name (TEXT, bypassing the
    FailureType enum) — same convention custom detectors already use.
    step_index is always 0: semantic evaluation judges a whole run, not one
    step. shadow=FALSE — semantic signals are alert-eligible immediately,
    gated by alerts_svc's per-evaluator confidence floor (Phase 1.4.1)
    instead of the shadow/LIVE_DETECTORS graduation structural detectors use.

    Returns the new row's id — the caller (worker._run_evaluators) needs it
    to record signal group membership (Phase 1.4.2).
    """
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow, co_signal_count, org_id, source)
            VALUES ($1, $2, $3, $4, $5, 0, $6, $7::jsonb, FALSE, 0, $8, 'semantic')
            RETURNING id
            """,
            evaluator,
            severity,
            run_id,
            agent_id,
            agent_version,
            confidence,
            json.dumps(evidence),
            org_id,
        )


async def record_signal_group_membership(
    org_id: str,
    agent_id: str,
    evaluator: str,
    reasoning: str,
    signal_id: int,
    run_id: str,
) -> int:
    """Groups a just-written semantic signal at emission time (Phase 1.4.2),
    not at query time: upserts the (org_id, agent_id, evaluator,
    root_cause_hash) group this signal's reasoning belongs to, links the
    signal as a member, and returns the group id.
    """
    from semantic_svc.grouping import normalize_root_cause, root_cause_hash

    rc_hash = root_cause_hash(reasoning)
    # Truncated further than the hash's own grouping prefix — this is just a
    # human-readable label for the dashboard, not part of the grouping key.
    sample = normalize_root_cause(reasoning)[:200]

    async with _pool.acquire() as conn:
        async with conn.transaction():
            group_id = await conn.fetchval(
                """
                INSERT INTO signal_groups
                    (org_id, agent_id, evaluator, root_cause_hash, root_cause_sample, signal_count)
                VALUES ($1, $2, $3, $4, $5, 1)
                ON CONFLICT (org_id, agent_id, evaluator, root_cause_hash) DO UPDATE
                    SET last_seen    = NOW(),
                        signal_count = signal_groups.signal_count + 1
                RETURNING id
                """,
                org_id,
                agent_id,
                evaluator,
                rc_hash,
                sample,
            )
            await conn.execute(
                """
                INSERT INTO signal_group_members (group_id, signal_id, run_id)
                VALUES ($1, $2, $3)
                """,
                group_id,
                signal_id,
                run_id,
            )
    return group_id
