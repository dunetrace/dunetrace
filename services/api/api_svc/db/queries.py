"""
Database queries for the customer API. Reads from events, failure_signals,
processed_runs, and api_keys. This service never writes.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json as _json_mod
import logging
import time
from typing import Any, AsyncGenerator, Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

from api_svc.config import settings
from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.explainer import explain

logger = logging.getLogger("dunetrace.api.db")
_pool = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

_MIGRATIONS_DDL = """
ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS co_signal_count INTEGER NOT NULL DEFAULT 0;
"""

_FIXES_DDL = """
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
-- org_id is nullable here: this table is also created by ingest_svc, which owns the
-- org_id backfill + NOT NULL migration (services/ingest/ingest_svc/db/postgres.py).
-- Whichever service starts first creates the base table; the other's ALTER ... ADD
-- COLUMN IF NOT EXISTS / backfill is a no-op or idempotent catch-up.
ALTER TABLE fixes ADD COLUMN IF NOT EXISTS org_id TEXT;
CREATE INDEX IF NOT EXISTS idx_fixes_signal_id ON fixes(signal_id);
CREATE INDEX IF NOT EXISTS idx_fixes_run_id    ON fixes(run_id, applied_at DESC);
"""

_POLICIES_DDL = """
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
-- See fixes.org_id comment above — same cross-service "whichever wins" ownership.
ALTER TABLE policies ADD COLUMN IF NOT EXISTS org_id TEXT;
CREATE INDEX IF NOT EXISTS idx_policies_agent ON policies(agent_id, enabled);
"""

_POLICY_SECURITY_DDL = """
ALTER TABLE policies ADD COLUMN IF NOT EXISTS signature TEXT NOT NULL DEFAULT '';
-- Which version of the HMAC canonical form a policy was signed under. Existing
-- rows default to 1 (the original form); policies using a condition.match
-- expression block are signed as 2. Verification is driven by this per-row value.
ALTER TABLE policies ADD COLUMN IF NOT EXISTS sig_version INT NOT NULL DEFAULT 1;

-- Policy evaluation observability (Phase 5). Owned/written by ingest_svc; created
-- defensively here (IF NOT EXISTS) so the API's read query never fails if the API
-- happens to start first. Kept in sync with ingest_svc's DDL.
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

CREATE TABLE IF NOT EXISTS policy_audit_log (
    id          BIGSERIAL    PRIMARY KEY,
    policy_id   BIGINT,
    action      TEXT         NOT NULL,
    customer_id TEXT         NOT NULL,
    before      JSONB,
    after       JSONB,
    changed_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'policy_audit_log' AND column_name = 'customer_id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'policy_audit_log' AND column_name = 'org_id'
    ) THEN
        ALTER TABLE policy_audit_log RENAME COLUMN customer_id TO org_id;
    END IF;
END $$;
ALTER TABLE policy_audit_log ADD COLUMN IF NOT EXISTS org_id TEXT;
CREATE INDEX IF NOT EXISTS idx_policy_audit_policy_id  ON policy_audit_log(policy_id);
CREATE INDEX IF NOT EXISTS idx_policy_audit_changed_at ON policy_audit_log(changed_at DESC);
"""

_FEEDBACK_DDL = """
ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS agent_detector_overrides (
    agent_id         TEXT        NOT NULL,
    failure_type     TEXT        NOT NULL,
    fp_count         INTEGER     NOT NULL DEFAULT 0,
    confidence_floor FLOAT       NOT NULL DEFAULT 0.0,
    silenced         BOOLEAN     NOT NULL DEFAULT FALSE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (agent_id, failure_type)
);
-- Widen the key to (org_id, agent_id, failure_type) — agent_id is not guaranteed
-- unique across orgs, so the 2-key PK could let one org's FP-marking silence another
-- org's identically-named agent's detector. Same fix as alerts_svc's alert_dedup.
ALTER TABLE agent_detector_overrides ADD COLUMN IF NOT EXISTS org_id TEXT;
UPDATE agent_detector_overrides SET org_id = 'default' WHERE org_id IS NULL;
ALTER TABLE agent_detector_overrides ALTER COLUMN org_id SET NOT NULL;
DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'agent_detector_overrides_pkey'
    ) THEN
        ALTER TABLE agent_detector_overrides DROP CONSTRAINT agent_detector_overrides_pkey;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ado_org_agent_failure_type_key'
    ) THEN
        ALTER TABLE agent_detector_overrides ADD CONSTRAINT ado_org_agent_failure_type_key
            PRIMARY KEY (org_id, agent_id, failure_type);
    END IF;
END $$;

-- Phase 4.1 — "Snooze this pattern" Slack button. Independent of fp_count/
-- confidence_floor/silenced above (snoozing is a deliberate, temporary
-- human decision — "I know about this, stop paging me for a day" — not
-- accumulated false-positive feedback). NULL/past means not snoozed.
ALTER TABLE agent_detector_overrides ADD COLUMN IF NOT EXISTS snoozed_until TIMESTAMPTZ;
"""

_KEYS_DDL = """
-- organizations/org_id replace companies/customer_id (see
-- services/ingest/ingest_svc/db/postgres.py for the rename migration that owns
-- this transition). This DDL only needs to land the NEW shape idempotently —
-- whichever of ingest_svc/api_svc starts first creates it.
CREATE TABLE IF NOT EXISTS organizations (
    id          TEXT PRIMARY KEY,
    name        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO organizations (id, name) VALUES ('default', 'Default Organization')
    ON CONFLICT (id) DO NOTHING;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS id BIGSERIAL;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS org_id TEXT;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rate_limit_rpm INTEGER NOT NULL DEFAULT 600;
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_id ON api_keys(id);
"""

_CUSTOM_DETECTORS_DDL = """
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
-- org_id ownership: see detector_svc/db.py, which also creates this table and
-- owns the org_id backfill + NOT NULL migration for it.
ALTER TABLE custom_detectors ADD COLUMN IF NOT EXISTS org_id TEXT;
CREATE INDEX IF NOT EXISTS idx_custom_detectors_agent ON custom_detectors(agent_id, status);

CREATE TABLE IF NOT EXISTS custom_detector_results (
    id           BIGSERIAL    PRIMARY KEY,
    detector_id  BIGINT       NOT NULL REFERENCES custom_detectors(id) ON DELETE CASCADE,
    run_id       TEXT         NOT NULL,
    agent_id     TEXT         NOT NULL,
    fired        BOOLEAN      NOT NULL,
    evaluated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
ALTER TABLE custom_detector_results ADD COLUMN IF NOT EXISTS org_id TEXT;
CREATE INDEX IF NOT EXISTS idx_cdr_detector ON custom_detector_results(detector_id, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS idx_cdr_run      ON custom_detector_results(run_id);
"""

# Phase 1.4.3 — semantic signal feedback capture. This service creates
# signal_groups/signal_group_members defensively (IF NOT EXISTS) even though
# semantic_svc is their primary owner — same dual-creation convention already
# used for custom_detectors/custom_detector_results (detector_svc + api_svc):
# whichever service starts first wins, and semantic_svc may never even be
# enabled (SEMANTIC_WORKER_ENABLED defaults to false) on a given install.
_SEMANTIC_FEEDBACK_DDL = """
-- Opt-in per org (default off) — the whole feedback loop (capture +
-- auto-suppress) is inert until an org turns it on. auto_suppress controls
-- what happens once a group crosses the false-positive threshold: FALSE
-- (default) just lowers future confidence by 0.3; TRUE stops writing new
-- signals for that group entirely. See semantic_svc/worker.py.
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_feedback_enabled BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_feedback_auto_suppress BOOLEAN NOT NULL DEFAULT FALSE;

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

CREATE TABLE IF NOT EXISTS signal_group_members (
    id         BIGSERIAL   PRIMARY KEY,
    group_id   BIGINT      NOT NULL REFERENCES signal_groups(id) ON DELETE CASCADE,
    signal_id  BIGINT      NOT NULL,
    run_id     TEXT        NOT NULL,
    added_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signal_group_members_group ON signal_group_members(group_id, added_at DESC);
CREATE INDEX IF NOT EXISTS idx_signal_group_members_signal ON signal_group_members(signal_id);

-- One row per group that has accumulated false-positive feedback. No row at
-- all means fp_count=0 — a group is only ever created here once its first
-- false_positive verdict arrives (see record_signal_feedback).
CREATE TABLE IF NOT EXISTS signal_group_overrides (
    group_id   BIGINT      PRIMARY KEY REFERENCES signal_groups(id) ON DELETE CASCADE,
    fp_count   INTEGER     NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per feedback submission (not per group) — (signal_id, org_id,
-- verdict, notes), exactly the shape in the Phase 1.4 brief. Aggregation
-- into signal_group_overrides.fp_count happens at write time in
-- record_signal_feedback, not by re-scanning this table on every read.
CREATE TABLE IF NOT EXISTS signal_feedback (
    id         BIGSERIAL   PRIMARY KEY,
    signal_id  BIGINT      NOT NULL,
    org_id     TEXT        NOT NULL,
    verdict    TEXT        NOT NULL,
    notes      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signal_feedback_signal ON signal_feedback(signal_id);
"""

# Phase 1.5 — semantic evaluation billing/quotas. Defensively duplicated from
# semantic_svc's own migration (services/semantic/semantic_svc/db.py), same
# whichever-starts-first convention as _SEMANTIC_FEEDBACK_DDL above — this
# service's new usage endpoint reads both tables without depending on
# semantic_svc (which may be disabled entirely, SEMANTIC_WORKER_ENABLED
# defaults to false) having ever run.
_SEMANTIC_QUOTA_DDL = """
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS semantic_evaluation_quota INTEGER NOT NULL DEFAULT 1000;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS allow_semantic_overage BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS org_semantic_evaluation_usage (
    org_id     TEXT    NOT NULL,
    month      TEXT    NOT NULL,
    eval_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, month)
);

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
"""

# Phase 2.1 — external evaluation integrations (Langfuse first; LangSmith/
# Braintrust reuse this same generic shape in 2.2/2.3, differing only by
# `provider` and whatever keys their own credentials JSON needs).
# Primarily owned by integrations_worker's own migration (not yet built as of
# this PR being the config-CRUD half); duplicated here defensively, same
# whichever-starts-first convention as every other cross-service table in
# this schema.
_INTEGRATIONS_DDL = """
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

-- Dedup: a poll's overlap window (to tolerate the provider's own indexing
-- lag) will re-fetch evaluations already seen — this is what prevents
-- writing a duplicate failure_signals row for the same external evaluation.
CREATE TABLE IF NOT EXISTS external_evaluation_processed (
    org_id       TEXT        NOT NULL,
    provider     TEXT        NOT NULL,
    external_id  TEXT        NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, provider, external_id)
);
"""

_ALERT_INTEGRATIONS_DDL = """
-- Phase 4.1 — per-org Slack/Linear alert destinations, both bring-your-own
-- (a customer's own Slack incoming webhook / Linear API key + webhook
-- secret), same encrypt-at-rest pattern as Phase 2.1's
-- external_evaluation_integrations. api_svc only ever encrypts (on config
-- submission) and never decrypts, EXCEPT for Linear's webhook_secret — see
-- api_svc/crypto.py::decrypt_credentials_for_webhook_verification's
-- docstring for why that one case is a deliberate, narrow exception.
-- alerts_svc is the only thing that decrypts webhook_url/api_key, to
-- actually call Slack/Linear's API.
CREATE TABLE IF NOT EXISTS org_alert_integrations (
    id                    BIGSERIAL    PRIMARY KEY,
    org_id                TEXT         NOT NULL,
    provider              TEXT         NOT NULL,   -- 'slack' | 'linear'
    encrypted_credentials TEXT         NOT NULL,   -- slack: {webhook_url}; linear: {api_key, webhook_secret}
    config_json           JSONB        NOT NULL DEFAULT '{}',  -- slack: {channel}; linear: {team_id, project_id}
    enabled               BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (org_id, provider)
);

-- Bi-directional sync (Linear issue closed -> Dunetrace signal resolved).
-- Written by alerts_svc (when it creates a Linear issue for a signal), read
-- by api_svc's webhook receiver (routers/linear_webhook.py) to find which
-- signal a given Linear issue corresponds to.
CREATE TABLE IF NOT EXISTS linear_issue_signals (
    id              BIGSERIAL    PRIMARY KEY,
    org_id          TEXT         NOT NULL,
    signal_id       BIGINT       NOT NULL,
    linear_issue_id TEXT         NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (linear_issue_id)
);
"""

# Phase 4.2 — MCP resolve_issue tool. `issues` is owned by detector_svc
# (services/detector/detector_svc/db.py); api_svc has only ever read it
# (list_issues) until now. This service becomes the first WRITER of these
# two specific columns (via resolve_issue below), so it defensively ensures
# they exist here — same "whichever service needs a column adds it
# defensively" convention detector_svc itself already uses for
# failure_signals.shadow/co_signal_count (owned by ingest_svc).
# Deliberately orthogonal to the existing auto-resolve/reopen-on-recurrence
# machinery (clean_runs_since) — a manually-resolved issue still reopens if
# the failure recurs later, same as an auto-resolved one; these two columns
# only record how a resolution happened, not a permanent lock.
#
# Guarded by an existence check, not a bare ALTER — api_svc doesn't create
# `issues` itself (detector_svc does), so on a fresh install where
# detector_svc hasn't started yet, a bare ALTER TABLE would fail with
# "relation issues does not exist" and crash this service's entire
# startup, taking down every other endpoint with it. This degrades to a
# no-op instead, retried (and eventually succeeding) on next restart —
# same tolerance list_issues already implicitly relies on.
_ISSUES_RESOLUTION_DDL = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'issues') THEN
        ALTER TABLE issues ADD COLUMN IF NOT EXISTS resolution_notes TEXT;
        ALTER TABLE issues ADD COLUMN IF NOT EXISTS manually_resolved BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;
END $$;
"""

# Phase 4.3 — GitHub App per-org config. installation_id isn't a secret (an
# App is one operator-level registration; per-org installs are just
# identifiers), so unlike the Slack/Linear integrations there's no
# encrypted_credentials column here at all — a genuine simplification, not
# an oversight.
_GITHUB_APP_DDL = """
CREATE TABLE IF NOT EXISTS org_github_integrations (
    id               BIGSERIAL    PRIMARY KEY,
    org_id           TEXT         NOT NULL,
    installation_id  BIGINT       NOT NULL,
    repos            JSONB        NOT NULL DEFAULT '[]',
    reviewers        JSONB        NOT NULL DEFAULT '[]',
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (org_id)
);

-- Tier-1 explicit source mapping (Phase 4.3). No central `agents` table
-- exists anywhere in this codebase — same side-table-keyed-by-(org_id,
-- agent_id) convention agent_semantic_config/agent_detector_overrides
-- already established.
CREATE TABLE IF NOT EXISTS agent_source_config (
    org_id      TEXT         NOT NULL,
    agent_id    TEXT         NOT NULL,
    repo        TEXT         NOT NULL,
    file_path   TEXT,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, agent_id)
);
"""

# failure_signals.source is owned by semantic_svc's migration (added there
# for Phase 1's semantic evaluators, reused by integrations_svc for
# Phase 2's external providers) — but Phase 4.4's agent_performance_trends()
# reads it directly from api_svc, and semantic_svc is disabled by default
# (SEMANTIC_WORKER_ENABLED). Found via a real 500 (UndefinedColumnError)
# against a deployment that had never started semantic_svc: api_svc must not
# assume a column owned by an optional service exists. Defensive
# ADD COLUMN IF NOT EXISTS, same "whichever starts first wins" convention
# already used for custom_detectors/custom_detector_results above.
_PERFORMANCE_TRENDS_DDL = """
ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'structural';
"""

# Pack activation (Phase 1.0). detector_svc owns packs (it seeds rows from
# PACK_REGISTRY at startup and is the sole reader of org_enabled_packs for
# detector selection) — created here too defensively, same "whichever starts
# first wins" convention as every other cross-service table, since api_svc
# is the write path for activation (POST/DELETE /v1/orgs/packs/{name}) and
# must not assume detector_svc has started first.
_PACKS_DDL = """
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

# Human-in-the-loop approvals (Capability 2). org_id TEXT NOT NULL, no FK to
# organizations(id) — same convention as every other org-scoped table here.
# status is stored as TEXT (validated against api_svc.approvals.ApprovalStatus
# in code, not a DB enum, so adding a status later needs no migration).
_APPROVALS_DDL = """
CREATE TABLE IF NOT EXISTS approvals (
    id               BIGSERIAL    PRIMARY KEY,
    org_id           TEXT         NOT NULL,
    run_id           TEXT         NOT NULL,
    agent_id         TEXT         NOT NULL,
    tool_name        TEXT         NOT NULL,
    tool_args        TEXT,
    status           TEXT         NOT NULL DEFAULT 'pending',
    requested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    expires_at       TIMESTAMPTZ,
    decided_at       TIMESTAMPTZ,
    decided_by       TEXT,
    decision_channel TEXT,
    delivered_at     TIMESTAMPTZ
);
-- delivered_at: set once alerts_svc has notified a human (Phase 2.3). Added
-- via ALTER too, so an approvals table created by an earlier build in this
-- sprint (before delivery existed) gains the column without a manual migration.
ALTER TABLE approvals ADD COLUMN IF NOT EXISTS delivered_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_approvals_org_status ON approvals(org_id, status);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id);
CREATE INDEX IF NOT EXISTS idx_approvals_undelivered
    ON approvals(requested_at) WHERE delivered_at IS NULL AND status = 'pending';
"""

# Per-run state metrics (Capability 3, Phase 3.3). Written by detector_svc,
# read here for cross-run analytics. Defensive copy (CREATE IF NOT EXISTS) —
# whichever service starts first wins, same pattern as the packs tables.
_RUN_STATE_METRICS_DDL = """
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


def _ts(v):
    """Normalize a DB timestamp (datetime or numeric) to a unix-epoch float, or
    None. Module-level so every row mapper shares one definition."""
    if v is None:
        return None
    return v.timestamp() if hasattr(v, "timestamp") else float(v)


async def init_pool() -> None:
    global _pool
    if asyncpg is None:
        return
    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=15,
        # See ingest_svc/db/postgres.py::init_pool for why this is required —
        # DATABASE_URL is Supabase's transaction-mode PgBouncer pooler, which
        # is incompatible with asyncpg's default prepared-statement cache.
        statement_cache_size=0,
    )
    async with _pool.acquire() as conn:
        await conn.execute(_MIGRATIONS_DDL)
        await conn.execute(_FIXES_DDL)
        await conn.execute(_POLICIES_DDL)
        await conn.execute(_POLICY_SECURITY_DDL)
        await conn.execute(_FEEDBACK_DDL)
        await conn.execute(_KEYS_DDL)
        await conn.execute(_CUSTOM_DETECTORS_DDL)
        await conn.execute(_SEMANTIC_FEEDBACK_DDL)
        await conn.execute(_SEMANTIC_QUOTA_DDL)
        await conn.execute(_INTEGRATIONS_DDL)
        await conn.execute(_ALERT_INTEGRATIONS_DDL)
        await conn.execute(_ISSUES_RESOLUTION_DDL)
        await conn.execute(_GITHUB_APP_DDL)
        await conn.execute(_PERFORMANCE_TRENDS_DDL)
        await conn.execute(_PACKS_DDL)
        await conn.execute(_APPROVALS_DDL)
        await conn.execute(_RUN_STATE_METRICS_DDL)
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


async def verify_api_key(key: str) -> Optional[str]:
    """Returns org_id if valid, None otherwise. Dev mode accepts anything."""
    if settings.is_dev:
        return "default"
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT org_id FROM api_keys WHERE key = $1 AND active = TRUE",
            key,
        )
    return row["org_id"] if row else None


# ── Agents ────────────────────────────────────────────────────────────────────


async def list_agents(org_id: str, offset: int, limit: int) -> tuple[list, int]:
    """Returns (rows, total_count). Each row has: agent_id, last_seen, run_count, signal_count, critical_count, high_count.

    Agents are discovered dynamically from events.org_id — an org-scoped key isn't
    1:1 with a single agent_id, so there's no api_keys join here at all."""
    if not _pool:
        return [], 0

    async with _pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(DISTINCT agent_id) FROM events WHERE org_id = $1",
            org_id,
        )

        rows = await conn.fetch(
            """
            WITH event_agg AS (
                SELECT
                    agent_id,
                    MAX(received_at)          AS last_seen,
                    COUNT(DISTINCT run_id)    AS run_count
                FROM events
                WHERE org_id = $1
                GROUP BY agent_id
            ),
            signal_agg AS (
                SELECT
                    agent_id,
                    COUNT(*) FILTER (WHERE shadow = FALSE)                          AS signal_count,
                    COUNT(*) FILTER (WHERE shadow = FALSE AND severity = 'CRITICAL') AS critical_count,
                    COUNT(*) FILTER (WHERE shadow = FALSE AND severity = 'HIGH')     AS high_count
                FROM failure_signals
                WHERE org_id = $1
                GROUP BY agent_id
            )
            SELECT
                e.agent_id,
                e.last_seen,
                e.run_count,
                COALESCE(s.signal_count, 0)   AS signal_count,
                COALESCE(s.critical_count, 0) AS critical_count,
                COALESCE(s.high_count, 0)     AS high_count
            FROM event_agg e
            LEFT JOIN signal_agg s ON s.agent_id = e.agent_id
            ORDER BY e.last_seen DESC
            LIMIT $2 OFFSET $3
            """,
            org_id,
            limit,
            offset,
        )

    return [dict(r) for r in rows], total or 0


# ── Failure type breakdown ────────────────────────────────────────────────────


async def agent_failure_type_counts(org_id: str) -> dict:
    """Live signal counts per agent broken down by failure type: { agent_id: { "TOOL_LOOP": 3, ... } }."""
    if not _pool:
        return {}

    from collections import defaultdict

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT agent_id, failure_type, COUNT(*) AS cnt
            FROM failure_signals
            WHERE shadow = FALSE
              AND org_id = $1
            GROUP BY agent_id, failure_type
            """,
            org_id,
        )

    result: dict = defaultdict(dict)
    for r in rows:
        result[r["agent_id"]][r["failure_type"]] = int(r["cnt"])
    return dict(result)


# ── Sparklines ────────────────────────────────────────────────────────────────


async def agent_signal_sparklines(org_id: str) -> dict:
    """7-day daily live signal counts per agent, oldest→newest: { agent_id: [day-6, ..., today] }."""
    if not _pool:
        return {}

    import datetime
    from collections import defaultdict

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                agent_id,
                DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC') AS day,
                COUNT(*)                                            AS cnt
            FROM failure_signals
            WHERE shadow = FALSE
              AND detected_at >= NOW() - INTERVAL '7 days'
              AND org_id = $1
            GROUP BY agent_id, day
            ORDER BY agent_id, day
            """,
            org_id,
        )

    # Build map: agent_id → {date: count}
    day_counts: dict = defaultdict(dict)
    for r in rows:
        d = r["day"].date() if hasattr(r["day"], "date") else r["day"]
        day_counts[r["agent_id"]][d] = int(r["cnt"])

    # Produce exactly 7 values per agent, oldest first
    today = datetime.date.today()
    return {
        agent_id: [
            day_counts[agent_id].get(today - datetime.timedelta(days=offset), 0)
            for offset in range(6, -1, -1)  # 6 days ago … today
        ]
        for agent_id in day_counts
    }


# ── Runs ──────────────────────────────────────────────────────────────────────


async def list_runs(
    org_id: str,
    agent_id: str,
    offset: int,
    limit: int,
    has_signals: Optional[bool] = None,
) -> tuple[list, int]:
    """List runs for an agent within org_id. Optionally filter to only runs with signals."""
    if not _pool:
        return [], 0

    signal_filter = ""
    if has_signals is True:
        signal_filter = "AND EXISTS (SELECT 1 FROM failure_signals s WHERE s.run_id = pr.run_id AND s.shadow = FALSE)"
    elif has_signals is False:
        signal_filter = "AND NOT EXISTS (SELECT 1 FROM failure_signals s WHERE s.run_id = pr.run_id AND s.shadow = FALSE)"

    async with _pool.acquire() as conn:
        total = await conn.fetchval(
            f"""
            SELECT COUNT(*) FROM processed_runs pr
            WHERE pr.org_id = $1 AND pr.agent_id = $2
            {signal_filter}
            """,
            org_id,
            agent_id,
        )

        rows = await conn.fetch(
            f"""
            WITH page AS (
                SELECT pr.run_id, pr.agent_id, pr.agent_version,
                       pr.trigger AS exit_reason, pr.processed_at
                FROM processed_runs pr
                WHERE pr.org_id = $1 AND pr.agent_id = $2
                {signal_filter}
                ORDER BY pr.processed_at DESC
                LIMIT $3 OFFSET $4
            ),
            run_events AS (
                -- Single pass over events for just this page's runs, instead of
                -- re-scanning events per-row (was 6 correlated subqueries/row).
                SELECT
                    e.run_id,
                    MIN(e.timestamp) FILTER (WHERE e.event_type = 'run.started') AS started_at,
                    MIN(e.timestamp) FILTER (
                        WHERE e.event_type IN ('run.completed', 'run.errored')
                    )                                                            AS completed_at,
                    MAX(e.step_index)                                           AS step_count,
                    SUM(
                        CASE WHEN e.event_type IN ('llm.called', 'llm.responded')
                             THEN COALESCE((e.payload->>'prompt_tokens')::integer, 0)
                             ELSE 0 END
                    )                                                            AS prompt_tokens,
                    SUM(
                        CASE WHEN e.event_type = 'llm.responded'
                                  AND e.payload->>'completion_tokens' IS NOT NULL
                             THEN COALESCE((e.payload->>'completion_tokens')::integer, 0)
                             ELSE 0 END
                    )                                                            AS completion_tokens,
                    (ARRAY_AGG(e.payload->>'model')
                        FILTER (WHERE e.event_type = 'llm.called'
                                       AND e.payload->>'model' IS NOT NULL))[1]  AS model
                FROM events e
                WHERE e.run_id IN (SELECT run_id FROM page)
                GROUP BY e.run_id
            ),
            run_signals AS (
                SELECT s.run_id, COUNT(*) AS signal_count
                FROM failure_signals s
                WHERE s.run_id IN (SELECT run_id FROM page) AND s.shadow = FALSE
                GROUP BY s.run_id
            )
            SELECT
                p.run_id, p.agent_id, p.agent_version, p.exit_reason, p.processed_at,
                re.started_at, re.completed_at, re.step_count,
                (COALESCE(re.prompt_tokens, 0) + COALESCE(re.completion_tokens, 0))
                    AS total_tokens,
                re.prompt_tokens, re.completion_tokens, re.model,
                COALESCE(rs.signal_count, 0) AS signal_count
            FROM page p
            LEFT JOIN run_events  re ON re.run_id = p.run_id
            LEFT JOIN run_signals rs ON rs.run_id = p.run_id
            ORDER BY p.processed_at DESC
            """,
            org_id,
            agent_id,
            limit,
            offset,
        )

    from explainer_svc.cost import estimate_cost

    results = []
    for r in rows:
        rd = dict(r)
        pt = int(rd.pop("prompt_tokens") or 0)
        ct = int(rd.pop("completion_tokens") or 0)
        model = rd.pop("model") or ""
        raw = estimate_cost(model, pt, ct) if (pt or ct) else None
        rd["cost_usd"] = round(raw, 6) if raw else None
        results.append(rd)
    return results, total or 0


async def get_run_detail(org_id: str, run_id: str, include_shadow: bool = False) -> Optional[dict]:
    """Full run detail: metadata + events + signals with explanations.
    Returns None if the run doesn't exist OR belongs to a different org."""
    if not _pool:
        return None

    import json

    async with _pool.acquire() as conn:
        pr = await conn.fetchrow(
            "SELECT run_id, agent_id, agent_version, trigger, processed_at "
            "FROM processed_runs WHERE run_id = $1 AND org_id = $2",
            run_id,
            org_id,
        )
        if not pr:
            return None

        events = await conn.fetch(
            """
            SELECT event_type, step_index, timestamp, payload, parent_run_id
            FROM events WHERE run_id = $1
            ORDER BY step_index ASC, timestamp ASC
            """,
            run_id,
        )

        shadow_filter = "" if include_shadow else "AND shadow = FALSE"
        signals = await conn.fetch(
            f"""
            SELECT id, failure_type, severity, step_index, confidence,
                   detected_at, evidence, shadow
            FROM failure_signals
            WHERE run_id = $1 {shadow_filter}
            ORDER BY shadow ASC, step_index ASC
            """,
            run_id,
        )

        # Phase 3.3 — "part of conversation X" navigation. runs is owned by
        # detector_svc (Phase 3.1); read directly, same trust relationship
        # this function already has with processed_runs (no defensive
        # CREATE TABLE here — this service never writes either table).
        # NULL for runs that predate this migration or never had a
        # conversation_id at all, same "instrumentation-dependent, may be
        # absent" tolerance as trace_id/system_prompt elsewhere.
        run_row = await conn.fetchrow("SELECT conversation_id FROM runs WHERE run_id = $1", run_id)
        conversation_id = run_row["conversation_id"] if run_row else None

    started_at = next((e["timestamp"] for e in events if e["event_type"] == "run.started"), None)
    completed_at = next(
        (e["timestamp"] for e in events if e["event_type"] in ("run.completed", "run.errored")),
        None,
    )

    # Build event list
    event_list = []
    for e in events:
        payload = e["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        event_list.append(
            {
                "event_type": e["event_type"],
                "step_index": e["step_index"],
                "timestamp": e["timestamp"],
                "payload": dict(payload) if payload else {},
                "parent_run_id": e["parent_run_id"],
            }
        )

    # Build signal list with explanations
    signal_list = []
    for s in signals:
        evidence = s["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)

        detected_at = s["detected_at"]
        if hasattr(detected_at, "timestamp"):
            detected_at = detected_at.timestamp()

        exp = None
        try:
            ft = FailureType(s["failure_type"])
            fs = FailureSignal(
                failure_type=ft,
                severity=Severity(s["severity"]),
                run_id=run_id,
                agent_id=dict(pr)["agent_id"],
                agent_version=dict(pr)["agent_version"],
                step_index=s["step_index"],
                confidence=s["confidence"],
                evidence=dict(evidence) if evidence else {},
                detected_at=detected_at,
            )
            exp = explain(fs)
        except ValueError:
            pass  # custom failure types (CUSTOM_*) are not in the FailureType enum
        except Exception as exc:
            logger.error("Explain failed for signal %d: %s", s["id"], exc)

        signal_list.append(
            {
                "id": s["id"],
                "failure_type": s["failure_type"],
                "severity": s["severity"],
                "step_index": s["step_index"],
                "confidence": s["confidence"],
                "detected_at": detected_at,
                "evidence": dict(evidence) if evidence else {},
                "shadow": s["shadow"],
                "title": exp.title if exp else s["failure_type"],
                "what": exp.what if exp else "",
                "why_it_matters": exp.why_it_matters if exp else "",
                "evidence_summary": exp.evidence_summary if exp else "",
                "suggested_fixes": [
                    {
                        "description": f.description,
                        "language": f.language,
                        "code": f.code,
                    }
                    for f in (exp.suggested_fixes if exp else [])
                ],
            }
        )

    llm_responded = [e for e in event_list if e["event_type"] == "llm.responded"]
    llm_called = [e for e in event_list if e["event_type"] == "llm.called"]
    llm_all = llm_called + llm_responded
    # prompt_tokens: direct SDK writes to llm.called; LangChain writes to llm.responded
    prompt_tokens = sum(e["payload"].get("prompt_tokens") or 0 for e in llm_all)
    completion_tokens = sum(e["payload"].get("completion_tokens") or 0 for e in llm_responded)
    total_tokens = (prompt_tokens + completion_tokens) or None
    model = next(
        (e["payload"].get("model") for e in llm_called if e["payload"].get("model")),
        None,
    )
    from explainer_svc.cost import estimate_cost

    raw_cost = (
        estimate_cost(model or "", prompt_tokens, completion_tokens) if total_tokens else None
    )
    cost_usd = round(raw_cost, 6) if raw_cost else None

    pr_dict = dict(pr)
    return {
        "run_id": run_id,
        "agent_id": pr_dict["agent_id"],
        "agent_version": pr_dict["agent_version"],
        "exit_reason": pr_dict["trigger"],
        "started_at": started_at,
        "completed_at": completed_at,
        "step_count": (max((e["step_index"] for e in event_list), default=0) if event_list else 0),
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "events": event_list,
        "signals": signal_list,
        "conversation_id": conversation_id,
    }


# ── Conversations (Phase 3.3) ───────────────────────────────────────────────────


async def get_conversation_detail(org_id: str, conversation_id: int) -> Optional[dict]:
    """Conversation metadata + its ordered run list + conversation-level
    signals. Returns None if the conversation doesn't exist OR belongs to a
    different org.

    conversations/runs are owned by detector_svc (Phase 3.1) — read directly,
    same no-defensive-copy trust relationship as processed_runs above."""
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        conv = await conn.fetchrow(
            """
            SELECT id, agent_id, user_id, external_id, first_run_at, last_run_at, run_count
            FROM conversations WHERE id = $1 AND org_id = $2
            """,
            conversation_id,
            org_id,
        )
        if not conv:
            return None

        runs = await conn.fetch(
            """
            SELECT run_id, agent_version, started_at
            FROM runs WHERE conversation_id = $1
            ORDER BY started_at ASC
            """,
            conversation_id,
        )

        # Generic: any ConversationEvaluator's finding (evidence.conversation_id
        # present), not hardcoded to USER_FRUSTRATION — a future second
        # ConversationEvaluator's signals show up here automatically.
        signals = await conn.fetch(
            """
            SELECT id, failure_type, severity, confidence, detected_at, evidence
            FROM failure_signals
            WHERE org_id = $1 AND evidence ? 'conversation_id'
              AND evidence->>'conversation_id' = $2
            ORDER BY detected_at DESC
            """,
            org_id,
            conv["external_id"],
        )

    return {
        "id": conv["id"],
        "agent_id": conv["agent_id"],
        "user_id": conv["user_id"],
        "external_id": conv["external_id"],
        "first_run_at": conv["first_run_at"].timestamp(),
        "last_run_at": conv["last_run_at"].timestamp(),
        "run_count": conv["run_count"],
        "runs": [
            {
                "run_id": r["run_id"],
                "agent_version": r["agent_version"],
                "started_at": r["started_at"].timestamp(),
            }
            for r in runs
        ],
        "signals": [
            {
                "id": s["id"],
                "failure_type": s["failure_type"],
                "severity": s["severity"],
                "confidence": s["confidence"],
                "detected_at": s["detected_at"].timestamp(),
                "evidence": (
                    _json_mod.loads(s["evidence"])
                    if isinstance(s["evidence"], str)
                    else dict(s["evidence"])
                ),
            }
            for s in signals
        ],
    }


async def search_conversations(
    org_id: str,
    agent_id: Optional[str],
    user_id: Optional[str],
    has_frustration_signal: Optional[bool],
    offset: int,
    limit: int,
) -> tuple[list[dict], int]:
    """Cross-conversation search. All filters optional. has_frustration_signal
    specifically checks for a USER_FRUSTRATION finding (the one
    ConversationEvaluator that exists today) — narrower than
    get_conversation_detail's evaluator-generic signal list, matching what
    the brief actually asked to search by. Same COUNT-then-paginated-query
    shape as list_runs, not a fetch-everything-then-slice-in-Python approach.

    user_id filtering is real, correct plumbing but currently a no-op in
    practice: nothing populates conversations.user_id yet (no SDK parameter
    for it — see BACKLOG.md's Phase 3.1 entry). Kept here so it starts
    working the moment a source for it exists, rather than needing this
    query rewritten later.
    """
    if not _pool:
        return [], 0

    frustration_filter = ""
    if has_frustration_signal is not None:
        frustration_filter = "WHERE scored.has_frustration_signal = $4"

    async with _pool.acquire() as conn:
        count_query = f"""
            WITH scored AS (
                SELECT c.id,
                       EXISTS (
                           SELECT 1 FROM failure_signals fs
                           WHERE fs.org_id = c.org_id
                             AND fs.evidence->>'conversation_id' = c.external_id
                             AND fs.failure_type = 'USER_FRUSTRATION'
                       ) AS has_frustration_signal
                FROM conversations c
                WHERE c.org_id = $1
                  AND ($2::text IS NULL OR c.agent_id = $2)
                  AND ($3::text IS NULL OR c.user_id = $3)
            )
            SELECT COUNT(*) FROM scored {frustration_filter}
        """
        count_args = [org_id, agent_id, user_id]
        if has_frustration_signal is not None:
            count_args.append(has_frustration_signal)
        total = await conn.fetchval(count_query, *count_args)

        page_query = f"""
            WITH scored AS (
                SELECT c.id, c.agent_id, c.user_id, c.external_id, c.last_run_at, c.run_count,
                       EXISTS (
                           SELECT 1 FROM failure_signals fs
                           WHERE fs.org_id = c.org_id
                             AND fs.evidence->>'conversation_id' = c.external_id
                             AND fs.failure_type = 'USER_FRUSTRATION'
                       ) AS has_frustration_signal
                FROM conversations c
                WHERE c.org_id = $1
                  AND ($2::text IS NULL OR c.agent_id = $2)
                  AND ($3::text IS NULL OR c.user_id = $3)
            )
            SELECT * FROM scored {frustration_filter}
            ORDER BY last_run_at DESC
            LIMIT ${len(count_args) + 1} OFFSET ${len(count_args) + 2}
        """
        page_args = list(count_args) + [limit, offset]
        rows = await conn.fetch(page_query, *page_args)

    return (
        [
            {
                "id": r["id"],
                "agent_id": r["agent_id"],
                "user_id": r["user_id"],
                "external_id": r["external_id"],
                "last_run_at": r["last_run_at"].timestamp(),
                "run_count": r["run_count"],
                "has_frustration_signal": r["has_frustration_signal"],
            }
            for r in rows
        ],
        total,
    )


# ── Signals ───────────────────────────────────────────────────────────────────


async def get_signal_by_id(org_id: str, signal_id: int) -> Optional[dict]:
    """Fetch a single signal row by primary key. Returns None if not found or owned by a different org."""
    if not _pool:
        return None

    import json

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, failure_type, severity, run_id, agent_id, agent_version,
                   step_index, confidence, detected_at, evidence, alerted, shadow,
                   COALESCE(co_signal_count, 0) AS co_signal_count,
                   COALESCE(source, 'structural') AS source
            FROM failure_signals
            WHERE id = $1 AND org_id = $2
            """,
            signal_id,
            org_id,
        )

    if row is None:
        return None

    evidence = row["evidence"]
    if isinstance(evidence, str):
        evidence = json.loads(evidence)

    detected_at = row["detected_at"]
    if hasattr(detected_at, "timestamp"):
        detected_at = detected_at.timestamp()

    try:
        fs = FailureSignal(
            failure_type=FailureType(row["failure_type"]),
            severity=Severity(row["severity"]),
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            agent_version=row["agent_version"],
            step_index=row["step_index"],
            confidence=row["confidence"],
            evidence=dict(evidence) if evidence else {},
            detected_at=detected_at,
        )
        exp = explain(fs)
    except Exception as exc:
        logger.error("Explain failed signal %d: %s", signal_id, exc)
        exp = None

    return {
        "id": row["id"],
        "failure_type": row["failure_type"],
        "severity": row["severity"],
        "run_id": row["run_id"],
        "agent_id": row["agent_id"],
        "agent_version": row["agent_version"],
        "step_index": row["step_index"],
        "confidence": row["confidence"],
        "detected_at": detected_at,
        "evidence": dict(evidence) if evidence else {},
        "alerted": row["alerted"],
        "shadow": row["shadow"],
        "co_signal_count": row["co_signal_count"],
        "source": row["source"],
        "title": exp.title if exp else row["failure_type"],
        "what": exp.what if exp else "",
        "why_it_matters": exp.why_it_matters if exp else "",
        "evidence_summary": exp.evidence_summary if exp else "",
        "suggested_fixes": [
            {"description": f.description, "language": f.language, "code": f.code}
            for f in (exp.suggested_fixes if exp else [])
        ],
    }


async def list_signals(
    org_id: str,
    agent_id: str,
    offset: int,
    limit: int,
    severity: Optional[str] = None,
    failure_type: Optional[str] = None,
    include_shadow: bool = False,
) -> tuple[list, int]:
    """List signals for an agent with optional filters. By default only live (non-shadow) signals are returned; pass include_shadow=True to include shadow signals too."""
    if not _pool:
        return [], 0

    import json

    where = ["org_id = $1", "agent_id = $2"]
    if not include_shadow:
        where.append("shadow = FALSE")
    params: list = [org_id, agent_id]

    if severity:
        params.append(severity.upper())
        where.append(f"severity = ${len(params)}")
    if failure_type:
        params.append(failure_type.upper())
        where.append(f"failure_type = ${len(params)}")

    where_clause = " AND ".join(where)

    async with _pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM failure_signals WHERE {where_clause}",
            *params,
        )

        params_paged = params + [limit, offset]
        rows = await conn.fetch(
            f"""
            SELECT id, failure_type, severity, run_id, agent_id, agent_version,
                   step_index, confidence, detected_at, evidence, alerted, shadow,
                   COALESCE(co_signal_count, 0) AS co_signal_count
            FROM failure_signals
            WHERE {where_clause}
            ORDER BY detected_at DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """,
            *params_paged,
        )

    results = []
    for s in rows:
        evidence = s["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)

        detected_at = s["detected_at"]
        if hasattr(detected_at, "timestamp"):
            detected_at = detected_at.timestamp()

        try:
            fs = FailureSignal(
                failure_type=FailureType(s["failure_type"]),
                severity=Severity(s["severity"]),
                run_id=s["run_id"],
                agent_id=s["agent_id"],
                agent_version=s["agent_version"],
                step_index=s["step_index"],
                confidence=s["confidence"],
                evidence=dict(evidence) if evidence else {},
                detected_at=detected_at,
            )
            exp = explain(fs)
        except Exception as exc:
            logger.error("Explain failed signal %d: %s", s["id"], exc)
            exp = None

        results.append(
            {
                "id": s["id"],
                "failure_type": s["failure_type"],
                "severity": s["severity"],
                "run_id": s["run_id"],
                "agent_id": s["agent_id"],
                "agent_version": s["agent_version"],
                "step_index": s["step_index"],
                "confidence": s["confidence"],
                "detected_at": detected_at,
                "evidence": dict(evidence) if evidence else {},
                "alerted": s["alerted"],
                "shadow": s["shadow"],
                "co_signal_count": s["co_signal_count"],
                "title": exp.title if exp else s["failure_type"],
                "what": exp.what if exp else "",
                "why_it_matters": exp.why_it_matters if exp else "",
                "evidence_summary": exp.evidence_summary if exp else "",
                "suggested_fixes": [
                    {
                        "description": f.description,
                        "language": f.language,
                        "code": f.code,
                    }
                    for f in (exp.suggested_fixes if exp else [])
                ],
            }
        )

    return results, total or 0


async def export_signals(
    org_id: str,
    agent_id: str,
    severity: Optional[str] = None,
    failure_type: Optional[str] = None,
    include_shadow: bool = False,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    batch_size: int = 500,
) -> AsyncGenerator[list, None]:
    """
    Async generator that yields batches of raw signal rows for streaming export.

    Uses keyset pagination on (detected_at DESC, id DESC) so performance is
    stable regardless of result set size — no OFFSET scan.

    Each yielded batch is a list of dicts with fields:
      id, failure_type, severity, run_id, agent_id, agent_version,
      step_index, confidence, detected_at (ISO-8601 UTC), evidence (dict)
    """
    if not _pool:
        return

    import json

    where = ["org_id = $1", "agent_id = $2"]
    params: list = [org_id, agent_id]

    if not include_shadow:
        where.append("shadow = FALSE")
    if severity:
        params.append(severity.upper())
        where.append(f"severity = ${len(params)}")
    if failure_type:
        params.append(failure_type.upper())
        where.append(f"failure_type = ${len(params)}")
    if from_ts is not None:
        params.append(datetime.datetime.fromtimestamp(from_ts, tz=datetime.timezone.utc))
        where.append(f"detected_at >= ${len(params)}")
    if to_ts is not None:
        params.append(datetime.datetime.fromtimestamp(to_ts, tz=datetime.timezone.utc))
        where.append(f"detected_at <= ${len(params)}")

    base_where = " AND ".join(where)
    # Keyset cursor added per-batch: (detected_at, id) < (cursor_ts, cursor_id)
    cursor_ts: Optional[datetime.datetime] = None
    cursor_id: Optional[int] = None

    while True:
        async with _pool.acquire() as conn:
            if cursor_ts is None:
                query = f"""
                    SELECT id, failure_type, severity, run_id, agent_id, agent_version,
                           step_index, confidence, detected_at, evidence
                    FROM failure_signals
                    WHERE {base_where}
                    ORDER BY detected_at DESC, id DESC
                    LIMIT {batch_size}
                """
                rows = await conn.fetch(query, *params)
            else:
                keyset_params = params + [cursor_ts, cursor_id]
                query = f"""
                    SELECT id, failure_type, severity, run_id, agent_id, agent_version,
                           step_index, confidence, detected_at, evidence
                    FROM failure_signals
                    WHERE {base_where}
                      AND (detected_at < ${len(params) + 1}
                           OR (detected_at = ${len(params) + 1} AND id < ${len(params) + 2}))
                    ORDER BY detected_at DESC, id DESC
                    LIMIT {batch_size}
                """
                rows = await conn.fetch(query, *keyset_params)

        if not rows:
            break

        batch = []
        for s in rows:
            evidence = s["evidence"]
            if isinstance(evidence, str):
                evidence = json.loads(evidence)
            detected_at = s["detected_at"]
            if hasattr(detected_at, "isoformat"):
                detected_at_str = detected_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            else:
                detected_at_str = str(detected_at)
            batch.append(
                {
                    "id": s["id"],
                    "failure_type": s["failure_type"],
                    "severity": s["severity"],
                    "run_id": s["run_id"],
                    "agent_id": s["agent_id"],
                    "agent_version": s["agent_version"],
                    "step_index": s["step_index"],
                    "confidence": round(float(s["confidence"]), 4),
                    "detected_at": detected_at_str,
                    "evidence": dict(evidence) if evidence else {},
                }
            )

        yield batch

        last = rows[-1]
        cursor_ts = last["detected_at"]
        cursor_id = last["id"]

        if len(rows) < batch_size:
            break


# ── Insights ───────────────────────────────────────────────────────────────────


async def agent_input_hash_patterns(org_id: str, agent_id: str) -> list:
    """Input texts that consistently produce specific failure types, grouped by an
    MD5 digest of the raw input (computed here, not transmitted) so identical inputs
    dedupe without the response carrying raw text. Only hashes seen ≥2 times so a
    single bad run doesn't dominate. Returns: [{input_hash, failure_type, triggered_count, total_runs, rate}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH run_inputs AS (
                SELECT e.run_id, md5(e.payload->>'input_text') AS input_hash
                FROM events e
                WHERE e.org_id = $1
                  AND e.agent_id = $2
                  AND e.event_type = 'run.started'
                  AND e.payload->>'input_text' IS NOT NULL
            ),
            hash_totals AS (
                SELECT input_hash, COUNT(DISTINCT run_id) AS total_runs
                FROM run_inputs
                GROUP BY input_hash
                HAVING COUNT(DISTINCT run_id) >= 2
            ),
            hash_signals AS (
                SELECT ri.input_hash, fs.failure_type,
                       COUNT(DISTINCT fs.run_id) AS triggered_count
                FROM run_inputs ri
                JOIN failure_signals fs ON fs.run_id = ri.run_id
                WHERE fs.shadow = FALSE AND fs.org_id = $1 AND fs.agent_id = $2
                GROUP BY ri.input_hash, fs.failure_type
            )
            SELECT
                hs.input_hash,
                hs.failure_type,
                hs.triggered_count::int,
                ht.total_runs::int,
                ROUND(hs.triggered_count::numeric / ht.total_runs, 2) AS rate
            FROM hash_signals hs
            JOIN hash_totals ht ON ht.input_hash = hs.input_hash
            ORDER BY rate DESC, triggered_count DESC
            LIMIT 20
            """,
            org_id,
            agent_id,
        )
    return [dict(r) for r in rows]


async def agent_signal_recurrence(org_id: str, agent_id: str) -> list:
    """Signal counts by failure_type × agent_version × day for the last 30 days. Returns: [{failure_type, agent_version, day (ISO str), count}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                failure_type,
                agent_version,
                DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC')::date AS day,
                COUNT(*) AS count
            FROM failure_signals
            WHERE org_id = $1
              AND agent_id = $2
              AND shadow = FALSE
              AND detected_at >= NOW() - INTERVAL '30 days'
            GROUP BY failure_type, agent_version, day
            ORDER BY day DESC, failure_type, agent_version
            LIMIT 300
            """,
            org_id,
            agent_id,
        )
    return [{**dict(r), "day": str(r["day"])} for r in rows]


async def agent_version_stats(org_id: str, agent_id: str) -> list:
    """Signal rate per version (runs_with_signals / total_runs), newest first. Returns: [{agent_version, run_count, runs_with_signals, signal_count, signal_rate, first_seen, last_seen}]."""
    if not _pool:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                pr.agent_version,
                COUNT(DISTINCT pr.run_id)                                              AS run_count,
                COUNT(DISTINCT fs.run_id) FILTER (WHERE fs.id IS NOT NULL)             AS runs_with_signals,
                COUNT(fs.id)                                                           AS signal_count,
                ROUND(
                    COUNT(DISTINCT fs.run_id) FILTER (WHERE fs.id IS NOT NULL)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0),
                    3
                )                                                                      AS signal_rate,
                MIN(pr.processed_at) AS first_seen,
                MAX(pr.processed_at) AS last_seen
            FROM processed_runs pr
            LEFT JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.org_id = pr.org_id
                AND fs.agent_id = pr.agent_id AND fs.shadow = FALSE
            WHERE pr.org_id = $1 AND pr.agent_id = $2
            GROUP BY pr.agent_version
            ORDER BY MAX(pr.processed_at) DESC
            LIMIT 10
            """,
            org_id,
            agent_id,
        )
    return [
        {
            "agent_version": r["agent_version"],
            "run_count": int(r["run_count"]),
            "runs_with_signals": int(r["runs_with_signals"]),
            "signal_count": int(r["signal_count"]),
            "signal_rate": float(r["signal_rate"] or 0),
            "first_seen": _ts(r["first_seen"]),
            "last_seen": _ts(r["last_seen"]),
        }
        for r in rows
    ]


async def agent_time_to_first_tool(org_id: str, agent_id: str) -> dict:
    """Steps before the first tool call — overall P25/P50/P75 plus a 14-day daily trend. Returns: {p25, p50, p75, avg_steps, runs_with_tool, total_runs, daily_trend}."""
    if not _pool:
        return {
            "p25": None,
            "p50": None,
            "p75": None,
            "avg_steps": None,
            "runs_with_tool": 0,
            "total_runs": 0,
            "daily_trend": [],
        }
    async with _pool.acquire() as conn:
        overall = await conn.fetchrow(
            """
            WITH first_tool AS (
                SELECT run_id, MIN(step_index) AS first_tool_step
                FROM events
                WHERE org_id = $1 AND agent_id = $2 AND event_type = 'tool.called'
                GROUP BY run_id
            )
            SELECT
                COUNT(pr.run_id)                                                      AS total_runs,
                COUNT(ft.run_id)                                                      AS runs_with_tool,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ft.first_tool_step)      AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ft.first_tool_step)      AS p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ft.first_tool_step)      AS p75,
                ROUND(AVG(ft.first_tool_step), 1)                                     AS avg_steps
            FROM processed_runs pr
            LEFT JOIN first_tool ft ON ft.run_id = pr.run_id
            WHERE pr.org_id = $1 AND pr.agent_id = $2
            """,
            org_id,
            agent_id,
        )
        daily = await conn.fetch(
            """
            WITH first_tool AS (
                SELECT run_id, MIN(step_index) AS first_tool_step
                FROM events
                WHERE org_id = $1 AND agent_id = $2 AND event_type = 'tool.called'
                GROUP BY run_id
            )
            SELECT
                DATE_TRUNC('day', pr.processed_at AT TIME ZONE 'UTC')::date AS day,
                COUNT(pr.run_id)              AS run_count,
                COUNT(ft.run_id)              AS runs_with_tool,
                ROUND(AVG(ft.first_tool_step), 1) AS avg_first_tool_step
            FROM processed_runs pr
            LEFT JOIN first_tool ft ON ft.run_id = pr.run_id
            WHERE pr.org_id = $1 AND pr.agent_id = $2
              AND pr.processed_at >= NOW() - INTERVAL '14 days'
            GROUP BY day
            ORDER BY day
            """,
            org_id,
            agent_id,
        )
    return {
        "p25": float(overall["p25"]) if overall["p25"] is not None else None,
        "p50": float(overall["p50"]) if overall["p50"] is not None else None,
        "p75": float(overall["p75"]) if overall["p75"] is not None else None,
        "avg_steps": (float(overall["avg_steps"]) if overall["avg_steps"] is not None else None),
        "runs_with_tool": int(overall["runs_with_tool"]),
        "total_runs": int(overall["total_runs"]),
        "daily_trend": [
            {
                "day": str(r["day"]),
                "run_count": int(r["run_count"]),
                "runs_with_tool": int(r["runs_with_tool"]),
                "avg_first_tool_step": (
                    float(r["avg_first_tool_step"])
                    if r["avg_first_tool_step"] is not None
                    else None
                ),
            }
            for r in daily
        ],
    }


async def agent_hourly_pattern(org_id: str, agent_id: str) -> list:
    """Signal rate by UTC hour of day over the last 30 days. Sparse — only hours with ≥1 run are returned; the UI fills gaps. Returns: [{hour_of_day, run_count, signal_count, signal_rate}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                EXTRACT(HOUR FROM pr.processed_at AT TIME ZONE 'UTC')::int AS hour_of_day,
                COUNT(DISTINCT pr.run_id)                                   AS run_count,
                COUNT(DISTINCT fs.run_id)                                   AS signal_count,
                ROUND(
                    COUNT(DISTINCT fs.run_id)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0),
                    3
                )                                                           AS signal_rate
            FROM processed_runs pr
            LEFT JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.org_id = pr.org_id
                AND fs.agent_id = pr.agent_id AND fs.shadow = FALSE
            WHERE pr.org_id = $1 AND pr.agent_id = $2
              AND pr.processed_at >= NOW() - INTERVAL '30 days'
            GROUP BY hour_of_day
            ORDER BY hour_of_day
            """,
            org_id,
            agent_id,
        )
    return [
        {
            "hour_of_day": int(r["hour_of_day"]),
            "run_count": int(r["run_count"]),
            "signal_count": int(r["signal_count"]),
            "signal_rate": float(r["signal_rate"] or 0),
        }
        for r in rows
    ]


async def list_issues(org_id: str, agent_id: str, status: Optional[str] = None) -> list:
    """Return persistent issues for an agent, ordered: open → reopened → resolved, then by last_seen desc."""
    if not _pool:
        return []

    where = "WHERE org_id = $1 AND agent_id = $2"
    params: list = [org_id, agent_id]
    if status:
        params.append(status.lower())
        where += f" AND status = ${len(params)}"

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, agent_id, failure_type, status,
                   first_seen, last_seen, resolved_at,
                   affected_runs, clean_runs_since
            FROM issues
            {where}
            ORDER BY
                CASE status WHEN 'open' THEN 0 WHEN 'reopened' THEN 1 ELSE 2 END,
                last_seen DESC
            """,
            *params,
        )
    return [
        {
            "id": r["id"],
            "agent_id": r["agent_id"],
            "failure_type": r["failure_type"],
            "status": r["status"],
            "first_seen": _ts(r["first_seen"]),
            "last_seen": _ts(r["last_seen"]),
            "resolved_at": _ts(r["resolved_at"]),
            "affected_runs": int(r["affected_runs"]),
            "clean_runs_since": int(r["clean_runs_since"]),
        }
        for r in rows
    ]


# ── Single-issue lookup, search, manual resolve (Phase 4.2 — MCP tools) ─────────


async def get_issue_by_id(org_id: str, issue_id: int) -> Optional[dict]:
    """Single issue by its own id, org-scoped — no such lookup existed
    before Phase 4.2 (list_issues above is agent-scoped listing only)."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, agent_id, failure_type, status,
                   first_seen, last_seen, resolved_at,
                   affected_runs, clean_runs_since,
                   resolution_notes, manually_resolved
            FROM issues
            WHERE id = $1 AND org_id = $2
            """,
            issue_id,
            org_id,
        )
    if row is None:
        return None
    return {
        "id": row["id"],
        "agent_id": row["agent_id"],
        "failure_type": row["failure_type"],
        "status": row["status"],
        "first_seen": _ts(row["first_seen"]),
        "last_seen": _ts(row["last_seen"]),
        "resolved_at": _ts(row["resolved_at"]),
        "affected_runs": int(row["affected_runs"]),
        "clean_runs_since": int(row["clean_runs_since"]),
        "resolution_notes": row["resolution_notes"],
        "manually_resolved": row["manually_resolved"],
    }


async def search_issues(
    org_id: str,
    q: str = "",
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
    failure_type: Optional[str] = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[dict], int]:
    """Cross-agent issue search. `q` is a plain substring match across
    agent_id/failure_type/resolution_notes — issues have no free-text
    title/description field, so this is not full-text search or relevance
    ranking, just a simple filter. Same COUNT-then-paginated-query shape as
    list_runs/search_conversations."""
    if not _pool:
        return [], 0

    where = ["org_id = $1"]
    params: list = [org_id]
    if q:
        params.append(f"%{q}%")
        where.append(
            f"(agent_id ILIKE ${len(params)} OR failure_type ILIKE ${len(params)} "
            f"OR resolution_notes ILIKE ${len(params)})"
        )
    if status:
        params.append(status.lower())
        where.append(f"status = ${len(params)}")
    if agent_id:
        params.append(agent_id)
        where.append(f"agent_id = ${len(params)}")
    if failure_type:
        params.append(failure_type.upper())
        where.append(f"failure_type = ${len(params)}")
    where_clause = " AND ".join(where)

    async with _pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM issues WHERE {where_clause}", *params)

        rows = await conn.fetch(
            f"""
            SELECT id, agent_id, failure_type, status,
                   first_seen, last_seen, resolved_at,
                   affected_runs, clean_runs_since,
                   resolution_notes, manually_resolved
            FROM issues
            WHERE {where_clause}
            ORDER BY last_seen DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """,
            *params,
            limit,
            offset,
        )

    return (
        [
            {
                "id": r["id"],
                "agent_id": r["agent_id"],
                "failure_type": r["failure_type"],
                "status": r["status"],
                "first_seen": _ts(r["first_seen"]),
                "last_seen": _ts(r["last_seen"]),
                "resolved_at": _ts(r["resolved_at"]),
                "affected_runs": int(r["affected_runs"]),
                "clean_runs_since": int(r["clean_runs_since"]),
                "resolution_notes": r["resolution_notes"],
                "manually_resolved": r["manually_resolved"],
            }
            for r in rows
        ],
        total,
    )


async def resolve_issue_manually(org_id: str, issue_id: int, resolution_notes: str) -> bool:
    """Manual resolve (Phase 4.2's resolve_issue MCP tool) — orthogonal to
    the existing auto-resolve-after-N-clean-runs mechanism (unchanged): a
    manually-resolved issue still reopens if the failure recurs later, same
    as an auto-resolved one. Returns False if the issue doesn't exist or
    belongs to a different org."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE issues
            SET status = 'resolved',
                resolved_at = NOW(),
                resolution_notes = $1,
                manually_resolved = TRUE
            WHERE id = $2 AND org_id = $3
            """,
            resolution_notes,
            issue_id,
            org_id,
        )
    return result != "UPDATE 0"


async def get_most_recent_signal_id(org_id: str, agent_id: str, failure_type: str) -> Optional[int]:
    """The most recent failure_signals row for this (agent_id, failure_type)
    pair — used to anchor get_issue's root-cause analysis onto a real
    signal, since native_explain operates on individual signals, not the
    aggregated issues row."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT id FROM failure_signals
            WHERE org_id = $1 AND agent_id = $2 AND failure_type = $3
            ORDER BY detected_at DESC
            LIMIT 1
            """,
            org_id,
            agent_id,
            failure_type,
        )


async def agent_failure_rates(org_id: str, agent_id: str) -> list:
    """Daily failure rate per failure_type over 30 days — affected_runs / total_runs.
    Returns: [{failure_type, day (ISO str), total_runs, affected_runs, rate}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH per_day AS (
                SELECT
                    DATE_TRUNC('day', processed_at AT TIME ZONE 'UTC')::date AS day,
                    COUNT(DISTINCT run_id) AS total_runs
                FROM processed_runs
                WHERE org_id = $1 AND agent_id = $2
                  AND processed_at >= NOW() - INTERVAL '30 days'
                GROUP BY day
            ),
            per_day_type AS (
                SELECT
                    DATE_TRUNC('day', pr.processed_at AT TIME ZONE 'UTC')::date AS day,
                    fs.failure_type,
                    COUNT(DISTINCT fs.run_id)::int AS affected_runs
                FROM processed_runs pr
                JOIN failure_signals fs
                    ON fs.run_id    = pr.run_id
                    AND fs.org_id   = pr.org_id
                    AND fs.agent_id = pr.agent_id
                    AND fs.shadow   = FALSE
                WHERE pr.org_id = $1 AND pr.agent_id = $2
                  AND pr.processed_at >= NOW() - INTERVAL '30 days'
                GROUP BY day, fs.failure_type
            )
            SELECT
                pdt.failure_type,
                pdt.day,
                pd.total_runs::int,
                pdt.affected_runs,
                ROUND(pdt.affected_runs::numeric / NULLIF(pd.total_runs, 0), 3) AS rate
            FROM per_day_type pdt
            JOIN per_day pd ON pd.day = pdt.day
            ORDER BY pdt.day DESC, pdt.affected_runs DESC
            LIMIT 300
            """,
            org_id,
            agent_id,
        )
    return [
        {
            "failure_type": r["failure_type"],
            "day": str(r["day"]),
            "total_runs": int(r["total_runs"]),
            "affected_runs": int(r["affected_runs"]),
            "rate": float(r["rate"] or 0),
        }
        for r in rows
    ]


async def agent_systemic_patterns(org_id: str, agent_id: str, rate_threshold: float = 0.10) -> list:
    """Failure types firing on >= rate_threshold of runs in the last 7 days.
    Returns: [{failure_type, total_runs, affected_runs, rate, first_seen, last_seen, is_systemic}].
    """
    if not _pool:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH total AS (
                SELECT COUNT(DISTINCT run_id) AS total_runs
                FROM processed_runs
                WHERE org_id = $1 AND agent_id = $2
                  AND processed_at >= NOW() - INTERVAL '7 days'
            )
            SELECT
                fs.failure_type,
                (SELECT total_runs FROM total)::int AS total_runs,
                COUNT(DISTINCT fs.run_id)::int       AS affected_runs,
                ROUND(
                    COUNT(DISTINCT fs.run_id)::numeric
                    / NULLIF((SELECT total_runs FROM total), 0),
                    3
                )                                   AS rate,
                MIN(fs.detected_at)                 AS first_seen,
                MAX(fs.detected_at)                 AS last_seen
            FROM failure_signals fs
            WHERE fs.org_id  = $1
              AND fs.agent_id = $2
              AND fs.shadow   = FALSE
              AND fs.run_id IN (
                  SELECT run_id FROM processed_runs
                  WHERE org_id = $1 AND agent_id = $2
                    AND processed_at >= NOW() - INTERVAL '7 days'
              )
            GROUP BY fs.failure_type
            ORDER BY affected_runs DESC
            """,
            org_id,
            agent_id,
        )
    return [
        {
            "failure_type": r["failure_type"],
            "total_runs": int(r["total_runs"]),
            "affected_runs": int(r["affected_runs"]),
            "rate": float(r["rate"] or 0),
            "first_seen": _ts(r["first_seen"]),
            "last_seen": _ts(r["last_seen"]),
            "is_systemic": float(r["rate"] or 0) >= rate_threshold,
        }
        for r in rows
    ]


async def agent_deploy_events(org_id: str, agent_id: str) -> list:
    """Deploy markers for an agent over the last 90 days.
    Returns: [{id, version, deployed_at (unix float), meta}]."""
    import json as _json

    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, version, deployed_at, meta
            FROM deploy_events
            WHERE org_id = $1 AND agent_id = $2
              AND deployed_at >= NOW() - INTERVAL '90 days'
            ORDER BY deployed_at ASC
            """,
            org_id,
            agent_id,
        )

    def _meta(v):
        if not v:
            return {}
        if isinstance(v, str):
            return _json.loads(v)
        return dict(v)

    return [
        {
            "id": int(r["id"]),
            "version": r["version"],
            "deployed_at": _ts(r["deployed_at"]),
            "meta": _meta(r["meta"]),
        }
        for r in rows
    ]


async def agent_failure_pattern(org_id: str, agent_id: str, failure_type: str) -> dict:
    """
    Cross-run deep-dive for one failure type.

    Returns overview stats, step distribution, evidence aggregates,
    14-day daily trend, co-occurring failure types, and top affected runs.
    """
    if not _pool:
        return {}

    def _ts(v):
        return v.isoformat() if v else None

    async with _pool.acquire() as conn:
        # 1. Overview
        overview_row = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT run_id)                          AS affected_runs,
                ROUND(AVG(confidence)::numeric, 3)              AS avg_confidence,
                MIN(detected_at)                                AS first_seen,
                MAX(detected_at)                                AS last_seen,
                COUNT(*) FILTER (WHERE severity = 'CRITICAL')  AS critical_count,
                COUNT(*) FILTER (WHERE severity = 'HIGH')       AS high_count,
                COUNT(*) FILTER (WHERE severity = 'MEDIUM')     AS medium_count,
                COUNT(*) FILTER (WHERE severity = 'LOW')        AS low_count
            FROM failure_signals
            WHERE org_id      = $1
              AND agent_id    = $2
              AND failure_type = $3
              AND shadow       = FALSE
            """,
            org_id,
            agent_id,
            failure_type,
        )

        total_row = await conn.fetchrow(
            "SELECT COUNT(DISTINCT run_id) AS total FROM processed_runs WHERE org_id = $1 AND agent_id = $2",
            org_id,
            agent_id,
        )

        # 2. Step distribution (where in the run does this fire)
        step_row = await conn.fetchrow(
            """
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY step_index) AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY step_index) AS p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY step_index) AS p75,
                ROUND(AVG(step_index)::numeric, 1)                        AS avg_step
            FROM failure_signals
            WHERE org_id      = $1
              AND agent_id    = $2
              AND failure_type = $3
              AND shadow       = FALSE
            """,
            org_id,
            agent_id,
            failure_type,
        )

        # 3. Evidence aggregates — extract the most useful keys per failure type
        evidence_rows = await conn.fetch(
            """
            SELECT
                -- Tool-based failures (TOOL_LOOP, RETRY_STORM, CASCADING_TOOL_FAILURE)
                evidence->>'tool'                                        AS tool,

                -- Loop / repetition
                ROUND(AVG((evidence->>'count')::numeric), 1)            AS avg_count,
                ROUND(AVG((evidence->>'consecutive_fails')::numeric), 1) AS avg_consecutive_fails,
                ROUND(AVG(
                    CASE WHEN evidence->>'args_identical' = 'true' THEN 1.0 ELSE 0.0 END
                ), 2)                                                    AS args_identical_rate,

                -- Context / token growth
                ROUND(AVG((evidence->>'growth_factor')::numeric), 2)    AS avg_growth_factor,
                ROUND(AVG((evidence->>'first_tokens')::numeric), 0)     AS avg_first_tokens,
                ROUND(AVG((evidence->>'last_tokens')::numeric), 0)      AS avg_last_tokens,

                -- Slow step
                ROUND(AVG((evidence->>'duration_ms')::numeric), 0)      AS avg_duration_ms,
                ROUND(AVG((evidence->>'threshold_ms')::numeric), 0)     AS avg_threshold_ms,
                ROUND(AVG((evidence->>'ratio')::numeric), 2)            AS avg_ratio,

                -- RAG
                evidence->>'index_name'                                  AS index_name,
                ROUND(AVG((evidence->>'top_score')::numeric), 3)        AS avg_top_score,
                ROUND(AVG((evidence->>'result_count')::numeric), 1)     AS avg_result_count,

                -- Step count inflation
                ROUND(AVG((evidence->>'inflation_ratio')::numeric), 2)  AS avg_inflation_ratio,
                ROUND(AVG((evidence->>'baseline_p75')::numeric), 1)     AS avg_baseline_p75,

                -- Reasoning / goal abandonment
                ROUND(AVG((evidence->>'stall_steps')::numeric), 1)      AS avg_stall_steps,

                COUNT(*) AS sample_count
            FROM failure_signals
            WHERE org_id      = $1
              AND agent_id    = $2
              AND failure_type = $3
              AND shadow       = FALSE
            GROUP BY
                evidence->>'tool',
                evidence->>'index_name'
            ORDER BY sample_count DESC
            LIMIT 10
            """,
            org_id,
            agent_id,
            failure_type,
        )

        # 4. Daily trend (14 days)
        trend_rows = await conn.fetch(
            """
            WITH days AS (
                SELECT generate_series(
                    DATE_TRUNC('day', NOW() - INTERVAL '13 days'),
                    DATE_TRUNC('day', NOW()),
                    INTERVAL '1 day'
                )::date AS day
            ),
            daily_affected AS (
                SELECT
                    DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC')::date AS day,
                    COUNT(DISTINCT run_id) AS affected_runs
                FROM failure_signals
                WHERE org_id      = $1
                  AND agent_id    = $2
                  AND failure_type = $3
                  AND shadow       = FALSE
                  AND detected_at >= NOW() - INTERVAL '14 days'
                GROUP BY 1
            ),
            daily_total AS (
                SELECT
                    DATE_TRUNC('day', processed_at AT TIME ZONE 'UTC')::date AS day,
                    COUNT(DISTINCT run_id) AS total_runs
                FROM processed_runs
                WHERE org_id      = $1
                  AND agent_id    = $2
                  AND processed_at >= NOW() - INTERVAL '14 days'
                GROUP BY 1
            )
            SELECT
                d.day,
                COALESCE(da.affected_runs, 0)  AS affected_runs,
                COALESCE(dt.total_runs, 0)     AS total_runs,
                CASE WHEN COALESCE(dt.total_runs, 0) > 0
                     THEN ROUND(COALESCE(da.affected_runs, 0)::numeric / dt.total_runs, 3)
                     ELSE 0 END                AS rate
            FROM days d
            LEFT JOIN daily_affected da ON da.day = d.day
            LEFT JOIN daily_total    dt ON dt.day = d.day
            ORDER BY d.day
            """,
            org_id,
            agent_id,
            failure_type,
        )

        # 5. Co-occurring failure types (same runs, last 30 days)
        co_rows = await conn.fetch(
            """
            WITH affected AS (
                SELECT DISTINCT run_id
                FROM failure_signals
                WHERE org_id      = $1
                  AND agent_id    = $2
                  AND failure_type = $3
                  AND shadow       = FALSE
                  AND detected_at >= NOW() - INTERVAL '30 days'
            )
            SELECT
                fs.failure_type,
                COUNT(DISTINCT fs.run_id)                                                   AS co_count,
                -- Denominator is total affected runs, not the per-group co-count.
                -- Using scalar subquery to avoid the JOIN collapsing the count to 1.0.
                ROUND(COUNT(DISTINCT fs.run_id)::numeric / NULLIF((SELECT COUNT(*) FROM affected), 0), 3) AS co_rate
            FROM affected a
            JOIN failure_signals fs ON fs.run_id = a.run_id
            WHERE fs.org_id      = $1
              AND fs.agent_id    = $2
              AND fs.failure_type != $3
              AND fs.shadow       = FALSE
            GROUP BY fs.failure_type
            ORDER BY co_rate DESC
            LIMIT 8
            """,
            org_id,
            agent_id,
            failure_type,
        )

        # 6. Top affected runs (highest confidence, most recent)
        run_rows = await conn.fetch(
            """
            SELECT
                run_id,
                confidence,
                severity,
                step_index,
                detected_at,
                evidence
            FROM failure_signals
            WHERE org_id      = $1
              AND agent_id    = $2
              AND failure_type = $3
              AND shadow       = FALSE
            ORDER BY confidence DESC, detected_at DESC
            LIMIT 5
            """,
            org_id,
            agent_id,
            failure_type,
        )

    total_runs = int(total_row["total"] or 0)
    affected_runs = int(overview_row["affected_runs"] or 0)

    return {
        "failure_type": failure_type,
        "overview": {
            "affected_runs": affected_runs,
            "total_runs": total_runs,
            "rate": round(affected_runs / total_runs, 3) if total_runs else 0.0,
            "avg_confidence": float(overview_row["avg_confidence"] or 0),
            "first_seen": _ts(overview_row["first_seen"]),
            "last_seen": _ts(overview_row["last_seen"]),
            "severity_breakdown": {
                "CRITICAL": int(overview_row["critical_count"] or 0),
                "HIGH": int(overview_row["high_count"] or 0),
                "MEDIUM": int(overview_row["medium_count"] or 0),
                "LOW": int(overview_row["low_count"] or 0),
            },
        },
        "step_distribution": {
            "p25": float(step_row["p25"]) if step_row["p25"] is not None else None,
            "p50": float(step_row["p50"]) if step_row["p50"] is not None else None,
            "p75": float(step_row["p75"]) if step_row["p75"] is not None else None,
            "avg_step": (float(step_row["avg_step"]) if step_row["avg_step"] is not None else None),
        },
        "evidence_aggregates": [
            {k: v for k, v in dict(r).items() if v is not None} for r in evidence_rows
        ],
        "daily_trend": [
            {
                "day": str(r["day"]),
                "affected_runs": int(r["affected_runs"]),
                "total_runs": int(r["total_runs"]),
                "rate": float(r["rate"] or 0),
            }
            for r in trend_rows
        ],
        "co_occurring": [
            {
                "failure_type": r["failure_type"],
                "co_count": int(r["co_count"]),
                "co_rate": float(r["co_rate"] or 0),
            }
            for r in co_rows
        ],
        "top_runs": [
            {
                "run_id": r["run_id"],
                "confidence": float(r["confidence"]),
                "severity": r["severity"],
                "step_index": int(r["step_index"]),
                "detected_at": _ts(r["detected_at"]),
                "evidence": dict(r["evidence"]),
            }
            for r in run_rows
        ],
    }


async def record_fix(
    org_id: str,
    signal_id: int,
    run_id: str,
    fix_content: str,
    applied_via: str,
    langfuse_prompt_name: Optional[str] = None,
    langfuse_version: Optional[int] = None,
) -> Optional[int]:
    """Insert a fix record and return its id. Returns None if pool unavailable."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO fixes (run_id, signal_id, fix_content, applied_via,
                               langfuse_prompt_name, langfuse_version, org_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            run_id,
            signal_id,
            fix_content,
            applied_via,
            langfuse_prompt_name,
            langfuse_version,
            org_id,
        )
    return int(row["id"]) if row else None


# ── Semantic feedback (Phase 1.4.3) ────────────────────────────────────────────


async def get_organization_semantic_feedback(org_id: str) -> Optional[dict]:
    """Returns {enabled, auto_suppress} for an org's feedback-loop settings, or
    None if the org doesn't exist (shouldn't happen for an org that passed
    require_org, but defensive)."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT semantic_feedback_enabled, semantic_feedback_auto_suppress "
            "FROM organizations WHERE id = $1",
            org_id,
        )
    if row is None:
        return None
    return {
        "enabled": row["semantic_feedback_enabled"],
        "auto_suppress": row["semantic_feedback_auto_suppress"],
    }


async def update_organization_semantic_feedback(
    org_id: str, enabled: bool, auto_suppress: bool
) -> None:
    """Toggle an org's opt-in feedback-loop settings (dashboard-facing)."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE organizations SET semantic_feedback_enabled = $2, "
            "semantic_feedback_auto_suppress = $3 WHERE id = $1",
            org_id,
            enabled,
            auto_suppress,
        )


async def get_org_semantic_usage(org_id: str, month: str) -> dict:
    """Returns {quota, allow_overage, used_this_month, cost_so_far_usd} for
    the given 'YYYY-MM' UTC month.

    used_this_month reads org_semantic_evaluation_usage — incremented on
    every sampled run regardless of per-agent budget config (Phase 1.5), not
    derived from failure_signals or the per-agent semantic_evaluation_usage
    table, either of which would undercount. cost_so_far_usd reads
    semantic_evaluation_log — every evaluate() call, fired or not; see that
    table's schema comment for why failure_signals alone isn't enough for
    honest billing math.
    """
    defaults = {"quota": 1000, "allow_overage": False}
    if not _pool:
        return {**defaults, "used_this_month": 0, "cost_so_far_usd": 0.0}

    async with _pool.acquire() as conn:
        org_row = await conn.fetchrow(
            "SELECT semantic_evaluation_quota, allow_semantic_overage FROM organizations WHERE id = $1",
            org_id,
        )
        usage_row = await conn.fetchrow(
            "SELECT eval_count FROM org_semantic_evaluation_usage WHERE org_id = $1 AND month = $2",
            org_id,
            month,
        )
        cost_row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(cost_usd), 0.0) AS total_cost
            FROM semantic_evaluation_log
            WHERE org_id = $1 AND to_char(evaluated_at, 'YYYY-MM') = $2
            """,
            org_id,
            month,
        )

    quota = org_row["semantic_evaluation_quota"] if org_row else defaults["quota"]
    allow_overage = org_row["allow_semantic_overage"] if org_row else defaults["allow_overage"]
    used_this_month = usage_row["eval_count"] if usage_row else 0
    cost_so_far_usd = float(cost_row["total_cost"]) if cost_row else 0.0

    return {
        "quota": quota,
        "allow_overage": allow_overage,
        "used_this_month": used_this_month,
        "cost_so_far_usd": cost_so_far_usd,
    }


# ── External evaluation integrations (Phase 2.1) ───────────────────────────────


async def upsert_external_integration(
    org_id: str,
    provider: str,
    endpoint_url: str,
    encrypted_credentials: str,
    poll_interval_secs: int,
) -> int:
    """Create or replace this org's config for one provider. Replacing
    (rather than requiring a separate update path) re-encrypts fresh
    credentials in one call — simplest correct behavior for "I rotated my
    Langfuse API key." Resets failure tracking, since new credentials
    deserve a clean slate rather than inheriting a stale outage streak.
    """
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO external_evaluation_integrations
                (org_id, provider, endpoint_url, encrypted_credentials, poll_interval_secs)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (org_id, provider) DO UPDATE
                SET endpoint_url          = EXCLUDED.endpoint_url,
                    encrypted_credentials = EXCLUDED.encrypted_credentials,
                    poll_interval_secs    = EXCLUDED.poll_interval_secs,
                    enabled               = TRUE,
                    consecutive_failures  = 0,
                    first_failure_at      = NULL,
                    updated_at            = NOW()
            RETURNING id
            """,
            org_id,
            provider,
            endpoint_url,
            encrypted_credentials,
            poll_interval_secs,
        )
    return row["id"]


async def get_external_integration_status(org_id: str, provider: str) -> Optional[dict]:
    """Configuration + health status — never the credential itself, even
    encrypted. `configured` distinguishes "never set up" (None) from "set up
    but currently failing" (a real status dict with consecutive_failures > 0)."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT endpoint_url, poll_interval_secs, enabled, last_polled_at,
                   last_success_at, consecutive_failures
            FROM external_evaluation_integrations
            WHERE org_id = $1 AND provider = $2
            """,
            org_id,
            provider,
        )
    if row is None:
        return None
    return {
        "endpoint_url": row["endpoint_url"],
        "poll_interval_secs": row["poll_interval_secs"],
        "enabled": row["enabled"],
        "last_polled_at": row["last_polled_at"],
        "last_success_at": row["last_success_at"],
        "consecutive_failures": row["consecutive_failures"],
    }


async def delete_external_integration(org_id: str, provider: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM external_evaluation_integrations WHERE org_id = $1 AND provider = $2",
            org_id,
            provider,
        )
    return result != "DELETE 0"


# ── Alert destination integrations (Phase 4.1) ─────────────────────────────────


async def upsert_org_alert_integration(
    org_id: str,
    provider: str,
    encrypted_credentials: str,
    config_json: dict,
) -> int:
    """Create or replace this org's config for one alert destination
    (slack | linear). Same replace-not-update-in-place semantics as Phase
    2.1's upsert_external_integration — rotating a customer's webhook URL
    or API key should just work with one call."""
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO org_alert_integrations (org_id, provider, encrypted_credentials, config_json)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (org_id, provider) DO UPDATE
                SET encrypted_credentials = EXCLUDED.encrypted_credentials,
                    config_json           = EXCLUDED.config_json,
                    enabled               = TRUE,
                    updated_at            = NOW()
            RETURNING id
            """,
            org_id,
            provider,
            encrypted_credentials,
            _json_mod.dumps(config_json),
        )
    return row["id"]


async def get_org_alert_integration_status(org_id: str, provider: str) -> Optional[dict]:
    """Configuration + enabled status — never the credential itself, even
    encrypted (matches Phase 2.1's own GET semantics)."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT config_json, enabled FROM org_alert_integrations WHERE org_id = $1 AND provider = $2",
            org_id,
            provider,
        )
    if row is None:
        return None
    config = row["config_json"]
    return {
        "config": _json_mod.loads(config) if isinstance(config, str) else dict(config),
        "enabled": row["enabled"],
    }


async def delete_org_alert_integration(org_id: str, provider: str) -> bool:
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM org_alert_integrations WHERE org_id = $1 AND provider = $2",
            org_id,
            provider,
        )
    return result != "DELETE 0"


# ── GitHub App integration + source mapping (Phase 4.3) ────────────────────────


async def upsert_org_github_installation(org_id: str, installation_id: int) -> None:
    """Called by the install-flow callback once GitHub redirects back with
    a real installation_id. repos/reviewers start empty — a separate config
    call (set_org_github_config) fills those in."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO org_github_integrations (org_id, installation_id)
            VALUES ($1, $2)
            ON CONFLICT (org_id) DO UPDATE
                SET installation_id = EXCLUDED.installation_id,
                    updated_at      = NOW()
            """,
            org_id,
            installation_id,
        )


async def set_org_github_config(org_id: str, repos: list, reviewers: list) -> bool:
    """Returns False if this org has no installation yet (install must
    happen before config — there's nothing to configure otherwise)."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE org_github_integrations
            SET repos = $1::jsonb, reviewers = $2::jsonb, updated_at = NOW()
            WHERE org_id = $3
            """,
            _json_mod.dumps(repos),
            _json_mod.dumps(reviewers),
            org_id,
        )
    return result != "UPDATE 0"


async def get_org_github_integration(org_id: str) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT installation_id, repos, reviewers FROM org_github_integrations WHERE org_id = $1",
            org_id,
        )
    if row is None:
        return None
    repos = row["repos"]
    reviewers = row["reviewers"]
    return {
        "installation_id": row["installation_id"],
        "repos": _json_mod.loads(repos) if isinstance(repos, str) else list(repos),
        "reviewers": _json_mod.loads(reviewers) if isinstance(reviewers, str) else list(reviewers),
    }


async def delete_org_github_integration(org_id: str) -> bool:
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute("DELETE FROM org_github_integrations WHERE org_id = $1", org_id)
    return result != "DELETE 0"


# ── Pack activation (Phase 1.0) ─────────────────────────────────────────────────


async def list_all_packs() -> list[dict]:
    """Every registered pack, regardless of activation status anywhere.
    detector_svc seeds this table from PACK_REGISTRY at startup — api_svc
    never writes to it, only reads it to validate a pack_name exists and to
    serve GET /v1/packs."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name, description, detector_names, added_at FROM packs ORDER BY name"
        )
    return [dict(r) for r in rows]


async def pack_exists(pack_name: str) -> bool:
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        return bool(await conn.fetchval("SELECT 1 FROM packs WHERE name = $1", pack_name))


async def list_org_enabled_packs(org_id: str) -> list[dict]:
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT pack_name, enabled_at, enabled_by
            FROM org_enabled_packs
            WHERE org_id = $1
            ORDER BY pack_name
            """,
            org_id,
        )
    return [dict(r) for r in rows]


async def activate_pack(org_id: str, pack_name: str, enabled_by: Optional[str]) -> None:
    """Idempotent — activating an already-active pack just refreshes
    enabled_at/enabled_by rather than erroring, since re-activation isn't a
    meaningfully different action from the caller's point of view."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO org_enabled_packs (org_id, pack_name, enabled_by)
            VALUES ($1, $2, $3)
            ON CONFLICT (org_id, pack_name) DO UPDATE
                SET enabled_at = NOW(),
                    enabled_by = EXCLUDED.enabled_by
            """,
            org_id,
            pack_name,
            enabled_by,
        )


async def deactivate_pack(org_id: str, pack_name: str) -> bool:
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM org_enabled_packs WHERE org_id = $1 AND pack_name = $2",
            org_id,
            pack_name,
        )
    return result != "DELETE 0"


# ── State analytics (Capability 3, Phase 3.3) ───────────────────────────────


async def agent_state_analytics(org_id: str, agent_id: str, window_days: int = 30) -> dict:
    """Fetch this agent's run_state_metrics within the window and reduce them to
    the analytics payload (aggregates, trends, outliers) via the pure module."""
    from datetime import datetime, timedelta, timezone

    from api_svc.state_analytics import compute_state_analytics

    if not _pool:
        return compute_state_analytics([], window_days=window_days)

    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, state, total_ms, run_started_at
            FROM run_state_metrics
            WHERE org_id = $1 AND agent_id = $2
              AND (run_started_at IS NULL OR run_started_at >= $3)
            """,
            org_id,
            agent_id,
            since,
        )
    return compute_state_analytics([dict(r) for r in rows], window_days=window_days)


# ── Approvals (Capability 2, Phase 2.1) ─────────────────────────────────────


async def create_approval(
    org_id: str,
    run_id: str,
    agent_id: str,
    tool_name: str,
    tool_args: Optional[str],
    expires_at,
) -> Optional[dict]:
    """Insert a pending approval and return the created row (including its id),
    or None if the pool isn't up."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO approvals (org_id, run_id, agent_id, tool_name, tool_args, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, org_id, run_id, agent_id, tool_name, tool_args,
                      status, requested_at, expires_at, decided_at, decided_by,
                      decision_channel
            """,
            org_id,
            run_id,
            agent_id,
            tool_name,
            tool_args,
            expires_at,
        )
    return dict(row) if row else None


async def list_approvals(org_id: str, status: Optional[str] = None, limit: int = 100) -> list[dict]:
    """Approvals for this org, newest first, optionally filtered by status.
    Org-scoped so the dashboard only ever sees its own org's approvals."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, org_id, run_id, agent_id, tool_name, tool_args, status,
                   requested_at, expires_at, decided_at, decided_by, decision_channel
            FROM approvals
            WHERE org_id = $1 AND ($2::text IS NULL OR status = $2)
            ORDER BY requested_at DESC
            LIMIT $3
            """,
            org_id,
            status,
            limit,
        )
    return [dict(r) for r in rows]


async def get_approval(org_id: str, approval_id: int) -> Optional[dict]:
    """Fetch one approval, scoped to org_id so one org can't read another's."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, org_id, run_id, agent_id, tool_name, tool_args, status,
                   requested_at, expires_at, decided_at, decided_by, decision_channel
            FROM approvals
            WHERE org_id = $1 AND id = $2
            """,
            org_id,
            approval_id,
        )
    return dict(row) if row else None


async def set_approval_decision(
    org_id: str,
    approval_id: int,
    new_status: str,
    decided_by: Optional[str],
    decision_channel: Optional[str],
) -> Optional[dict]:
    """Move a *pending* approval to a terminal status. The `status = 'pending'`
    guard in the WHERE clause makes this a no-op on an already-decided approval
    (late Slack click, double submit) — it returns None rather than overwriting
    the recorded outcome. Caller should treat None as 'already decided or not
    found' and re-read to see the actual state. Transition legality is enforced
    here structurally (only pending → terminal is reachable); the full rule set
    lives in api_svc.approvals.is_valid_transition()."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE approvals
            SET status = $3,
                decided_at = NOW(),
                decided_by = $4,
                decision_channel = $5
            WHERE org_id = $1 AND id = $2 AND status = 'pending'
            RETURNING id, org_id, run_id, agent_id, tool_name, tool_args, status,
                      requested_at, expires_at, decided_at, decided_by, decision_channel
            """,
            org_id,
            approval_id,
            new_status,
            decided_by,
            decision_channel,
        )
    return dict(row) if row else None


async def upsert_agent_source_config(
    org_id: str, agent_id: str, repo: str, file_path: Optional[str]
) -> None:
    """Tier-1 explicit source mapping. file_path may be omitted (repo-only)
    — see source_resolution.py for how that combines with tier-2 SDK
    auto-detection."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_source_config (org_id, agent_id, repo, file_path)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (org_id, agent_id) DO UPDATE
                SET repo       = EXCLUDED.repo,
                    file_path  = EXCLUDED.file_path,
                    updated_at = NOW()
            """,
            org_id,
            agent_id,
            repo,
            file_path,
        )


async def get_agent_source_config(org_id: str, agent_id: str) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT repo, file_path FROM agent_source_config WHERE org_id = $1 AND agent_id = $2",
            org_id,
            agent_id,
        )
    return {"repo": row["repo"], "file_path": row["file_path"]} if row else None


async def delete_agent_source_config(org_id: str, agent_id: str) -> bool:
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM agent_source_config WHERE org_id = $1 AND agent_id = $2",
            org_id,
            agent_id,
        )
    return result != "DELETE 0"


async def get_latest_run_started_payload(org_id: str, agent_id: str) -> Optional[dict]:
    """Most recent run.started event's payload for this agent — used to
    read the SDK's tier-2 auto-detected source_file (see
    packages/sdk-py/dunetrace/run_context.py), the same way native_explain
    already reads system_prompt off this same event."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT payload FROM events
            WHERE org_id = $1 AND agent_id = $2 AND event_type = 'run.started'
            ORDER BY received_at DESC
            LIMIT 1
            """,
            org_id,
            agent_id,
        )
    if row is None:
        return None
    payload = row["payload"]
    return _json_mod.loads(payload) if isinstance(payload, str) else dict(payload)


async def get_org_linear_webhook_secret(org_id: str) -> Optional[str]:
    """Returns the encrypted_credentials token for this org's Linear
    integration, for routers/linear_webhook.py to decrypt (see
    api_svc/crypto.py::decrypt_credentials_for_webhook_verification's
    docstring — this is the one deliberate exception to api_svc's
    encrypt-only rule)."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT encrypted_credentials FROM org_alert_integrations "
            "WHERE org_id = $1 AND provider = 'linear' AND enabled = TRUE",
            org_id,
        )


async def get_signal_id_for_linear_issue(linear_issue_id: str) -> Optional[dict]:
    """Returns {org_id, signal_id} for a Linear issue previously created by
    alerts_svc (see alerts_svc/db.py::record_linear_issue_mapping), or None
    if this issue wasn't one Dunetrace created."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT org_id, signal_id FROM linear_issue_signals WHERE linear_issue_id = $1",
            linear_issue_id,
        )
    return dict(row) if row else None


# ── Generic external signal push (Phase 2.4) ───────────────────────────────────


async def has_processed_external(org_id: str, provider: str, external_id: str) -> bool:
    """Reuses the same external_evaluation_processed table Phase 2.1-2.3's
    pull integrations use for dedup, keyed the same way — a push caller's
    own external_id plays the same idempotency-key role a pulled score's own
    id does."""
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM external_evaluation_processed "
            "WHERE org_id = $1 AND provider = $2 AND external_id = $3)",
            org_id,
            provider,
            external_id,
        )


async def mark_processed_external(org_id: str, provider: str, external_id: str) -> None:
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


async def fetch_run_by_trace_id_for_org(org_id: str, trace_id: str) -> Optional[dict]:
    """Correlates a pushed evaluation back to the Dunetrace run it's about —
    scoped to the caller's own org_id, unlike integrations_svc's
    fetch_run_by_trace_id (which trusts that a configured pull integration
    only ever sees its own org's trace_ids). This endpoint takes a
    caller-supplied trace_id directly over the wire, so it deliberately
    doesn't extend that same trust — a customer must not be able to probe
    for another org's trace_id by reading this endpoint's response."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT run_id, agent_id, agent_version
            FROM events
            WHERE trace_id = $1 AND org_id = $2
            LIMIT 1
            """,
            trace_id,
            org_id,
        )
    return dict(row) if row else None


async def write_pushed_external_signal(
    org_id: str,
    agent_id: str,
    agent_version: str,
    run_id: str,
    provider: str,
    failure_type: str,
    confidence: float,
    evidence: dict,
) -> int:
    """Same shadow=FALSE convention Phase 2.1's write_external_signal
    established — the caller explicitly pushed this evaluation, so they're
    trusted to have judged its quality. Returns the new row's id (unlike the
    pull-integration version, which has no caller waiting on a response to
    hand it back to)."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO failure_signals
                (failure_type, severity, run_id, agent_id, agent_version,
                 step_index, confidence, evidence, shadow, co_signal_count, org_id, source)
            VALUES ($1, 'MEDIUM', $2, $3, $4, 0, $5, $6::jsonb, FALSE, 0, $7, $8)
            RETURNING id
            """,
            failure_type,
            run_id,
            agent_id,
            agent_version,
            confidence,
            _json_mod.dumps(evidence),
            org_id,
            provider,
        )
    return row["id"]


async def record_signal_feedback(
    signal_id: int, org_id: str, verdict: str, notes: Optional[str]
) -> int:
    """Records one feedback submission, and — for a false_positive verdict —
    increments the false-positive counter for whichever signal_group this
    signal belongs to (via signal_group_members). A signal with no group yet
    (e.g. written before Phase 1.4.2 shipped) simply doesn't affect any
    group's count; the feedback row itself is still recorded either way.
    Returns the new signal_feedback row's id.
    """
    async with _pool.acquire() as conn:
        async with conn.transaction():
            feedback_id = await conn.fetchval(
                """
                INSERT INTO signal_feedback (signal_id, org_id, verdict, notes)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                signal_id,
                org_id,
                verdict,
                notes,
            )
            if verdict == "false_positive":
                group_id = await conn.fetchval(
                    "SELECT group_id FROM signal_group_members WHERE signal_id = $1",
                    signal_id,
                )
                if group_id is not None:
                    await conn.execute(
                        """
                        INSERT INTO signal_group_overrides (group_id, fp_count)
                        VALUES ($1, 1)
                        ON CONFLICT (group_id) DO UPDATE
                            SET fp_count   = signal_group_overrides.fp_count + 1,
                                updated_at = NOW()
                        """,
                        group_id,
                    )
    return feedback_id


async def get_signal_fix_status(
    org_id: str,
    agent_id: str,
    failure_type: str,
    signal_id: int,
) -> dict:
    """
    Check whether the most recent fix for a signal has reduced recurrence.

    Looks up the earliest fix for signal_id, then counts runs and signal
    recurrences on the same agent+failure_type after that fix was applied.
    """
    if not _pool:
        return {}

    async with _pool.acquire() as conn:
        fix_row = await conn.fetchrow(
            """
            SELECT applied_at, langfuse_prompt_name, langfuse_version, applied_via
            FROM fixes WHERE signal_id = $1 AND org_id = $2
            ORDER BY applied_at ASC LIMIT 1
            """,
            signal_id,
            org_id,
        )
        if not fix_row:
            return {"fix_applied": False}

        applied_at = fix_row["applied_at"]

        runs_after = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT run_id) FROM events
            WHERE org_id = $1 AND agent_id = $2 AND received_at > $3
            """,
            org_id,
            agent_id,
            applied_at,
        )
        recurrences = await conn.fetchval(
            """
            SELECT COUNT(*) FROM failure_signals
            WHERE org_id = $1 AND agent_id = $2 AND failure_type = $3 AND detected_at > $4
            """,
            org_id,
            agent_id,
            failure_type,
            applied_at,
        )

    rec = int(recurrences or 0)
    runs = int(runs_after or 0)
    return {
        "fix_applied": True,
        "applied_at": applied_at.timestamp() if applied_at else None,
        "applied_via": fix_row["applied_via"],
        "langfuse_prompt_name": fix_row["langfuse_prompt_name"],
        "langfuse_version": fix_row["langfuse_version"],
        "runs_after_fix": runs,
        "recurrences_after_fix": rec,
        "verdict": (
            "verified"
            if runs >= 10 and rec == 0
            else (
                "likely_fixed"
                if runs >= 5 and rec == 0
                else "still_occurring"
                if rec > 0
                else "insufficient_data"
            )
        ),
    }


# ── Policies ─────────────────────────────────────────────────────────────────


def _policy_row(r: Any) -> dict:
    import json as _json

    cond = r["condition"]
    act = r["action"]
    if isinstance(cond, str):
        cond = _json.loads(cond)
    if isinstance(act, str):
        act = _json.loads(act)
    ca = r["created_at"]
    ua = r["updated_at"]
    return {
        "id": r["id"],
        "agent_id": r["agent_id"],
        "name": r["name"],
        "condition": dict(cond) if cond else {},
        "action": dict(act) if act else {},
        "enabled": r["enabled"],
        "priority": r["priority"],
        "signature": r.get("signature", "") or "",
        "sig_version": r.get("sig_version", 1) or 1,
        "created_at": ca.timestamp() if hasattr(ca, "timestamp") else ca,
        "updated_at": ua.timestamp() if hasattr(ua, "timestamp") else ua,
    }


def _policy_canonical(
    version: int,
    policy_id: int,
    agent_id: str,
    name: str,
    condition: dict,
    action: dict,
    enabled: bool,
    priority: int,
) -> str:
    """Versioned canonical string for the policy HMAC. MUST stay in exact sync
    with the SDK's ``_policy_canonical`` (dunetrace/policies/__init__.py):
      v1 — original 7 fields, null-byte separated (byte-identical to pre-feature).
      v2 — same fields with an authenticated "v2" domain-separation prefix; used
           for policies carrying a condition.match expression block.
    condition is JSON-dumped with sort_keys, so the nested `match` block is
    already covered by the signature under either version."""
    fields = [
        str(policy_id),
        agent_id,
        name,
        _json_mod.dumps(condition, sort_keys=True),
        _json_mod.dumps(action, sort_keys=True),
        str(enabled),
        str(priority),
    ]
    if version >= 2:
        fields.insert(0, "v%d" % version)
    return "\x00".join(fields)


def _sig_version_for(condition: dict) -> int:
    """The minimum canonical-form version representing this condition: v2 when it
    uses a `match` expression block, else v1 (keeps legacy policies byte-identical
    and older SDKs able to verify them)."""
    if isinstance(condition, dict) and condition.get("match") is not None:
        return 2
    return 1


def _sign_policy(
    policy_id: int,
    agent_id: str,
    name: str,
    condition: dict,
    action: dict,
    enabled: bool,
    priority: int,
    secret: str,
) -> tuple:
    """HMAC-SHA256 over the versioned canonical policy fields. Returns
    ``(signature, sig_version)``; signature is '' when secret is empty (dev mode),
    but sig_version is still reported so it can be recorded consistently.

    Must stay in sync with _verify_policy_signature in the SDK's policies.py.
    """
    version = _sig_version_for(condition)
    if not secret:
        return "", version
    canonical = _policy_canonical(
        version, policy_id, agent_id, name, condition, action, enabled, priority
    )
    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest(), version


async def log_policy_audit(
    policy_id: Optional[int],
    action: str,
    org_id: str,
    before: Optional[dict],
    after: Optional[dict],
) -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO policy_audit_log (policy_id, action, org_id, before, after)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
            """,
            policy_id,
            action,
            org_id,
            _json_mod.dumps(before) if before is not None else None,
            _json_mod.dumps(after) if after is not None else None,
        )


async def list_policies(org_id: str, agent_id: Optional[str] = None) -> list:
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        if agent_id:
            rows = await conn.fetch(
                "SELECT * FROM policies WHERE org_id = $1 AND (agent_id = $2 OR agent_id = '*') "
                "ORDER BY priority, id",
                org_id,
                agent_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM policies WHERE org_id = $1 ORDER BY priority, id", org_id
            )
    return [_policy_row(r) for r in rows]


async def get_policy_by_id(org_id: str, policy_id: int) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM policies WHERE id = $1 AND org_id = $2", policy_id, org_id
        )
    return _policy_row(row) if row else None


def _policy_eval_row(r: Any) -> dict:
    import json as _json

    conds = r["conditions"]
    if isinstance(conds, str):
        conds = _json.loads(conds)
    ea = r["evaluated_at"]
    return {
        "id": r["id"],
        "policy_id": r["policy_id"],
        "policy_name": r["policy_name"],
        "agent_id": r["agent_id"],
        "run_id": r["run_id"],
        "trigger": r["trigger_name"],
        "trigger_matched": r["trigger_matched"],
        "fired": r["fired"],
        "sampled": r["sampled"],
        "reason": r["reason"],
        "conditions": conds if conds is not None else [],
        "evaluated_at": ea.timestamp() if hasattr(ea, "timestamp") else ea,
    }


async def fetch_policy_evaluations(org_id: str, policy_id: int, limit: int = 100) -> list:
    """Recent policy.evaluated observability records for one policy, newest
    first. Powers GET /v1/policies/{id}/evaluations — the "why did/didn't my
    policy fire?" debug view. Org-scoped so one org can't read another's."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, policy_id, policy_name, agent_id, run_id, trigger_name,
                   trigger_matched, fired, sampled, reason, conditions, evaluated_at
            FROM policy_evaluations
            WHERE org_id = $1 AND policy_id = $2
            ORDER BY evaluated_at DESC
            LIMIT $3
            """,
            org_id,
            policy_id,
            limit,
        )
    return [_policy_eval_row(r) for r in rows]


async def create_policy(
    org_id: str,
    name: str,
    agent_id: str,
    condition: dict,
    action: dict,
    priority: int = 100,
    enabled: bool = True,
) -> dict:
    if not _pool:
        raise RuntimeError("DB pool not available")
    async with _pool.acquire() as conn:
        # Insert first to get the id, then back-fill the signature in one transaction.
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO policies (name, agent_id, condition, action, priority, enabled, org_id)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7)
                RETURNING *
                """,
                name,
                agent_id,
                _json_mod.dumps(condition),
                _json_mod.dumps(action),
                priority,
                enabled,
                org_id,
            )
            # Signature is over policy identity/behavior fields only — org_id is a
            # tenancy filter, not part of the policy's cryptographic identity, and
            # must stay out of the canonical string so existing signed policies
            # (and the SDK's _verify_policy_signature) don't need to be re-signed.
            sig, sig_version = _sign_policy(
                row["id"],
                agent_id,
                name,
                condition,
                action,
                enabled,
                priority,
                settings.POLICY_SIGNING_SECRET,
            )
            row = await conn.fetchrow(
                "UPDATE policies SET signature = $1, sig_version = $2 WHERE id = $3 RETURNING *",
                sig,
                sig_version,
                row["id"],
            )
    return _policy_row(row)


async def update_policy(org_id: str, policy_id: int, fields: dict) -> dict:
    if not _pool:
        raise RuntimeError("DB pool not available")

    set_parts = []
    params: list = []
    for key, value in fields.items():
        if key not in {
            "name",
            "agent_id",
            "condition",
            "action",
            "priority",
            "enabled",
        }:
            continue
        params.append(value if key not in {"condition", "action"} else _json_mod.dumps(value))
        cast = "::jsonb" if key in {"condition", "action"} else ""
        set_parts.append(f"{key} = ${len(params)}{cast}")

    if not set_parts:
        return await get_policy_by_id(org_id, policy_id)  # type: ignore[return-value]

    params.append(policy_id)
    params.append(org_id)
    async with _pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                f"UPDATE policies SET {', '.join(set_parts)}, updated_at = NOW() "
                f"WHERE id = ${len(params) - 1} AND org_id = ${len(params)} RETURNING *",
                *params,
            )
            if row is None:
                return None  # type: ignore[return-value]
            # Recompute signature over the full updated record (org_id excluded — see create_policy).
            cond = row["condition"]
            act = row["action"]
            if isinstance(cond, str):
                cond = _json_mod.loads(cond)
            if isinstance(act, str):
                act = _json_mod.loads(act)
            sig, sig_version = _sign_policy(
                row["id"],
                row["agent_id"],
                row["name"],
                dict(cond),
                dict(act),
                row["enabled"],
                row["priority"],
                settings.POLICY_SIGNING_SECRET,
            )
            row = await conn.fetchrow(
                "UPDATE policies SET signature = $1, sig_version = $2 WHERE id = $3 RETURNING *",
                sig,
                sig_version,
                policy_id,
            )
    return _policy_row(row)


async def delete_policy(org_id: str, policy_id: int) -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("DELETE FROM policies WHERE id = $1 AND org_id = $2", policy_id, org_id)


# ── Agent Health Score ────────────────────────────────────────────────────────


async def get_agent_health_score(org_id: str, agent_id: str) -> dict:
    """
    Composite 0–100 health score for an agent based on the last 30 days.

    Components:
      failure_rate  (0–40): proportion of runs with any signal (40 = zero failures)
      loop_avoidance (0–25): proportion of runs free of looping signals
      token_efficiency (0–20): avg prompt tokens — lower is more efficient
      latency (0–15): avg LLM latency_ms — lower is faster

    Returns {"score": int|None, "components": {...}, "sample_runs": int}.
    score is None when fewer than 3 runs are available.
    """
    if not _pool:
        return {"score": None, "components": {}, "sample_runs": 0}

    async with _pool.acquire() as conn:
        stats = await conn.fetchrow(
            """
            WITH runs_30d AS (
                SELECT DISTINCT run_id
                FROM processed_runs
                WHERE org_id = $1 AND agent_id = $2
                  AND processed_at >= NOW() - INTERVAL '30 days'
            ),
            signal_runs AS (
                SELECT DISTINCT run_id
                FROM failure_signals
                WHERE org_id = $1 AND agent_id = $2
                  AND shadow = FALSE
                  AND detected_at >= NOW() - INTERVAL '30 days'
            ),
            loop_runs AS (
                SELECT DISTINCT run_id
                FROM failure_signals
                WHERE org_id = $1 AND agent_id = $2
                  AND shadow = FALSE
                  AND detected_at >= NOW() - INTERVAL '30 days'
                  AND failure_type IN (
                      'TOOL_LOOP','TOOL_THRASHING','TOOL_AVOIDANCE',
                      'STEP_COUNT_INFLATION','LLM_TRUNCATION_LOOP'
                  )
            ),
            token_data AS (
                -- prompt_tokens is in llm.called for direct SDK, llm.responded for LangChain
                SELECT
                    AVG((payload->>'prompt_tokens')::float)                                        AS avg_prompt_tokens,
                    COUNT(*)                                                                        AS token_sample
                FROM events
                WHERE org_id = $1 AND agent_id = $2
                  AND event_type IN ('llm.called', 'llm.responded')
                  AND payload->>'prompt_tokens' IS NOT NULL
                  AND received_at >= NOW() - INTERVAL '30 days'
            ),
            latency_data AS (
                SELECT
                    AVG((payload->>'latency_ms')::float)                                           AS avg_latency_ms,
                    COUNT(*)                                                                        AS latency_sample
                FROM events
                WHERE org_id = $1 AND agent_id = $2
                  AND event_type = 'llm.responded'
                  AND payload->>'latency_ms' IS NOT NULL
                  AND received_at >= NOW() - INTERVAL '30 days'
            ),
            token_baseline AS (
                -- P75 of per-run average token usage, computed from 30–90 days ago.
                -- Excludes the 30-day scoring window so the baseline is independent
                -- of the period being measured — avoids the boiling-frog problem where
                -- chronic degradation inflates the reference point.
                SELECT
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_tokens) AS p75_tokens,
                    COUNT(*)                                                   AS baseline_sample
                FROM (
                    SELECT run_id, AVG((payload->>'prompt_tokens')::float) AS avg_tokens
                    FROM events
                    WHERE org_id = $1 AND agent_id = $2
                      AND event_type IN ('llm.called', 'llm.responded')
                      AND payload->>'prompt_tokens' IS NOT NULL
                      AND received_at >= NOW() - INTERVAL '90 days'
                      AND received_at <  NOW() - INTERVAL '30 days'
                    GROUP BY run_id
                ) per_run
            ),
            latency_baseline AS (
                -- P75 of per-run average LLM latency, same 30–90 day reference window.
                SELECT
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_latency) AS p75_latency_ms,
                    COUNT(*)                                                    AS baseline_sample
                FROM (
                    SELECT run_id, AVG((payload->>'latency_ms')::float) AS avg_latency
                    FROM events
                    WHERE org_id = $1 AND agent_id = $2
                      AND event_type = 'llm.responded'
                      AND payload->>'latency_ms' IS NOT NULL
                      AND received_at >= NOW() - INTERVAL '90 days'
                      AND received_at <  NOW() - INTERVAL '30 days'
                    GROUP BY run_id
                ) per_run
            )
            SELECT
                (SELECT COUNT(*)           FROM runs_30d)            AS total_runs,
                (SELECT COUNT(*)           FROM signal_runs)         AS runs_with_signals,
                (SELECT COUNT(*)           FROM loop_runs)           AS runs_with_loops,
                (SELECT avg_prompt_tokens  FROM token_data)          AS avg_prompt_tokens,
                (SELECT avg_latency_ms     FROM latency_data)        AS avg_latency_ms,
                (SELECT p75_tokens         FROM token_baseline)      AS p75_tokens,
                (SELECT baseline_sample    FROM token_baseline)      AS token_baseline_sample,
                (SELECT p75_latency_ms     FROM latency_baseline)    AS p75_latency_ms,
                (SELECT baseline_sample    FROM latency_baseline)    AS latency_baseline_sample
            """,
            org_id,
            agent_id,
        )

    _BASELINE_MIN_RUNS = 30  # runs needed before token/latency components leave neutral

    total = int(stats["total_runs"] or 0)
    if total < 3:
        return {
            "score": None,
            "components": {},
            "sample_runs": total,
            "baseline_ready": False,
        }

    baseline_ready = total >= _BASELINE_MIN_RUNS

    failure_rate = (int(stats["runs_with_signals"] or 0)) / total
    loop_rate = (int(stats["runs_with_loops"] or 0)) / total
    avg_tokens = stats["avg_prompt_tokens"]
    avg_latency = stats["avg_latency_ms"]
    p75_tokens = stats["p75_tokens"] if (stats["token_baseline_sample"] or 0) >= 20 else None
    p75_latency = stats["p75_latency_ms"] if (stats["latency_baseline_sample"] or 0) >= 20 else None

    # Failure component (0–40): 0% failure rate = 40 pts. Active from run 1.
    failure_score = round((1.0 - failure_rate) * 40)

    # Loop component (0–25): 0% loop rate = 25 pts. Active from run 1.
    loop_score = round((1.0 - loop_rate) * 25)

    # Token efficiency (0–20).
    # Neutral (15) only when avg_tokens is None — no data at all.
    # Young agents (no P75 baseline yet) fall back to global absolutes rather than neutral,
    # because 4,000 tokens/call is expensive regardless of history.
    # Once a P75 baseline exists: full score ≤ 1.5× P75; linear decay 1.5×–4×; 0 at 4×.
    if avg_tokens is None:
        token_score = 15  # neutral — no token data recorded
    elif p75_tokens is not None:
        healthy_ceil = p75_tokens * 1.5
        penalty_ceil = p75_tokens * 4.0
        if avg_tokens <= healthy_ceil:
            token_score = 20
        elif avg_tokens >= penalty_ceil:
            token_score = 0
        else:
            token_score = round(
                20.0 * (1.0 - (avg_tokens - healthy_ceil) / (penalty_ceil - healthy_ceil))
            )
    else:
        # No pre-window baseline (young agent or no LLM events in 30–90 day window).
        # Use global absolutes: < 500 tokens = full score, > 4,000 = 0.
        token_score = max(0, min(20, round(20.0 * (1.0 - max(0.0, avg_tokens - 500) / 3500))))

    # Latency component (0–15). Same logic: neutral only when avg_latency is None.
    # Young agents with data scored against global absolutes: < 1,000ms = full, > 8,000ms = 0.
    if avg_latency is None:
        latency_score = 10  # neutral — no latency data recorded
    elif p75_latency is not None:
        healthy_ceil = p75_latency * 1.5
        penalty_ceil = p75_latency * 4.0
        if avg_latency <= healthy_ceil:
            latency_score = 15
        elif avg_latency >= penalty_ceil:
            latency_score = 0
        else:
            latency_score = round(
                15.0 * (1.0 - (avg_latency - healthy_ceil) / (penalty_ceil - healthy_ceil))
            )
    else:
        # No pre-window baseline — global absolutes: < 1,000ms = full, > 8,000ms = 0.
        latency_score = max(0, min(15, round(15.0 * (1.0 - max(0.0, avg_latency - 1000) / 7000))))

    score = failure_score + loop_score + token_score + latency_score

    return {
        "score": score,
        "components": {
            "failure_rate": {
                "score": failure_score,
                "max": 40,
                "value": round(failure_rate * 100, 1),
                "label": "% runs with failures",
            },
            "loop_avoidance": {
                "score": loop_score,
                "max": 25,
                "value": round(loop_rate * 100, 1),
                "label": "% runs with loops",
            },
            "token_efficiency": {
                "score": token_score,
                "max": 20,
                "value": round(avg_tokens) if avg_tokens is not None else None,
                "baseline": round(p75_tokens) if p75_tokens is not None else None,
                "label": "avg prompt tokens",
            },
            "latency": {
                "score": latency_score,
                "max": 15,
                "value": round(avg_latency) if avg_latency is not None else None,
                "baseline": round(p75_latency) if p75_latency is not None else None,
                "label": "avg LLM latency ms",
            },
        },
        "sample_runs": total,
        "baseline_ready": baseline_ready,
    }


# ── Cost stats ────────────────────────────────────────────────────────────────


async def agent_cost_stats(org_id: str, agent_id: str) -> dict:
    """
    Estimated API cost for an agent over the last 30 days.

    Returns:
      total_cost_usd     — all runs combined
      wasted_cost_usd    — runs that had at least one live signal
      wasted_pct         — wasted / total (0–1)
      cost_by_failure_type — [{failure_type, wasted_usd, affected_runs}]
    """
    if not _pool:
        return {
            "total_cost_usd": 0.0,
            "wasted_cost_usd": 0.0,
            "wasted_pct": 0.0,
            "cost_by_failure_type": [],
        }

    from explainer_svc.cost import estimate_cost

    async with _pool.acquire() as conn:
        # Prompt tokens + model per run. prompt_tokens may be in llm.called (direct SDK)
        # or llm.responded (LangChain) — sum both; model always comes from llm.called.
        prompt_rows = await conn.fetch(
            """
            SELECT run_id,
                   MAX(CASE WHEN event_type = 'llm.called' THEN payload->>'model' END) AS model,
                   SUM(COALESCE((payload->>'prompt_tokens')::int, 0)) AS prompt_tokens
            FROM events
            WHERE org_id = $1 AND agent_id = $2
              AND event_type IN ('llm.called', 'llm.responded')
              AND received_at >= NOW() - INTERVAL '30 days'
            GROUP BY run_id
            """,
            org_id,
            agent_id,
        )

        # Completion tokens per run
        completion_rows = await conn.fetch(
            """
            SELECT run_id,
                   SUM(COALESCE((payload->>'completion_tokens')::int, 0)) AS completion_tokens
            FROM events
            WHERE org_id = $1 AND agent_id = $2
              AND event_type = 'llm.responded'
              AND payload->>'completion_tokens' IS NOT NULL
              AND received_at >= NOW() - INTERVAL '30 days'
            GROUP BY run_id
            """,
            org_id,
            agent_id,
        )

        # Runs with signals and their failure types
        signal_rows = await conn.fetch(
            """
            SELECT run_id, failure_type
            FROM failure_signals
            WHERE org_id = $1 AND agent_id = $2
              AND shadow = FALSE
              AND detected_at >= NOW() - INTERVAL '30 days'
            """,
            org_id,
            agent_id,
        )

    # Build lookup maps
    completion: dict[str, int] = {r["run_id"]: int(r["completion_tokens"]) for r in completion_rows}
    # run_id → set of failure types
    run_signals: dict[str, set] = {}
    for r in signal_rows:
        run_signals.setdefault(r["run_id"], set()).add(r["failure_type"])

    total_cost = 0.0
    wasted_cost = 0.0
    # failure_type → wasted_usd, affected_run_ids
    ft_cost: dict[str, dict] = {}

    for r in prompt_rows:
        run_id = r["run_id"]
        model = r["model"] or "unknown"
        prompt = int(r["prompt_tokens"])
        comp = completion.get(run_id, 0)
        cost = estimate_cost(model, prompt, comp)
        total_cost += cost
        if run_id in run_signals:
            wasted_cost += cost
            for ft in run_signals[run_id]:
                entry = ft_cost.setdefault(ft, {"wasted_usd": 0.0, "run_ids": set()})
                entry["wasted_usd"] += cost
                entry["run_ids"].add(run_id)

    cost_by_ft = sorted(
        [
            {
                "failure_type": ft,
                "wasted_usd": round(v["wasted_usd"], 4),
                "affected_runs": len(v["run_ids"]),
            }
            for ft, v in ft_cost.items()
        ],
        key=lambda x: x["wasted_usd"],
        reverse=True,
    )

    return {
        "total_cost_usd": round(total_cost, 4),
        "wasted_cost_usd": round(wasted_cost, 4),
        "wasted_pct": round(wasted_cost / total_cost, 3) if total_cost else 0.0,
        "cost_by_failure_type": cost_by_ft,
    }


async def agent_token_stats(org_id: str, agent_id: str) -> dict:
    """
    Per-window token usage stats for 1d / 7d / 30d.

    Three distinct spend metrics, because "wasted" means different things:

      • failed_run_tokens — total tokens on runs that had ≥1 live signal. Backward-
        looking attribution ("spend on runs that had a problem"). Exposed under the
        legacy ``wasted_*`` keys too, for compatibility.
      • excess_tokens — the *avoidable* portion: tokens above a healthy baseline
        (P75 of that agent version's clean runs) on failed runs. A 596k-token
        runaway whose healthy self is 6k contributes ~590k; a 6k run that tripped
        one signal contributes ~0.
      • prevented_tokens — tokens we actually *stopped from being spent*, only for
        runs a policy/approval halted in-path (``policy.triggered`` action_type=stop,
        ``approval.denied``/``timeout``). Post-hoc detectors prevent nothing, so they
        never count here. Estimated as (healthy baseline − tokens already spent).

    Also projects avoidable waste forward (Model 3): projected_monthly_excess_* is
    the window's excess run-rate extrapolated to 30 days.

    Returns:
      windows: {"1d": {...}, "7d": {...}, "30d": {...}}
      waste_by_failure_type: [{failure_type, wasted_tokens, wasted_cost_usd, affected_runs}]
        (tokens on failed runs, by type; sorted by wasted_cost_usd desc)
    """
    if not _pool:
        return {"windows": {}, "waste_by_failure_type": []}

    from explainer_svc.cost import estimate_cost

    async with _pool.acquire() as conn:
        # Per-run token totals + run timestamp.
        # prompt_tokens: direct SDK → llm.called, LangChain → llm.responded.
        # completion_tokens: always in llm.responded.
        token_rows = await conn.fetch(
            """
            SELECT
                run_id,
                MAX(agent_version) AS agent_version,
                MAX(CASE WHEN event_type = 'llm.called' THEN payload->>'model' END) AS model,
                SUM(COALESCE((payload->>'prompt_tokens')::int, 0)) AS prompt_tokens,
                SUM(CASE WHEN event_type = 'llm.responded'
                    THEN COALESCE((payload->>'completion_tokens')::int, 0) ELSE 0 END) AS completion_tokens,
                SUM(CASE WHEN event_type = 'llm.responded'
                    THEN COALESCE((payload->>'reasoning_tokens')::int, 0) ELSE 0 END) AS reasoning_tokens,
                MIN(received_at) AS run_start
            FROM events
            WHERE org_id = $1 AND agent_id = $2
              AND event_type IN ('llm.called', 'llm.responded')
              AND received_at >= NOW() - INTERVAL '30 days'
            GROUP BY run_id
            """,
            org_id,
            agent_id,
        )

        signal_rows = await conn.fetch(
            """
            SELECT run_id, failure_type
            FROM failure_signals
            WHERE org_id = $1 AND agent_id = $2
              AND shadow = FALSE
              AND detected_at >= NOW() - INTERVAL '30 days'
            """,
            org_id,
            agent_id,
        )

        # Runs a policy/approval actually halted in-path — the only runs with a
        # genuine "we prevented this spend" counterfactual. A structural detector
        # that fires on run.completed cannot prevent tokens already spent.
        block_rows = await conn.fetch(
            """
            SELECT DISTINCT run_id
            FROM events
            WHERE org_id = $1 AND agent_id = $2
              AND received_at >= NOW() - INTERVAL '30 days'
              AND (
                    (event_type = 'policy.triggered' AND payload->>'action_type' = 'stop')
                 OR event_type IN ('approval.denied', 'approval.timeout')
              )
            """,
            org_id,
            agent_id,
        )

    now = time.time()
    cutoffs = {"1d": now - 86400, "7d": now - 7 * 86400, "30d": now - 30 * 86400}
    window_days = {"1d": 1, "7d": 7, "30d": 30}

    run_signals: dict[str, set] = {}
    for r in signal_rows:
        run_signals.setdefault(r["run_id"], set()).add(r["failure_type"])

    blocked_run_ids = {r["run_id"] for r in block_rows}

    runs = []
    for r in token_rows:
        prompt = int(r["prompt_tokens"])
        comp = int(r["completion_tokens"])
        reasoning = int(r["reasoning_tokens"])
        cost = estimate_cost(r["model"] or "unknown", prompt, comp, reasoning)
        ts = r["run_start"].timestamp() if r["run_start"] else 0.0
        runs.append(
            {
                "run_id": r["run_id"],
                "agent_version": r["agent_version"] or "",
                "prompt_tokens": prompt,
                "completion_tokens": comp,
                "total_tokens": prompt + comp + reasoning,
                "cost": cost,
                "ts": ts,
                "failure_types": run_signals.get(r["run_id"], set()),
                "blocked": r["run_id"] in blocked_run_ids,
            }
        )

    # ── Healthy baselines. Isolate the *avoidable* (excess) portion of failed-run
    # spend and value what an in-path block prevented. Preference order:
    #   1. P75 of this agent version's clean (no-signal) runs
    #   2. P75 of the agent's clean runs across versions
    #   3. P50 (median) of ALL the agent's runs — robust last resort so heavily-
    #      failing agents (which lack a clean sample) still get a "typical run"
    #      floor. Median ignores the runaway tail, so excess still surfaces.
    _MIN_BASELINE_RUNS = 3

    def _pct(values: list, q: float) -> "Optional[float]":
        vals = sorted(values)
        n = len(vals)
        if n == 0:
            return None
        rank = q * (n - 1)  # linear interpolation — matches PERCENTILE_CONT
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        return float(vals[lo] + (vals[hi] - vals[lo]) * (rank - lo))

    clean_by_version: dict[str, list] = {}
    clean_all: list = []
    for r in runs:
        if not r["failure_types"]:
            clean_by_version.setdefault(r["agent_version"], []).append(r["total_tokens"])
            clean_all.append(r["total_tokens"])
    version_baseline = {
        v: _pct(tt, 0.75) for v, tt in clean_by_version.items() if len(tt) >= _MIN_BASELINE_RUNS
    }
    agent_baseline = _pct(clean_all, 0.75) if len(clean_all) >= _MIN_BASELINE_RUNS else None
    all_tokens = [r["total_tokens"] for r in runs]
    fallback_baseline = _pct(all_tokens, 0.50) if len(all_tokens) >= _MIN_BASELINE_RUNS else None

    def _baseline_for(version: str) -> "Optional[float]":
        b = version_baseline.get(version)
        if b is None:
            b = agent_baseline
        if b is None:
            b = fallback_baseline
        return b

    # Effective $/token across the window — used to price hypothetical tokens for
    # in-path blocks that stopped before any billable LLM call (no rate of their own).
    _tot_tok = sum(r["total_tokens"] for r in runs)
    _tot_cost = sum(r["cost"] for r in runs)
    avg_rate = (_tot_cost / _tot_tok) if _tot_tok else 0.0

    def _excess(r: dict) -> "tuple[int, float]":
        """Avoidable (tokens, cost) for one run: spend above its healthy baseline."""
        if not r["failure_types"]:
            return 0, 0.0
        base = _baseline_for(r["agent_version"])
        tt = r["total_tokens"]
        if base is None or tt <= base:
            return 0, 0.0
        ex = tt - base
        rate = (r["cost"] / tt) if tt else avg_rate
        return int(ex), rate * ex

    def _aggregate(filtered: list, days: int) -> dict:
        total_tokens = prompt_tokens = completion_tokens = 0
        failed_tokens = excess_tokens = prevented_tokens = 0
        total_cost = failed_cost = excess_cost = prevented_cost = 0.0
        failed_run_count = blocked_run_count = 0
        for r in filtered:
            tt = r["total_tokens"]
            total_tokens += tt
            prompt_tokens += r["prompt_tokens"]
            completion_tokens += r["completion_tokens"]
            total_cost += r["cost"]
            rate = (r["cost"] / tt) if tt else avg_rate
            base = _baseline_for(r["agent_version"])
            if r["failure_types"]:
                failed_tokens += tt
                failed_cost += r["cost"]
                failed_run_count += 1
                # Avoidable = spend above a healthy run of the same agent version.
                ex_tok, ex_cost = _excess(r)
                excess_tokens += ex_tok
                excess_cost += ex_cost
            if r["blocked"]:
                blocked_run_count += 1
                # Counterfactual: absent the block it would have run to at least a
                # normal completion. Prevented = normal spend − spend so far.
                if base is not None:
                    prev = base - tt
                    if prev > 0:
                        prevented_tokens += int(prev)
                        prevented_cost += (rate or avg_rate) * prev
        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_cost_usd": round(total_cost, 4),
            "run_count": len(filtered),
            # Attribution: all tokens on runs that had ≥1 signal.
            "failed_run_tokens": failed_tokens,
            "failed_run_cost_usd": round(failed_cost, 4),
            "failed_run_count": failed_run_count,
            "failed_pct": round(failed_cost / total_cost, 3) if total_cost else 0.0,
            # Avoidable: spend above the healthy baseline on failed runs.
            "excess_tokens": excess_tokens,
            "excess_cost_usd": round(excess_cost, 4),
            "baseline_tokens": (
                round(agent_baseline)
                if agent_baseline is not None
                else round(fallback_baseline)
                if fallback_baseline is not None
                else None
            ),
            # Prevented: in-path blocks that stopped a run before it ran on.
            "prevented_tokens": prevented_tokens,
            "prevented_cost_usd": round(prevented_cost, 4),
            "blocked_run_count": blocked_run_count,
            # Forward projection (Model 3): a single rate, filled in below so it is
            # identical across windows (a longer window must not project *less*).
            "projected_monthly_excess_tokens": 0,
            "projected_monthly_excess_cost_usd": 0.0,
            # ── Back-compat aliases: legacy "wasted_*" == failed-run attribution ──
            "wasted_tokens": failed_tokens,
            "wasted_cost_usd": round(failed_cost, 4),
            "wasted_pct": round(failed_cost / total_cost, 3) if total_cost else 0.0,
            "wasted_run_count": failed_run_count,
        }

    windows = {
        w: _aggregate([r for r in runs if r["ts"] >= cut], window_days[w])
        for w, cut in cutoffs.items()
    }

    # Forward projection: "if this continues unfixed, ~$X/month." It is a *rate*, so
    # it must not depend on which window the user is viewing. Base it on the recent
    # 7-day run-rate of avoidable spend, normalised by how many days were actually
    # observed (guards agents with <7 days of history), then scale to 30 days.
    recent = [r for r in runs if r["ts"] >= cutoffs["7d"]]
    recent_ex_tok = 0
    recent_ex_cost = 0.0
    for r in recent:
        et, ec = _excess(r)
        recent_ex_tok += et
        recent_ex_cost += ec
    recent_ts = [r["ts"] for r in recent if r["ts"] > 0]
    # Floor the observed span at 3 days so a single-day burst can't extrapolate ×30.
    observed_days = min(7.0, max(3.0, (now - min(recent_ts)) / 86400)) if recent_ts else 7.0
    proj_scale = 30.0 / observed_days
    proj_tokens = int(recent_ex_tok * proj_scale)
    proj_cost = round(recent_ex_cost * proj_scale, 4)
    for w in windows.values():
        w["projected_monthly_excess_tokens"] = proj_tokens
        w["projected_monthly_excess_cost_usd"] = proj_cost

    ft_stats: dict[str, dict] = {}
    for r in runs:
        for ft in r["failure_types"]:
            e = ft_stats.setdefault(ft, {"wasted_tokens": 0, "wasted_cost": 0.0, "run_ids": set()})
            e["wasted_tokens"] += r["total_tokens"]
            e["wasted_cost"] += r["cost"]
            e["run_ids"].add(r["run_id"])

    waste_by_ft = sorted(
        [
            {
                "failure_type": ft,
                "wasted_tokens": v["wasted_tokens"],
                "wasted_cost_usd": round(v["wasted_cost"], 4),
                "affected_runs": len(v["run_ids"]),
            }
            for ft, v in ft_stats.items()
        ],
        key=lambda x: x["wasted_cost_usd"],
        reverse=True,
    )

    return {"windows": windows, "waste_by_failure_type": waste_by_ft}


# ── Deploy regression check ────────────────────────────────────────────────────


async def deploy_regression_check(org_id: str, agent_id: str) -> list:
    """
    For each deploy event in the last 7 days, compare overall signal rates in the
    2-hour window before vs after the deploy.

    Skips deploys with fewer than 3 runs in either window — not enough data.
    Returns: [{deploy_id, version, deployed_at, before_runs, before_signals,
               before_rate, after_runs, after_signals, after_rate, delta_rate, is_regression}]
    """
    if not _pool:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH deploys AS (
                SELECT id, version, deployed_at
                FROM deploy_events
                WHERE org_id = $1 AND agent_id = $2
                  AND deployed_at >= NOW() - INTERVAL '7 days'
                ORDER BY deployed_at DESC
                LIMIT 10
            ),
            run_windows AS (
                SELECT
                    d.id          AS deploy_id,
                    d.version,
                    d.deployed_at,
                    pr.run_id,
                    CASE
                        WHEN pr.processed_at < d.deployed_at THEN 'before'
                        ELSE 'after'
                    END           AS window
                FROM deploys d
                JOIN processed_runs pr
                    ON pr.org_id = $1 AND pr.agent_id = $2
                    AND pr.processed_at >= d.deployed_at - INTERVAL '2 hours'
                    AND pr.processed_at <= d.deployed_at + INTERVAL '2 hours'
            ),
            signal_flags AS (
                SELECT DISTINCT run_id
                FROM failure_signals
                WHERE org_id = $1 AND agent_id = $2 AND shadow = FALSE
            )
            SELECT
                rw.deploy_id,
                rw.version,
                rw.deployed_at,
                rw.window,
                COUNT(DISTINCT rw.run_id)     AS total_runs,
                COUNT(DISTINCT sf.run_id)     AS signal_runs
            FROM run_windows rw
            LEFT JOIN signal_flags sf ON sf.run_id = rw.run_id
            GROUP BY rw.deploy_id, rw.version, rw.deployed_at, rw.window
            ORDER BY rw.deployed_at DESC, rw.window
            """,
            org_id,
            agent_id,
        )

    # Group by deploy_id
    from collections import defaultdict

    by_deploy: dict = defaultdict(dict)
    for r in rows:
        by_deploy[(r["deploy_id"], r["version"], r["deployed_at"])][r["window"]] = {
            "total_runs": int(r["total_runs"]),
            "signal_runs": int(r["signal_runs"]),
        }

    results = []
    for (deploy_id, version, deployed_at), windows in by_deploy.items():
        before = windows.get("before", {"total_runs": 0, "signal_runs": 0})
        after = windows.get("after", {"total_runs": 0, "signal_runs": 0})

        # Need at least 3 runs per window for a meaningful comparison
        if before["total_runs"] < 3 or after["total_runs"] < 3:
            continue

        before_rate = before["signal_runs"] / before["total_runs"]
        after_rate = after["signal_runs"] / after["total_runs"]
        delta = after_rate - before_rate

        ts = deployed_at.timestamp() if hasattr(deployed_at, "timestamp") else float(deployed_at)
        results.append(
            {
                "deploy_id": int(deploy_id),
                "version": version,
                "deployed_at": ts,
                "before_runs": before["total_runs"],
                "before_signals": before["signal_runs"],
                "before_rate": round(before_rate, 3),
                "after_runs": after["total_runs"],
                "after_signals": after["signal_runs"],
                "after_rate": round(after_rate, 3),
                "delta_rate": round(delta, 3),
                "is_regression": delta > 0.15,  # >15pp increase is a regression
            }
        )

    return sorted(results, key=lambda x: x["deployed_at"], reverse=True)


# ── Agent fixes list ───────────────────────────────────────────────────────────


async def list_agent_fixes(org_id: str, agent_id: str) -> list:
    """
    All fixes applied to signals for this agent, with recurrence status computed inline.

    Returns: [{id, signal_id, run_id, failure_type, severity, fix_type, applied_via,
               langfuse_prompt_name, langfuse_version, applied_at,
               runs_after, recurrences_after, verdict}]
    """
    if not _pool:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                f.id,
                f.run_id,
                f.signal_id,
                f.fix_type,
                f.applied_via,
                f.langfuse_prompt_name,
                f.langfuse_version,
                f.applied_at,
                fs.failure_type,
                fs.severity,
                fs.agent_version,
                -- Runs completed after this fix was applied
                (
                    SELECT COUNT(DISTINCT e.run_id)
                    FROM events e
                    WHERE e.org_id = $1 AND e.agent_id = $2
                      AND e.received_at > f.applied_at
                ) AS runs_after,
                -- Recurrences of the same failure type after fix
                (
                    SELECT COUNT(*)
                    FROM failure_signals fs2
                    WHERE fs2.org_id     = $1
                      AND fs2.agent_id   = $2
                      AND fs2.failure_type = fs.failure_type
                      AND fs2.detected_at  > f.applied_at
                      AND fs2.shadow       = FALSE
                ) AS recurrences_after
            FROM fixes f
            JOIN failure_signals fs ON fs.id = f.signal_id
            WHERE fs.org_id = $1 AND fs.agent_id = $2
            ORDER BY f.applied_at DESC
            LIMIT 100
            """,
            org_id,
            agent_id,
        )

    def _ts(v):
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    results = []
    for r in rows:
        runs = int(r["runs_after"] or 0)
        recs = int(r["recurrences_after"] or 0)
        verdict = (
            "verified"
            if runs >= 10 and recs == 0
            else (
                "likely_fixed"
                if runs >= 5 and recs == 0
                else "still_occurring"
                if recs > 0
                else "insufficient_data"
            )
        )
        results.append(
            {
                "id": int(r["id"]),
                "signal_id": int(r["signal_id"]),
                "run_id": r["run_id"],
                "failure_type": r["failure_type"],
                "severity": r["severity"],
                "agent_version": r["agent_version"],
                "fix_type": r["fix_type"],
                "applied_via": r["applied_via"],
                "langfuse_prompt_name": r["langfuse_prompt_name"],
                "langfuse_version": r["langfuse_version"],
                "applied_at": _ts(r["applied_at"]),
                "runs_after": runs,
                "recurrences_after": recs,
                "verdict": verdict,
            }
        )
    return results


# ── User impact ────────────────────────────────────────────────────────────────


async def agent_user_impact(org_id: str, agent_id: str) -> list:
    """
    Unique users (proxied by an MD5 digest of input_text from run.started, computed
    here rather than transmitted) affected per failure type over the last 30 days.

    Returns: [{failure_type, affected_users, total_users, user_impact_rate}]
    """
    if not _pool:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH run_inputs AS (
                SELECT
                    e.run_id,
                    md5(e.payload->>'input_text') AS input_hash
                FROM events e
                WHERE e.org_id      = $1
                  AND e.agent_id    = $2
                  AND e.event_type  = 'run.started'
                  AND e.payload->>'input_text' IS NOT NULL
                  AND e.received_at >= NOW() - INTERVAL '30 days'
            ),
            total_users AS (
                SELECT COUNT(DISTINCT input_hash) AS cnt FROM run_inputs
            ),
            affected AS (
                SELECT
                    fs.failure_type,
                    COUNT(DISTINCT ri.input_hash) AS affected_users
                FROM failure_signals fs
                JOIN run_inputs ri ON ri.run_id = fs.run_id
                WHERE fs.org_id     = $1
                  AND fs.agent_id   = $2
                  AND fs.shadow     = FALSE
                  AND fs.detected_at >= NOW() - INTERVAL '30 days'
                GROUP BY fs.failure_type
            )
            SELECT
                a.failure_type,
                a.affected_users::int        AS affected_users,
                tu.cnt::int                  AS total_users,
                ROUND(a.affected_users::numeric / NULLIF(tu.cnt, 0), 3) AS user_impact_rate
            FROM affected a
            CROSS JOIN total_users tu
            ORDER BY a.affected_users DESC
            """,
            org_id,
            agent_id,
        )

    return [
        {
            "failure_type": r["failure_type"],
            "affected_users": int(r["affected_users"]),
            "total_users": int(r["total_users"]),
            "user_impact_rate": float(r["user_impact_rate"] or 0),
        }
        for r in rows
    ]


# ── Performance trends (Phase 4.4) ──────────────────────────────────────────────


async def agent_performance_trends(org_id: str, agent_id: str, window_days: int) -> dict:
    """
    Per-agent performance trends: daily time series (structural/semantic
    signal rate, cost, latency), failure-mode deltas (current window_days vs
    the immediately preceding window_days), and a self-baseline comparison
    (ALWAYS last 30 days vs this agent's own rate 90-30 days ago — the same
    fixed reference get_agent_health_score uses, independent of window_days;
    see performance_trends.py's module docstring for why it's pinned rather
    than parameterized).

    All arithmetic lives in api_svc/performance_trends.py (pure, unit-tested
    without a DB) — this function only fetches raw rows and hands them over.
    """
    empty = {"points": [], "failure_mode_deltas": [], "baseline_comparisons": []}
    if not _pool:
        return empty

    from api_svc.performance_trends import (
        build_day_buckets,
        compute_baseline_comparisons,
        compute_daily_points,
        compute_failure_mode_deltas,
    )
    from explainer_svc.cost import estimate_cost

    window_str = str(window_days)
    double_window_str = str(window_days * 2)

    async with _pool.acquire() as conn:
        runs_rows = await conn.fetch(
            """
            SELECT DATE_TRUNC('day', processed_at AT TIME ZONE 'UTC')::date AS day,
                   COUNT(DISTINCT run_id)::int AS total_runs
            FROM processed_runs
            WHERE org_id = $1 AND agent_id = $2
              AND processed_at >= NOW() - ($3 || ' days')::interval
            GROUP BY day
            """,
            org_id,
            agent_id,
            window_str,
        )
        runs_by_day = {str(r["day"]): r["total_runs"] for r in runs_rows}

        signal_rows = await conn.fetch(
            """
            SELECT DATE_TRUNC('day', pr.processed_at AT TIME ZONE 'UTC')::date AS day,
                   (fs.source = 'structural') AS is_structural,
                   COUNT(DISTINCT fs.run_id)::int AS affected_runs
            FROM processed_runs pr
            JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.org_id = pr.org_id AND fs.agent_id = pr.agent_id
            WHERE pr.org_id = $1 AND pr.agent_id = $2
              AND fs.shadow = FALSE
              AND pr.processed_at >= NOW() - ($3 || ' days')::interval
            GROUP BY day, is_structural
            """,
            org_id,
            agent_id,
            window_str,
        )
        structural_by_day: dict = {}
        semantic_by_day: dict = {}
        for r in signal_rows:
            day = str(r["day"])
            (structural_by_day if r["is_structural"] else semantic_by_day)[day] = r["affected_runs"]

        token_rows = await conn.fetch(
            """
            SELECT e.run_id,
                   DATE_TRUNC('day', pr.processed_at AT TIME ZONE 'UTC')::date AS day,
                   MAX(CASE WHEN e.event_type = 'llm.called' THEN e.payload->>'model' END) AS model,
                   SUM(COALESCE((e.payload->>'prompt_tokens')::int, 0)) AS prompt_tokens,
                   SUM(CASE WHEN e.event_type = 'llm.responded'
                            THEN COALESCE((e.payload->>'completion_tokens')::int, 0) ELSE 0 END)
                       AS completion_tokens
            FROM events e
            JOIN processed_runs pr
                ON pr.run_id = e.run_id AND pr.org_id = e.org_id AND pr.agent_id = e.agent_id
            WHERE e.org_id = $1 AND e.agent_id = $2
              AND e.event_type IN ('llm.called', 'llm.responded')
              AND pr.processed_at >= NOW() - ($3 || ' days')::interval
            GROUP BY e.run_id, day
            """,
            org_id,
            agent_id,
            window_str,
        )
        cost_by_day: dict = {}
        for r in token_rows:
            day = str(r["day"])
            cost = estimate_cost(
                r["model"] or "unknown",
                int(r["prompt_tokens"] or 0),
                int(r["completion_tokens"] or 0),
            )
            cost_by_day[day] = cost_by_day.get(day, 0.0) + cost

        latency_rows = await conn.fetch(
            """
            SELECT DATE_TRUNC('day', pr.processed_at AT TIME ZONE 'UTC')::date AS day,
                   AVG((e.payload->>'latency_ms')::float) AS avg_latency_ms
            FROM events e
            JOIN processed_runs pr
                ON pr.run_id = e.run_id AND pr.org_id = e.org_id AND pr.agent_id = e.agent_id
            WHERE e.org_id = $1 AND e.agent_id = $2
              AND e.event_type = 'llm.responded'
              AND e.payload->>'latency_ms' IS NOT NULL
              AND pr.processed_at >= NOW() - ($3 || ' days')::interval
            GROUP BY day
            """,
            org_id,
            agent_id,
            window_str,
        )
        latency_by_day = {
            str(r["day"]): (float(r["avg_latency_ms"]) if r["avg_latency_ms"] is not None else None)
            for r in latency_rows
        }

        buckets = build_day_buckets(window_days)
        points = compute_daily_points(
            buckets, runs_by_day, structural_by_day, semantic_by_day, cost_by_day, latency_by_day
        )

        # Current window_days vs the immediately preceding window_days — total
        # run counts and per-failure-type counts, each via one query with a
        # CASE-based period label rather than two near-duplicate queries.
        period_totals = await conn.fetch(
            """
            SELECT
                CASE WHEN processed_at >= NOW() - ($3 || ' days')::interval
                     THEN 'current' ELSE 'previous' END AS period,
                COUNT(DISTINCT run_id)::int AS total_runs
            FROM processed_runs
            WHERE org_id = $1 AND agent_id = $2
              AND processed_at >= NOW() - ($4 || ' days')::interval
            GROUP BY period
            """,
            org_id,
            agent_id,
            window_str,
            double_window_str,
        )
        period_total_map = {r["period"]: r["total_runs"] for r in period_totals}
        current_total = period_total_map.get("current", 0)
        previous_total = period_total_map.get("previous", 0)

        period_counts = await conn.fetch(
            """
            SELECT
                fs.failure_type,
                CASE WHEN pr.processed_at >= NOW() - ($3 || ' days')::interval
                     THEN 'current' ELSE 'previous' END AS period,
                COUNT(DISTINCT fs.run_id)::int AS affected_runs
            FROM failure_signals fs
            JOIN processed_runs pr
                ON pr.run_id = fs.run_id AND pr.org_id = fs.org_id AND pr.agent_id = fs.agent_id
            WHERE fs.org_id = $1 AND fs.agent_id = $2 AND fs.shadow = FALSE
              AND pr.processed_at >= NOW() - ($4 || ' days')::interval
            GROUP BY fs.failure_type, period
            """,
            org_id,
            agent_id,
            window_str,
            double_window_str,
        )
        current_counts: dict = {}
        previous_counts: dict = {}
        for r in period_counts:
            (current_counts if r["period"] == "current" else previous_counts)[r["failure_type"]] = (
                r["affected_runs"]
            )

        deltas = compute_failure_mode_deltas(
            current_counts, previous_counts, current_total, previous_total
        )

        # Self-baseline: fixed last-30d vs 90-30d-ago, independent of window_days.
        baseline_totals = await conn.fetch(
            """
            SELECT
                CASE WHEN processed_at >= NOW() - INTERVAL '30 days' THEN 'current' ELSE 'baseline' END
                    AS period,
                COUNT(DISTINCT run_id)::int AS total_runs
            FROM processed_runs
            WHERE org_id = $1 AND agent_id = $2
              AND processed_at >= NOW() - INTERVAL '90 days'
            GROUP BY period
            """,
            org_id,
            agent_id,
        )
        baseline_total_map = {r["period"]: r["total_runs"] for r in baseline_totals}
        baseline_current_total = baseline_total_map.get("current", 0)
        baseline_ref_total = baseline_total_map.get("baseline", 0)

        baseline_period_counts = await conn.fetch(
            """
            SELECT
                fs.failure_type,
                CASE WHEN pr.processed_at >= NOW() - INTERVAL '30 days' THEN 'current' ELSE 'baseline' END
                    AS period,
                COUNT(DISTINCT fs.run_id)::int AS affected_runs
            FROM failure_signals fs
            JOIN processed_runs pr
                ON pr.run_id = fs.run_id AND pr.org_id = fs.org_id AND pr.agent_id = fs.agent_id
            WHERE fs.org_id = $1 AND fs.agent_id = $2 AND fs.shadow = FALSE
              AND pr.processed_at >= NOW() - INTERVAL '90 days'
            GROUP BY fs.failure_type, period
            """,
            org_id,
            agent_id,
        )
        baseline_current_counts: dict = {}
        baseline_ref_counts: dict = {}
        for r in baseline_period_counts:
            (baseline_current_counts if r["period"] == "current" else baseline_ref_counts)[
                r["failure_type"]
            ] = r["affected_runs"]

        current_rates = {
            ft: round(count / baseline_current_total, 4) if baseline_current_total else 0.0
            for ft, count in baseline_current_counts.items()
        }
        baseline_comparisons = compute_baseline_comparisons(
            current_rates, baseline_ref_counts, baseline_ref_total
        )

    return {
        "points": points,
        "failure_mode_deltas": deltas,
        "baseline_comparisons": baseline_comparisons,
    }


# ── Cross-run patterns ─────────────────────────────────────────────────────────


def _is_trending_up(daily_counts: list) -> bool:
    if len(daily_counts) < 3:
        return False
    recent = sum(daily_counts[-3:]) / 3
    earlier = sum(daily_counts[:3]) / 3
    return recent > earlier * 1.3


async def cross_run_patterns(org_id: str) -> list:
    """
    Per (agent_id × failure_type) 7-day signal frequency. Only live signals (shadow=FALSE).
    Returns a list of agent dicts, each containing detector rows with daily buckets and summary stats.
    Filters out pairs with only 1 total occurrence (noise).
    """
    if not _pool:
        return []

    import datetime
    from collections import defaultdict

    today = datetime.date.today()
    days = [today - datetime.timedelta(days=i) for i in range(6, -1, -1)]  # oldest → newest

    async with _pool.acquire() as conn:
        sig_rows = await conn.fetch(
            """
            SELECT
                agent_id,
                failure_type,
                DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC')::date AS day,
                COUNT(*)                  AS signal_count,
                COUNT(DISTINCT run_id)    AS affected_runs
            FROM failure_signals
            WHERE shadow = FALSE
              AND detected_at >= NOW() - INTERVAL '7 days'
              AND org_id = $1
            GROUP BY agent_id, failure_type, DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC')::date
            ORDER BY agent_id, failure_type, day
            """,
            org_id,
        )

        run_rows = await conn.fetch(
            """
            SELECT agent_id, COUNT(DISTINCT run_id) AS total_runs
            FROM processed_runs
            WHERE processed_at >= NOW() - INTERVAL '7 days'
              AND org_id = $1
            GROUP BY agent_id
            """,
            org_id,
        )

    # agent_id → total runs in period
    agent_total_runs: dict = {r["agent_id"]: int(r["total_runs"]) for r in run_rows}

    # (agent_id, failure_type, date) → {signal_count, affected_runs}
    daily: dict = defaultdict(lambda: defaultdict(lambda: {"signal_count": 0, "affected_runs": 0}))
    for r in sig_rows:
        d = r["day"] if isinstance(r["day"], datetime.date) else r["day"].date()
        daily[(r["agent_id"], r["failure_type"])][d] = {
            "signal_count": int(r["signal_count"]),
            "affected_runs": int(r["affected_runs"]),
        }

    # Build per-agent result, filtering single-occurrence noise
    agent_map: dict = defaultdict(list)
    for (agent_id, failure_type), day_map in daily.items():
        buckets = [
            {
                "date": d.isoformat(),
                "signal_count": day_map[d]["signal_count"] if d in day_map else 0,
                "affected_runs": day_map[d]["affected_runs"] if d in day_map else 0,
            }
            for d in days
        ]
        total_occurrences = sum(b["signal_count"] for b in buckets)
        total_affected_runs = sum(b["affected_runs"] for b in buckets)

        if total_occurrences <= 1:
            continue

        total_runs = agent_total_runs.get(agent_id, 0)
        pct_of_runs = round(total_affected_runs / total_runs, 4) if total_runs else 0.0

        trending_up = _is_trending_up([b["signal_count"] for b in buckets])

        agent_map[agent_id].append(
            {
                "failure_type": failure_type,
                "days": buckets,
                "total_occurrences": total_occurrences,
                "total_affected_runs": total_affected_runs,
                "total_runs": total_runs,
                "pct_of_runs": pct_of_runs,
                "trending_up": trending_up,
            }
        )

    # Sort rows within each agent by total_occurrences desc
    return [
        {
            "agent_id": agent_id,
            "rows": sorted(rows, key=lambda r: r["total_occurrences"], reverse=True),
        }
        for agent_id, rows in sorted(agent_map.items())
    ]


# ── User feedback (Slack buttons) ──────────────────────────────────────────────


async def mark_signal_resolved(org_id: str, signal_id: int) -> bool:
    """Set resolved_at=NOW() on the signal. Returns True if the row was found."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE failure_signals SET resolved_at = NOW() "
            "WHERE id = $1 AND org_id = $2 AND resolved_at IS NULL",
            signal_id,
            org_id,
        )
    return result.split()[-1] != "0"


async def record_false_positive(
    org_id: str,
    signal_id: int,
    agent_id: str,
    failure_type: str,
) -> dict:
    """Increment fp_count and raise confidence_floor by 0.1.
    Sets silenced=TRUE when fp_count reaches 3.
    Returns the updated override row."""
    if not _pool:
        return {}
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO agent_detector_overrides (org_id, agent_id, failure_type, fp_count, confidence_floor, silenced)
            VALUES ($1, $2, $3, 1, 0.1, FALSE)
            ON CONFLICT (org_id, agent_id, failure_type) DO UPDATE
              SET fp_count         = agent_detector_overrides.fp_count + 1,
                  confidence_floor = LEAST(1.0, agent_detector_overrides.confidence_floor + 0.1),
                  silenced         = (agent_detector_overrides.fp_count + 1) >= 3,
                  updated_at       = NOW()
            RETURNING org_id, agent_id, failure_type, fp_count, confidence_floor, silenced
            """,
            org_id,
            agent_id,
            failure_type,
        )
    return dict(row) if row else {}


async def snooze_pattern(
    org_id: str,
    agent_id: str,
    failure_type: str,
    hours: int = 24,
) -> dict:
    """ "Snooze this pattern" — sets snoozed_until = NOW() + hours, leaving
    fp_count/confidence_floor/silenced untouched (a deliberate temporary
    mute is a different signal than accumulated false-positive feedback;
    the two mechanisms don't interact). Returns the updated row."""
    if not _pool:
        return {}
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO agent_detector_overrides (org_id, agent_id, failure_type, snoozed_until)
            VALUES ($1, $2, $3, NOW() + ($4 || ' hours')::interval)
            ON CONFLICT (org_id, agent_id, failure_type) DO UPDATE
              SET snoozed_until = NOW() + ($4 || ' hours')::interval,
                  updated_at    = NOW()
            RETURNING org_id, agent_id, failure_type, snoozed_until
            """,
            org_id,
            agent_id,
            failure_type,
            str(hours),
        )
    return dict(row) if row else {}


async def reset_detector_override(org_id: str, agent_id: str, failure_type: str) -> bool:
    """Reset false-positive suppression for a detector on an agent."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE agent_detector_overrides
            SET fp_count=0, confidence_floor=0.0, silenced=FALSE, updated_at=NOW()
            WHERE org_id=$1 AND agent_id=$2 AND failure_type=$3
            """,
            org_id,
            agent_id,
            failure_type,
        )
    return result.split()[-1] != "0"


async def get_detector_override(org_id: str, agent_id: str, failure_type: str) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM agent_detector_overrides WHERE org_id=$1 AND agent_id=$2 AND failure_type=$3",
            org_id,
            agent_id,
            failure_type,
        )
    return dict(row) if row else None


# ── API key management ─────────────────────────────────────────────────────────


def _mask_key(key: str) -> str:
    """Return first 10 + last 4 chars with ellipsis — never expose the full secret."""
    if len(key) <= 14:
        return key[:4] + "…"
    return key[:10] + "…" + key[-4:]


async def list_api_keys(
    org_id: Optional[str] = None,
    active_only: bool = True,
    limit: int = 100,
) -> list:
    """List API keys. org_id scopes to one org's own keys; omit only for admin/dev tooling."""
    if not _pool:
        return []
    conditions = []
    params: list = []
    if active_only:
        params.append(True)
        conditions.append(f"k.active = ${len(params)}")
    if org_id:
        params.append(org_id)
        conditions.append(f"k.org_id = ${len(params)}")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT k.id, k.org_id, k.active,
                   k.rate_limit_rpm, k.created_at, o.name AS org_name
            FROM api_keys k
            LEFT JOIN organizations o ON o.id = k.org_id
            {where}
            ORDER BY k.id DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
    return [
        {
            "id": r["id"],
            "org_id": r["org_id"],
            "org_name": r["org_name"],
            "active": r["active"],
            "rate_limit_rpm": r["rate_limit_rpm"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


async def create_api_key(
    org_id: str,
    org_name: Optional[str] = None,
    rate_limit_rpm: int = 600,
) -> dict:
    """Create an org-scoped API key. Not tied to a single agent_id — an org can
    have many agents, discovered dynamically on first ingest."""
    import secrets as _sec

    key = "dt_" + _sec.token_urlsafe(32)
    name = org_name or org_id
    if not _pool:
        raise RuntimeError("DB pool not ready")
    async with _pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO organizations (id, name) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                org_id,
                name,
            )
            row = await conn.fetchrow(
                """
                INSERT INTO api_keys (key, org_id, rate_limit_rpm)
                VALUES ($1, $2, $3)
                RETURNING id, created_at
                """,
                key,
                org_id,
                rate_limit_rpm,
            )
    return {
        "id": row["id"],
        "key": key,
        "key_prefix": _mask_key(key),
        "org_id": org_id,
        "org_name": name,
        "rate_limit_rpm": rate_limit_rpm,
        "created_at": row["created_at"].isoformat(),
    }


async def revoke_api_key(org_id: str, key_id: int) -> bool:
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE api_keys SET active = FALSE WHERE id = $1 AND org_id = $2 AND active = TRUE",
            key_id,
            org_id,
        )
    return result.split()[-1] != "0"


# ── Custom detectors ───────────────────────────────────────────────────────────


def _custom_detector_row(r) -> dict:
    return {
        "id": r["id"],
        "agent_id": r["agent_id"],
        "name": r["name"],
        "description": r["description"],
        "config": r["config_json"]
        if isinstance(r["config_json"], dict)
        else _json_mod.loads(r["config_json"]),
        "status": r["status"],
        "created_at": r["created_at"].isoformat(),
        "total_runs": r["total_runs"],
        "shadow_fire_count": r["shadow_fire_count"],
    }


async def list_custom_detectors(org_id: str, agent_id: Optional[str] = None) -> list[dict]:
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        if agent_id:
            rows = await conn.fetch(
                "SELECT * FROM custom_detectors WHERE org_id = $1 AND (agent_id = $2 OR agent_id = '*') "
                "ORDER BY created_at DESC",
                org_id,
                agent_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM custom_detectors WHERE org_id = $1 ORDER BY created_at DESC", org_id
            )
    return [_custom_detector_row(r) for r in rows]


async def get_custom_detector(org_id: str, detector_id: int) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM custom_detectors WHERE id = $1 AND org_id = $2", detector_id, org_id
        )
    return _custom_detector_row(row) if row else None


async def create_custom_detector(
    org_id: str, agent_id: str, name: str, description: str, config: dict
) -> dict:
    if not _pool:
        raise RuntimeError("DB pool not initialized")
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO custom_detectors (agent_id, name, description, config_json, status, org_id)
            VALUES ($1, $2, $3, $4::jsonb, 'shadow', $5)
            RETURNING *
            """,
            agent_id,
            name,
            description,
            _json_mod.dumps(config),
            org_id,
        )
    return _custom_detector_row(row)


async def update_custom_detector_status(
    org_id: str, detector_id: int, status: str
) -> Optional[dict]:
    if not _pool:
        raise RuntimeError("DB pool not initialized")
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE custom_detectors SET status = $3 WHERE id = $1 AND org_id = $2 RETURNING *",
            detector_id,
            org_id,
            status,
        )
    return _custom_detector_row(row) if row else None


async def delete_custom_detector(org_id: str, detector_id: int) -> bool:
    if not _pool:
        raise RuntimeError("DB pool not initialized")
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM custom_detectors WHERE id = $1 AND org_id = $2", detector_id, org_id
        )
    return result.split()[-1] != "0"


async def get_custom_detector_shadow_stats(org_id: str, detector_id: int) -> dict:
    """Return shadow evaluation stats: total runs evaluated and sample firing runs."""
    if not _pool:
        return {"total_runs": 0, "fire_count": 0, "fire_rate": 0.0, "sample_runs": []}
    async with _pool.acquire() as conn:
        stats = await conn.fetchrow(
            """
            SELECT
                COUNT(*)                                   AS total_runs,
                COUNT(*) FILTER (WHERE fired = TRUE)       AS fire_count
            FROM custom_detector_results
            WHERE detector_id = $1 AND org_id = $2
            """,
            detector_id,
            org_id,
        )
        samples = await conn.fetch(
            """
            SELECT run_id, agent_id, evaluated_at
            FROM custom_detector_results
            WHERE detector_id = $1 AND org_id = $2 AND fired = TRUE
            ORDER BY evaluated_at DESC
            LIMIT 5
            """,
            detector_id,
            org_id,
        )
    total = stats["total_runs"] or 0
    fires = stats["fire_count"] or 0
    return {
        "total_runs": total,
        "fire_count": fires,
        "fire_rate": round(fires / total, 4) if total > 0 else 0.0,
        "sample_runs": [
            {
                "run_id": r["run_id"],
                "agent_id": r["agent_id"],
                "evaluated_at": r["evaluated_at"].isoformat(),
            }
            for r in samples
        ],
    }
