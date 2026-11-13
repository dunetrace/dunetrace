"""
Ordered, versioned database migrations — the owner of the shared schema.

WHY THIS EXISTS. Dunetrace's services communicate only through one Postgres
database, which makes the schema the inter-service interface. That interface had
no owner: every service declared the tables it touched with
``CREATE TABLE IF NOT EXISTS`` and grew them with ``ALTER TABLE ... ADD COLUMN IF
NOT EXISTS``, so "whichever service starts first wins" was the entire contract.
It worked only because every change so far had been additive, and it already
misfired — ``policies.signature`` was declared by exactly one service while
another service's query selected it inside a try/except that returned an empty
list, so a cold start in the wrong order silently disabled every runtime
guardrail in the fleet.

Additive-only is also a ceiling, not just a risk: changing a primary key across
two services cannot be expressed as start-order-independent idempotent DDL.
That is what forced this module, and the composite ``(org_id, run_id)`` key in
migration 2 is the first change that needed it.

THE RULE. A table touched by more than one service belongs here. A table with a
single owner may stay in that service's own ``ensure_schema`` — the point is
that shared definitions have one home, not that every ``CREATE TABLE`` moves.

HOW IT RUNS. Every service calls ``apply_migrations()`` before its own schema
setup. A Postgres advisory lock serialises concurrent startups, so N replicas
booting together apply each migration exactly once; each migration runs in its
own transaction, so a failure leaves the database at the last good version
rather than half-applied. Services that depend on a specific shape call
``require_schema_version()`` and refuse to start below it, which turns the
old silent-wrong-order failure into a loud one.

This package is on every service's PYTHONPATH and depends only on the driver
connection passed in, so it stays usable from ingest (which does not have the
SDK) as well as from the SDK-carrying services.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, AsyncIterator, List, Tuple

logger = logging.getLogger("dunetrace.migrations")

# Arbitrary but fixed: two processes must pick the same lock id to serialise.
_MIGRATION_LOCK_ID = 8_431_002

# (version, name, sql). Append only — never renumber or edit an applied
# migration, or databases at different versions diverge silently.
MIGRATIONS: List[Tuple[int, str, str]] = [
    (
        1,
        "baseline",
        # Deliberately empty. Version 1 records "this database is under
        # migration control" without asserting anything about what is already
        # there, so an existing deployment adopts the runner without a rewrite.
        "SELECT 1;",
    ),
    (
        2,
        "composite_run_key",
        # run_id is caller-supplied — the SDK exposes run_id= and the OTLP path
        # derives it from a caller-supplied trace id — so a bare
        # `run_id TEXT PRIMARY KEY` is a cross-tenant collision: two orgs
        # emitting run_id="session-1" meant one silently lost its run to
        # ON CONFLICT DO NOTHING, and the survivor's run detail returned the
        # other tenant's system prompts and tool arguments.
        #
        # (org_id, run_id) is the shape this codebase already uses one table
        # over, in conversations' UNIQUE (org_id, agent_id, external_id).
        #
        # These two tables are CREATEd here, not merely altered: they are
        # touched by both the detector and the API, so by this module's rule
        # they belong here — and creating them makes the migration independent
        # of which service happens to boot first. Statements that touch tables
        # owned by a single service are guarded on existence instead.
        """
        CREATE TABLE IF NOT EXISTS processed_runs (
            run_id           TEXT        NOT NULL,
            agent_id         TEXT        NOT NULL,
            agent_version    TEXT        NOT NULL,
            org_id           TEXT,
            processed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            signal_count     INTEGER     NOT NULL DEFAULT 0,
            trigger          TEXT        NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id          TEXT        NOT NULL,
            org_id          TEXT,
            agent_id        TEXT        NOT NULL,
            agent_version   TEXT        NOT NULL,
            conversation_id BIGINT,
            started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- A row with no org cannot be attributed and cannot be part of a
        -- composite key. There is no correct owner to guess, so it goes.
        DELETE FROM processed_runs WHERE org_id IS NULL;
        DELETE FROM runs           WHERE org_id IS NULL;

        ALTER TABLE processed_runs ALTER COLUMN org_id SET NOT NULL;
        ALTER TABLE runs           ALTER COLUMN org_id SET NOT NULL;

        ALTER TABLE processed_runs DROP CONSTRAINT IF EXISTS processed_runs_pkey;
        ALTER TABLE processed_runs ADD  CONSTRAINT processed_runs_pkey
            PRIMARY KEY (org_id, run_id);

        ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_pkey;
        ALTER TABLE runs ADD  CONSTRAINT runs_pkey PRIMARY KEY (org_id, run_id);

        -- events and failure_signals belong to ingest, which may not have run
        -- its own DDL yet on a cold start.
        DO $$
        BEGIN
            IF to_regclass('public.events') IS NOT NULL THEN
                CREATE INDEX IF NOT EXISTS idx_events_org_run ON events(org_id, run_id);
            END IF;
            IF to_regclass('public.failure_signals') IS NOT NULL THEN
                CREATE INDEX IF NOT EXISTS idx_signals_org_run
                    ON failure_signals(org_id, run_id);
            END IF;
        END $$;
        """,
    ),
    (
        3,
        "hash_api_keys",
        # The raw secret was the plaintext PRIMARY KEY of api_keys, so any read
        # of that table — a pg_dump, a read replica, a support query, an
        # incident export — handed over live working credentials for every
        # tenant. The codebase already encrypts stored third-party credentials;
        # its own were in the clear.
        #
        # Existing rows cannot be migrated: a hash cannot be derived from a key
        # nobody stored reversibly, and here the plaintext IS the key. They are
        # deactivated rather than deleted, so the audit trail survives and the
        # failure reads as "revoked" rather than "vanished". Owned by ingest, so
        # guarded on existence.
        """
        -- The identity tables are shared by ingest and the API, so they are
        -- owned here rather than guarded on existence: an API-first boot on an
        -- empty database used to crash with
        -- `relation "api_keys" does not exist`. Shapes match ingest's, with the
        -- legacy per-agent columns already relaxed to nullable — ingest's own
        -- multi-tenancy DDL drops them, and requiring them here would make a
        -- migrations-first boot reject rows the running code writes.
        CREATE TABLE IF NOT EXISTS organizations (
            id          TEXT PRIMARY KEY,
            name        TEXT        NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        INSERT INTO organizations (id, name) VALUES ('default', 'Default Organization')
        ON CONFLICT (id) DO NOTHING;

        CREATE TABLE IF NOT EXISTS api_keys (
            key            TEXT PRIMARY KEY,
            org_id         TEXT,
            active         BOOLEAN     NOT NULL DEFAULT TRUE,
            rate_limit_rpm INTEGER     NOT NULL DEFAULT 600,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        DO $$
        BEGIN
            ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_hash   TEXT;
            ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_prefix TEXT;

            UPDATE api_keys
               SET active     = FALSE,
                   key_prefix = COALESCE(key_prefix, LEFT(key, 12))
             WHERE key_hash IS NULL;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_hash
                ON api_keys(key_hash) WHERE key_hash IS NOT NULL;
        END $$;
        """,
    ),
    (
        4,
        "api_key_scopes",
        # The authz model was one flat capability: a single key could submit
        # events, mint further keys, create a `stop` policy that terminates live
        # agent runs, open GitHub PRs — and grant its own approval requests.
        #
        # That last one inverts the whole point of human-in-the-loop approval.
        # The docs describe it as gating "sending a customer email, deleting
        # data, wiring money", but the agent process being gated held the very
        # credential that could approve it, so the control defended against
        # everything except the thing it exists for.
        #
        # Scopes split the credential the agent runtime holds from the one a
        # human decision needs. 'ingest' is the default and is what an SDK key
        # gets; granting an approval needs 'approve', which an agent never has.
        """
        DO $$
        BEGIN
            IF to_regclass('public.api_keys') IS NULL THEN
                RETURN;
            END IF;
            ALTER TABLE api_keys
                ADD COLUMN IF NOT EXISTS scopes TEXT[] NOT NULL DEFAULT ARRAY['ingest'];
        END $$;
        """,
    ),
    (
        5,
        "failure_signals_ownership",
        # failure_signals is the system's most important table and had no owning
        # service: ingest CREATEd it with 11 columns and 8 more arrived by
        # ALTER from four other services, three of them declared in exactly one
        # place. The real shape of the table was written down nowhere, and the
        # schema-parity test could not see it — it compares CREATE TABLE and
        # explicitly skips ALTER, which is where the coupling actually lived.
        #
        # Every column is declared here now. The per-service ALTERs stay for the
        # moment as harmless no-ops (IF NOT EXISTS against a column that already
        # exists), but this is the definition of record: add a column here, not
        # in whichever service happened to need it first.
        """
        -- CREATEd here, not guarded on existence. failure_signals is touched by
        -- five services, so by this module's rule it belongs here — and a
        -- verified detector-first boot on an empty database used to crash with
        -- `relation "failure_signals" does not exist`, because the detector
        -- ALTERs a table only ingest created. Owning the CREATE is what makes
        -- boot order genuinely irrelevant rather than merely usually fine.
        CREATE TABLE IF NOT EXISTS failure_signals (
            id             BIGSERIAL   PRIMARY KEY,
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
        CREATE INDEX IF NOT EXISTS idx_signals_agent
            ON failure_signals(agent_id, detected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_signals_unalerted
            ON failure_signals(alerted) WHERE alerted = FALSE;

        DO $$
        BEGIN
            -- Multi-tenancy
            ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS org_id TEXT;

            -- Detection lifecycle. shadow had ONE declarer (the detector), so a
            -- deployment that brought up the API before the detector had every
            -- signal read fail on a missing column.
            ALTER TABLE failure_signals
                ADD COLUMN IF NOT EXISTS shadow BOOLEAN NOT NULL DEFAULT TRUE;
            ALTER TABLE failure_signals
                ADD COLUMN IF NOT EXISTS co_signal_count INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

            -- Which pipeline produced the signal: structural | semantic | external.
            ALTER TABLE failure_signals
                ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'structural';

            -- Alerts worker claim protocol.
            ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS alert_claimed_at TIMESTAMPTZ;
            ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS alert_claimed_by TEXT;
        END $$;
        """,
    ),
    (
        6,
        "identity_tables",
        # organizations and api_keys were declared by migration 3 and then grown
        # by four services, each with an ALTER on a table it did not declare:
        # ingest added the per-org OTel kill switch, the API and the semantic
        # worker both added the semantic-feedback flags and quotas, and the API
        # added api_keys.id / rate_limit_rpm plus the unique index the per-agent
        # rate quotas key on. The real shape of the two tenancy tables was
        # spread over five files. This is now the definition of record; the
        # service copies (three CREATE TABLE organizations, one CREATE TABLE
        # api_keys, thirteen ALTERs) are deleted.
        #
        # Decisions:
        # - api_keys.org_id stays nullable. ingest's startup still recovers a
        #   pre-v0.5.0 `customer_id` into org_id and only then promotes it to
        #   NOT NULL; a NOT NULL or a 'default' backfill in a run-once migration
        #   would pre-empt that recovery and re-home real tenants under
        #   'default'. The FK to organizations IS declared here — it is part of
        #   the shape, and NULLs pass it.
        # - api_keys.id is ADD COLUMN, not part of a CREATE: migration 3 already
        #   created the table without it, and a CREATE here would never run.
        """
        -- organizations: per-org feature flags and quotas. Defaults are the
        -- ones the service copies carried.
        ALTER TABLE organizations
            ADD COLUMN IF NOT EXISTS semantic_feedback_enabled BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE organizations
            ADD COLUMN IF NOT EXISTS semantic_feedback_auto_suppress BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE organizations
            ADD COLUMN IF NOT EXISTS otel_ingestion_enabled BOOLEAN NOT NULL DEFAULT TRUE;
        ALTER TABLE organizations
            ADD COLUMN IF NOT EXISTS semantic_evaluation_quota INTEGER NOT NULL DEFAULT 1000;
        ALTER TABLE organizations
            ADD COLUMN IF NOT EXISTS allow_semantic_overage BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE organizations
            ADD COLUMN IF NOT EXISTS conversation_evaluation_quota INTEGER NOT NULL DEFAULT 200;
        ALTER TABLE organizations
            ADD COLUMN IF NOT EXISTS allow_conversation_overage BOOLEAN NOT NULL DEFAULT FALSE;

        -- api_keys: what a table created by the pre-migration ingest copy
        -- (key, agent_id, customer_id, active, created_at, company_id) lacks.
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS org_id TEXT;
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rate_limit_rpm INTEGER NOT NULL DEFAULT 600;
        -- Surrogate id: agent_rate_quotas.key_id points at it (deliberately
        -- not as a DB foreign key — see ingest's agent_rate_quotas comment).
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS id BIGSERIAL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_id ON api_keys(id);

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE table_name = 'api_keys' AND constraint_name = 'api_keys_org_id_fkey'
            ) THEN
                ALTER TABLE api_keys
                    ADD CONSTRAINT api_keys_org_id_fkey
                    FOREIGN KEY (org_id) REFERENCES organizations(id);
            END IF;
        END $$;
        """,
    ),
    (
        7,
        "policy_tables",
        # fixes, policies and policy_evaluations were each declared twice —
        # ingest (which writes evaluations and serves policies to the SDK) and
        # the API (which writes policies and fixes and reads evaluations). The
        # two copies happened to agree, but only because someone kept them in
        # sync by hand: the `policies.signature` outage described in this
        # module's docstring is what happens when that lapses.
        #
        # Decisions:
        # - fixes.org_id / policies.org_id are NOT NULL in the CREATE (the fresh
        #   install shape, and what ingest's legacy backfill promotes them to)
        #   but ADDed nullable for a table an older copy created. The promotion
        #   for those stays with ingest's backfill, for the reason given in
        #   migration 6: a run-once 'default' fill would pre-empt the real
        #   attribution.
        # - policy_evaluations gains policy_bundle_stale / policy_bundle_age_s.
        #   The SDK has sent both on every policy.evaluated record since the
        #   policy-fetch resilience work (a stale bundle means the agent was
        #   enforcing policies older than its fetch TTL because the server was
        #   unreachable) and ingest dropped them on the floor. Nullable: older
        #   SDKs do not send them, and NULL must read as "unknown", not "fresh".
        """
        CREATE TABLE IF NOT EXISTS fixes (
            id                    BIGSERIAL   PRIMARY KEY,
            org_id                TEXT        NOT NULL,
            run_id                TEXT        NOT NULL,
            signal_id             BIGINT      NOT NULL,
            fix_content           TEXT        NOT NULL,
            fix_type              TEXT        NOT NULL DEFAULT 'prompt_addition',
            applied_via           TEXT        NOT NULL,
            langfuse_prompt_name  TEXT,
            langfuse_version      INTEGER,
            applied_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        ALTER TABLE fixes ADD COLUMN IF NOT EXISTS org_id TEXT;
        CREATE INDEX IF NOT EXISTS idx_fixes_signal_id ON fixes(signal_id);
        CREATE INDEX IF NOT EXISTS idx_fixes_run_id    ON fixes(run_id, applied_at DESC);
        CREATE INDEX IF NOT EXISTS idx_fixes_org       ON fixes(org_id);

        CREATE TABLE IF NOT EXISTS policies (
            id          BIGSERIAL   PRIMARY KEY,
            org_id      TEXT        NOT NULL,
            agent_id    TEXT        NOT NULL DEFAULT '*',
            name        TEXT        NOT NULL,
            condition   JSONB       NOT NULL,
            action      JSONB       NOT NULL,
            enabled     BOOLEAN     NOT NULL DEFAULT TRUE,
            priority    INT         NOT NULL DEFAULT 100,
            -- HMAC over the canonical policy form, and which version of that
            -- form it was signed under (2 = condition.match expression block).
            signature   TEXT        NOT NULL DEFAULT '',
            sig_version INT         NOT NULL DEFAULT 1,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        ALTER TABLE policies ADD COLUMN IF NOT EXISTS org_id      TEXT;
        ALTER TABLE policies ADD COLUMN IF NOT EXISTS signature   TEXT NOT NULL DEFAULT '';
        ALTER TABLE policies ADD COLUMN IF NOT EXISTS sig_version INT  NOT NULL DEFAULT 1;
        CREATE INDEX IF NOT EXISTS idx_policies_agent ON policies(agent_id, enabled);
        CREATE INDEX IF NOT EXISTS idx_policies_org   ON policies(org_id, enabled);

        -- One row per shipped policy.evaluated record (rate-limited SDK-side).
        -- trigger_name avoids the SQL reserved word `trigger`.
        CREATE TABLE IF NOT EXISTS policy_evaluations (
            id                  BIGSERIAL   PRIMARY KEY,
            org_id              TEXT,
            policy_id           BIGINT,
            policy_name         TEXT        NOT NULL DEFAULT '',
            agent_id            TEXT        NOT NULL DEFAULT '',
            run_id              TEXT,
            trigger_name        TEXT,
            trigger_matched     BOOLEAN,
            fired               BOOLEAN,
            sampled             BOOLEAN     NOT NULL DEFAULT FALSE,
            reason              TEXT,
            conditions          JSONB,
            policy_bundle_stale BOOLEAN,
            policy_bundle_age_s DOUBLE PRECISION,
            evaluated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        ALTER TABLE policy_evaluations ADD COLUMN IF NOT EXISTS policy_bundle_stale BOOLEAN;
        ALTER TABLE policy_evaluations ADD COLUMN IF NOT EXISTS policy_bundle_age_s DOUBLE PRECISION;
        CREATE INDEX IF NOT EXISTS idx_policy_evals_policy
            ON policy_evaluations(org_id, policy_id, evaluated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_policy_evals_agent
            ON policy_evaluations(org_id, agent_id, evaluated_at DESC);
        """,
    ),
    (
        8,
        "detector_shared_tables",
        # Everything the detector and the API both touched. runs and
        # processed_runs were already here (migration 2) but the detector kept
        # its own copy plus two extra columns and five indexes; custom
        # detectors, packs and run_state_metrics were declared verbatim in both
        # services under an explicit "whichever starts first wins" comment.
        #
        # Decisions:
        # - runs.conversation_id loses its REFERENCES conversations(id). The
        #   detector copy declared it, but migration 2 creates runs before the
        #   detector's DDL runs, so the FK has been absent on every install
        #   since migration 2 landed. conversations is detector-owned and may
        #   not exist when this runs, so the FK cannot be declared here either.
        #   Recorded rather than silently continued.
        # - custom_detectors.org_id / custom_detector_results.org_id: NOT NULL
        #   in the CREATE, ADDed nullable for older tables; the detector's
        #   events-based backfill promotes those (same reasoning as 7).
        # - idx_signals_org_run is re-issued here. Migration 2 could only
        #   create it if ingest had already created failure_signals, which
        #   ingest no longer does — so on a fresh install migration 2 skipped
        #   it and nothing revisited that.
        """
        ALTER TABLE processed_runs ADD COLUMN IF NOT EXISTS event_count INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE processed_runs ADD COLUMN IF NOT EXISTS processing_error TEXT;
        CREATE INDEX IF NOT EXISTS idx_processed_runs_agent_version
            ON processed_runs(agent_id, agent_version, processed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_processed_runs_org_agent_version
            ON processed_runs(org_id, agent_id, agent_version, processed_at DESC);
        -- Same leading columns minus agent_version, which sits between agent_id
        -- and processed_at above and so cannot serve an ORDER BY processed_at
        -- within an agent (the API's run listing ranks per agent).
        CREATE INDEX IF NOT EXISTS idx_processed_runs_org_agent_time
            ON processed_runs(org_id, agent_id, processed_at DESC);

        CREATE INDEX IF NOT EXISTS idx_runs_conversation
            ON runs(conversation_id) WHERE conversation_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_runs_org_agent
            ON runs(org_id, agent_id, started_at DESC);

        CREATE INDEX IF NOT EXISTS idx_signals_org_run
            ON failure_signals(org_id, run_id);
        CREATE INDEX IF NOT EXISTS idx_signals_org_agent
            ON failure_signals(org_id, agent_id, detected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_failure_signals_agent_shadow_alerted
            ON failure_signals(agent_id, shadow, alerted, detected_at DESC);

        CREATE TABLE IF NOT EXISTS custom_detectors (
            id                BIGSERIAL   PRIMARY KEY,
            org_id            TEXT        NOT NULL,
            agent_id          TEXT        NOT NULL DEFAULT '*',
            name              TEXT        NOT NULL,
            description       TEXT        NOT NULL,
            config_json       JSONB       NOT NULL,
            status            TEXT        NOT NULL DEFAULT 'shadow',
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            total_runs        INTEGER     NOT NULL DEFAULT 0,
            shadow_fire_count INTEGER     NOT NULL DEFAULT 0
        );
        ALTER TABLE custom_detectors ADD COLUMN IF NOT EXISTS org_id TEXT;
        CREATE INDEX IF NOT EXISTS idx_custom_detectors_agent
            ON custom_detectors(agent_id, status);
        CREATE INDEX IF NOT EXISTS idx_custom_detectors_org_agent
            ON custom_detectors(org_id, agent_id, status);

        CREATE TABLE IF NOT EXISTS custom_detector_results (
            id           BIGSERIAL   PRIMARY KEY,
            org_id       TEXT        NOT NULL,
            detector_id  BIGINT      NOT NULL REFERENCES custom_detectors(id) ON DELETE CASCADE,
            run_id       TEXT        NOT NULL,
            agent_id     TEXT        NOT NULL,
            fired        BOOLEAN     NOT NULL,
            evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        ALTER TABLE custom_detector_results ADD COLUMN IF NOT EXISTS org_id TEXT;
        CREATE INDEX IF NOT EXISTS idx_cdr_detector
            ON custom_detector_results(detector_id, evaluated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_cdr_run ON custom_detector_results(run_id);

        -- packs is Dunetrace's own registry of detector-pack classes (seeded by
        -- the detector from PACK_REGISTRY); org_enabled_packs is the per-org
        -- activation side table the API writes and the detector reads.
        CREATE TABLE IF NOT EXISTS packs (
            name           TEXT        PRIMARY KEY,
            description    TEXT        NOT NULL,
            detector_names TEXT[]      NOT NULL DEFAULT '{}',
            added_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS org_enabled_packs (
            org_id     TEXT        NOT NULL,
            pack_name  TEXT        NOT NULL REFERENCES packs(name),
            enabled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            enabled_by TEXT,
            PRIMARY KEY (org_id, pack_name)
        );
        CREATE INDEX IF NOT EXISTS idx_org_enabled_packs_org ON org_enabled_packs(org_id);

        -- One row per (run, state), written by the detector, read by the API's
        -- state analytics. run_started_at is the run's own time so trends
        -- bucket by when the run happened, not when it was processed.
        CREATE TABLE IF NOT EXISTS run_state_metrics (
            run_id         TEXT        NOT NULL,
            org_id         TEXT        NOT NULL,
            agent_id       TEXT        NOT NULL,
            state          TEXT        NOT NULL,
            total_ms       BIGINT      NOT NULL,
            segment_count  INT         NOT NULL,
            run_started_at TIMESTAMPTZ,
            computed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, state)
        );
        CREATE INDEX IF NOT EXISTS idx_rsm_agent
            ON run_state_metrics(org_id, agent_id, run_started_at);
        """,
    ),
    (
        9,
        "alerting_tables",
        # org_alert_integrations and linear_issue_signals were declared twice —
        # the API (which writes the per-org Slack/Linear config and reads the
        # Linear-issue -> signal mapping in its webhook receiver) and the
        # alerts worker (which reads the config to deliver, and writes the
        # mapping when it opens a Linear issue). The alerts copy called itself
        # a "defensive copy" of the API's. The two agreed column-for-column,
        # so nothing changes shape here; the point is one declaration.
        #
        # Also here: the alerts worker's claim-scan index on failure_signals.
        # The claim columns are migration 5's, but the index that makes the
        # claim scan cheap was created by the worker alone, so an install that
        # never ran alerts had the columns and not the index. An index on a
        # migrations-owned table belongs with the table.
        #
        # The per-service ALTERs on failure_signals that migration 5 left as
        # "harmless no-ops" — source in alerts, semantic and (twice) the
        # integrations worker; alert_claimed_at/_by in alerts — are deleted
        # with this migration. 5 has been the definition of record for those
        # columns since it landed; a no-op that a reader has to prove is a
        # no-op is still a second declaration.
        """
        -- Per-org Slack/Linear alert destinations, bring-your-own credentials
        -- encrypted at rest. The API only ever encrypts (on config submission);
        -- the alerts worker is what decrypts to call Slack/Linear — except the
        -- Linear webhook_secret, which the API's webhook receiver needs to
        -- verify signatures (see api_svc/crypto.py).
        CREATE TABLE IF NOT EXISTS org_alert_integrations (
            id                    BIGSERIAL   PRIMARY KEY,
            org_id                TEXT        NOT NULL,
            provider              TEXT        NOT NULL,   -- 'slack' | 'linear'
            encrypted_credentials TEXT        NOT NULL,   -- slack: {webhook_url}; linear: {api_key, webhook_secret}
            config_json           JSONB       NOT NULL DEFAULT '{}',  -- slack: {channel}; linear: {team_id, project_id}
            enabled               BOOLEAN     NOT NULL DEFAULT TRUE,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (org_id, provider)
        );

        -- Bi-directional sync (Linear issue closed -> Dunetrace signal
        -- resolved). Written by the alerts worker when it creates the issue,
        -- read by the API's webhook receiver (routers/linear_webhook.py).
        CREATE TABLE IF NOT EXISTS linear_issue_signals (
            id              BIGSERIAL   PRIMARY KEY,
            org_id          TEXT        NOT NULL,
            signal_id       BIGINT      NOT NULL,
            linear_issue_id TEXT        NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (linear_issue_id)
        );

        -- Sized for the alerts worker's claim scan: the driving predicate is
        -- alerted = FALSE, ordered by detected_at, with the claim state read
        -- per candidate row (alerts_svc/db.py::claim_unalerted_signals).
        CREATE INDEX IF NOT EXISTS idx_signals_alert_claim
            ON failure_signals (detected_at, alert_claimed_at)
            WHERE alerted = FALSE;
        """,
    ),
    (
        10,
        "semantic_tables",
        # signal_groups, signal_group_members, signal_group_overrides,
        # org_semantic_evaluation_usage and semantic_evaluation_log were
        # declared twice — the semantic worker (which writes all five) and the
        # API (which reads them for the feedback and usage endpoints, and
        # writes signal_group_overrides.fp_count on a false-positive verdict).
        # The API's copy existed because SEMANTIC_WORKER_ENABLED defaults to
        # false, so on most installs the "primary owner" never starts at all —
        # which is exactly the case the runner exists for: these tables exist
        # from the first boot of any service, whether or not semantic ever
        # runs. The copies agreed column-for-column.
        #
        # What stays put: signal_feedback (one row per verdict, API-only) and
        # the per-agent semantic config/usage tables plus the conversation
        # quota counter (semantic-only) are single-owner and remain inline.
        """
        -- Signal grouping/dedup: one row per distinct recurring pattern,
        -- (org_id, agent_id, evaluator, root_cause_hash). root_cause_hash is a
        -- crude hash of the evaluator's reasoning text (semantic_svc/grouping.py)
        -- — a different concept from the detector's `issues` table, which is
        -- keyed on the closed FailureType enum. root_cause_sample is a
        -- human-readable (normalised, truncated) copy of the first reasoning
        -- seen, for display; the hash itself is not reversible.
        CREATE TABLE IF NOT EXISTS signal_groups (
            id                BIGSERIAL   PRIMARY KEY,
            org_id            TEXT        NOT NULL,
            agent_id          TEXT        NOT NULL,
            evaluator         TEXT        NOT NULL,
            root_cause_hash   TEXT        NOT NULL,
            root_cause_sample TEXT        NOT NULL,
            first_seen        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            signal_count      INTEGER     NOT NULL DEFAULT 0,
            UNIQUE (org_id, agent_id, evaluator, root_cause_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_signal_groups_org_agent
            ON signal_groups(org_id, agent_id);

        -- Membership: which failure_signals rows belong to which group.
        -- signal_id deliberately has no FK to failure_signals(id): nothing in
        -- the system deletes a signal row (the retention pass scrubs evidence
        -- in place), so the constraint would only ever cost.
        CREATE TABLE IF NOT EXISTS signal_group_members (
            id        BIGSERIAL   PRIMARY KEY,
            group_id  BIGINT      NOT NULL REFERENCES signal_groups(id) ON DELETE CASCADE,
            signal_id BIGINT      NOT NULL,
            run_id    TEXT        NOT NULL,
            added_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_signal_group_members_group
            ON signal_group_members(group_id, added_at DESC);
        CREATE INDEX IF NOT EXISTS idx_signal_group_members_signal
            ON signal_group_members(signal_id);

        -- One row per group that has accumulated false-positive feedback. No
        -- row means fp_count = 0; the API creates the row on the first
        -- false_positive verdict and the semantic worker reads it to lower
        -- confidence on later signals in the group.
        CREATE TABLE IF NOT EXISTS signal_group_overrides (
            group_id   BIGINT      PRIMARY KEY REFERENCES signal_groups(id) ON DELETE CASCADE,
            fp_count   INTEGER     NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Org-level monthly evaluation counter behind
        -- organizations.semantic_evaluation_quota (migration 6). Incremented
        -- on every sampled run regardless of per-agent config; 'YYYY-MM' UTC
        -- calendar-month bucket.
        CREATE TABLE IF NOT EXISTS org_semantic_evaluation_usage (
            org_id     TEXT    NOT NULL,
            month      TEXT    NOT NULL,
            eval_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (org_id, month)
        );

        -- Per-evaluation cost/token log, fired or not. failure_signals only
        -- records evaluations that fired, so billing built from it would
        -- badly undercount; this table is the honest source for usage.
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
        CREATE INDEX IF NOT EXISTS idx_semantic_evaluation_log_org_time
            ON semantic_evaluation_log(org_id, evaluated_at);
        """,
    ),
    (
        11,
        "external_evaluation_tables",
        # The four tables behind the pull integrations — external_evaluation_*
        # (Langfuse / LangSmith / Braintrust evaluation results) and
        # elevenlabs_* (TTS generations) — were declared by the API (config
        # CRUD and the run/call read paths) and by the integrations worker
        # (the pollers), each calling its own the primary and the other's a
        # defensive copy. Neither was: the worker never called
        # apply_migrations at all, so a worker-first boot on an empty database
        # would have found no failure_signals to write into.
        #
        # Decisions:
        # - elevenlabs_generations declares unmatched_reason / run_id /
        #   agent_id in the CREATE and ADDs them IF NOT EXISTS as well: the
        #   worker's copy carried those ALTERs for installs whose table
        #   predates correlation (unmatched_reason) and the denormalised
        #   run/agent columns the API's read paths filter on. The API's copy
        #   ADDed only the latter two.
        # - The three read-path indexes on elevenlabs_generations move here.
        #   idx_events_tts_correlation does NOT: it is an index on events,
        #   ingest's single-owner table, so ingest declares it beside the
        #   table (an index lives with its table).
        """
        -- Per-org external evaluation provider config (Langfuse, LangSmith,
        -- Braintrust share this shape, differing by `provider` and by what
        -- their encrypted credentials JSON holds).
        CREATE TABLE IF NOT EXISTS external_evaluation_integrations (
            id                    BIGSERIAL   PRIMARY KEY,
            org_id                TEXT        NOT NULL,
            provider              TEXT        NOT NULL,
            endpoint_url          TEXT        NOT NULL,
            encrypted_credentials TEXT        NOT NULL,
            poll_interval_secs    INTEGER     NOT NULL DEFAULT 60,
            enabled               BOOLEAN     NOT NULL DEFAULT TRUE,
            last_polled_at        TIMESTAMPTZ,
            last_success_at       TIMESTAMPTZ,
            consecutive_failures  INTEGER     NOT NULL DEFAULT 0,
            first_failure_at      TIMESTAMPTZ,
            last_alerted_at       TIMESTAMPTZ,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (org_id, provider)
        );
        CREATE INDEX IF NOT EXISTS idx_ext_integrations_enabled
            ON external_evaluation_integrations(enabled) WHERE enabled = TRUE;

        -- Dedup: a poll's overlap window (tolerating the provider's indexing
        -- lag) re-fetches evaluations already seen; this is what prevents a
        -- duplicate failure_signals row for the same external evaluation.
        CREATE TABLE IF NOT EXISTS external_evaluation_processed (
            org_id       TEXT        NOT NULL,
            provider     TEXT        NOT NULL,
            external_id  TEXT        NOT NULL,
            processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (org_id, provider, external_id)
        );

        -- ElevenLabs is not an evaluation provider (it yields TTS generations
        -- correlated to tts.generated events by timestamp / character count /
        -- voice, not scores joined by trace_id), so it gets its own config
        -- table with a generation high-water mark and no endpoint_url. The
        -- failure-tracking columns match external_evaluation_integrations so
        -- the poller's backoff behaviour is the same.
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
            created_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
            updated_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_elevenlabs_integrations_enabled
            ON elevenlabs_integrations(enabled) WHERE enabled = TRUE;

        -- One row per generation the poller fetched. cost_credits is stored
        -- separately from character_count even though standard TTS plans
        -- bill 1 credit/char, so a different plan can be represented without
        -- a migration. correlated_* are set on a match; unmatched_reason when
        -- the generation is old enough that a match should have arrived and
        -- none did (kept as drift, not retried). run_id/agent_id are
        -- denormalised from the matched event so the read paths never need
        -- an events.id join (events is a big partitioned table with no
        -- standalone id index).
        CREATE TABLE IF NOT EXISTS elevenlabs_generations (
            id                     BIGSERIAL        PRIMARY KEY,
            org_id                 TEXT             NOT NULL,
            generation_id          TEXT             NOT NULL,
            voice_id               TEXT,
            voice_name             TEXT,
            model                  TEXT,
            character_count        INTEGER          NOT NULL DEFAULT 0,
            cost_credits           INTEGER,
            text                   TEXT,
            source                 TEXT,
            generated_at           DOUBLE PRECISION NOT NULL,
            correlated_to_event_id BIGINT,
            correlation_method     TEXT,
            correlation_confidence REAL,
            correlated_at          TIMESTAMPTZ,
            unmatched_reason       TEXT,
            run_id                 TEXT,
            agent_id               TEXT,
            fetched_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
            UNIQUE (org_id, generation_id)
        );
        ALTER TABLE elevenlabs_generations ADD COLUMN IF NOT EXISTS unmatched_reason TEXT;
        ALTER TABLE elevenlabs_generations ADD COLUMN IF NOT EXISTS run_id           TEXT;
        ALTER TABLE elevenlabs_generations ADD COLUMN IF NOT EXISTS agent_id         TEXT;
        CREATE INDEX IF NOT EXISTS idx_elevenlabs_gen_org_time
            ON elevenlabs_generations(org_id, generated_at DESC);
        -- The correlation pass scans only generations still awaiting a decision.
        CREATE INDEX IF NOT EXISTS idx_elevenlabs_gen_uncorrelated
            ON elevenlabs_generations(org_id) WHERE correlated_to_event_id IS NULL;
        -- Read paths: generations for a run, and the filter/high-cost listing.
        CREATE INDEX IF NOT EXISTS idx_elevenlabs_gen_run
            ON elevenlabs_generations(org_id, run_id);
        """,
    ),
    (
        12,
        "issues_table",
        # issues was CREATEd by the detector (which upserts a row per fired
        # failure type and auto-resolves it after N clean runs) and then grown
        # by the API, which ALTERed in resolution_notes / manually_resolved for
        # the MCP resolve_issue tool — inside an existence check, because on
        # an API-first boot the table was not there to ALTER. That guard is
        # the tell: a table one service creates and another extends is shared,
        # and the API's list/search/resolve paths had been relying on "the
        # detector started first" all along.
        #
        # Decisions:
        # - org_id is NOT NULL in the CREATE and ADDed nullable for a table
        #   the old detector copy created; the promotion stays with the
        #   detector's events-based backfill, for the reason given in 6-8.
        # - The UNIQUE key is (org_id, agent_id, failure_type), declared under
        #   the name issues_org_agent_failure_type_key. Widening a
        #   pre-multi-tenancy (agent_id, failure_type) key happens here too,
        #   and is safe to do before org_id is populated: NULLs are distinct
        #   under UNIQUE, and the narrow key guaranteed there are no
        #   (agent_id, failure_type) duplicates to collide once the backfill
        #   sets them all to 'default'.
        # - idx_issues_org_agent moves with the table.
        """
        -- Persistent issue tracking: one row per (org_id, agent_id, failure_type).
        -- status: open | resolved | reopened. clean_runs_since is the run of
        -- consecutive signal-free runs (reset on each hit); the detector
        -- resolves at CLEAN_RUNS_THRESHOLD. resolution_notes / manually_resolved
        -- record how a resolution happened (the API's resolve_issue) and do
        -- not lock it: a manually-resolved issue still reopens on recurrence.
        CREATE TABLE IF NOT EXISTS issues (
            id                BIGSERIAL   PRIMARY KEY,
            org_id            TEXT        NOT NULL,
            agent_id          TEXT        NOT NULL,
            failure_type      TEXT        NOT NULL,
            status            TEXT        NOT NULL DEFAULT 'open',
            first_seen        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            resolved_at       TIMESTAMPTZ,
            affected_runs     INTEGER     NOT NULL DEFAULT 1,
            clean_runs_since  INTEGER     NOT NULL DEFAULT 0,
            resolution_notes  TEXT,
            manually_resolved BOOLEAN     NOT NULL DEFAULT FALSE,
            CONSTRAINT issues_org_agent_failure_type_key
                UNIQUE (org_id, agent_id, failure_type)
        );
        ALTER TABLE issues ADD COLUMN IF NOT EXISTS org_id            TEXT;
        ALTER TABLE issues ADD COLUMN IF NOT EXISTS resolution_notes  TEXT;
        ALTER TABLE issues
            ADD COLUMN IF NOT EXISTS manually_resolved BOOLEAN NOT NULL DEFAULT FALSE;

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

        CREATE INDEX IF NOT EXISTS idx_issues_org_agent ON issues(org_id, agent_id);
        """,
    ),
    (
        13,
        "run_state_metrics keyed by org",
        """
        -- run_state_metrics was declared in migration 8 with PRIMARY KEY
        -- (run_id, state) while carrying an org_id NOT NULL column — the same
        -- defect migration 2 fixed on runs/processed_runs, carried in one
        -- table later. run_id is caller-supplied (the SDK exposes run_id=, the
        -- OTLP path derives it from a caller-supplied trace id), so two
        -- tenants legitimately hold the same one and the upsert in
        -- detector_svc.write_run_state_metrics collided:
        --   ON CONFLICT (run_id, state) DO UPDATE SET total_ms = EXCLUDED...
        -- deliberately leaves org_id and agent_id alone, so the row KEPT the
        -- first tenant's owner while taking the second's timings. The second
        -- org never got a row at all (agent_state_analytics is org-scoped, so
        -- its State Analytics page simply lost the run), and the first org's
        -- page reported another tenant's numbers under its own agent.
        --
        -- Rows whose org cannot be trusted are dropped rather than guessed:
        -- there is no way to tell which tenant's timings a collided row holds,
        -- and this table is a derived cache the detector rewrites on the next
        -- pass. Only collided run_ids are touched.
        DELETE FROM run_state_metrics rsm
         WHERE EXISTS (
             SELECT 1 FROM run_state_metrics other
              WHERE other.run_id = rsm.run_id
                AND other.state  = rsm.state
                AND other.org_id <> rsm.org_id
         );

        ALTER TABLE run_state_metrics DROP CONSTRAINT IF EXISTS run_state_metrics_pkey;
        ALTER TABLE run_state_metrics
            ADD CONSTRAINT run_state_metrics_pkey PRIMARY KEY (org_id, run_id, state);
        """,
    ),
    (
        14,
        "policy_dry_run",
        # Dry run: a policy evaluates and records what it WOULD have done, and
        # executes nothing. The reason nobody arms a policy is that they cannot
        # see what it will do first; this is that preview.
        #
        # `policies.mode` defaults to 'enforcing', so every existing policy keeps
        # behaving exactly as it does today and a new policy is armed unless its
        # author says otherwise. Dry run is opt-in, deliberately: a guardrail
        # that silently does nothing by default is the failure mode this whole
        # workstream exists to remove.
        #
        # The verdict columns extend policy_evaluations rather than starting a
        # second table. A dry-run verdict IS an evaluation, and one table keeps
        # the promotion story intact: flipping mode changes nothing about the
        # rows already written, so the history from before enforcement stays
        # readable next to the history after it.
        #
        # Every added column is NULLable with no default backfill. Rows written
        # before this migration genuinely do not know their step or their
        # would-be action, and inventing a value would make an old row
        # indistinguishable from a new one that really recorded those facts.
        # NULL reads as "not recorded", which is the truth.
        """
        ALTER TABLE policies
            ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'enforcing';

        ALTER TABLE policy_evaluations
            ADD COLUMN IF NOT EXISTS mode               TEXT,
            ADD COLUMN IF NOT EXISTS step_index         INTEGER,
            ADD COLUMN IF NOT EXISTS would_action_type  TEXT,
            ADD COLUMN IF NOT EXISTS would_action_params JSONB,
            ADD COLUMN IF NOT EXISTS matched_branch     TEXT;

        -- The dashboard's dry-run view asks one question: for this policy, over
        -- this window, how many times would it have fired and on which runs.
        -- Partial, because dry-run rows are the only ones it reads and they are
        -- a small slice of a table dominated by enforcing evaluations.
        CREATE INDEX IF NOT EXISTS idx_policy_evals_dry_run
            ON policy_evaluations(org_id, policy_id, evaluated_at DESC)
            WHERE mode = 'dry_run' AND fired = TRUE;
        """,
    ),
]

CURRENT_SCHEMA_VERSION = MIGRATIONS[-1][0]

_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INT         PRIMARY KEY,
    name        TEXT        NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def current_version(conn: Any) -> int:
    """Highest applied migration, or 0 when the database is unmanaged."""
    await conn.execute(_VERSION_TABLE)
    return await conn.fetchval("SELECT COALESCE(MAX(version), 0) FROM schema_version")


@contextlib.asynccontextmanager
async def schema_connection(dsn: str) -> AsyncIterator[Any]:
    """A connection for startup schema work, with NO command_timeout.

    Startup DDL must not run on a pooled connection. Every service builds its
    pool with ``command_timeout`` of 10-15s and asyncpg applies that to each
    ``execute()`` — including migration bodies that build an index or promote a
    column to NOT NULL, and the org_id backfills that scan ``events`` whole.
    Those costs are proportional to table size, and none can use CONCURRENTLY
    because each migration runs inside a transaction.

    On any database holding real data the statement was cancelled, the
    migration rolled back, and the service exited. It then failed identically
    on every restart, because the timeout is deterministic and the table only
    grows — and every OTHER service refused to start too, since
    ``schema_version`` never advanced past the failing migration. Fresh
    installs and CI never saw it, because every table is empty.

    Use this for `apply_migrations`, for a service's own `CREATE TABLE`/index
    DDL, and for backfills. Ordinary queries keep the pool and its timeout,
    which is there to stop a slow query pinning a connection.
    """
    import asyncpg  # lazy: dunetrace_schemas must not require asyncpg to import

    conn = await asyncpg.connect(dsn)
    try:
        yield conn
    finally:
        await conn.close()


# How long to wait for another replica's migration run before giving up. Long
# enough for a genuinely slow migration on a large table (index builds run
# without a statement timeout — see schema_connection), short enough that a
# leaked lock is reported within one deploy rather than hanging forever.
MIGRATION_LOCK_WAIT_S = 600.0
_LOCK_POLL_INTERVAL_S = 1.0


async def _acquire_migration_lock(conn: Any) -> None:
    """Take the migration advisory lock, or raise after MIGRATION_LOCK_WAIT_S.

    Polls pg_try_advisory_lock instead of blocking on pg_advisory_lock, which
    waits forever. See apply_migrations' docstring for why that matters.
    """
    deadline = time.monotonic() + MIGRATION_LOCK_WAIT_S
    waited = False
    while True:
        if await conn.fetchval("SELECT pg_try_advisory_lock($1)", _MIGRATION_LOCK_ID):
            if waited:
                logger.info("Acquired the migration lock after waiting for another replica.")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out after "
                f"{MIGRATION_LOCK_WAIT_S:.0f}s waiting for the schema migration lock "
                f"({_MIGRATION_LOCK_ID}). Either another replica is still migrating a "
                "large table, or a previous run left the lock held — check "
                "pg_locks for locktype='advisory', and see whether DATABASE_URL "
                "points at a transaction-mode connection pooler (the lock is "
                "session-scoped and does not survive one)."
            )
        if not waited:
            waited = True
            logger.info("Another replica holds the migration lock; waiting.")
        await asyncio.sleep(_LOCK_POLL_INTERVAL_S)


async def apply_migrations(conn: Any) -> int:
    """Bring the database up to CURRENT_SCHEMA_VERSION. Returns the version.

    Safe to call from every service on every start, and safe to call
    concurrently: the advisory lock means the second caller waits and then finds
    nothing to do. Each migration commits on its own, so a failure stops at the
    last good version instead of leaving a half-applied schema.

    The lock is acquired by POLLING pg_try_advisory_lock against a deadline
    rather than by a blocking pg_advisory_lock. Blocking had no timeout, so a
    lock that was never released left every service hanging here forever with no
    log line, no exception, and a container that stays up but never becomes
    ready — indistinguishable from a slow migration. That is reachable: the lock
    is session-scoped, and both ingest and the API document their
    statement_cache_size=0 as being required because DATABASE_URL may point at a
    transaction-mode PgBouncer pooler, where the lock, the migrations and the
    unlock are not guaranteed to share a backend. A bounded wait turns that into
    a named error an operator can act on. (schema_connection also opens its own
    connection and closes it afterwards, which releases a leaked session lock.)
    """
    await _acquire_migration_lock(conn)
    try:
        version = await current_version(conn)
        for number, name, sql in MIGRATIONS:
            if number <= version:
                continue
            logger.info("Applying migration %d (%s)", number, name)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_version (version, name) VALUES ($1, $2) "
                    "ON CONFLICT (version) DO NOTHING",
                    number,
                    name,
                )
            version = number
        return version
    finally:
        try:
            await conn.execute("SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID)
        except Exception:  # the connection is closing anyway; the lock dies with it
            logger.debug("advisory unlock failed", exc_info=True)


async def require_schema_version(conn: Any, minimum: int, service: str) -> None:
    """Refuse to start against a database older than this service needs.

    The failure this replaces was silent: a service whose query referenced a
    column another service had not yet created logged one line and returned
    empty results. Crashing on a startup check is strictly better than running
    with a schema that cannot answer the questions the service will ask.
    """
    version = await current_version(conn)
    if version < minimum:
        raise RuntimeError(
            f"{service} needs schema version >= {minimum} but the database is at "
            f"{version}. Run migrations first (any service's startup applies them); "
            f"this service will not start against an older schema."
        )
