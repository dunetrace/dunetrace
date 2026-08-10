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

import logging
from typing import Any, List, Tuple

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


async def apply_migrations(conn: Any) -> int:
    """Bring the database up to CURRENT_SCHEMA_VERSION. Returns the version.

    Safe to call from every service on every start, and safe to call
    concurrently: the advisory lock means the second caller waits and then finds
    nothing to do. Each migration commits on its own, so a failure stops at the
    last good version instead of leaving a half-applied schema.
    """
    await conn.execute("SELECT pg_advisory_lock($1)", _MIGRATION_LOCK_ID)
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
        await conn.execute("SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID)


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
