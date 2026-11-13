"""
The migration runner — the thing that gives the shared schema an owner.

Run: PYTHONPATH=packages/schemas-py python -m pytest packages/schemas-py/tests/test_migrations.py -v
"""

from __future__ import annotations

import re
import unittest
from unittest.mock import patch
from pathlib import Path

from dunetrace_schemas import migrations
from dunetrace_schemas.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    apply_migrations,
    current_version,
    require_schema_version,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# The six service schema files — the same set scripts/check_schema_owners.py
# scans. A table owned by migrations must not be declared in any of them.
_SERVICE_SCHEMA_FILES = (
    "services/ingest/ingest_svc/db/postgres.py",
    "services/detector/detector_svc/db.py",
    "services/api/api_svc/db/queries.py",
    "services/alerts/alerts_svc/db.py",
    "services/semantic/semantic_svc/db.py",
    "services/integrations/integrations_svc/db.py",
)

# Every table migrations own, and the migration whose CREATE TABLE declares it.
# Append here when a table moves in; the tests below fail if a service still
# declares one of these, or if two migrations CREATE the same table.
MIGRATION_OWNED_TABLES = {
    "processed_runs": 2,
    "runs": 2,
    "organizations": 3,
    "api_keys": 3,
    "failure_signals": 5,
    "fixes": 7,
    "policies": 7,
    "policy_evaluations": 7,
    "custom_detectors": 8,
    "custom_detector_results": 8,
    "packs": 8,
    "org_enabled_packs": 8,
    "run_state_metrics": 8,
    "org_alert_integrations": 9,
    "linear_issue_signals": 9,
    "signal_groups": 10,
    "signal_group_members": 10,
    "signal_group_overrides": 10,
    "org_semantic_evaluation_usage": 10,
    "semantic_evaluation_log": 10,
    "external_evaluation_integrations": 11,
    "external_evaluation_processed": 11,
    "elevenlabs_integrations": 11,
    "elevenlabs_generations": 11,
    "issues": 12,
}

# Columns the deleted per-service copies carried that the original migration
# lacked, and which migration folds each in. A regression that dropped one of
# these would only show up as an UndefinedColumnError in some other service.
FOLDED_COLUMNS = {
    6: {
        "organizations": (
            "semantic_feedback_enabled",
            "semantic_feedback_auto_suppress",
            "otel_ingestion_enabled",
            "semantic_evaluation_quota",
            "allow_semantic_overage",
            "conversation_evaluation_quota",
            "allow_conversation_overage",
        ),
        "api_keys": ("id", "org_id", "rate_limit_rpm"),
    },
    7: {
        "fixes": ("org_id",),
        "policies": ("org_id", "signature", "sig_version"),
        "policy_evaluations": ("policy_bundle_stale", "policy_bundle_age_s"),
    },
    8: {
        "processed_runs": ("event_count", "processing_error"),
        "custom_detectors": ("org_id",),
        "custom_detector_results": ("org_id",),
    },
    11: {
        "elevenlabs_generations": ("unmatched_reason", "run_id", "agent_id"),
    },
    12: {
        "issues": ("org_id", "resolution_notes", "manually_resolved"),
    },
}

_CREATE_TABLE_RE = re.compile(r"CREATE TABLE(?: IF NOT EXISTS)? (?!IF\b)(\w+)\s*\(")
_ADD_COLUMN_RE = re.compile(r"ALTER TABLE\s+(\w+)\s+ADD COLUMN(?: IF NOT EXISTS)?\s+(\w+)", re.S)


def _sql_for(version: int) -> str:
    return next(sql for v, _, sql in MIGRATIONS if v == version)


class FakeConn:
    """Records every statement, so ordering and transactionality are assertable
    without a database."""

    def __init__(self, applied=()):
        self.statements = []
        self.fetched = []
        self._applied = list(applied)
        self.transactions = 0

    async def execute(self, sql, *args):
        self.statements.append((sql, args))

    async def fetchval(self, sql, *args):
        self.fetched.append(sql)
        if "MAX(version)" in sql:
            return max(self._applied) if self._applied else 0
        if "pg_try_advisory_lock" in sql:
            # Real Postgres returns a boolean here. apply_migrations polls this
            # against a deadline instead of blocking on pg_advisory_lock, which
            # waits forever — a lock nothing released used to hang every service
            # on startup with no log line and no exception.
            return self.lock_granted
        return None

    lock_granted = True

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self_inner):
                conn.transactions += 1

            async def __aexit__(self_inner, *exc):
                return False

        return _Tx()


class TestMigrationList(unittest.TestCase):
    def test_versions_are_contiguous_and_ordered(self):
        versions = [v for v, _, _ in MIGRATIONS]
        self.assertEqual(versions, sorted(versions))
        self.assertEqual(versions, list(range(1, len(versions) + 1)))

    def test_versions_are_unique(self):
        versions = [v for v, _, _ in MIGRATIONS]
        self.assertEqual(len(versions), len(set(versions)))

    def test_current_version_is_the_last(self):
        self.assertEqual(CURRENT_SCHEMA_VERSION, MIGRATIONS[-1][0])

    def test_every_migration_has_sql_and_a_name(self):
        for version, name, sql in MIGRATIONS:
            self.assertTrue(name.strip(), version)
            self.assertTrue(sql.strip(), version)


class TestApplyMigrations(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_database_applies_everything(self):
        conn = FakeConn()
        version = await apply_migrations(conn)
        self.assertEqual(version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(conn.transactions, len(MIGRATIONS))

    async def test_already_current_applies_nothing(self):
        conn = FakeConn(applied=[CURRENT_SCHEMA_VERSION])
        version = await apply_migrations(conn)
        self.assertEqual(version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(conn.transactions, 0)

    async def test_partially_migrated_applies_only_the_remainder(self):
        conn = FakeConn(applied=[1])
        await apply_migrations(conn)
        self.assertEqual(conn.transactions, len(MIGRATIONS) - 1)

    async def test_each_migration_runs_in_its_own_transaction(self):
        """A failure must leave the database at the last good version rather
        than half-applied."""
        conn = FakeConn()
        await apply_migrations(conn)
        self.assertEqual(conn.transactions, len(MIGRATIONS))

    async def test_concurrent_startups_are_serialised_by_an_advisory_lock(self):
        conn = FakeConn()
        await apply_migrations(conn)
        # The lock is TAKEN with pg_try_advisory_lock (a fetchval, polled
        # against a deadline) and RELEASED with pg_advisory_unlock (an execute).
        # Blocking on pg_advisory_lock had no timeout, so a lock nothing
        # released hung every service on startup, silently.
        self.assertIn("pg_try_advisory_lock", " ".join(conn.fetched))
        self.assertIn("pg_advisory_unlock", " ".join(s for s, _ in conn.statements))

    async def test_lock_is_released_even_when_a_migration_fails(self):
        class Boom(FakeConn):
            async def execute(self, sql, *args):
                await super().execute(sql, *args)
                if "CREATE TABLE IF NOT EXISTS processed_runs" in sql:
                    raise RuntimeError("nope")

        conn = Boom()
        with self.assertRaises(RuntimeError):
            await apply_migrations(conn)
        self.assertIn("pg_advisory_unlock", " ".join(s for s, _ in conn.statements))


class TestSharedTableOwnership(unittest.TestCase):
    """Migrations are the ONLY declaration of every shared table.

    `scripts/check_schema_owners.py` ratchets the repo-wide count down; this
    pins the tables that have already moved so they cannot quietly move back
    (a service re-adding a "defensive" CREATE TABLE IF NOT EXISTS is exactly
    the "whichever starts first wins" contract the runner replaced).
    """

    def test_migrations_six_to_twelve_are_the_expected_groups(self):
        names = {v: n for v, n, _ in MIGRATIONS}
        self.assertEqual(names[6], "identity_tables")
        self.assertEqual(names[7], "policy_tables")
        self.assertEqual(names[8], "detector_shared_tables")
        self.assertEqual(names[9], "alerting_tables")
        self.assertEqual(names[10], "semantic_tables")
        self.assertEqual(names[11], "external_evaluation_tables")
        self.assertEqual(names[12], "issues_table")

    def test_each_shared_table_is_created_by_exactly_one_migration(self):
        for table, expected in MIGRATION_OWNED_TABLES.items():
            creators = [
                v
                for v, _, sql in MIGRATIONS
                if re.search(rf"CREATE TABLE IF NOT EXISTS {table}\s*\(", sql)
            ]
            self.assertEqual(creators, [expected], table)

    def test_no_service_declares_a_migration_owned_table(self):
        """Includes the "defensive copies" — the alerts, semantic and
        integrations workers and the API each kept a CREATE TABLE IF NOT
        EXISTS of the other's tables so a boot in the wrong order would not
        crash. Those were the drift risk, not the protection: a copy that
        agrees today is one ADD COLUMN away from not agreeing."""
        offenders = []
        for rel in _SERVICE_SCHEMA_FILES:
            source = (_REPO_ROOT / rel).read_text()
            for m in _CREATE_TABLE_RE.finditer(source):
                if m.group(1) in MIGRATION_OWNED_TABLES:
                    offenders.append(f"{rel}: {m.group(1)}")
        self.assertEqual(offenders, [], "a service re-declared a migrations-owned table")

    def test_no_service_adds_columns_to_the_tables_moved_here(self):
        """An ADD COLUMN in a service on a table it does not declare is a
        stray: the column belongs in a new migration. That includes
        failure_signals — the alerts, semantic and integrations workers each
        used to ALTER `source` (and alerts the claim columns) in "defensively",
        long after migration 5 declared them; those ALTERs are gone."""
        watched = set(MIGRATION_OWNED_TABLES)
        offenders = []
        for rel in _SERVICE_SCHEMA_FILES:
            source = (_REPO_ROOT / rel).read_text()
            for m in _ADD_COLUMN_RE.finditer(source):
                if m.group(1) in watched:
                    offenders.append(f"{rel}: {m.group(1)}.{m.group(2)}")
        self.assertEqual(offenders, [])

    def test_folded_columns_are_declared_by_the_migration_that_owns_them(self):
        for version, tables in FOLDED_COLUMNS.items():
            sql = _sql_for(version)
            for table, columns in tables.items():
                for column in columns:
                    self.assertRegex(
                        sql,
                        rf"ALTER TABLE {table}\s+ADD COLUMN IF NOT EXISTS {column}\b",
                        f"migration {version} must ADD COLUMN IF NOT EXISTS {table}.{column} "
                        "so a table created by the narrower service copy gains it",
                    )

    def test_policy_evaluations_bundle_columns_have_the_sdk_types(self):
        """The SDK sends policy_bundle_stale (bool) and policy_bundle_age_s
        (float seconds, None when there was never a fetch). Nullable on both:
        NULL means 'not reported', which an old SDK must not be able to turn
        into 'fresh'."""
        sql = _sql_for(7)
        self.assertRegex(sql, r"policy_bundle_stale\s+BOOLEAN,")
        self.assertRegex(sql, r"policy_bundle_age_s\s+DOUBLE PRECISION,")
        self.assertNotRegex(sql, r"policy_bundle_stale\s+BOOLEAN\s+NOT NULL")
        self.assertNotRegex(sql, r"policy_bundle_age_s\s+DOUBLE PRECISION\s+NOT NULL")

    def test_new_migrations_are_idempotent_by_construction(self):
        """6 onwards must be re-runnable against a database in any prior
        state: every CREATE and ADD COLUMN carries IF NOT EXISTS, and nothing
        drops a table, column or index. (A guarded DROP CONSTRAINT is allowed
        — see test_issues_key_widening_is_guarded — because swapping a key is
        the one shape change IF NOT EXISTS cannot express.)"""
        for version in range(6, CURRENT_SCHEMA_VERSION + 1):
            sql = _sql_for(version)
            for m in re.finditer(
                r"CREATE (?:UNIQUE )?(?:TABLE|INDEX)\s+(?!IF NOT EXISTS)(\w+)", sql
            ):
                self.fail(f"migration {version}: CREATE without IF NOT EXISTS: {m.group(0)}")
            for m in re.finditer(r"ADD COLUMN\s+(?!IF NOT EXISTS)(\w+)", sql):
                self.fail(f"migration {version}: ADD COLUMN without IF NOT EXISTS: {m.group(0)}")
            self.assertNotRegex(
                sql, r"DROP (TABLE|COLUMN|INDEX)", f"migration {version} must not drop"
            )

    def test_issues_key_widening_is_guarded(self):
        """A table the old detector copy created has UNIQUE (agent_id,
        failure_type); the tenant-scoped key is (org_id, agent_id,
        failure_type). The swap is a DROP + ADD, so it must be guarded on the
        constraint names in both directions to be re-runnable, and the CREATE
        must declare the wide key under the same name so a fresh table is a
        no-op for the guard rather than a second, duplicate constraint."""
        sql = _sql_for(12)
        self.assertRegex(
            sql,
            r"CONSTRAINT issues_org_agent_failure_type_key\s+UNIQUE \(org_id, agent_id, failure_type\)",
        )
        self.assertRegex(
            sql,
            r"IF EXISTS \(\s*SELECT 1 FROM pg_constraint WHERE conname = "
            r"'issues_agent_id_failure_type_key'\s*\) THEN\s+"
            r"ALTER TABLE issues DROP CONSTRAINT issues_agent_id_failure_type_key",
        )
        self.assertRegex(
            sql,
            r"IF NOT EXISTS \(\s*SELECT 1 FROM pg_constraint WHERE conname = "
            r"'issues_org_agent_failure_type_key'\s*\) THEN",
        )
        # NULLs are distinct under UNIQUE, so the wide key can go on before the
        # detector's backfill populates org_id — but only if it is nullable at
        # that point. The NOT NULL belongs to the CREATE (fresh installs) and to
        # the detector's promotion, never to the ADD COLUMN.
        self.assertRegex(sql, r"ADD COLUMN IF NOT EXISTS org_id\s+TEXT;")

    def test_shared_indexes_moved_with_their_tables(self):
        """Indexes a service used to create on a table it no longer declares.
        idx_events_tts_correlation is deliberately absent: it is an index on
        `events`, ingest's single-owner table, and ingest declares it beside
        the table."""
        expected = {
            6: ("idx_api_keys_id",),
            7: (
                "idx_fixes_signal_id",
                "idx_fixes_run_id",
                "idx_fixes_org",
                "idx_policies_agent",
                "idx_policies_org",
                "idx_policy_evals_policy",
                "idx_policy_evals_agent",
            ),
            8: (
                "idx_processed_runs_agent_version",
                "idx_processed_runs_org_agent_version",
                "idx_processed_runs_org_agent_time",
                "idx_runs_conversation",
                "idx_runs_org_agent",
                "idx_signals_org_run",
                "idx_signals_org_agent",
                "idx_failure_signals_agent_shadow_alerted",
                "idx_custom_detectors_agent",
                "idx_custom_detectors_org_agent",
                "idx_cdr_detector",
                "idx_cdr_run",
                "idx_org_enabled_packs_org",
                "idx_rsm_agent",
            ),
            9: ("idx_signals_alert_claim",),
            10: (
                "idx_signal_groups_org_agent",
                "idx_signal_group_members_group",
                "idx_signal_group_members_signal",
                "idx_semantic_evaluation_log_org_time",
            ),
            11: (
                "idx_ext_integrations_enabled",
                "idx_elevenlabs_integrations_enabled",
                "idx_elevenlabs_gen_org_time",
                "idx_elevenlabs_gen_uncorrelated",
                "idx_elevenlabs_gen_run",
            ),
            12: ("idx_issues_org_agent",),
        }
        for version, names in expected.items():
            sql = _sql_for(version)
            for name in names:
                self.assertIn(f"INDEX IF NOT EXISTS {name}", sql, f"migration {version}")


class TestRequireSchemaVersion(unittest.IsolatedAsyncioTestCase):
    async def test_raises_below_the_minimum(self):
        """The failure this replaces was silent — a query referencing a column
        another service had not created returned empty results."""
        with self.assertRaises(RuntimeError) as ctx:
            await require_schema_version(FakeConn(applied=[1]), 3, "detector")
        self.assertIn("detector", str(ctx.exception))

    async def test_passes_at_or_above_the_minimum(self):
        await require_schema_version(FakeConn(applied=[3]), 3, "detector")


if __name__ == "__main__":
    unittest.main()


class TestMigrationLockIsBounded(unittest.IsolatedAsyncioTestCase):
    """A lock that is never granted must raise, not hang forever.

    pg_advisory_lock has no timeout. A session-scoped lock that was never
    released — reachable when DATABASE_URL points at a transaction-mode
    connection pooler, which both ingest and the API document as a supported
    target — left every service blocked on startup with no log line, no
    exception, and a container that stays up but never becomes ready.
    """

    async def test_a_lock_never_granted_raises_after_the_deadline(self):
        conn = FakeConn()
        conn.lock_granted = False
        with (
            patch.object(migrations, "MIGRATION_LOCK_WAIT_S", 0.05),
            patch.object(migrations, "_LOCK_POLL_INTERVAL_S", 0.01),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await apply_migrations(conn)
        message = str(ctx.exception)
        self.assertIn("migration lock", message)
        # The message has to tell an operator where to look.
        self.assertIn("pg_locks", message)
        self.assertIn("pooler", message)

    async def test_a_lock_granted_after_waiting_still_applies(self):
        conn = FakeConn()
        conn.lock_granted = False

        real_sleep = migrations.asyncio.sleep

        async def grant_then_sleep(delay):
            conn.lock_granted = True
            await real_sleep(0)

        with (
            patch.object(migrations, "MIGRATION_LOCK_WAIT_S", 5.0),
            patch.object(migrations, "_LOCK_POLL_INTERVAL_S", 0.01),
            patch.object(migrations.asyncio, "sleep", grant_then_sleep),
        ):
            version = await apply_migrations(conn)
        self.assertEqual(version, CURRENT_SCHEMA_VERSION)
