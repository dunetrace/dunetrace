"""
The alerts worker's schema bootstrap — what it declares (only its own two
tables) and what it applies (the shared migrations, before anything else).

This worker used to ALTER failure_signals.source and the alert_claimed_*
columns in "defensively", and carry a CREATE TABLE copy of the API's
org_alert_integrations / linear_issue_signals, all under the
whichever-service-starts-first convention. Every one of those is a migration's
now (5 and 9), so each ensure_* entry point worker.py calls at startup must
resolve to apply_migrations and must not run DDL of its own on those tables.

No DB.

Run:
    cd services/alerts
    python -m unittest tests.test_schema_bootstrap -v
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import unittest
from unittest.mock import AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/schemas-py"),
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alerts_svc.db as db_module  # noqa: E402
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION  # noqa: E402

_SHARED_ENSURE_FUNCS = (
    db_module.ensure_semantic_signal_column,
    db_module.ensure_alert_claim_columns,
    db_module.ensure_alert_integrations_schema,
)

_MIGRATION_OWNED = (
    "failure_signals",
    "org_alert_integrations",
    "linear_issue_signals",
    "processed_runs",
    "organizations",
)


class _Conn:
    def __init__(self, log):
        self._log = log

    async def execute(self, sql, *args):
        self._log.append(("execute", sql))

    async def fetchval(self, sql, *args):
        # require_schema_version's read of the version table.
        if "MAX(version)" in sql:
            return CURRENT_SCHEMA_VERSION
        return None


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def __init__(self, log):
        self._conn = _Conn(log)

    def acquire(self):
        return _Acquire(self._conn)


def _fake_schema_connection(db_module):
    """Stand in for migrations.schema_connection, backed by the patched pool.

    The ensure_* entry points now run migrations on a DEDICATED connection with
    no command_timeout — a pooled one cancelled index builds and backfills on
    any populated database and crash-looped the service, invisibly to CI where
    every table is empty. That connection comes from asyncpg.connect(), so
    patching `_pool` no longer intercepts it; this resolves `_pool` lazily so it
    picks up whatever the test installed.
    """
    import contextlib

    @contextlib.asynccontextmanager
    async def _cm(dsn):
        async with db_module._pool.acquire() as conn:
            yield conn

    return _cm


class TestSharedEnsureFunctionsApplyMigrations(unittest.IsolatedAsyncioTestCase):
    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            for fn in _SHARED_ENSURE_FUNCS:
                await fn()  # must not raise, must not import-and-fail

    async def test_each_entry_point_applies_the_shared_migrations(self):
        """worker.py calls all three at startup; whichever runs first must
        bring the shared schema up, so each is an apply_migrations call."""
        for fn in _SHARED_ENSURE_FUNCS:
            log = []
            applied = AsyncMock(return_value=0)
            with (
                patch.object(db_module, "_pool", _Pool(log)),
                patch("dunetrace_schemas.migrations.apply_migrations", applied),
                patch(
                    "dunetrace_schemas.migrations.schema_connection",
                    _fake_schema_connection(db_module),
                ),
            ):
                await fn()
            self.assertEqual(applied.await_count, 1, fn.__name__)
            # And nothing else: the only statement after it is the gate's own
            # read of schema_version (require_schema_version, see
            # test_schema_version_gate.py) — no local DDL.
            self.assertEqual(
                [sql for _kind, sql in log if "schema_version" not in sql], [], fn.__name__
            )


class TestNoLocalDdlOnSharedTables(unittest.TestCase):
    """The source-level guarantee behind the runtime one above: this module
    contains no CREATE TABLE / ALTER TABLE / CREATE INDEX on a table it does
    not own. Its own two tables (digest_log, alert_dedup) still live here."""

    def test_no_ddl_names_a_migration_owned_table(self):
        source = inspect.getsource(db_module)
        for table in _MIGRATION_OWNED:
            for verb in (
                r"CREATE TABLE(?: IF NOT EXISTS)?",
                r"ALTER TABLE",
                r"CREATE INDEX(?: IF NOT EXISTS)? \w+\s+ON",
            ):
                self.assertIsNone(
                    re.search(rf"{verb}\s+{table}\b", source),
                    f"{table} is migrations-owned; alerts_svc/db.py must not {verb} it",
                )

    def test_own_tables_are_still_declared_here(self):
        source = inspect.getsource(db_module)
        self.assertIn("CREATE TABLE IF NOT EXISTS digest_log", source)
        self.assertIn("CREATE TABLE IF NOT EXISTS alert_dedup", source)


if __name__ == "__main__":
    unittest.main()
