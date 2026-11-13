"""
The schema-version gate in the alerts worker's schema bootstrap.

_apply_shared_migrations applies, then REQUIRES CURRENT_SCHEMA_VERSION.
ensure_digest_schema and ensure_dedup_schema — the two entry points that
declare this worker's own tables, and the first two worker.py calls at
startup — run it before their own DDL, so a too-old schema raises before any
DDL at all, naming the service. The three shared entry points resolve to the
same call (see test_schema_bootstrap.py).

No DB.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

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


class _Conn:
    def __init__(self, log, version):
        self._log = log
        self._version = version

    async def execute(self, sql, *args):
        self._log.append(("execute", sql))

    async def fetchval(self, sql, *args):
        if "MAX(version)" in sql:
            return self._version
        return None


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def __init__(self, log, version):
        self._conn = _Conn(log, version)

    def acquire(self):
        return _Acquire(self._conn)


def _ddl(log):
    return [sql for kind, sql in log if kind == "execute" and "schema_version" not in sql]


_OWN_DDL_ENTRY_POINTS = (
    (db_module.ensure_digest_schema, "digest_log"),
    (db_module.ensure_dedup_schema, "alert_dedup"),
)


def _fake_schema_connection(db_module):
    """Stand in for migrations.schema_connection, backed by the patched pool.

    ensure_*_schema() now runs its DDL, the shared migrations and any backfill
    on a DEDICATED connection with no command_timeout. A pooled one cancelled
    index builds and full-table backfills on any populated database, rolling the
    migration back and crash-looping the service on every restart — while fresh
    installs and CI never saw it, because every table is empty.

    That connection comes from asyncpg.connect(), so patching `_pool` no longer
    intercepts it. This resolves `_pool` lazily, at call time, so it picks up
    whatever the test installed.
    """
    import contextlib

    @contextlib.asynccontextmanager
    async def _cm(dsn):
        async with db_module._pool.acquire() as conn:
            yield conn

    return _cm


class TestSharedMigrationsGate(unittest.IsolatedAsyncioTestCase):
    async def test_shared_entry_points_require_the_current_version(self):
        for fn in (
            db_module.ensure_semantic_signal_column,
            db_module.ensure_alert_integrations_schema,
            db_module.ensure_alert_claim_columns,
        ):
            log = []

            async def _applied(conn):
                log.append(("apply_migrations", None))
                return CURRENT_SCHEMA_VERSION - 1

            with (
                patch.object(db_module, "_pool", _Pool(log, CURRENT_SCHEMA_VERSION - 1)),
                patch("dunetrace_schemas.migrations.apply_migrations", _applied),
                patch(
                    "dunetrace_schemas.migrations.schema_connection",
                    _fake_schema_connection(db_module),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    await fn()
            self.assertIn("alerts", str(ctx.exception), fn.__name__)
            self.assertEqual(log[0], ("apply_migrations", None), fn.__name__)

    async def test_own_ddl_entry_points_apply_and_require_before_their_ddl(self):
        for fn, table in _OWN_DDL_ENTRY_POINTS:
            log = []

            async def _applied(conn):
                log.append(("apply_migrations", None))
                return CURRENT_SCHEMA_VERSION

            with (
                patch.object(db_module, "_pool", _Pool(log, CURRENT_SCHEMA_VERSION)),
                patch("dunetrace_schemas.migrations.apply_migrations", _applied),
                patch(
                    "dunetrace_schemas.migrations.schema_connection",
                    _fake_schema_connection(db_module),
                ),
            ):
                await fn()
            self.assertEqual(log[0], ("apply_migrations", None), fn.__name__)
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", _ddl(log)[0], fn.__name__)

    async def test_own_ddl_entry_points_raise_on_too_old_schema_before_any_ddl(self):
        for fn, _table in _OWN_DDL_ENTRY_POINTS:
            log = []

            async def _applied(conn):
                log.append(("apply_migrations", None))
                return CURRENT_SCHEMA_VERSION - 1

            with (
                patch.object(db_module, "_pool", _Pool(log, CURRENT_SCHEMA_VERSION - 1)),
                patch("dunetrace_schemas.migrations.apply_migrations", _applied),
                patch(
                    "dunetrace_schemas.migrations.schema_connection",
                    _fake_schema_connection(db_module),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    await fn()
            self.assertIn("alerts", str(ctx.exception), fn.__name__)
            self.assertEqual(_ddl(log), [], fn.__name__)

    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            for fn, _ in _OWN_DDL_ENTRY_POINTS:
                await fn()


if __name__ == "__main__":
    unittest.main()
