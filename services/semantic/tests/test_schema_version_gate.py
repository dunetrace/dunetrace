"""
The schema-version gate in ensure_semantic_schema().

Apply, then require: after apply_migrations the semantic worker refuses to
run its own DDL unless the database is at CURRENT_SCHEMA_VERSION. A replica
that could not apply (lock timeout, read-only standby) raises at startup,
naming the service, rather than writing signals into a table shaped for an
older version.

No DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import semantic_svc.db as db_module
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION


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


class TestEnsureSemanticSchemaGate(unittest.IsolatedAsyncioTestCase):
    async def test_current_schema_runs_local_ddl_after_migrations(self):
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
            await db_module.ensure_semantic_schema()
        self.assertEqual(log[0], ("apply_migrations", None))
        self.assertEqual(_ddl(log), [db_module._SEMANTIC_SCHEMA])

    async def test_too_old_schema_raises_before_any_local_ddl(self):
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
                await db_module.ensure_semantic_schema()
        self.assertIn("semantic", str(ctx.exception))
        self.assertEqual(log[0], ("apply_migrations", None))
        self.assertEqual(_ddl(log), [])

    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            await db_module.ensure_semantic_schema()


if __name__ == "__main__":
    unittest.main()
