"""
The schema-version gate in the Customer API's init_pool().

init_pool opens the pool, applies the shared migrations, then REQUIRES the
database to be at CURRENT_SCHEMA_VERSION before the API's own DDL runs —
apply, then require. A replica that could not apply (lock timeout, read-only
standby) raises here, naming the service, instead of serving requests against
a schema that cannot answer them.

No DB: asyncpg.create_pool is replaced with a recording fake.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import api_svc.db.queries as db_module
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

    async def close(self):
        pass


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


class TestInitPoolGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved_pool = db_module._pool

    def tearDown(self):
        db_module._pool = self._saved_pool

    async def _run(self, version):
        log = []

        async def _applied(conn):
            log.append(("apply_migrations", None))
            return version

        with (
            patch.object(
                db_module.asyncpg, "create_pool", AsyncMock(return_value=_Pool(log, version))
            ),
            patch("dunetrace_schemas.migrations.apply_migrations", _applied),
            patch(
                "dunetrace_schemas.migrations.schema_connection",
                _fake_schema_connection(db_module),
            ),
        ):
            await db_module.init_pool()
        return log

    async def test_current_schema_runs_local_ddl_after_migrations(self):
        log = await self._run(CURRENT_SCHEMA_VERSION)
        self.assertEqual(log[0], ("apply_migrations", None))
        ddl = _ddl(log)
        self.assertEqual(ddl[0], db_module._POLICY_AUDIT_DDL)
        self.assertIn(db_module._APPROVALS_DDL, ddl)

    async def test_too_old_schema_raises_before_any_local_ddl(self):
        log = []

        async def _applied(conn):
            log.append(("apply_migrations", None))
            return CURRENT_SCHEMA_VERSION - 1

        with (
            patch.object(
                db_module.asyncpg,
                "create_pool",
                AsyncMock(return_value=_Pool(log, CURRENT_SCHEMA_VERSION - 1)),
            ),
            patch("dunetrace_schemas.migrations.apply_migrations", _applied),
            patch(
                "dunetrace_schemas.migrations.schema_connection",
                _fake_schema_connection(db_module),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await db_module.init_pool()
        self.assertIn("api", str(ctx.exception))
        self.assertEqual(log[0], ("apply_migrations", None))
        self.assertEqual(_ddl(log), [])


if __name__ == "__main__":
    unittest.main()
