"""
The schema-version gate in ensure_detector_schema().

Apply, then require: after apply_migrations the detector refuses to run its
own DDL (or the org_id backfill, or the pack seeding) unless the database is
at CURRENT_SCHEMA_VERSION. A replica that could not apply — lock timeout,
read-only standby — used to fall through to queries that read columns the
schema lacked; now it raises at startup, naming the service.

No DB.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/schemas-py"),
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/detector"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detector_svc.db as db_module  # noqa: E402
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


class TestEnsureDetectorSchemaGate(unittest.IsolatedAsyncioTestCase):
    async def _run(self, version):
        log = []

        async def _applied(conn):
            log.append(("apply_migrations", None))
            return version

        backfill, seed = AsyncMock(), AsyncMock()
        with (
            patch.object(db_module, "_pool", _Pool(log, version)),
            patch.object(db_module, "_backfill_org_id", backfill),
            patch.object(db_module, "_seed_packs", seed),
            patch("dunetrace_schemas.migrations.apply_migrations", _applied),
            patch(
                "dunetrace_schemas.migrations.schema_connection",
                _fake_schema_connection(db_module),
            ),
        ):
            await db_module.ensure_detector_schema()
        return log, backfill, seed

    async def test_current_schema_runs_local_ddl_after_migrations(self):
        log, backfill, seed = await self._run(CURRENT_SCHEMA_VERSION)
        self.assertEqual(log[0], ("apply_migrations", None))
        self.assertEqual(_ddl(log)[0], db_module._DETECTOR_SCHEMA)
        self.assertEqual(backfill.await_count, 1)
        self.assertEqual(seed.await_count, 1)

    async def test_too_old_schema_raises_before_any_local_ddl(self):
        log = []

        async def _applied(conn):
            log.append(("apply_migrations", None))
            return CURRENT_SCHEMA_VERSION - 1

        backfill, seed = AsyncMock(), AsyncMock()
        with (
            patch.object(db_module, "_pool", _Pool(log, CURRENT_SCHEMA_VERSION - 1)),
            patch.object(db_module, "_backfill_org_id", backfill),
            patch.object(db_module, "_seed_packs", seed),
            patch("dunetrace_schemas.migrations.apply_migrations", _applied),
            patch(
                "dunetrace_schemas.migrations.schema_connection",
                _fake_schema_connection(db_module),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await db_module.ensure_detector_schema()
        self.assertIn("detector", str(ctx.exception))
        self.assertEqual(log[0], ("apply_migrations", None))
        self.assertEqual(_ddl(log), [])
        self.assertEqual(backfill.await_count, 0)
        self.assertEqual(seed.await_count, 0)

    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            await db_module.ensure_detector_schema()


if __name__ == "__main__":
    unittest.main()
