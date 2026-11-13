"""
The schema-version gate in the integrations worker's schema bootstrap.

Both containers' entry points (ensure_integrations_schema for the evaluation
pollers, ensure_elevenlabs_schema for the voice poller) apply the shared
migrations and then REQUIRE CURRENT_SCHEMA_VERSION. This worker declares no
DDL of its own, so the gate is the whole of its startup contract: a replica
that could not apply (lock timeout, read-only standby) raises here, naming
the service, rather than polling into tables shaped for an older version.

No DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import integrations_svc.db as db_module
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


_ENTRY_POINTS = (db_module.ensure_integrations_schema, db_module.ensure_elevenlabs_schema)


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


class TestEnsureSchemaGate(unittest.IsolatedAsyncioTestCase):
    async def test_current_schema_passes(self):
        for fn in _ENTRY_POINTS:
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

    async def test_too_old_schema_raises_naming_the_service(self):
        for fn in _ENTRY_POINTS:
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
            self.assertIn("integrations", str(ctx.exception), fn.__name__)
            self.assertEqual(log[0], ("apply_migrations", None), fn.__name__)


if __name__ == "__main__":
    unittest.main()
