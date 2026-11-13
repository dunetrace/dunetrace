"""
The schema-version gate in ingest's ensure_schema().

ensure_schema runs ingest's own DDL, applies the shared migrations, then
REQUIRES the database to be at CURRENT_SCHEMA_VERSION before the legacy org_id
backfill runs — apply, then require. A returned apply is not proof the database
is current (a replica that lost the advisory-lock race to a peer whose apply
then failed, or one pointed at a read-only standby), and the backfill reads
columns that migrations 5 and 7 add. A too-old schema must raise, naming the
service, before that backfill.

Also here: the ElevenLabs correlation index on `events` is ingest's DDL now.

No DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import ingest_svc.db.postgres as db_module
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


def _fake_schema_connection(conn):
    """Stand in for migrations.schema_connection.

    ensure_schema() runs its DDL, the migrations and the org_id backfill on a
    dedicated connection with NO command_timeout — a pooled one cancelled index
    builds and the backfill's full events scan on any populated database, and
    crash-looped the service. That connection is opened with asyncpg.connect(),
    so patching the pool no longer intercepts it.
    """
    import contextlib

    @contextlib.asynccontextmanager
    async def _cm(dsn):
        yield conn

    return _cm


def _kinds(log):
    """Statement kinds in order, with the version-table bookkeeping that
    require_schema_version's read performs filtered out."""
    return [(kind, sql) for kind, sql in log if not (kind == "execute" and "schema_version" in sql)]


class TestEnsureSchemaGate(unittest.IsolatedAsyncioTestCase):
    async def _run(self, version):
        log = []

        async def _applied(conn):
            log.append(("apply_migrations", None))
            return version

        backfill = AsyncMock(side_effect=lambda conn: log.append(("backfill", None)))
        partitions = AsyncMock()
        pool = _Pool(log, version)
        with (
            patch.object(db_module, "_pool", pool),
            patch.object(db_module, "_ensure_event_partitions", partitions),
            patch.object(db_module, "_backfill_org_id", backfill),
            patch("dunetrace_schemas.migrations.apply_migrations", _applied),
            patch(
                "dunetrace_schemas.migrations.schema_connection",
                _fake_schema_connection(pool._conn),
            ),
        ):
            await db_module.ensure_schema()
        return log, backfill

    async def test_current_schema_applies_then_backfills(self):
        log, backfill = await self._run(CURRENT_SCHEMA_VERSION)
        kinds = _kinds(log)
        # Own DDL first (ingest-owned tables), then the shared migrations, then
        # the backfill that depends on them.
        self.assertEqual(kinds[0], ("execute", db_module._SCHEMA))
        self.assertEqual(kinds[1], ("execute", db_module._MULTI_TENANCY_DDL))
        self.assertEqual(kinds[2], ("apply_migrations", None))
        self.assertEqual(kinds[-1], ("backfill", None))
        self.assertEqual(backfill.await_count, 1)

    async def test_too_old_schema_raises_before_the_backfill(self):
        with self.assertRaises(RuntimeError) as ctx:
            await self._run(CURRENT_SCHEMA_VERSION - 1)
        self.assertIn("ingest", str(ctx.exception))
        self.assertIn(str(CURRENT_SCHEMA_VERSION), str(ctx.exception))

    async def test_too_old_schema_never_runs_the_backfill(self):
        log = []

        async def _applied(conn):
            log.append(("apply_migrations", None))
            return 0

        backfill = AsyncMock()
        pool = _Pool(log, 0)
        with (
            patch.object(db_module, "_pool", pool),
            patch.object(db_module, "_ensure_event_partitions", AsyncMock()),
            patch.object(db_module, "_backfill_org_id", backfill),
            patch("dunetrace_schemas.migrations.apply_migrations", _applied),
            patch(
                "dunetrace_schemas.migrations.schema_connection",
                _fake_schema_connection(pool._conn),
            ),
        ):
            with self.assertRaises(RuntimeError):
                await db_module.ensure_schema()
        self.assertEqual([k for k, _ in _kinds(log)][-1], "apply_migrations")
        self.assertEqual(backfill.await_count, 0)

    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            await db_module.ensure_schema()


class TestEventsCorrelationIndexIsIngests(unittest.TestCase):
    def test_tts_correlation_index_declared_with_events(self):
        """The ElevenLabs worker used to create this index guarded on `events`
        existing, so a worker-first boot ran without it until the worker's
        next restart. events is ingest's table; the index lives with it."""
        self.assertIn(
            "CREATE INDEX IF NOT EXISTS idx_events_tts_correlation",
            db_module._MULTI_TENANCY_DDL,
        )
        self.assertIn("WHERE event_type = 'tts.generated'", db_module._MULTI_TENANCY_DDL)


if __name__ == "__main__":
    unittest.main()
