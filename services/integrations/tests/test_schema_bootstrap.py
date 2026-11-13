"""
The integrations worker's schema bootstrap.

This worker never called apply_migrations: it carried its own CREATE TABLE
copies of the four migration-11 tables and a "defensive" ALTER on
failure_signals.source instead, so a worker-first boot on an empty database
had none of the tables it does not declare (failure_signals among them). Both
ensure_* entry points — one per container (integrations / elevenlabs) — apply
the shared migrations now, and the module carries no DDL at all: the one
index it used to create on `events` (idx_events_tts_correlation, guarded on
ingest's table existing, so a worker-first boot ran without it until the
worker's next restart) is declared by ingest, whose table it is.

No DB.

Run:
    cd services/integrations
    python -m unittest tests.test_schema_bootstrap -v
"""

from __future__ import annotations

import inspect
import re
import unittest
from unittest.mock import AsyncMock, patch

import integrations_svc.db as db_module
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

_MIGRATION_OWNED = (
    "external_evaluation_integrations",
    "external_evaluation_processed",
    "elevenlabs_integrations",
    "elevenlabs_generations",
    "failure_signals",
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


def _ddl(log):
    """Everything executed other than the gate's own schema_version read."""
    return [sql for _kind, sql in log if "schema_version" not in sql]


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


class TestEnsureSchemaAppliesMigrations(unittest.IsolatedAsyncioTestCase):
    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            await db_module.ensure_integrations_schema()
            await db_module.ensure_elevenlabs_schema()

    async def test_integrations_container_applies_migrations_and_nothing_else(self):
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
            await db_module.ensure_integrations_schema()
        self.assertEqual(applied.await_count, 1)
        self.assertEqual(_ddl(log), [])

    async def test_elevenlabs_container_applies_migrations_and_nothing_else(self):
        """This entry point used to follow apply_migrations with a guarded
        CREATE INDEX on ingest's events table — which on a worker-first boot
        found no table and silently left the worker without its correlation
        index until its next restart. The index is ingest's now."""
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
            await db_module.ensure_elevenlabs_schema()
        self.assertEqual(applied.await_count, 1)
        self.assertEqual(_ddl(log), [])


class TestNoLocalDdl(unittest.TestCase):
    def test_no_ddl_of_any_kind_in_the_module(self):
        source = inspect.getsource(db_module)
        self.assertIsNone(re.search(r"CREATE TABLE", source))
        self.assertIsNone(re.search(r"ALTER TABLE", source))
        self.assertIsNone(re.search(r"CREATE (?:UNIQUE )?INDEX", source))

    def test_events_index_is_not_declared_here(self):
        """idx_events_tts_correlation is on `events`, ingest's single-owner
        table, and is declared there. A guarded copy here is the boot-order
        dependency the migration runner exists to remove."""
        source = inspect.getsource(db_module)
        self.assertNotIn("CREATE INDEX IF NOT EXISTS idx_events_tts_correlation", source)


if __name__ == "__main__":
    unittest.main()
