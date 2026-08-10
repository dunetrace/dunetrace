"""
The migration runner — the thing that gives the shared schema an owner.

Run: PYTHONPATH=packages/schemas-py python -m pytest packages/schemas-py/tests/test_migrations.py -v
"""

from __future__ import annotations

import unittest

from dunetrace_schemas.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    apply_migrations,
    current_version,
    require_schema_version,
)


class FakeConn:
    """Records every statement, so ordering and transactionality are assertable
    without a database."""

    def __init__(self, applied=()):
        self.statements = []
        self._applied = list(applied)
        self.transactions = 0

    async def execute(self, sql, *args):
        self.statements.append((sql, args))

    async def fetchval(self, sql, *args):
        if "MAX(version)" in sql:
            return max(self._applied) if self._applied else 0
        return None

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
        sql = " ".join(s for s, _ in conn.statements)
        self.assertIn("pg_advisory_lock", sql)
        self.assertIn("pg_advisory_unlock", sql)

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
