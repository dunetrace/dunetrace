"""
Tests for the approval DB query helpers (Capability 2, Phase 2.1). No real DB
— a fake pool yields a recording connection so we can assert the SQL params
and the no-pool safety without Postgres.

Run: PYTHONPATH=../../packages/sdk-py:../explainer:. python -m unittest tests.test_approvals_queries -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

import api_svc.db.queries as q


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


class _PoolPatch:
    """Swap q._pool for a fake pool backed by `conn` for the duration of a test."""

    def __init__(self, conn):
        self._conn = conn
        self._orig = None

    def __enter__(self):
        self._orig = q._pool
        q._pool = _FakePool(self._conn)
        return self._conn

    def __exit__(self, *exc):
        q._pool = self._orig
        return False


class TestNoPoolSafety(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig = q._pool
        q._pool = None

    def tearDown(self):
        q._pool = self._orig

    async def test_create_returns_none(self):
        self.assertIsNone(await q.create_approval("org", "run", "agent", "wire_money", "{}", None))

    async def test_get_returns_none(self):
        self.assertIsNone(await q.get_approval("org", 1))

    async def test_set_decision_returns_none(self):
        self.assertIsNone(await q.set_approval_decision("org", 1, "granted", "u", "slack"))


class TestCreateApproval(unittest.IsolatedAsyncioTestCase):
    async def test_inserts_pending_row_and_returns_it(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 7, "org_id": "org", "status": "pending"})
        with _PoolPatch(conn):
            row = await q.create_approval(
                "org", "run-1", "agent", "wire_money", '{"amt": 500}', None
            )

        self.assertEqual(row["id"], 7)
        args = conn.fetchrow.await_args.args
        self.assertIn("INSERT INTO approvals", args[0])
        # positional params: org, run, agent, tool, args, expires_at
        self.assertEqual(args[1:6], ("org", "run-1", "agent", "wire_money", '{"amt": 500}'))


class TestGetApproval(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_by_org(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 3, "org_id": "org", "status": "pending"})
        with _PoolPatch(conn):
            await q.get_approval("org", 3)
        args = conn.fetchrow.await_args.args
        self.assertIn("WHERE org_id = $1 AND id = $2", args[0])
        self.assertEqual(args[1:3], ("org", 3))

    async def test_missing_returns_none(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        with _PoolPatch(conn):
            self.assertIsNone(await q.get_approval("org", 999))


class TestSetDecision(unittest.IsolatedAsyncioTestCase):
    async def test_guards_on_pending_status(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 5, "status": "granted"})
        with _PoolPatch(conn):
            row = await q.set_approval_decision("org", 5, "granted", "alice", "slack")

        args = conn.fetchrow.await_args.args
        # The status='pending' guard is what makes a late/double decision a no-op.
        self.assertIn("status = 'pending'", args[0])
        self.assertEqual(args[1:6], ("org", 5, "granted", "alice", "slack"))
        self.assertEqual(row["status"], "granted")

    async def test_already_decided_returns_none(self):
        # UPDATE ... WHERE status='pending' matched nothing → RETURNING yields
        # no row → None.
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        with _PoolPatch(conn):
            self.assertIsNone(await q.set_approval_decision("org", 5, "granted", "alice", "slack"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
