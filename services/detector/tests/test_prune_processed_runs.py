"""
Tests for processed_runs retention.

`processed_runs` is the detector's idempotency ledger — one row per run, and the
anti-join target in fetch_completed_runs. Nothing pruned it, so it grew forever
while `events` dropped partitions at EVENT_RETENTION_DAYS.

The ordering invariant is what these tests pin down: a processed_runs row may only
be deleted once its run's events are gone. Delete it early and the run reads as
unprocessed, every detector runs against it again, and a duplicate set of signals
lands in failure_signals. The SQL enforces that with NOT EXISTS rather than by
trusting a retention constant, so it holds whatever ingest_svc is configured to do.

No DB — the SQL is asserted structurally.

Run:
    cd services/detector
    python -m unittest tests.test_prune_processed_runs -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/detector"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detector_svc.db as db_module


class _Conn:
    def __init__(self, deleted=0):
        self.deleted = deleted
        self.sql = None
        self.args = None

    async def fetchval(self, sql, *args):
        self.sql = sql
        self.args = args
        return self.deleted


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Acquire:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _Acquire()


class TestPruneProcessedRuns(unittest.IsolatedAsyncioTestCase):
    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            self.assertEqual(await db_module.prune_processed_runs(), 0)

    async def test_only_deletes_rows_whose_events_are_gone(self):
        """The invariant. Without this guard, pruning a row while its events still
        exist makes the run look unprocessed and it gets re-detected, duplicating
        every signal it produced."""
        conn = _Conn()
        with patch.object(db_module, "_pool", _Pool(conn)):
            await db_module.prune_processed_runs()
        self.assertIn("NOT EXISTS", conn.sql)
        self.assertIn("FROM events e WHERE e.run_id = p.run_id", conn.sql)

    async def test_age_bound_and_batch_are_parameterized(self):
        conn = _Conn()
        with patch.object(db_module, "_pool", _Pool(conn)):
            await db_module.prune_processed_runs(min_age_days=45, batch_size=250)
        self.assertEqual(conn.args, ("45", 250))
        # Age is an interval built from a bound parameter, never interpolated.
        self.assertIn("($1 || ' days')::INTERVAL", conn.sql)
        self.assertIn("LIMIT $2", conn.sql)

    async def test_returns_deleted_count(self):
        conn = _Conn(deleted=37)
        with patch.object(db_module, "_pool", _Pool(conn)):
            self.assertEqual(await db_module.prune_processed_runs(), 37)

    async def test_null_count_treated_as_zero(self):
        conn = _Conn(deleted=None)
        with patch.object(db_module, "_pool", _Pool(conn)):
            self.assertEqual(await db_module.prune_processed_runs(), 0)

    async def test_default_age_exceeds_event_retention_default(self):
        """The scan bound must sit beyond ingest's 90-day event default, or the
        anti-join does pointless work on rows that can't qualify yet."""
        conn = _Conn()
        with patch.object(db_module, "_pool", _Pool(conn)):
            await db_module.prune_processed_runs()
        self.assertGreater(int(conn.args[0]), 90)


class TestPruneLoopShardGuard(unittest.IsolatedAsyncioTestCase):
    """processed_runs isn't shard-partitioned, so only one replica should prune —
    otherwise N replicas contend on the same rows for no extra throughput."""

    async def _run_one_pass(self, shard_index, deleted_seq):
        import detector_svc.worker as worker_module

        prune_mock = AsyncMock(side_effect=deleted_seq)

        # Break out of the infinite loop after the first pass by making the
        # inter-pass sleep raise.
        async def _sleep(_secs):
            raise asyncio.CancelledError

        import asyncio

        with (
            patch("detector_svc.worker.prune_processed_runs", prune_mock),
            patch("detector_svc.worker.settings") as mock_settings,
            patch("detector_svc.worker.asyncio.sleep", AsyncMock(side_effect=_sleep)),
        ):
            mock_settings.SHARD_INDEX = shard_index
            mock_settings.PROCESSED_RUNS_RETENTION_DAYS = 120
            mock_settings.PRUNE_BATCH_SIZE = 100
            with self.assertRaises(asyncio.CancelledError):
                await worker_module._prune_loop()
        return prune_mock

    async def test_shard_zero_prunes(self):
        prune_mock = await self._run_one_pass(0, [5])
        prune_mock.assert_awaited()

    async def test_nonzero_shard_does_not_prune(self):
        prune_mock = await self._run_one_pass(3, [5])
        prune_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
