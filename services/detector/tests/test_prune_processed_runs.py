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

    async def test_batch_size_is_parameterized(self):
        conn = _Conn()
        with patch.object(db_module, "_pool", _Pool(conn)):
            await db_module.prune_processed_runs(batch_size=250)
        self.assertEqual(conn.args, (250,))
        self.assertIn("LIMIT $1", conn.sql)

    async def test_candidates_are_chosen_by_event_absence_not_age(self):
        """`processed_at` is refreshed whenever late events trigger a
        re-detection, so an age bound over it never expires rows whose events
        aged out long ago — they accumulate as permanently hollow run-list
        entries. Selecting on event absence also guarantees each pass makes
        progress rather than returning a batch that all still have events."""
        conn = _Conn()
        with patch.object(db_module, "_pool", _Pool(conn)):
            await db_module.prune_processed_runs()
        self.assertNotIn("processed_at <", conn.sql)
        self.assertNotIn("days')::INTERVAL", conn.sql)
        # The candidate CTE itself must filter on absence of events.
        candidate_cte = conn.sql.split("removed AS")[0]
        self.assertIn("NOT EXISTS", candidate_cte)

    async def test_returns_deleted_count(self):
        conn = _Conn(deleted=37)
        with patch.object(db_module, "_pool", _Pool(conn)):
            self.assertEqual(await db_module.prune_processed_runs(), 37)

    async def test_null_count_treated_as_zero(self):
        conn = _Conn(deleted=None)
        with patch.object(db_module, "_pool", _Pool(conn)):
            self.assertEqual(await db_module.prune_processed_runs(), 0)


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
