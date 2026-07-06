"""
Tests for DurableRetryEmitter — the disk-backed queue wrapping any
BatchingEmitter so failed batches survive a backend outage across process
restarts, rather than being dropped.

No network required. Uses a real SQLite file per test (tempfile), not a mock —
this is exactly the kind of persistence-correctness logic that's worth
verifying against the real thing.

Run: python -m unittest tests.test_durable_retry_emitter -v
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from dunetrace.emitters import BatchingEmitter, DEFAULT_QUEUE_PATH, DurableRetryEmitter
from dunetrace.models import AgentEvent, EventType


def _event(run_id: str = "run-1") -> AgentEvent:
    return AgentEvent(
        event_type=EventType.RUN_STARTED,
        run_id=run_id,
        agent_id="agent-1",
        agent_version="v1",
        step_index=0,
        payload={"k": "v"},
    )


class _ScriptedEmitter(BatchingEmitter):
    """Returns canned True/False results in order; records every batch it saw."""

    def __init__(self, results):
        self._results = list(results)
        self.received: list = []

    def ship(self, batch):
        self.received.append([e.run_id for e in batch])
        if not self._results:
            return True
        return self._results.pop(0)


class _AlwaysFails(BatchingEmitter):
    def __init__(self):
        self.call_count = 0

    def ship(self, batch):
        self.call_count += 1
        return False


class _AlwaysSucceeds(BatchingEmitter):
    def __init__(self):
        self.received: list = []

    def ship(self, batch):
        self.received.extend(batch)
        return True


class _TempQueueTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.queue_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.queue_path)  # DurableRetryEmitter must create it fresh

    def tearDown(self):
        if os.path.exists(self.queue_path):
            os.unlink(self.queue_path)

    def _row_count(self) -> int:
        conn = sqlite3.connect(self.queue_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
        finally:
            conn.close()


# ── Basic ship() delegation and queuing on failure ──────────────────────────────


class TestShipDelegatesToInner(_TempQueueTestCase):
    def test_inner_success_is_passed_through(self):
        inner = _AlwaysSucceeds()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        result = emitter.ship([_event()])
        self.assertTrue(result)
        self.assertEqual(len(inner.received), 1)
        self.assertEqual(self._row_count(), 0)  # nothing queued — it succeeded

    def test_inner_failure_is_queued_and_ship_still_returns_true(self):
        """Once durably queued, ship() reports success — the caller's ring
        buffer doesn't need to hold onto it anymore."""
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        result = emitter.ship([_event()])
        self.assertTrue(result)
        self.assertEqual(self._row_count(), 1)

    def test_queue_persists_across_emitter_instances(self):
        """The whole point: a batch queued by one instance is retried by a
        fresh instance pointed at the same path — simulating a process restart."""
        inner1 = _AlwaysFails()
        emitter1 = DurableRetryEmitter(inner1, queue_path=self.queue_path)
        emitter1.ship([_event(run_id="r1")])
        self.assertEqual(self._row_count(), 1)

        inner2 = _AlwaysSucceeds()
        emitter2 = DurableRetryEmitter(inner2, queue_path=self.queue_path)
        emitter2._next_retry_at = 0  # force the backlog check to run immediately
        emitter2.ship([_event(run_id="r2")])

        self.assertEqual(self._row_count(), 0)  # backlog drained
        run_ids = [e.run_id for e in inner2.received]
        self.assertIn("r1", run_ids)
        self.assertIn("r2", run_ids)


# ── Backlog retry ordering and cadence ──────────────────────────────────────────


class TestBacklogRetry(_TempQueueTestCase):
    def test_backlog_drained_oldest_first(self):
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        emitter.ship([_event(run_id="r1")])
        emitter.ship([_event(run_id="r2")])
        emitter.ship([_event(run_id="r3")])
        self.assertEqual(self._row_count(), 3)

        recording = _ScriptedEmitter([True, True, True])
        emitter._inner = recording
        emitter._next_retry_at = 0
        emitter.ship([_event(run_id="r4")])  # triggers backlog drain, then ships r4

        # First three ship() calls on the inner emitter are the backlog, in order.
        self.assertEqual(recording.received[0], ["r1"])
        self.assertEqual(recording.received[1], ["r2"])
        self.assertEqual(recording.received[2], ["r3"])
        self.assertEqual(self._row_count(), 0)

    def test_retry_stops_at_first_failure_preserving_order(self):
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        emitter.ship([_event(run_id="r1")])
        emitter.ship([_event(run_id="r2")])
        self.assertEqual(self._row_count(), 2)

        # r1 (backlog) succeeds, r2 (backlog) fails again -> retry stops, doesn't
        # skip ahead. r3 (the new ship() call's own batch) also fails -> queued.
        recording = _ScriptedEmitter([True, False, False])
        emitter._inner = recording
        emitter._next_retry_at = 0
        emitter.ship([_event(run_id="r3")])

        self.assertEqual(self._row_count(), 2)  # r2 (still queued) and r3 (newly queued)

    def test_retry_not_attempted_before_interval_elapses(self):
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(
            inner, queue_path=self.queue_path, retry_interval_s=30.0, retry_jitter_s=5.0
        )
        emitter.ship([_event(run_id="r1")])
        self.assertEqual(self._row_count(), 1)

        recording = _AlwaysSucceeds()
        emitter._inner = recording
        # _next_retry_at was set in the future by the first ship() call — don't reset it.
        emitter.ship([_event(run_id="r2")])

        # r1 must still be queued — the retry interval hasn't elapsed.
        self.assertEqual(self._row_count(), 1)
        run_ids = [e.run_id for e in recording.received]
        self.assertEqual(run_ids, ["r2"])

    def test_next_retry_at_is_jittered_around_interval(self):
        emitter = DurableRetryEmitter(
            _AlwaysSucceeds(),
            queue_path=self.queue_path,
            retry_interval_s=30.0,
            retry_jitter_s=5.0,
        )
        before = time.monotonic()
        emitter.ship([_event()])
        after = time.monotonic()
        # next_retry_at should land within [25, 35] seconds from "now"
        self.assertGreaterEqual(emitter._next_retry_at, before + 25.0)
        self.assertLessEqual(emitter._next_retry_at, after + 35.0)


# ── Bounded queue + eviction ─────────────────────────────────────────────────────


class TestBoundedQueueEviction(_TempQueueTestCase):
    def test_evicts_oldest_when_event_cap_exceeded(self):
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path, max_queue_events=2)
        emitter.ship([_event(run_id="r1")])
        emitter.ship([_event(run_id="r2")])
        emitter.ship([_event(run_id="r3")])  # should evict r1

        conn = sqlite3.connect(self.queue_path)
        try:
            payloads = [
                r[0] for r in conn.execute("SELECT payload FROM queue ORDER BY id").fetchall()
            ]
        finally:
            conn.close()
        self.assertEqual(len(payloads), 2)
        self.assertNotIn('"r1"', payloads[0])
        self.assertIn("r2", payloads[0])
        self.assertIn("r3", payloads[1])

    def test_evicts_oldest_when_byte_cap_exceeded(self):
        from dunetrace.emitters import _batch_to_json

        one_batch_size = len(_batch_to_json([_event(run_id="r1")]).encode())
        inner = _AlwaysFails()
        # Cap large enough for exactly one batch, too small for two.
        emitter = DurableRetryEmitter(
            inner, queue_path=self.queue_path, max_queue_bytes=one_batch_size + 10
        )
        emitter.ship([_event(run_id="r1")])
        self.assertEqual(self._row_count(), 1)
        emitter.ship([_event(run_id="r2")])
        self.assertEqual(self._row_count(), 1)  # r1 evicted, only r2 (the newest) survives

    def test_eviction_logs_warning(self):
        # Mocks logger.warning directly rather than using assertLogs — this
        # doesn't depend on the "dunetrace" logger's propagation/handler
        # state, which is what made this test CI-flaky (assertLogs failed to
        # observe the record in a fresh environment where some dependency's
        # own import-time logging setup put the logger in a state assertLogs
        # didn't see through, even though the warning genuinely fired).
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path, max_queue_events=1)
        emitter.ship([_event(run_id="r1")])
        with patch("dunetrace.emitters.logger.warning") as mock_warning:
            emitter.ship([_event(run_id="r2")])
        mock_warning.assert_called_once()
        self.assertIn("evicted", mock_warning.call_args[0][0])

    def test_eviction_warning_rate_limited_to_once_per_minute(self):
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path, max_queue_events=1)
        emitter.ship([_event(run_id="r1")])  # fills the single slot, no eviction yet

        # Simulate the rate-limit window already being "fresh" (just warned).
        emitter._last_eviction_warning = time.monotonic()

        import logging

        handler_calls = []
        logger = logging.getLogger("dunetrace")

        class _Counter(logging.Handler):
            def emit(self, record):
                handler_calls.append(record)

        h = _Counter()
        logger.addHandler(h)
        try:
            emitter.ship([_event(run_id="r2")])  # evicts r1 — but warning is rate-limited
            warnings = [
                r for r in handler_calls if r.levelname == "WARNING" and "evicted" in r.getMessage()
            ]
            self.assertEqual(len(warnings), 0)
        finally:
            logger.removeHandler(h)

        # The eviction still happened even though the warning was suppressed.
        self.assertEqual(self._row_count(), 1)


# ── Graceful degradation ─────────────────────────────────────────────────────────


class TestGracefulDegradation(unittest.TestCase):
    def test_unwritable_queue_path_does_not_crash(self):
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path="/nonexistent-root-xyz/queue.db")
        self.assertFalse(emitter._db_ok)
        result = emitter.ship([_event()])
        self.assertFalse(result)  # can't queue, can't deliver — honest failure, no crash


# ── Path resolution ──────────────────────────────────────────────────────────────


class TestQueuePathResolution(_TempQueueTestCase):
    def test_explicit_path_wins(self):
        emitter = DurableRetryEmitter(_AlwaysSucceeds(), queue_path=self.queue_path)
        self.assertEqual(emitter._path, self.queue_path)

    def test_env_var_used_when_no_explicit_path(self):
        os.environ["DUNETRACE_QUEUE_PATH"] = self.queue_path
        try:
            emitter = DurableRetryEmitter(_AlwaysSucceeds())
            self.assertEqual(emitter._path, self.queue_path)
        finally:
            del os.environ["DUNETRACE_QUEUE_PATH"]

    def test_default_path_used_when_nothing_specified(self):
        """Verifies path resolution only — must not touch the real home
        directory, so DEFAULT_QUEUE_PATH is patched to a temp location before
        __init__ (which eagerly initializes the DB file) ever runs."""
        os.environ.pop("DUNETRACE_QUEUE_PATH", None)
        with patch("dunetrace.emitters.DEFAULT_QUEUE_PATH", self.queue_path):
            emitter = DurableRetryEmitter(_AlwaysSucceeds())
        self.assertEqual(emitter._path, self.queue_path)

    def test_default_path_is_under_home_dunetrace(self):
        self.assertEqual(DEFAULT_QUEUE_PATH, os.path.expanduser("~/.dunetrace/queue.db"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
