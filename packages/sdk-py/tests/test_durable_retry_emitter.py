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

from dunetrace.emitters import (
    BatchingEmitter,
    DEFAULT_QUEUE_PATH,
    DurableRetryEmitter,
    ShipOutcome,
)
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


class _OutcomeByRunId(BatchingEmitter):
    """Returns a per-run_id ShipOutcome — lets a test say "the backend refuses
    r1 outright but would happily take r2"."""

    def __init__(self, outcomes):
        self._outcomes = dict(outcomes)
        self.received: list = []

    def ship(self, batch):
        run_id = batch[0].run_id
        self.received.append(run_id)
        return self._outcomes.get(run_id, ShipOutcome.DELIVERED)


class _AlwaysRejects(BatchingEmitter):
    """Stands in for a stale DUNETRACE_API_KEY: every batch 401s."""

    def __init__(self):
        self.call_count = 0

    def ship(self, batch):
        self.call_count += 1
        return ShipOutcome.REJECTED


class _LegacyBoolEmitter(BatchingEmitter):
    """A third-party emitter written against the old bool contract."""

    def __init__(self, result):
        self._result = result
        self.call_count = 0

    def ship(self, batch):
        self.call_count += 1
        return self._result


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
        # Mocks logger.warning directly rather than using assertLogs so the
        # assertion doesn't depend on logger propagation/handler state.
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path, max_queue_events=1)
        emitter.ship([_event(run_id="r1")])
        with patch("dunetrace.emitters.logger.warning") as mock_warning:
            emitter.ship([_event(run_id="r2")])
        mock_warning.assert_called_once()
        self.assertIn("evicted", mock_warning.call_args[0][0])

    def test_eviction_warning_fires_when_monotonic_clock_starts_near_zero(self):
        # Regression for a real CI-only failure: _last_eviction_warning used
        # to be seeded to 0.0 and compared as `now - last >= 60`. time.monotonic()'s
        # epoch is unspecified (often time-since-boot on Linux), so on a freshly
        # booted CI runner `now` itself can be under 60, silently suppressing the
        # very first eviction warning — while passing on any dev machine whose
        # uptime is already well past 60s. Pinning time.monotonic() low reproduces
        # that environment regardless of the host's actual uptime.
        with patch("time.monotonic", return_value=5.0):
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


# ── Permanently-rejected batches must not be persisted (head-of-line blocking) ──


class TestRejectedBatchesAreNotQueued(_TempQueueTestCase):
    """HttpBatchingEmitter logs a 401/400/413/422 batch as "not retried, events
    dropped" and then used to hand back a bare False, which this wrapper read as
    "transient" and wrote to disk. Because the backlog is drained oldest-first
    and stops at the first failure, one such batch at the head blocked every
    deliverable batch behind it — forever, at one doomed attempt every 30s,
    until the 100k-event / 100MB cap silently evicted the good batches. Nothing
    was ever delivered and the log line said the opposite of what happened."""

    def test_rejected_batch_is_not_written_to_disk(self):
        inner = _AlwaysRejects()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        outcome = emitter.ship([_event()])
        self.assertIs(outcome, ShipOutcome.REJECTED)
        self.assertEqual(self._row_count(), 0)

    def test_a_stale_api_key_does_not_fill_the_disk_queue(self):
        """The whole reported shape: every batch 401s, none is retained."""
        inner = _AlwaysRejects()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        for i in range(25):
            emitter.ship([_event(run_id=f"r{i}")])
        self.assertEqual(self._row_count(), 0)
        self.assertEqual(inner.call_count, 25)  # each tried exactly once, never re-tried

    def test_retryable_batch_is_still_queued(self):
        """The fix must not disable the feature it is fixing."""
        emitter = DurableRetryEmitter(_AlwaysFails(), queue_path=self.queue_path)
        self.assertIs(emitter.ship([_event()]), ShipOutcome.DELIVERED)
        self.assertEqual(self._row_count(), 1)

    def test_queued_batch_that_turns_permanent_is_dropped_not_retried_forever(self):
        """A batch queued while the backend was merely down, and refused
        outright once it came back (the key was rotated meanwhile)."""
        emitter = DurableRetryEmitter(_AlwaysFails(), queue_path=self.queue_path)
        emitter.ship([_event(run_id="r1")])
        self.assertEqual(self._row_count(), 1)

        rejecting = _AlwaysRejects()
        emitter2 = DurableRetryEmitter(rejecting, queue_path=self.queue_path)
        emitter2._next_retry_at = 0
        emitter2.retry_pending()

        self.assertEqual(self._row_count(), 0)
        self.assertEqual(rejecting.call_count, 1)  # attempted once, then dropped

    def test_rejected_head_does_not_block_the_batch_behind_it(self):
        """The head-of-line property, stated directly."""
        blocked = DurableRetryEmitter(_AlwaysFails(), queue_path=self.queue_path)
        blocked.ship([_event(run_id="r1")])
        blocked._next_retry_at = float("inf")  # don't let ship() drain r1 first
        blocked.ship([_event(run_id="r2")])
        self.assertEqual(self._row_count(), 2)

        inner = _OutcomeByRunId({"r1": ShipOutcome.REJECTED, "r2": ShipOutcome.DELIVERED})
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        emitter._next_retry_at = 0
        delivered = emitter.retry_pending()

        self.assertEqual(inner.received, ["r1", "r2"])
        self.assertEqual(delivered, 1)  # r2 got through; r1 was never deliverable
        self.assertEqual(self._row_count(), 0)

    def test_transient_failure_still_stops_the_drain_to_preserve_order(self):
        """Only REJECTED short-circuits. A still-down backend must not let
        later batches jump ahead of earlier ones."""
        seed = DurableRetryEmitter(_AlwaysFails(), queue_path=self.queue_path)
        seed.ship([_event(run_id="r1")])
        seed._next_retry_at = float("inf")
        seed.ship([_event(run_id="r2")])
        self.assertEqual(self._row_count(), 2)

        inner = _OutcomeByRunId({"r1": ShipOutcome.RETRYABLE, "r2": ShipOutcome.DELIVERED})
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        emitter._next_retry_at = 0
        emitter.retry_pending()

        self.assertEqual(inner.received, ["r1"])  # stopped at r1, r2 never attempted
        self.assertEqual(self._row_count(), 2)

    def test_third_party_bool_false_is_treated_as_retryable(self):
        """An emitter that still returns a bool cannot say "permanent", so its
        batches must keep being queued — never silently dropped for it."""
        emitter = DurableRetryEmitter(_LegacyBoolEmitter(False), queue_path=self.queue_path)
        self.assertIs(emitter.ship([_event()]), ShipOutcome.DELIVERED)
        self.assertEqual(self._row_count(), 1)

    def test_third_party_bool_true_is_delivered(self):
        emitter = DurableRetryEmitter(_LegacyBoolEmitter(True), queue_path=self.queue_path)
        self.assertIs(emitter.ship([_event()]), ShipOutcome.DELIVERED)
        self.assertEqual(self._row_count(), 0)


class TestUndecodableQueueRows(_TempQueueTestCase):
    """A row this build cannot parse is as undeliverable as a rejected one, and
    it used to escape _maybe_retry_backlog's sqlite3.Error-only handler
    entirely — up into retry_pending(), where the drain thread swallows it at
    DEBUG. Invisible at default log level, and the queue never drained again."""

    def test_corrupt_row_is_discarded_and_does_not_block_the_queue(self):
        seed = DurableRetryEmitter(_AlwaysFails(), queue_path=self.queue_path)
        seed.ship([_event(run_id="r1")])
        conn = sqlite3.connect(self.queue_path)
        try:
            conn.execute("UPDATE queue SET payload = ?", ("{not json at all",))
            conn.commit()
        finally:
            conn.close()
        seed.ship([_event(run_id="r2")])
        self.assertEqual(self._row_count(), 2)

        inner = _AlwaysSucceeds()
        emitter = DurableRetryEmitter(inner, queue_path=self.queue_path)
        emitter._next_retry_at = 0
        with self.assertLogs("dunetrace", level="WARNING") as cm:
            delivered = emitter.retry_pending()

        self.assertTrue(any("undecodable" in m for m in cm.output))
        self.assertEqual(delivered, 1)  # r2 still got through
        self.assertEqual(self._row_count(), 0)
        self.assertEqual([e.run_id for e in inner.received], ["r2"])


# ── Graceful degradation ─────────────────────────────────────────────────────────


class TestGracefulDegradation(unittest.TestCase):
    def test_unwritable_queue_path_does_not_crash(self):
        inner = _AlwaysFails()
        emitter = DurableRetryEmitter(inner, queue_path="/nonexistent-root-xyz/queue.db")
        self.assertFalse(emitter._db_ok)
        result = emitter.ship([_event()])
        self.assertFalse(result)  # can't queue, can't deliver — honest failure, no crash
        self.assertIs(result, ShipOutcome.RETRYABLE)

    def test_sqlite_operational_error_is_not_an_oserror(self):
        """The premise of the bug, asserted so it cannot silently change: the
        failure _init_db exists to absorb does not derive from OSError, so the
        original `except OSError` never saw it."""
        self.assertFalse(issubclass(sqlite3.OperationalError, OSError))
        self.assertTrue(issubclass(sqlite3.OperationalError, Exception))

    def test_queue_path_that_is_a_directory_does_not_raise_into_the_caller(self):
        """A real, unmocked reproduction: makedirs succeeds, sqlite3.connect
        then raises OperationalError("unable to open database file") — which is
        not an OSError, so the constructor used to propagate straight into the
        customer's agent process at import/init time."""
        with tempfile.TemporaryDirectory() as d:
            with self.assertLogs("dunetrace", level="WARNING") as cm:
                emitter = DurableRetryEmitter(_AlwaysFails(), queue_path=d)
            self.assertFalse(emitter._db_ok)
            self.assertTrue(any("could not initialize queue" in m for m in cm.output))
            # Degraded, not dead: the client keeps shipping, just without a
            # disk queue behind it.
            self.assertIs(emitter.ship([_event()]), ShipOutcome.RETRYABLE)
            self.assertEqual(emitter.retry_pending(), 0)

    def test_unwritable_home_does_not_raise_into_the_caller(self):
        """The documented trigger — read-only root filesystem, non-root
        container user, $HOME unset (AWS Lambda, distroless, Kubernetes
        readOnlyRootFilesystem) — where the default path is ~/.dunetrace."""
        boom = sqlite3.OperationalError("unable to open database file")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "home", ".dunetrace", "queue.db")
            with patch("dunetrace.emitters.sqlite3.connect", side_effect=boom):
                with self.assertLogs("dunetrace", level="WARNING") as cm:
                    emitter = DurableRetryEmitter(_AlwaysFails(), queue_path=path)
        self.assertFalse(emitter._db_ok)
        self.assertTrue(any("DUNETRACE_QUEUE_PATH" in m for m in cm.output))

    def test_a_non_sqlite_error_during_retry_does_not_escape(self):
        """_maybe_retry_backlog is reached through retry_pending(), which the
        drain thread only guards at DEBUG — an escape there is invisible at the
        default log level and the queue never drains again. `except
        sqlite3.Error` covered neither a third-party inner emitter breaking the
        non-raising ship() contract nor an undecodable payload."""

        class _RaisingEmitter(BatchingEmitter):
            def ship(self, batch):
                raise RuntimeError("third-party emitter broke the contract")

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "queue.db")
            DurableRetryEmitter(_AlwaysFails(), queue_path=path).ship([_event()])

            emitter = DurableRetryEmitter(_RaisingEmitter(), queue_path=path)
            emitter._next_retry_at = 0
            with self.assertLogs("dunetrace", level="WARNING") as cm:
                self.assertEqual(emitter.retry_pending(), 0)  # no raise
        self.assertTrue(any("retry attempt failed" in m for m in cm.output))


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
