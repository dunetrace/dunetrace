"""
Tests for HttpBatchingEmitter's failure classification and bounded in-memory
retry queue, and the drain thread's idle retry hook.

No network and no real sleeps: urllib.request.urlopen is mocked throughout and
the emitter takes an injected clock, so backoff is driven by advancing a fake
time rather than waiting.

Run: python -m pytest tests/test_http_emitter_retry.py -v
"""

from __future__ import annotations

import email.message
import io
import json
import logging
import os
import threading
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from dunetrace.client import Dunetrace
from dunetrace.emitters import (
    SERIALIZATION_ERROR_KEY,
    BatchingEmitter,
    ConsoleBatchingEmitter,
    FileBatchingEmitter,
    HttpBatchingEmitter,
    NoopBatchingEmitter,
    ShipOutcome,
    as_ship_outcome,
    _parse_retry_after,
)
from dunetrace.models import AgentEvent, EventType


def _event(run_id: str = "run-1", agent_id: str = "agent-1") -> AgentEvent:
    return AgentEvent(
        event_type=EventType.RUN_STARTED,
        run_id=run_id,
        agent_id=agent_id,
        agent_version="v1",
        step_index=0,
        payload={"k": "v"},
    )


def _batch(*run_ids: str) -> list:
    return [_event(run_id=r) for r in run_ids]


class _Clock:
    """Injected in place of time.monotonic so tests advance time explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _http_error(
    code: int, headers: dict | None = None, body: bytes = b""
) -> urllib.error.HTTPError:
    """What urlopen raises for a non-2xx response."""
    hdrs = email.message.Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    return urllib.error.HTTPError(
        url="http://localhost:8001/v1/ingest", code=code, msg="err", hdrs=hdrs, fp=io.BytesIO(body)
    )


def _ok_response(status: int = 202) -> MagicMock:
    """A context-manager response as urlopen returns for a 2xx."""
    cm = MagicMock()
    cm.__enter__.return_value.status = status
    return cm


def _status_response(status: int, headers: dict | None = None) -> MagicMock:
    """A non-2xx delivered as a response object rather than an exception —
    exercises the explicit resp.status check."""
    cm = MagicMock()
    resp = cm.__enter__.return_value
    resp.status = status
    resp.headers = dict(headers or {})
    return cm


def _run_ids_in(mock_urlopen, call_index: int) -> list:
    req = mock_urlopen.call_args_list[call_index][0][0]
    return [e["run_id"] for e in json.loads(req.data)["events"]]


def _emitter(clock: _Clock | None = None, **kwargs) -> HttpBatchingEmitter:
    return HttpBatchingEmitter(
        "http://localhost:8001", api_key="dt_test", clock=clock or _Clock(), **kwargs
    )


_NO_JITTER = patch("dunetrace.emitters.random.uniform", return_value=0.0)


# ── ship() returns a classification, not a bare bool ────────────────────────────


class TestShipOutcomeClassification(unittest.TestCase):
    """A bare bool could not tell a wrapper that a batch will NEVER be
    accepted. HttpBatchingEmitter already classified the failure and logged a
    permanent one as "events dropped"; it just threw the classification away on
    the way out, so DurableRetryEmitter persisted a 401 batch to disk and
    retried it every 30s forever."""

    def test_2xx_is_delivered(self):
        emitter = _emitter()
        with patch("urllib.request.urlopen", return_value=_ok_response(202)):
            self.assertIs(emitter.ship(_batch("r1")), ShipOutcome.DELIVERED)

    def test_503_is_retryable(self):
        emitter = _emitter()
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            self.assertIs(emitter.ship(_batch("r1")), ShipOutcome.RETRYABLE)

    def test_429_and_408_are_retryable(self):
        for code in (429, 408):
            with self.subTest(code=code):
                emitter = _emitter()
                with patch("urllib.request.urlopen", side_effect=_http_error(code)):
                    self.assertIs(emitter.ship(_batch("r1")), ShipOutcome.RETRYABLE)

    def test_permanent_4xx_are_rejected(self):
        """The exact set the log line calls "not retried, events dropped"."""
        for code in (400, 401, 403, 413, 422):
            with self.subTest(code=code):
                emitter = _emitter()
                with patch("urllib.request.urlopen", side_effect=_http_error(code)):
                    self.assertIs(emitter.ship(_batch("r1")), ShipOutcome.REJECTED)

    def test_connection_error_is_retryable(self):
        emitter = _emitter()
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("nope")):
            self.assertIs(emitter.ship(_batch("r1")), ShipOutcome.RETRYABLE)

    def test_rejected_batch_is_not_queued_in_memory_either(self):
        emitter = _emitter()
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            emitter.ship(_batch("r1"))
        self.assertEqual(emitter.pending_events, 0)

    def test_unexpected_exception_is_retryable_not_rejected(self):
        """An exception from our own code says nothing about the batch. The
        outcome that loses no data is the right default."""
        emitter = _emitter()
        with patch.object(emitter, "_post", side_effect=RuntimeError("boom")):
            self.assertIs(emitter.ship(_batch("r1")), ShipOutcome.RETRYABLE)

    def test_outcome_truthiness_matches_the_old_bool_contract(self):
        self.assertTrue(ShipOutcome.DELIVERED)
        self.assertFalse(ShipOutcome.RETRYABLE)
        self.assertFalse(ShipOutcome.REJECTED)

    def test_as_ship_outcome_normalises_a_third_party_bool(self):
        """emitter= is a public extension point and the old contract was a
        bool. False must normalise to RETRYABLE, never REJECTED: an emitter
        that cannot say a batch is permanently refused must not have it
        silently dropped on its behalf."""
        self.assertIs(as_ship_outcome(True), ShipOutcome.DELIVERED)
        self.assertIs(as_ship_outcome(False), ShipOutcome.RETRYABLE)
        self.assertIs(as_ship_outcome(None), ShipOutcome.RETRYABLE)
        self.assertIs(as_ship_outcome(ShipOutcome.REJECTED), ShipOutcome.REJECTED)

    def test_builtin_emitters_return_the_enum(self):
        self.assertIs(NoopBatchingEmitter().ship(_batch("r1")), ShipOutcome.DELIVERED)
        with patch("builtins.print"):
            self.assertIs(ConsoleBatchingEmitter().ship(_batch("r1")), ShipOutcome.DELIVERED)
        emitter = FileBatchingEmitter("/nonexistent-root-xyz/events.ndjson")
        with self.assertLogs("dunetrace", level="WARNING"):
            self.assertIs(emitter.ship(_batch("r1")), ShipOutcome.RETRYABLE)


# ── Success and failure classification ──────────────────────────────────────────


class TestShipOutcome(unittest.TestCase):
    def test_202_counts_as_success(self):
        emitter = _emitter()
        with patch("urllib.request.urlopen", return_value=_ok_response(202)):
            self.assertTrue(emitter.ship(_batch("r1")))
        self.assertEqual(emitter.pending_events, 0)

    def test_non_2xx_status_without_exception_is_failure(self):
        """A transport that hands back a 5xx as a response rather than raising
        (a mock, an unusual opener) must still read as failure — the old code
        only looked at exceptions and would have logged 'Shipped'."""
        emitter = _emitter()
        with patch("urllib.request.urlopen", return_value=_status_response(500)):
            self.assertFalse(emitter.ship(_batch("r1")))
        self.assertEqual(emitter.pending_events, 1)  # 5xx → queued for retry

    def test_non_2xx_status_permanent_without_exception_is_dropped(self):
        emitter = _emitter()
        with patch("urllib.request.urlopen", return_value=_status_response(400)):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                self.assertFalse(emitter.ship(_batch("r1")))
        self.assertEqual(emitter.pending_events, 0)
        self.assertTrue(any("HTTP 400" in m and "dropped" in m for m in logs.output))

    def test_response_without_status_is_failure_not_success(self):
        emitter = _emitter()
        cm = MagicMock()  # .status is a MagicMock, not an int
        with patch("urllib.request.urlopen", return_value=cm):
            self.assertFalse(emitter.ship(_batch("r1")))

    def test_5xx_429_408_are_retryable(self):
        for code in (500, 502, 503, 504, 429, 408):
            with self.subTest(code=code):
                emitter = _emitter()
                with patch("urllib.request.urlopen", side_effect=_http_error(code)):
                    self.assertFalse(emitter.ship(_batch("r1")))
                self.assertEqual(emitter.pending_events, 1)

    def test_other_4xx_not_retried(self):
        """401 (bad key) and 413 (too large) would fail identically on retry:
        dropped now, with a WARNING naming the status, nothing queued."""
        for code in (400, 401, 403, 404, 413, 422):
            with self.subTest(code=code):
                emitter = _emitter()
                with patch("urllib.request.urlopen", side_effect=_http_error(code)):
                    with self.assertLogs("dunetrace", level="WARNING") as logs:
                        self.assertFalse(emitter.ship(_batch("r1", "r2")))
                self.assertEqual(emitter.pending_events, 0)
                self.assertEqual(len(logs.output), 1)
                self.assertIn(f"HTTP {code}", logs.output[0])
                self.assertIn("2 events", logs.output[0])
                self.assertIn("not retried", logs.output[0])

    def test_permanent_4xx_warning_includes_body_excerpt(self):
        emitter = _emitter()
        err = _http_error(401, body=b'{"detail":"invalid api key"}')
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                emitter.ship(_batch("r1"))
        self.assertIn("invalid api key", logs.output[0])

    def test_connection_refused_is_retryable_with_hint(self):
        emitter = _emitter()
        err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                self.assertFalse(emitter.ship(_batch("r1")))
        self.assertEqual(emitter.pending_events, 1)
        self.assertIn("docker compose up", logs.output[0])

    def test_timeout_and_reset_are_retryable(self):
        for exc in (TimeoutError("timed out"), ConnectionResetError("reset"), OSError("boom")):
            with self.subTest(exc=type(exc).__name__):
                emitter = _emitter()
                with patch("urllib.request.urlopen", side_effect=exc):
                    self.assertFalse(emitter.ship(_batch("r1")))
                self.assertEqual(emitter.pending_events, 1)

    def test_unexpected_exception_is_failure_not_retried_and_never_raises(self):
        emitter = _emitter()
        with patch("urllib.request.urlopen", side_effect=ValueError("unknown url type")):
            with self.assertLogs("dunetrace", level="WARNING"):
                self.assertFalse(emitter.ship(_batch("r1")))
        self.assertEqual(emitter.pending_events, 0)

    def test_unserialisable_payload_is_repaired_and_shipped_not_dropped(self):
        """One bad value costs its own payload, not the batch it rode in on.

        This used to fail the whole encode and drop everything: a single
        `run.external_signal(..., response=<a requests.Response>)` discarded up
        to 99 unrelated events, other runs' terminals among them, logged as if
        the server had rejected them. The event now ships with its payload
        replaced by a marker naming the cause.
        """
        emitter = _emitter()
        bad = _event("r1")
        # A circular reference: json.dumps(default=str) cannot coerce this, so
        # it is the case the repair path actually exists for. (An ORM entity or
        # a datetime is handled by default=str without any repair — see
        # test_uncoercible_value_is_stringified_not_repaired below.)
        circular: dict = {}
        circular["self"] = circular
        bad.payload = circular
        good = _event("r2")

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = req.data.decode()
            return _ok_response(202)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertLogs("dunetrace", level="WARNING"):
                self.assertTrue(emitter.ship([bad, good]))

        body = captured["body"]
        self.assertIn(SERIALIZATION_ERROR_KEY, body)
        # The innocent event in the same batch is untouched and still shipped.
        self.assertIn('"r2"', body)
        self.assertEqual(emitter.pending_events, 0)

    def test_uncoercible_value_is_stringified_not_repaired(self):
        """The common case — a Response, a datetime, a dataclass — costs nothing.

        default=str coerces it, so the payload ships with a readable string
        rather than triggering the repair path or dropping anything. This is
        what stops one odd value in external_signal() killing the drain thread.
        """
        emitter = _emitter()
        ev = _event("r1")
        ev.payload = {"obj": object()}
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = req.data.decode()
            return _ok_response(202)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertTrue(emitter.ship([ev]))
        self.assertIn("<object object at", captured["body"])
        self.assertNotIn(SERIALIZATION_ERROR_KEY, captured["body"])
        self.assertEqual(emitter.pending_events, 0)


# ── Retry: re-send, backoff, Retry-After, give-up ───────────────────────────────


class TestRetry(unittest.TestCase):
    def test_resends_after_503_and_delivers_exactly_once(self):
        clock = _Clock()
        emitter = _emitter(clock)
        side_effects = [_http_error(503, {"Retry-After": "2"}), _ok_response()]
        with patch("urllib.request.urlopen", side_effect=side_effects) as mock_urlopen:
            self.assertFalse(emitter.ship(_batch("r1")))
            self.assertEqual(emitter.pending_events, 1)

            # Not due yet: nothing is re-sent.
            clock.advance(1.0)
            self.assertEqual(emitter.retry_pending(), 0)
            self.assertEqual(mock_urlopen.call_count, 1)

            clock.advance(30.0)  # past any jitter
            self.assertEqual(emitter.retry_pending(), 1)

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(_run_ids_in(mock_urlopen, 1), ["r1"])
        self.assertEqual(emitter.pending_events, 0)

        # And never again.
        with patch("urllib.request.urlopen", return_value=_ok_response()) as later:
            clock.advance(60.0)
            self.assertEqual(emitter.retry_pending(), 0)
        later.assert_not_called()

    def test_due_retries_run_before_the_new_batch_on_ship(self):
        clock = _Clock()
        emitter = _emitter(clock)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.ship(_batch("r1"))
        clock.advance(60.0)
        with patch("urllib.request.urlopen", return_value=_ok_response()) as mock_urlopen:
            self.assertTrue(emitter.ship(_batch("r2")))
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(_run_ids_in(mock_urlopen, 0), ["r1"])  # backlog first
        self.assertEqual(_run_ids_in(mock_urlopen, 1), ["r2"])

    def test_retry_after_is_minimum_delay(self):
        clock = _Clock()
        emitter = _emitter(clock)
        with (
            _NO_JITTER,
            patch("urllib.request.urlopen", side_effect=_http_error(503, {"Retry-After": "10"})),
        ):
            emitter.ship(_batch("r1"))
        pending = emitter._queue[0]
        self.assertGreaterEqual(pending.due_at, clock() + 10.0)

    def test_retry_after_on_429_is_honoured(self):
        clock = _Clock()
        emitter = _emitter(clock)
        with (
            _NO_JITTER,
            patch("urllib.request.urlopen", side_effect=_http_error(429, {"Retry-After": "7"})),
        ):
            emitter.ship(_batch("r1"))
        self.assertGreaterEqual(emitter._queue[0].due_at, clock() + 7.0)

    def test_retry_after_is_capped_at_backoff_cap(self):
        clock = _Clock()
        emitter = _emitter(clock, backoff_cap_s=30.0)
        with patch("urllib.request.urlopen", side_effect=_http_error(503, {"Retry-After": "600"})):
            emitter.ship(_batch("r1"))
        self.assertLessEqual(emitter._queue[0].due_at, clock() + 30.0)

    def test_retry_after_read_from_status_response_headers(self):
        clock = _Clock()
        emitter = _emitter(clock)
        with (
            _NO_JITTER,
            patch(
                "urllib.request.urlopen", return_value=_status_response(503, {"Retry-After": "12"})
            ),
        ):
            emitter.ship(_batch("r1"))
        self.assertGreaterEqual(emitter._queue[0].due_at, clock() + 12.0)

    def test_backoff_doubles_per_attempt_and_caps(self):
        clock = _Clock()
        emitter = _emitter(clock, max_retries=10, backoff_base_s=1.0, backoff_cap_s=30.0)
        delays = []
        with _NO_JITTER, patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.ship(_batch("r1"))
            for _ in range(7):
                delays.append(emitter._queue[0].due_at - clock())
                clock.advance(delays[-1])
                emitter.retry_pending()
        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0])

    def test_jitter_keeps_delay_within_one_to_two_times_base(self):
        clock = _Clock()
        for _ in range(20):
            emitter = _emitter(clock, backoff_base_s=1.0)
            with patch("urllib.request.urlopen", side_effect=_http_error(503)):
                emitter.ship(_batch("r1"))
            delay = emitter._queue[0].due_at - clock()
            self.assertGreaterEqual(delay, 1.0)
            self.assertLess(delay, 2.0)

    def test_gives_up_after_max_retries_with_one_warning(self):
        clock = _Clock()
        emitter = _emitter(clock, max_retries=3)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)) as mock_urlopen:
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                emitter.ship(_batch("r1", "r2"))
                for _ in range(10):  # more passes than retries: it must stop on its own
                    clock.advance(60.0)
                    emitter.retry_pending()
        # 1 initial attempt + 3 retries, then dropped.
        self.assertEqual(mock_urlopen.call_count, 4)
        self.assertEqual(emitter.pending_events, 0)
        give_ups = [m for m in logs.output if "Dropping 2 events after 4 failed attempt" in m]
        self.assertEqual(len(give_ups), 1)
        self.assertIn("HTTP 503", give_ups[0])

    def test_max_retries_zero_disables_retry_and_drops_with_warning(self):
        emitter = _emitter(max_retries=0)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                self.assertFalse(emitter.ship(_batch("r1")))
        self.assertFalse(emitter.retry_enabled)
        self.assertEqual(emitter.pending_events, 0)
        self.assertTrue(any("Dropping 1 events" in m for m in logs.output))

    def test_transient_failure_warning_is_rate_limited(self):
        """An outage must not log a WARNING per drain cycle — one per minute,
        carrying the count since the last, with detail at DEBUG."""
        clock = _Clock()
        emitter = _emitter(clock)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                for i in range(5):
                    emitter.ship(_batch(f"r{i}"))
                    clock.advance(0.2)
        self.assertEqual(len(logs.output), 1)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            clock.advance(61.0)
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                emitter.ship(_batch("r9"))
        self.assertEqual(len(logs.output), 1)
        self.assertIn("5 failed attempt(s) since last warning", logs.output[0])

    def test_ship_never_sleeps(self):
        emitter = _emitter()
        with (
            patch("dunetrace.emitters.time.sleep") as mock_sleep,
            patch("urllib.request.urlopen", side_effect=_http_error(503, {"Retry-After": "30"})),
        ):
            emitter.ship(_batch("r1"))
            emitter.retry_pending()
        mock_sleep.assert_not_called()


# ── Retry queue: ordering and bound ─────────────────────────────────────────────


class TestRetryQueue(unittest.TestCase):
    def test_oldest_first_and_stops_at_first_failure(self):
        clock = _Clock()
        emitter = _emitter(clock)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            for r in ("r1", "r2", "r3"):
                emitter.ship(_batch(r))
        clock.advance(60.0)
        # r1 succeeds, r2 fails again → stop; r3 is not attempted.
        with patch(
            "urllib.request.urlopen", side_effect=[_ok_response(), _http_error(503)]
        ) as mock_urlopen:
            self.assertEqual(emitter.retry_pending(), 1)
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(_run_ids_in(mock_urlopen, 0), ["r1"])
        self.assertEqual(_run_ids_in(mock_urlopen, 1), ["r2"])
        # r2 went back to the front, still ahead of r3.
        self.assertEqual([p.batch[0].run_id for p in emitter._queue], ["r2", "r3"])
        self.assertEqual(emitter._queue[0].attempts, 2)

    def test_head_not_due_blocks_newer_batches(self):
        """Order is preserved even when a newer batch's backoff is shorter."""
        clock = _Clock()
        emitter = _emitter(clock)
        with (
            _NO_JITTER,
            patch("urllib.request.urlopen", side_effect=_http_error(503, {"Retry-After": "20"})),
        ):
            emitter.ship(_batch("r1"))
        with _NO_JITTER, patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.ship(_batch("r2"))  # due in 1s, but behind r1
        clock.advance(5.0)
        with patch("urllib.request.urlopen", return_value=_ok_response()) as mock_urlopen:
            self.assertEqual(emitter.retry_pending(), 0)
        mock_urlopen.assert_not_called()

    def test_queue_bound_drops_oldest_with_warning(self):
        clock = _Clock()
        emitter = _emitter(clock, max_queued_events=2)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.ship(_batch("r1"))
            emitter.ship(_batch("r2"))
            with patch("dunetrace.emitters.logger.warning") as mock_warning:
                emitter.ship(_batch("r3"))  # over the cap: r1 evicted
        self.assertEqual([p.batch[0].run_id for p in emitter._queue], ["r2", "r3"])
        self.assertEqual(emitter.pending_events, 2)
        drop_calls = [c for c in mock_warning.call_args_list if "dropped" in c[0][0]]
        self.assertEqual(len(drop_calls), 1)
        self.assertIn("retry-queue cap", drop_calls[0][0][0])

    def test_queue_bound_warning_rate_limited_to_once_per_minute(self):
        clock = _Clock()
        emitter = _emitter(clock, max_queued_events=1)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.ship(_batch("r1"))
            with patch("dunetrace.emitters.logger.warning") as mock_warning:
                emitter.ship(_batch("r2"))  # evicts r1 → warns
                clock.advance(10.0)
                emitter.ship(_batch("r3"))  # evicts r2 → suppressed
                clock.advance(10.0)
                emitter.ship(_batch("r4"))  # evicts r3 → suppressed
                clock.advance(61.0)
                emitter.ship(_batch("r5"))  # evicts r4 → warns, with the count since last
        drop_calls = [c for c in mock_warning.call_args_list if "dropped" in c[0][0]]
        self.assertEqual(len(drop_calls), 2)
        self.assertEqual(drop_calls[1][0][1], 3)  # batches dropped since the first warning
        self.assertEqual(emitter.pending_events, 1)

    def test_single_batch_larger_than_cap_is_dropped(self):
        emitter = _emitter(max_queued_events=1)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.ship(_batch("r1", "r2"))
        self.assertEqual(emitter.pending_events, 0)


# ── disable_retry() — wrapper composition ───────────────────────────────────────


class TestDisableRetry(unittest.TestCase):
    def test_disabled_emitter_returns_false_and_queues_nothing(self):
        emitter = _emitter()
        emitter.disable_retry()
        self.assertFalse(emitter.retry_enabled)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            self.assertFalse(emitter.ship(_batch("r1")))
        self.assertEqual(emitter.pending_events, 0)

    def test_batches_queued_before_handoff_are_still_finished_here(self):
        clock = _Clock()
        emitter = _emitter(clock)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.ship(_batch("r1"))
        emitter.disable_retry()
        clock.advance(60.0)
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            emitter.retry_pending()
        self.assertEqual(emitter.pending_events, 1)  # requeued, not dropped


# ── Configuration ───────────────────────────────────────────────────────────────


class TestConfiguration(unittest.TestCase):
    def test_defaults(self):
        emitter = _emitter()
        self.assertEqual(emitter._max_retries, 5)
        self.assertEqual(emitter._backoff_base_s, 1.0)
        self.assertEqual(emitter._backoff_cap_s, 30.0)
        self.assertEqual(emitter._max_queued_events, 10_000)
        self.assertTrue(emitter.retry_enabled)

    def test_env_overrides_apply_when_kwarg_omitted(self):
        env = {
            "DUNETRACE_SHIP_MAX_RETRIES": "2",
            "DUNETRACE_SHIP_BACKOFF_BASE_S": "0.5",
            "DUNETRACE_SHIP_BACKOFF_CAP_S": "9",
            "DUNETRACE_SHIP_MAX_QUEUED_EVENTS": "50",
        }
        with patch.dict(os.environ, env):
            emitter = _emitter()
        self.assertEqual(emitter._max_retries, 2)
        self.assertEqual(emitter._backoff_base_s, 0.5)
        self.assertEqual(emitter._backoff_cap_s, 9.0)
        self.assertEqual(emitter._max_queued_events, 50)

    def test_explicit_kwarg_wins_over_env(self):
        with patch.dict(os.environ, {"DUNETRACE_SHIP_MAX_RETRIES": "2"}):
            emitter = _emitter(max_retries=7)
        self.assertEqual(emitter._max_retries, 7)

    def test_invalid_env_value_warns_and_uses_default(self):
        with patch.dict(os.environ, {"DUNETRACE_SHIP_MAX_RETRIES": "lots"}):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                emitter = _emitter()
        self.assertEqual(emitter._max_retries, 5)
        self.assertIn("DUNETRACE_SHIP_MAX_RETRIES", logs.output[0])

    def test_default_emitter_built_by_client_reads_env(self):
        with patch.dict(os.environ, {"DUNETRACE_SHIP_MAX_RETRIES": "1"}):
            client = Dunetrace(api_key="dt_test")
        try:
            self.assertEqual(client._emitter._max_retries, 1)
        finally:
            client.shutdown(timeout=1)


class TestParseRetryAfter(unittest.TestCase):
    def test_delta_seconds(self):
        self.assertEqual(_parse_retry_after("12"), 12.0)
        self.assertEqual(_parse_retry_after(" 1.5 "), 1.5)

    def test_negative_clamped_to_zero(self):
        self.assertEqual(_parse_retry_after("-3"), 0.0)

    def test_absent_or_garbage_is_none(self):
        self.assertIsNone(_parse_retry_after(None))
        self.assertIsNone(_parse_retry_after(""))
        self.assertIsNone(_parse_retry_after("soon"))
        self.assertIsNone(_parse_retry_after(MagicMock()))

    def test_http_date_in_the_future(self):
        from datetime import datetime, timedelta, timezone

        when = datetime.now(timezone.utc) + timedelta(seconds=90)
        value = email.utils.format_datetime(when, usegmt=True)
        parsed = _parse_retry_after(value)
        self.assertIsNotNone(parsed)
        self.assertGreater(parsed, 80.0)
        self.assertLessEqual(parsed, 90.0)

    def test_http_date_in_the_past_is_zero(self):
        self.assertEqual(_parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), 0.0)


# ── BatchingEmitter.retry_pending() default and the drain thread's idle hook ────


class TestRetryPendingHook(unittest.TestCase):
    def test_base_default_is_zero(self):
        self.assertEqual(NoopBatchingEmitter().retry_pending(), 0)

        class Minimal(BatchingEmitter):
            def ship(self, batch):
                return True

        self.assertEqual(Minimal().retry_pending(), 0)

    def test_drain_loop_calls_retry_pending_when_idle(self):
        called = threading.Event()

        class Recording(BatchingEmitter):
            def __init__(self):
                self.shipped = []

            def ship(self, batch):
                self.shipped.extend(batch)
                return True

            def retry_pending(self):
                called.set()
                return 0

        emitter = Recording()
        client = Dunetrace(emitter=emitter, flush_interval_ms=10)
        try:
            self.assertTrue(called.wait(timeout=2.0), "idle drain loop never called retry_pending")
        finally:
            client.shutdown(timeout=1)

    def test_drain_loop_survives_retry_pending_raising(self):
        raised = threading.Event()

        class Exploding(BatchingEmitter):
            def __init__(self):
                self.shipped = []

            def ship(self, batch):
                self.shipped.extend(batch)
                return True

            def retry_pending(self):
                raised.set()
                raise RuntimeError("boom")

        emitter = Exploding()
        client = Dunetrace(emitter=emitter, flush_interval_ms=10)
        try:
            self.assertTrue(raised.wait(timeout=2.0))
            self.assertTrue(client._drain_thread.is_alive())
            with client.run("agent-1"):
                pass
            client.flush(block=True, timeout=2)
        finally:
            client.shutdown(timeout=1)
        self.assertGreaterEqual(len(emitter.shipped), 2)  # still shipping after the raise

    def test_idle_retry_delivers_due_batch_without_new_events(self):
        """End to end through the real client: a 503'd batch is re-sent while
        the agent is quiet, once its backoff elapses, with no new event to
        trigger a ship()."""
        clock = _Clock()
        emitter = HttpBatchingEmitter("http://localhost:8001", clock=clock, backoff_base_s=0.0)
        delivered = threading.Event()

        def _urlopen(req, timeout=None):
            if _urlopen.calls == 0:
                _urlopen.calls += 1
                raise _http_error(503)
            _urlopen.calls += 1
            delivered.set()
            return _ok_response()

        _urlopen.calls = 0

        with patch("urllib.request.urlopen", side_effect=_urlopen):
            client = Dunetrace(emitter=emitter, flush_interval_ms=10)
            try:
                with client.run("agent-1"):
                    pass
                client.flush(block=True, timeout=2)
                # backoff_base_s=0 → due immediately; the idle loop must pick it up.
                self.assertTrue(delivered.wait(timeout=2.0), "due retry never attempted while idle")
            finally:
                client.shutdown(timeout=1)
        self.assertEqual(emitter.pending_events, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
