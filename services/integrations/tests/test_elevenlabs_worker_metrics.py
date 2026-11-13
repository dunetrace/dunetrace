"""
Metrics and readiness for the ElevenLabs worker (hardening Phase 3.1/3.2).

The second of the two workers built from the integrations image, with its own
enable flag, its own metrics port and its own /ready. Mirrors
test_worker_metrics.py's structure for the evaluation-provider worker; the one
behaviour that has no counterpart there is the correlation pass, which runs
after each poll and is counted under its own failures_total{where} label so a
correlation bug is never mistaken for a polling one.

Metric values are read back through the prometheus registry, which is
process-global, so every assertion is a delta (see _Meter).

Readiness is driven through the worker's own surface — _mark_poll_started /
_mark_poll_ok to record a cycle, settings.WAKE_INTERVAL to size the staleness
window — rather than by reaching into the freshness bookkeeping.

No DB, no credentials, no network beyond 127.0.0.1.

Run:
    PYTHONPATH=packages/schemas-py:services/integrations \
        python -m pytest services/integrations/tests/test_elevenlabs_worker_metrics.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import types
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import integrations_svc.elevenlabs_worker as worker  # import before patch() resolves paths
from dunetrace_schemas import metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION
from integrations_svc.config import Settings

_needs_prometheus = unittest.skipUnless(
    metrics.AVAILABLE, "prometheus_client not installed; every metric is a no-op"
)


# ── Reading metric samples ────────────────────────────────────────────────────


def _sample(name: str, labels: dict | None = None) -> float:
    value = metrics.REGISTRY.get_sample_value(name, labels or {})
    return float(value or 0.0)


class _Meter:
    """Snapshot metric samples before an action, read the deltas after."""

    def __init__(self, *specs: tuple):
        self._specs = [(n, dict(lbl) if lbl else {}) for n, lbl in specs]
        self._before = {self._key(n, lbl): _sample(n, lbl) for n, lbl in self._specs}

    @staticmethod
    def _key(name: str, labels: dict) -> tuple:
        return (name, tuple(sorted(labels.items())))

    def delta(self, name: str, labels: dict | None = None) -> float:
        labels = labels or {}
        return _sample(name, labels) - self._before[self._key(name, labels)]


BACKLOG = "dunetrace_elevenlabs_backlog"
PROCESSED = "dunetrace_elevenlabs_processed_total"
FAILURES = "dunetrace_elevenlabs_failures_total"
EXTERNAL_SECONDS_COUNT = "dunetrace_elevenlabs_external_call_seconds_count"
EXTERNAL_CALLS = "dunetrace_elevenlabs_external_calls_total"
POLL_SECONDS_COUNT = "dunetrace_elevenlabs_poll_seconds_count"
SCHEMA_GAUGE = "dunetrace_schema_version"


# ── Fake pool: the real db_ready() runs against this ──────────────────────────


class _Conn:
    def __init__(self, version, error):
        self._version = version
        self._error = error

    async def fetchval(self, sql, *args):
        if self._error is not None:
            raise self._error
        if "MAX(version)" in sql:
            return self._version
        return 1


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    """Enough of an asyncpg pool for dunetrace_schemas.metrics.db_ready()."""

    def __init__(self, version: int | None = None, error: Exception | None = None):
        self._conn = _Conn(version, error)

    def acquire(self):
        return _Acquire(self._conn)


# ── Poll-freshness state ──────────────────────────────────────────────────────

_FRESHNESS_GLOBALS = ("_last_poll_ok_at", "_poll_started_at", "_poll_finished_at")


def _clear_poll_state(mod: types.ModuleType = worker) -> None:
    """Put the worker back in its "no cycle has ever run" state."""
    for name in _FRESHNESS_GLOBALS:
        if hasattr(mod, name):
            setattr(mod, name, None)


def _start_cycle(mod: types.ModuleType = worker) -> None:
    """A cycle has begun and has not finished (the worker is busy)."""
    start = getattr(mod, "_mark_poll_started", None)
    if callable(start):
        start()


def _complete_cycle(mod: types.ModuleType = worker) -> None:
    """One cycle start to finish, through the worker's own bookkeeping."""
    _start_cycle(mod)
    mod._mark_poll_ok()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _integration(id=1, org_id="org-1"):
    return {
        "id": id,
        "org_id": org_id,
        "encrypted_credentials": "encrypted-blob",
        "poll_interval_secs": 300,
        "last_seen_generation_at": None,
        "first_failure_at": None,
        "last_alerted_at": None,
        "created_at": datetime.now(timezone.utc),
    }


def _provider_cls(generations=(), error=None):
    """A stand-in ElevenLabsProvider class — _poll_one calls
    ElevenLabsProvider(**creds), so this has to be callable."""
    instance = MagicMock()
    instance.fetch_generations = AsyncMock(
        side_effect=error if error is not None else None,
        return_value=list(generations),
    )
    cls = MagicMock(side_effect=lambda *a, **kw: instance)
    cls._instance = instance
    return cls


# ── _poll_one: the per-integration outcome counters ───────────────────────────


@_needs_prometheus
class TestPollOneMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_successful_poll_counts_result_ok(self):
        m = _Meter(
            (PROCESSED, {"result": "ok"}),
            (PROCESSED, {"result": "error"}),
            (FAILURES, {"where": "integration"}),
        )
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch("integrations_svc.elevenlabs_worker.ElevenLabsProvider", _provider_cls()),
            patch("integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()),
        ):
            await worker._poll_one(_integration())
        self.assertEqual(m.delta(PROCESSED, {"result": "ok"}), 1.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "error"}), 0.0)
        self.assertEqual(m.delta(FAILURES, {"where": "integration"}), 0.0)

    async def test_failing_poll_counts_result_error_and_a_contained_failure(self):
        m = _Meter(
            (PROCESSED, {"result": "ok"}),
            (PROCESSED, {"result": "error"}),
            (FAILURES, {"where": "integration"}),
        )
        state = {"first_failure_at": None, "last_alerted_at": None, "consecutive_failures": 1}
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch(
                "integrations_svc.elevenlabs_worker.ElevenLabsProvider",
                _provider_cls(error=RuntimeError("elevenlabs 503")),
            ),
            patch("integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()),
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_poll_failure",
                AsyncMock(return_value=state),
            ),
        ):
            await worker._poll_one(_integration())  # contained, must not raise
        self.assertEqual(m.delta(PROCESSED, {"result": "error"}), 1.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "ok"}), 0.0)
        self.assertEqual(m.delta(FAILURES, {"where": "integration"}), 1.0)


# ── _observed_call: the outbound-fetch histogram and counter ──────────────────


@_needs_prometheus
class TestExternalCallMetrics(unittest.IsolatedAsyncioTestCase):
    def _meter(self, provider: str) -> _Meter:
        return _Meter(
            (EXTERNAL_SECONDS_COUNT, {"provider": provider}),
            (EXTERNAL_CALLS, {"provider": provider, "status": "ok"}),
            (EXTERNAL_CALLS, {"provider": provider, "status": "error"}),
        )

    async def test_successful_fetch_observes_ok_once_for_its_provider(self):
        m = self._meter(worker._PROVIDER)
        out = await worker._observed_call(worker._PROVIDER, AsyncMock(return_value=["gen"]), 0.0)
        self.assertEqual(out, ["gen"])
        self.assertEqual(m.delta(EXTERNAL_SECONDS_COUNT, {"provider": worker._PROVIDER}), 1.0)
        self.assertEqual(
            m.delta(EXTERNAL_CALLS, {"provider": worker._PROVIDER, "status": "ok"}), 1.0
        )
        self.assertEqual(
            m.delta(EXTERNAL_CALLS, {"provider": worker._PROVIDER, "status": "error"}), 0.0
        )

    async def test_failed_fetch_observes_error_and_re_raises(self):
        m = self._meter(worker._PROVIDER)
        with self.assertRaises(RuntimeError):
            await worker._observed_call(
                worker._PROVIDER, AsyncMock(side_effect=RuntimeError("429")), 0.0
            )
        # A call that failed still took time and still cost a slot.
        self.assertEqual(m.delta(EXTERNAL_SECONDS_COUNT, {"provider": worker._PROVIDER}), 1.0)
        self.assertEqual(
            m.delta(EXTERNAL_CALLS, {"provider": worker._PROVIDER, "status": "error"}), 1.0
        )
        self.assertEqual(
            m.delta(EXTERNAL_CALLS, {"provider": worker._PROVIDER, "status": "ok"}), 0.0
        )

    async def test_missing_provider_is_labelled_unknown_not_none(self):
        """The label set must stay closed and stringly-typed — a None provider
        must not become a `None` time series."""
        m = self._meter("unknown")
        await worker._observed_call(None, AsyncMock(return_value=[]), 0.0)
        self.assertEqual(m.delta(EXTERNAL_CALLS, {"provider": "unknown", "status": "ok"}), 1.0)

    async def test_an_integration_poll_observes_its_provider_fetch(self):
        """The counter measures what it claims: one outbound fetch per poll."""
        m = self._meter(worker._PROVIDER)
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch("integrations_svc.elevenlabs_worker.ElevenLabsProvider", _provider_cls()),
            patch("integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()),
        ):
            await worker._poll_one(_integration())
        self.assertEqual(
            m.delta(EXTERNAL_CALLS, {"provider": worker._PROVIDER, "status": "ok"}), 1.0
        )


# ── poll_once: the backlog gauge ──────────────────────────────────────────────


@_needs_prometheus
class TestPollOnceMetrics(unittest.IsolatedAsyncioTestCase):
    def _due(self, n):
        return patch(
            "integrations_svc.elevenlabs_worker.fetch_due_elevenlabs_integrations",
            AsyncMock(return_value=[_integration(id=i) for i in range(n)]),
        )

    async def test_empty_wake_sets_backlog_zero(self):
        with self._due(0):
            self.assertEqual(await worker.poll_once(), 0)
        self.assertEqual(_sample(BACKLOG), 0.0)

    async def test_backlog_is_every_due_account(self):
        with self._due(3), patch("integrations_svc.elevenlabs_worker._poll_one", AsyncMock()):
            self.assertEqual(await worker.poll_once(), 3)
        self.assertEqual(_sample(BACKLOG), 3.0)

    async def test_backlog_drops_back_to_zero_on_the_next_empty_wake(self):
        with self._due(2), patch("integrations_svc.elevenlabs_worker._poll_one", AsyncMock()):
            await worker.poll_once()
        self.assertEqual(_sample(BACKLOG), 2.0)
        with self._due(0):
            await worker.poll_once()
        self.assertEqual(_sample(BACKLOG), 0.0)


# ── The loop in run_worker: poll timing, cycle/correlation failures, gauge ────


@_needs_prometheus
class TestRunWorkerLoopMetrics(unittest.IsolatedAsyncioTestCase):
    """One or two turns of run_worker's loop, then out via CancelledError —
    which the inner `except Exception` deliberately does not catch."""

    def setUp(self):
        self.addCleanup(_clear_poll_state)
        for name, value in (
            ("ELEVENLABS_WORKER_ENABLED", True),
            ("WAKE_INTERVAL", 0),
            ("ELEVENLABS_METRICS_PORT", 0),
        ):
            p = patch.object(worker.settings, name, value)
            p.start()
            self.addCleanup(p.stop)

    async def _run_loop(self, poll_side_effect, correlate=None):
        close_mock = AsyncMock()
        with (
            patch("integrations_svc.elevenlabs_worker.init_pool", AsyncMock()),
            patch("integrations_svc.elevenlabs_worker.ensure_elevenlabs_schema", AsyncMock()),
            patch("integrations_svc.elevenlabs_worker.close_pool", close_mock),
            patch(
                "integrations_svc.elevenlabs_worker.poll_once",
                AsyncMock(side_effect=poll_side_effect),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.correlate_once",
                correlate if correlate is not None else AsyncMock(),
            ),
        ):
            await worker.run_worker()
        return close_mock

    async def test_a_completed_cycle_is_timed_and_the_schema_gauge_is_set(self):
        m = _Meter((POLL_SECONDS_COUNT, None))
        close_mock = await self._run_loop([0, asyncio.CancelledError()])
        self.assertEqual(m.delta(POLL_SECONDS_COUNT), 1.0)
        self.assertEqual(
            _sample(SCHEMA_GAUGE, {"service": "elevenlabs"}), float(CURRENT_SCHEMA_VERSION)
        )
        close_mock.assert_awaited_once()

    async def test_a_failing_cycle_counts_where_cycle_and_keeps_the_loop_alive(self):
        m = _Meter(
            (FAILURES, {"where": "cycle"}),
            (FAILURES, {"where": "correlation"}),
            (POLL_SECONDS_COUNT, None),
        )
        correlate = AsyncMock()
        await self._run_loop([RuntimeError("db gone"), 0, asyncio.CancelledError()], correlate)
        self.assertEqual(m.delta(FAILURES, {"where": "cycle"}), 1.0)
        self.assertEqual(m.delta(FAILURES, {"where": "correlation"}), 0.0)
        # Both cycles were timed — a failed cycle still took wall-clock time.
        self.assertEqual(m.delta(POLL_SECONDS_COUNT), 2.0)
        # A failed poll must not skip correlation over already-stored data.
        self.assertEqual(correlate.await_count, 2)

    async def test_a_failing_correlation_pass_has_its_own_label(self):
        """Correlation runs after polling over already-stored generations; a
        bug there must be distinguishable from a polling failure, and must not
        stop the loop."""
        m = _Meter((FAILURES, {"where": "correlation"}), (FAILURES, {"where": "cycle"}))
        await self._run_loop(
            [0, asyncio.CancelledError()], AsyncMock(side_effect=RuntimeError("boom"))
        )
        self.assertEqual(m.delta(FAILURES, {"where": "correlation"}), 1.0)
        self.assertEqual(m.delta(FAILURES, {"where": "cycle"}), 0.0)

    async def test_a_completed_cycle_makes_ready_report_a_fresh_poll(self):
        _clear_poll_state()
        await self._run_loop([0, asyncio.CancelledError()])
        with (
            patch.object(worker.settings, "WAKE_INTERVAL", 10),
            patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)),
        ):
            ok, info = await worker._ready_check()
        self.assertTrue(ok)
        self.assertEqual(info["poll"], "ok")


class TestDisabledWorkerRegistersNothing(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_worker_never_starts_a_metrics_server(self):
        with (
            patch.object(worker.settings, "ELEVENLABS_WORKER_ENABLED", False),
            patch("integrations_svc.elevenlabs_worker.init_pool", AsyncMock()) as init_mock,
            patch.object(worker._metrics, "start_metrics_server") as start_mock,
        ):
            await worker.run_worker()
        init_mock.assert_not_awaited()
        start_mock.assert_not_called()


# ── _ready_check: the states a compose healthcheck sees ───────────────────────


class TestReadyCheck(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _clear_poll_state()
        self.addCleanup(_clear_poll_state)
        p = patch.object(worker.settings, "WAKE_INTERVAL", 10)
        p.start()
        self.addCleanup(p.stop)

    async def test_not_ready_when_there_is_no_pool(self):
        _complete_cycle()
        with patch.object(worker._db, "_pool", None):
            ok, info = await worker._ready_check()
        self.assertFalse(ok)
        self.assertEqual(info["db"], "no_pool")

    async def test_not_ready_when_the_db_is_unreachable(self):
        _complete_cycle()
        pool = _FakePool(error=ConnectionRefusedError("connection refused"))
        with patch.object(worker._db, "_pool", pool):
            ok, info = await worker._ready_check()
        self.assertFalse(ok)
        self.assertEqual(info["db"], "ConnectionRefusedError")
        # The poll loop is fine; only the DB is not.
        self.assertEqual(info["poll"], "ok")

    async def test_not_ready_when_the_applied_schema_is_too_old(self):
        _complete_cycle()
        pool = _FakePool(version=CURRENT_SCHEMA_VERSION - 1)
        with patch.object(worker._db, "_pool", pool):
            ok, info = await worker._ready_check()
        self.assertFalse(ok)
        self.assertEqual(info["db"], "ok")  # it answered; it is just behind
        self.assertEqual(info["schema_version"], CURRENT_SCHEMA_VERSION - 1)
        self.assertEqual(info["required"], CURRENT_SCHEMA_VERSION)

    async def test_not_ready_when_no_cycle_has_ever_run(self):
        with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)):
            ok, info = await worker._ready_check()
        self.assertFalse(ok)
        self.assertEqual(info["db"], "ok")
        self.assertNotEqual(info["poll"], "ok")

    async def test_not_ready_when_the_last_completed_cycle_is_long_past(self):
        """A worker whose loop has stopped turning between cycles is not
        healthy just because the process is still alive. Sized by shrinking the
        interval rather than by moving the clock, so only elapsed real time is
        involved."""
        with patch.object(worker.settings, "WAKE_INTERVAL", 0.001):
            _complete_cycle()
            await asyncio.sleep(0.05)
            with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)):
                ok, info = await worker._ready_check()
        self.assertFalse(ok)
        self.assertEqual(info["db"], "ok")
        self.assertEqual(info["poll"], "stale")

    async def test_ready_when_the_db_is_current_and_a_cycle_just_completed(self):
        _complete_cycle()
        with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)):
            ok, info = await worker._ready_check()
        self.assertTrue(ok)
        self.assertEqual(info["db"], "ok")
        self.assertEqual(info["poll"], "ok")
        self.assertEqual(info["schema_version"], CURRENT_SCHEMA_VERSION)

    async def test_the_grace_window_spans_more_than_one_cycle(self):
        """Documented behaviour of /ready: a worker gets more than one missed
        cycle before it is called dead, and says so in the payload."""
        _complete_cycle()
        with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)):
            _ok, info = await worker._ready_check()
        self.assertGreaterEqual(info["stale_after_seconds"], 2 * 10)

    async def test_a_schema_ahead_of_this_build_is_still_ready(self):
        _complete_cycle()
        with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION + 1)):
            ok, _ = await worker._ready_check()
        self.assertTrue(ok)

    async def test_a_cycle_still_running_is_not_called_dead(self):
        """A wake that is still paging an account's TTS history outlives
        3 × WAKE_INTERVAL routinely; reporting unhealthy then invites an
        orchestrator to restart the busiest worker mid-cycle."""
        with patch.object(worker.settings, "WAKE_INTERVAL", 0.001):
            _complete_cycle()
            _start_cycle()  # ... and this one has not finished
            await asyncio.sleep(0.05)  # well past 3 × WAKE_INTERVAL
            with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)):
                ok, info = await worker._ready_check()
        self.assertTrue(ok)
        self.assertEqual(info["poll"], "ok")

    @unittest.skipUnless(
        hasattr(worker, "_INFLIGHT_GRACE_SECS"), "this build has no in-flight grace window"
    )
    async def test_the_in_flight_grace_is_a_bound_not_an_exemption(self):
        """A genuinely wedged cycle still goes stale — just later."""
        with (
            patch.object(worker.settings, "WAKE_INTERVAL", 0.001),
            patch.object(worker, "_INFLIGHT_GRACE_SECS", 0.001),
        ):
            _complete_cycle()
            _start_cycle()
            await asyncio.sleep(0.05)
            with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)):
                ok, info = await worker._ready_check()
        self.assertFalse(ok)
        self.assertEqual(info["poll"], "stale")


# ── The metrics server: never fatal, and it serves what it claims ─────────────


class TestMetricsServer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _clear_poll_state()
        self.addCleanup(_clear_poll_state)

    async def test_port_zero_disables_the_server(self):
        self.assertIsNone(
            worker._metrics.start_metrics_server(0, worker._ready_check, asyncio.get_running_loop())
        )

    async def test_bind_failure_is_logged_not_raised(self):
        """Both containers run from this one image; a single-container
        deployment binding both ports in one network namespace is exactly how
        this collides. It must cost the scrape endpoint, not the worker."""
        # Loopback on BOTH sides. Binding the blocker to 0.0.0.0 was the only
        # way to guarantee the collision while the server defaulted to all
        # interfaces — but a test has no business listening off-host, and
        # bandit flags it (B104). Pinning the server to 127.0.0.1 makes the
        # addresses identical, so the collision is exact and guaranteed.
        blocker = socket.socket()
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 0))
        port = blocker.getsockname()[1]
        blocker.listen(1)
        self.addCleanup(blocker.close)
        with self.assertLogs("dunetrace.metrics", level="WARNING") as logs:
            server = worker._metrics.start_metrics_server(
                port, worker._ready_check, asyncio.get_running_loop(), host="127.0.0.1"
            )
        self.assertIsNone(server)
        self.assertTrue(any("could not bind" in line for line in logs.output))

    async def test_worker_startup_survives_a_metrics_server_that_never_came_up(self):
        """With no server, run_worker must still reach its loop and shut down
        cleanly — observability is not a dependency."""
        poll_mock = AsyncMock(side_effect=[0, asyncio.CancelledError()])
        close_mock = AsyncMock()
        with (
            patch.object(worker.settings, "ELEVENLABS_WORKER_ENABLED", True),
            patch.object(worker.settings, "WAKE_INTERVAL", 0),
            patch.object(worker._metrics, "start_metrics_server", MagicMock(return_value=None)),
            patch("integrations_svc.elevenlabs_worker.init_pool", AsyncMock()),
            patch("integrations_svc.elevenlabs_worker.ensure_elevenlabs_schema", AsyncMock()),
            patch("integrations_svc.elevenlabs_worker.close_pool", close_mock),
            patch("integrations_svc.elevenlabs_worker.correlate_once", AsyncMock()),
            patch("integrations_svc.elevenlabs_worker.poll_once", poll_mock),
        ):
            await worker.run_worker()  # must not raise
        self.assertEqual(poll_mock.await_count, 2)
        close_mock.assert_awaited_once()

    @_needs_prometheus
    async def test_server_serves_metrics_ready_and_health(self):
        port = _free_port()
        metrics.register_standard("elevenlabs", worker.settings.APP_VERSION)
        metrics.set_schema_version(CURRENT_SCHEMA_VERSION)
        with patch.object(worker.settings, "WAKE_INTERVAL", 10):
            server = metrics.start_metrics_server(
                port, worker._ready_check, asyncio.get_running_loop(), host="127.0.0.1"
            )
            self.assertIsNotNone(server)
            try:
                base = f"http://127.0.0.1:{server.port}"
                status, _ = await asyncio.to_thread(_get, base + "/health")
                self.assertEqual(status, 200)

                # No cycle has run yet → 503 with the reason.
                with patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)):
                    status, body = await asyncio.to_thread(_get, base + "/ready")
                    self.assertEqual(status, 503)
                    self.assertEqual(json.loads(body)["status"], "not_ready")

                    # After a cycle → 200.
                    _complete_cycle()
                    status, body = await asyncio.to_thread(_get, base + "/ready")
                    self.assertEqual(status, 200)
                    payload = json.loads(body)
                    self.assertEqual(payload["status"], "ok")
                    self.assertEqual(payload["schema_version"], CURRENT_SCHEMA_VERSION)

                status, body = await asyncio.to_thread(_get, base + "/metrics")
                self.assertEqual(status, 200)
                text = body.decode()
                self.assertIn('dunetrace_build_info{service="elevenlabs"', text)
                self.assertIn(
                    f'dunetrace_schema_version{{service="elevenlabs"}} '
                    f"{float(CURRENT_SCHEMA_VERSION)}",
                    text,
                )
                for name in (
                    BACKLOG,
                    PROCESSED,
                    FAILURES,
                    "dunetrace_elevenlabs_external_call_seconds",
                    EXTERNAL_CALLS,
                    "dunetrace_elevenlabs_poll_seconds",
                ):
                    self.assertIn(f"# TYPE {name}", text, name)
            finally:
                server.shutdown()
                server.server_close()

    async def test_db_down_at_startup_still_answers_ready_with_503(self):
        """Exactly when a probe needs an answer, the DB is the thing that is
        missing — /ready must exist anyway."""
        port = _free_port()
        _complete_cycle()
        server = metrics.start_metrics_server(
            port, worker._ready_check, asyncio.get_running_loop(), host="127.0.0.1"
        )
        self.assertIsNotNone(server)
        try:
            with patch.object(worker._db, "_pool", None):
                status, body = await asyncio.to_thread(_get, f"http://127.0.0.1:{port}/ready")
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["db"], "no_pool")


# ── Config ────────────────────────────────────────────────────────────────────


class TestConfig(unittest.TestCase):
    def test_metrics_port_default(self):
        self.assertEqual(
            Settings.ELEVENLABS_METRICS_PORT, int(os.getenv("ELEVENLABS_METRICS_PORT", "9105"))
        )

    def test_app_version_present(self):
        self.assertTrue(Settings.APP_VERSION)


if __name__ == "__main__":
    unittest.main()
