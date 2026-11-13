"""
Metrics and readiness for the detector worker (Phase 3.1/3.2).

DB is mocked the same way test_worker.py does it. Metric values are read back
through the prometheus registry, so every assertion is a delta — the registry
is process-global and other tests in the same session move the same counters.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/detector \
        python -m pytest services/detector/tests/test_worker_metrics.py -v
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import socket
import sys
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import AsyncMock, MagicMock, patch

import detector_svc.worker as worker
from dunetrace.models import FailureSignal, FailureType, Severity
from dunetrace_schemas import metrics as shared_metrics

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.require_test_deps import require_prometheus_client  # noqa: E402

# A missing prometheus_client makes every metric in dunetrace_schemas.metrics a
# no-op, so these assertions would prove nothing. This used to be a module-level
# `raise unittest.SkipTest(...)`, which pytest reports as `1 skipped` and exit
# code 0 — dropping the dependency deleted the whole file and CI stayed green.
# It is now a hard error unless DUNETRACE_ALLOW_MISSING_TEST_DEPS is set.
REGISTRY = require_prometheus_client("services/detector/requirements.txt").REGISTRY
assert shared_metrics.AVAILABLE, "prometheus_client imports but metrics degraded to no-ops"


# ── Event factories (same shapes as test_worker.py) ───────────────────────────


def evt(event_type: str, step_index: int = 1, payload: dict | None = None) -> dict:
    return {
        "event_type": event_type,
        "run_id": "run-m-1",
        "agent_id": "agent-test",
        "agent_version": "abc12345",
        "step_index": step_index,
        "timestamp": time.time(),
        "payload": payload or {},
        "parent_run_id": None,
    }


def tool_evt(tool_name: str, step: int) -> dict:
    return evt("tool.called", step, {"tool_name": tool_name, "args": "aa"})


def run_started() -> dict:
    return evt(
        "run.started",
        0,
        {"input_text": "abc123", "model": "gpt-4o", "tools": ["web_search", "calculator"]},
    )


def run_completed(step: int = 10) -> dict:
    return evt("run.completed", step, {"exit_reason": "final_answer", "total_steps": step})


def sample(name: str, labels: dict | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else value


def free_port() -> int:
    """A port that was free a moment ago — see the retry loop that uses it.

    Probing and binding are two steps, so the port can be taken in between;
    `start_metrics_server` swallows the resulting bind failure and returns
    None. Callers must retry rather than trust one draw.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# start_metrics_server defaults to host="0.0.0.0" and the detector worker does
# not override it, so without this the port is probed on loopback and bound on
# every interface. Tests have no business listening off-host.
loopback_metrics_server = functools.partial(shared_metrics.start_metrics_server, host="127.0.0.1")


def http_get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _loop_events() -> list[dict]:
    """A run that fires TOOL_LOOP (a live detector)."""
    events = [run_started()]
    for i in range(1, 9):
        events.append(tool_evt("web_search", i))
    events.append(run_completed(step=9))
    return events


# ── process_run ───────────────────────────────────────────────────────────────


class TestProcessRunMetrics(unittest.IsolatedAsyncioTestCase):
    def _base_patches(self, write_mock=None):
        return (
            # Built-ins only: other test modules leave plugin classes in
            # CUSTOM_DETECTOR_REGISTRY and packs in the TTL cache, and a
            # leftover plugin firing would turn a "clean" run into a signal.
            patch("detector_svc.detectors._build_plugin_detectors", return_value=[]),
            patch("detector_svc.packs.get_enabled_packs", AsyncMock(return_value=set())),
            patch(
                "detector_svc.worker.fetch_run_state",
                AsyncMock(
                    return_value={
                        "processed": False,
                        "event_count": None,
                        "signal_types": set(),
                    }
                ),
            ),
            patch("detector_svc.worker.fetch_run_lineage", AsyncMock(return_value=None)),
            patch(
                "detector_svc.worker.write_signals",
                write_mock if write_mock is not None else AsyncMock(return_value=1),
            ),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()),
            patch("detector_svc.worker.clear_processing_failures", AsyncMock()),
            patch("detector_svc.worker.upsert_fired_issues", AsyncMock()),
            patch("detector_svc.worker.advance_clean_runs", AsyncMock()),
            patch("detector_svc.worker.upsert_run_and_conversation", AsyncMock()),
            patch("detector_svc.worker.fetch_custom_detectors", AsyncMock(return_value=[])),
            patch("detector_svc.worker.write_baseline_metrics", AsyncMock()),
            patch("detector_svc.worker.write_run_state_metrics", AsyncMock()),
        )

    async def _run(self, events, extra=(), write_mock=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch("detector_svc.worker.fetch_run_events", AsyncMock(return_value=events))
            )
            for p in self._base_patches(write_mock):
                stack.enter_context(p)
            for p in extra:
                stack.enter_context(p)
            return await worker.process_run("run-m-1", "agent-test", "abc12345", "completed", "org")

    async def test_signal_run_counts_result_signals_and_signal_type(self):
        runs_before = sample("dunetrace_detector_runs_processed_total", {"result": "signals"})
        sig_before = sample(
            "dunetrace_detector_signals_total", {"failure_type": "TOOL_LOOP", "shadow": "false"}
        )
        count = await self._run(
            _loop_events(), extra=(patch("detector_svc.worker.LIVE_DETECTORS", {"TOOL_LOOP"}),)
        )
        self.assertGreaterEqual(count, 1)
        self.assertEqual(
            sample("dunetrace_detector_runs_processed_total", {"result": "signals"}),
            runs_before + 1,
        )
        self.assertEqual(
            sample(
                "dunetrace_detector_signals_total",
                {"failure_type": "TOOL_LOOP", "shadow": "false"},
            ),
            sig_before + 1,
        )

    async def test_shadow_signal_is_labelled_shadow_true(self):
        before = sample(
            "dunetrace_detector_signals_total", {"failure_type": "TOOL_LOOP", "shadow": "true"}
        )
        await self._run(_loop_events(), extra=(patch("detector_svc.worker.LIVE_DETECTORS", set()),))
        self.assertEqual(
            sample(
                "dunetrace_detector_signals_total", {"failure_type": "TOOL_LOOP", "shadow": "true"}
            ),
            before + 1,
        )

    async def test_signals_total_follows_rows_written_not_signals_found(self):
        """write_signals returns the row count; a 0 (deduped) write is not a signal."""
        before = sample(
            "dunetrace_detector_signals_total", {"failure_type": "TOOL_LOOP", "shadow": "false"}
        )
        clean_before = sample("dunetrace_detector_runs_processed_total", {"result": "clean"})
        await self._run(
            _loop_events(),
            extra=(patch("detector_svc.worker.LIVE_DETECTORS", {"TOOL_LOOP"}),),
            write_mock=AsyncMock(return_value=0),
        )
        self.assertEqual(
            sample(
                "dunetrace_detector_signals_total",
                {"failure_type": "TOOL_LOOP", "shadow": "false"},
            ),
            before,
        )
        # Nothing written → the run counts as clean.
        self.assertEqual(
            sample("dunetrace_detector_runs_processed_total", {"result": "clean"}),
            clean_before + 1,
        )

    async def test_clean_run_counts_result_clean(self):
        before = sample("dunetrace_detector_runs_processed_total", {"result": "clean"})
        count = await self._run([run_started(), tool_evt("web_search", 1), run_completed(step=2)])
        self.assertEqual(count, 0)
        self.assertEqual(
            sample("dunetrace_detector_runs_processed_total", {"result": "clean"}),
            before + 1,
        )

    async def test_run_with_no_events_counts_clean(self):
        before = sample("dunetrace_detector_runs_processed_total", {"result": "clean"})
        await self._run([])
        self.assertEqual(
            sample("dunetrace_detector_runs_processed_total", {"result": "clean"}),
            before + 1,
        )

    async def test_transient_failure_counts_retry_and_exception(self):
        retry_before = sample("dunetrace_detector_runs_processed_total", {"result": "retry"})
        exc_before = sample("dunetrace_detector_exceptions_total", {"where": "process_run_detect"})
        await self._run(
            [run_started()],
            extra=(
                patch(
                    "detector_svc.worker.build_run_state",
                    MagicMock(side_effect=RuntimeError("pool timeout")),
                ),
                patch("detector_svc.worker.record_processing_failure", AsyncMock(return_value=1)),
            ),
        )
        self.assertEqual(
            sample("dunetrace_detector_runs_processed_total", {"result": "retry"}),
            retry_before + 1,
        )
        self.assertEqual(
            sample("dunetrace_detector_exceptions_total", {"where": "process_run_detect"}),
            exc_before + 1,
        )

    async def test_exhausted_budget_counts_failed(self):
        failed_before = sample("dunetrace_detector_runs_processed_total", {"result": "failed"})
        clean_before = sample("dunetrace_detector_runs_processed_total", {"result": "clean"})
        await self._run(
            [run_started()],
            extra=(
                patch(
                    "detector_svc.worker.build_run_state",
                    MagicMock(side_effect=RuntimeError("pool timeout")),
                ),
                patch(
                    "detector_svc.worker.record_processing_failure",
                    AsyncMock(return_value=worker.MAX_PROCESSING_ATTEMPTS),
                ),
            ),
        )
        self.assertEqual(
            sample("dunetrace_detector_runs_processed_total", {"result": "failed"}),
            failed_before + 1,
        )
        # A failed run must never read as clean.
        self.assertEqual(
            sample("dunetrace_detector_runs_processed_total", {"result": "clean"}), clean_before
        )

    async def test_custom_detector_signal_and_exception_are_counted(self):
        detector = {
            "id": 7,
            "shadow": True,
            "config": {
                "failure_type": "MY_CUSTOM",
                "severity": "MEDIUM",
                "conditions": [{"metric": "tool_call_count", "operator": ">=", "value": 1}],
            },
        }
        fired = {
            "failure_type": "MY_CUSTOM",
            "severity": "MEDIUM",
            "step_index": 0,
            "confidence": 0.8,
            "evidence": {},
        }
        before = sample(
            "dunetrace_detector_signals_total", {"failure_type": "custom", "shadow": "true"}
        )
        await self._run(
            [run_started(), tool_evt("web_search", 1), run_completed(step=2)],
            extra=(
                patch(
                    "detector_svc.worker.fetch_custom_detectors", AsyncMock(return_value=[detector])
                ),
                patch("detector_svc.worker.evaluate_custom_detector", return_value=fired),
                patch("detector_svc.worker.write_custom_signal", AsyncMock()),
                patch("detector_svc.worker.record_custom_detector_results", AsyncMock()),
            ),
        )
        self.assertEqual(
            sample(
                "dunetrace_detector_signals_total", {"failure_type": "custom", "shadow": "true"}
            ),
            before + 1,
        )

        exc_before = sample("dunetrace_detector_exceptions_total", {"where": "custom_detector"})
        await self._run(
            [run_started(), tool_evt("web_search", 1), run_completed(step=2)],
            extra=(
                patch(
                    "detector_svc.worker.fetch_custom_detectors",
                    AsyncMock(side_effect=RuntimeError("boom")),
                ),
            ),
        )
        self.assertEqual(
            sample("dunetrace_detector_exceptions_total", {"where": "custom_detector"}),
            exc_before + 1,
        )

    async def test_plugin_signal_is_counted_as_custom_not_by_its_name(self):
        """A plugin's name is caller-supplied free TEXT on a shared worker, so
        it must not become a metric label — 200 orgs x 5 detectors is 1000+
        registry children held for the process lifetime, never reclaimed when
        the detector is deleted. Per-detector counts live in
        custom_detector_results, which is org-scoped and queryable."""
        from dunetrace.detectors import BaseDetector

        class _FakePlugin(BaseDetector):
            name = "my_plugin"
            SHADOW_BY_DEFAULT = False

            def on_run_completion(self, state):
                return FailureSignal(
                    failure_type=FailureType.CUSTOM,
                    severity=Severity.LOW,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                    agent_version=state.agent_version,
                    step_index=0,
                    confidence=0.6,
                    evidence={"detector_name": "my_plugin"},
                )

        before = sample(
            "dunetrace_detector_signals_total", {"failure_type": "custom", "shadow": "false"}
        )
        await self._run(
            [run_started(), run_completed(step=1)],
            extra=(
                patch("detector_svc.worker.get_detectors", AsyncMock(return_value=[_FakePlugin()])),
                patch("detector_svc.worker.CUSTOM_DETECTOR_REGISTRY", {"my_plugin": _FakePlugin}),
                patch("detector_svc.worker.write_custom_signal", AsyncMock()),
            ),
        )
        self.assertEqual(
            sample(
                "dunetrace_detector_signals_total", {"failure_type": "custom", "shadow": "false"}
            ),
            before + 1,
        )


# ── poll_once ─────────────────────────────────────────────────────────────────


class TestPollMetrics(unittest.IsolatedAsyncioTestCase):
    def _patches(self, completed, stalled, batch_size=100):
        mock_settings = MagicMock()
        mock_settings.BATCH_SIZE = batch_size
        mock_settings.SHARD_COUNT = 1
        mock_settings.SHARD_INDEX = 0
        mock_settings.STALL_TIMEOUT_SECS = 90
        mock_settings.WATERMARK_GRACE_SECS = 3600
        mock_settings.DETECTOR_CONCURRENCY = 8
        mock_settings.POLL_INTERVAL = 5
        return (
            patch("detector_svc.worker.fetch_completed_runs", AsyncMock(return_value=completed)),
            patch("detector_svc.worker.fetch_stalled_runs", AsyncMock(return_value=stalled)),
            patch("detector_svc.worker.get_watermark", AsyncMock(return_value=None)),
            patch("detector_svc.worker.advance_watermark", AsyncMock()),
            patch("detector_svc.worker.process_run", AsyncMock(return_value=0)),
            patch("detector_svc.worker.settings", mock_settings),
        )

    @staticmethod
    def _runs(n, prefix):
        return [
            {"run_id": f"{prefix}-{i}", "agent_id": "a", "agent_version": "v", "org_id": "o"}
            for i in range(n)
        ]

    async def _poll(self, completed, stalled, batch_size=100):
        with contextlib.ExitStack() as stack:
            for p in self._patches(completed, stalled, batch_size):
                stack.enter_context(p)
            return await worker.poll_once()

    async def test_backlog_gauge_is_runs_found_and_poll_is_timed(self):
        count_before = sample("dunetrace_detector_poll_seconds_count")
        runs, _ = await self._poll(self._runs(3, "c"), self._runs(2, "s"))
        self.assertEqual(runs, 5)
        self.assertEqual(sample("dunetrace_detector_backlog_runs"), 5)
        self.assertEqual(sample("dunetrace_detector_poll_saturated"), 0)
        self.assertEqual(sample("dunetrace_detector_poll_seconds_count"), count_before + 1)

    async def test_empty_poll_zeroes_backlog_and_still_observes_duration(self):
        count_before = sample("dunetrace_detector_poll_seconds_count")
        await self._poll(self._runs(4, "c"), [])
        self.assertEqual(sample("dunetrace_detector_backlog_runs"), 4)
        await self._poll([], [])
        self.assertEqual(sample("dunetrace_detector_backlog_runs"), 0)
        self.assertEqual(sample("dunetrace_detector_poll_seconds_count"), count_before + 2)

    async def test_saturated_when_completed_query_fills_the_batch(self):
        await self._poll(self._runs(2, "c"), [], batch_size=2)
        self.assertEqual(sample("dunetrace_detector_poll_saturated"), 1)

    async def test_saturated_when_stalled_query_fills_the_batch(self):
        await self._poll([], self._runs(2, "s"), batch_size=2)
        self.assertEqual(sample("dunetrace_detector_poll_saturated"), 1)

    async def test_saturation_clears_once_a_poll_drains(self):
        await self._poll(self._runs(2, "c"), [], batch_size=2)
        self.assertEqual(sample("dunetrace_detector_poll_saturated"), 1)
        await self._poll(self._runs(1, "c"), [], batch_size=2)
        self.assertEqual(sample("dunetrace_detector_poll_saturated"), 0)

    async def test_poll_duration_is_observed_when_the_poll_raises(self):
        count_before = sample("dunetrace_detector_poll_seconds_count")
        with contextlib.ExitStack() as stack:
            for p in self._patches([], []):
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "detector_svc.worker.get_watermark",
                    AsyncMock(side_effect=RuntimeError("db down")),
                )
            )
            with self.assertRaises(RuntimeError):
                await worker.poll_once()
        self.assertEqual(sample("dunetrace_detector_poll_seconds_count"), count_before + 1)


class TestPollCycle(unittest.IsolatedAsyncioTestCase):
    """_poll_cycle is one loop iteration: it refreshes the liveness stamp on
    success and counts (never raises) on failure."""

    def setUp(self):
        self._saved = worker._last_poll_at

    def tearDown(self):
        worker._last_poll_at = self._saved

    async def test_success_refreshes_last_poll_at(self):
        worker._last_poll_at = 0.0
        with patch("detector_svc.worker.poll_once", AsyncMock(return_value=(0, 0))):
            await worker._poll_cycle()
        self.assertGreater(worker._last_poll_at, 0.0)
        self.assertLess(time.monotonic() - worker._last_poll_at, 5.0)

    async def test_failure_counts_poll_exception_and_does_not_refresh(self):
        worker._last_poll_at = 0.0
        before = sample("dunetrace_detector_exceptions_total", {"where": "poll"})
        with patch("detector_svc.worker.poll_once", AsyncMock(side_effect=RuntimeError("db down"))):
            await worker._poll_cycle()  # must not raise
        self.assertEqual(
            sample("dunetrace_detector_exceptions_total", {"where": "poll"}), before + 1
        )
        # A failed poll is not a heartbeat.
        self.assertEqual(worker._last_poll_at, 0.0)


# ── Readiness ─────────────────────────────────────────────────────────────────


class TestReadiness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = (worker._last_poll_at, worker._poll_started_at)
        # Readiness reads TWO stamps now — the last completed poll and the
        # current one's start — so both must be reset, or a leftover start
        # stamp puts the worker in the in-flight branch and grants it the
        # longer grace.
        worker._poll_started_at = None

    def tearDown(self):
        worker._last_poll_at, worker._poll_started_at = self._saved

    def _settings(self, poll_interval=5):
        s = MagicMock()
        s.POLL_INTERVAL = poll_interval
        return patch("detector_svc.worker.settings", s)

    def test_poll_is_stale_before_any_poll(self):
        worker._last_poll_at = None
        with self._settings():
            self.assertTrue(worker.poll_is_stale())

    def test_poll_is_stale_after_three_intervals(self):
        now = time.monotonic()
        with self._settings(poll_interval=5):
            worker._last_poll_at = now - 14.0
            self.assertFalse(worker.poll_is_stale(now))
            worker._last_poll_at = now - 16.0
            self.assertTrue(worker.poll_is_stale(now))

    def test_a_long_running_poll_is_not_stale_while_in_flight(self):
        """A busy cycle must not read as a dead worker.

        This is the regression that mattered: readiness was computed only from
        the last COMPLETED poll against 3 x POLL_INTERVAL, so a detector
        working through a backlog — the one time it genuinely has work — was
        reported unhealthy for most of every cycle.
        """
        now = time.monotonic()
        with self._settings(poll_interval=5):
            worker._last_poll_at = now - 600.0  # last completion long ago
            worker._poll_started_at = now - 60.0  # but a poll is running now
            self.assertFalse(worker.poll_is_stale(now))

    def test_an_in_flight_poll_still_goes_stale_eventually(self):
        """The grace is a bound, not an exemption — a wedged loop must fail."""
        now = time.monotonic()
        with self._settings(poll_interval=5):
            worker._last_poll_at = now - 10_000.0
            worker._poll_started_at = now - (worker.POLL_INFLIGHT_GRACE_SECS + 1)
            self.assertTrue(worker.poll_is_stale(now))

    async def test_ready_when_db_ok_and_poll_fresh(self):
        worker._last_poll_at = time.monotonic()
        with (
            self._settings(),
            patch(
                "detector_svc.worker.db_ready",
                AsyncMock(return_value=(True, {"db": "ok", "schema_version": 9, "required": 9})),
            ),
        ):
            ok, info = await worker.readiness_check()
        self.assertTrue(ok)
        self.assertEqual(info["poll"], "ok")
        self.assertEqual(info["db"], "ok")
        self.assertEqual(info["poll_stale_after_seconds"], 15.0)

    async def test_not_ready_when_last_poll_is_stale(self):
        worker._last_poll_at = time.monotonic() - 1000
        with (
            self._settings(),
            patch(
                "detector_svc.worker.db_ready",
                AsyncMock(return_value=(True, {"db": "ok", "schema_version": 9, "required": 9})),
            ),
        ):
            ok, info = await worker.readiness_check()
        self.assertFalse(ok)
        self.assertEqual(info["poll"], "stale")
        self.assertGreater(info["last_poll_age_seconds"], 900)

    async def test_not_ready_when_db_is_not_ready(self):
        worker._last_poll_at = time.monotonic()
        with (
            self._settings(),
            patch(
                "detector_svc.worker.db_ready",
                AsyncMock(
                    return_value=(False, {"db": "ConnectionRefusedError", "schema_version": None})
                ),
            ),
        ):
            ok, info = await worker.readiness_check()
        self.assertFalse(ok)
        self.assertEqual(info["poll"], "ok")
        self.assertEqual(info["db"], "ConnectionRefusedError")


# ── Metrics server wiring ─────────────────────────────────────────────────────


class TestStartMetrics(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = worker._last_poll_at

    def tearDown(self):
        worker._last_poll_at = self._saved

    def _settings(self, port):
        s = MagicMock()
        s.METRICS_PORT = port
        s.POLL_INTERVAL = 5
        return patch("detector_svc.worker.settings", s)

    async def test_port_zero_disables_the_server(self):
        with self._settings(0):
            self.assertIsNone(worker.start_metrics(asyncio.get_running_loop()))

    async def test_start_failure_is_logged_not_raised(self):
        with (
            self._settings(9101),
            patch(
                "detector_svc.worker.start_metrics_server",
                MagicMock(side_effect=RuntimeError("bind")),
            ),
        ):
            self.assertIsNone(worker.start_metrics(asyncio.get_running_loop()))

    async def test_server_serves_metrics_and_ready(self):
        db_ok = AsyncMock(return_value=(True, {"db": "ok", "schema_version": 9, "required": 9}))
        mock_settings = MagicMock()
        mock_settings.POLL_INTERVAL = 5
        with (
            patch("detector_svc.worker.settings", mock_settings),
            patch("detector_svc.worker.db_ready", db_ok),
            patch("detector_svc.worker.start_metrics_server", loopback_metrics_server),
        ):
            # Both stamps: readiness reads the completed-poll time AND the
            # in-flight start, and a value left by an earlier test in this
            # module would otherwise decide which branch runs.
            worker._last_poll_at = time.monotonic()
            worker._poll_started_at = None
            # Redraw the port on a lost race instead of failing: another process
            # can take the probed port before the server binds it.
            server = None
            for _ in range(8):
                mock_settings.METRICS_PORT = free_port()
                server = worker.start_metrics(asyncio.get_running_loop())
                if server is not None:
                    break
            self.assertIsNotNone(server, "could not bind a free loopback port in 8 attempts")
            try:
                base = f"http://127.0.0.1:{server.port}"
                # /ready runs the coroutine on this loop from the server thread,
                # so the request itself must not block the loop.
                status, body = await asyncio.to_thread(http_get, base + "/ready")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["status"], "ok")

                status, body = await asyncio.to_thread(http_get, base + "/metrics")
                self.assertEqual(status, 200)
                text = body.decode()
                self.assertIn('dunetrace_build_info{service="detector"', text)
                self.assertIn('dunetrace_schema_version{service="detector"}', text)
                for name in (
                    "dunetrace_detector_backlog_runs",
                    "dunetrace_detector_poll_saturated",
                    "dunetrace_detector_poll_seconds",
                    "dunetrace_detector_runs_processed_total",
                    "dunetrace_detector_exceptions_total",
                    "dunetrace_detector_signals_total",
                ):
                    self.assertIn(name, text)

                status, _ = await asyncio.to_thread(http_get, base + "/health")
                self.assertEqual(status, 200)

                # Wedged loop → stale → 503.
                worker._last_poll_at = time.monotonic() - 1000
                status, body = await asyncio.to_thread(http_get, base + "/ready")
                self.assertEqual(status, 503)
                self.assertEqual(json.loads(body)["poll"], "stale")
            finally:
                server.shutdown()
                server.server_close()


class TestSchemaVersionGauge(unittest.IsolatedAsyncioTestCase):
    async def test_records_the_version_the_worker_observed(self):
        conn = MagicMock()
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        with (
            patch.object(worker._db, "_pool", pool),
            patch("detector_svc.worker.current_version", AsyncMock(return_value=42)),
        ):
            await worker._record_schema_version()
        self.assertEqual(sample("dunetrace_schema_version", {"service": "detector"}), 42)

    async def test_falls_back_to_current_when_the_read_fails(self):
        pool = MagicMock()
        pool.acquire.side_effect = RuntimeError("no db")
        with patch.object(worker._db, "_pool", pool):
            await worker._record_schema_version()  # must not raise
        self.assertEqual(
            sample("dunetrace_schema_version", {"service": "detector"}),
            worker.CURRENT_SCHEMA_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
