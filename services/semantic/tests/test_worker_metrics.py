"""
Metrics and readiness for the semantic worker (hardening Phase 3.1/3.2).

The semantic worker is opt-in and its /ready is what a compose healthcheck
calls, so an inverted boolean, a forgotten set_schema_version or a mislabelled
counter would ship green without these. Covers, for each metric, the path it
claims to measure; the three ways _ready_check must say NOT ready (DB
unreachable, schema too old, poll loop stale) and the one way it says ready;
and that a metrics-server bind failure is logged rather than raised into
startup.

Metric values are read back through the prometheus registry, which is
process-global, so every assertion is a delta (see _Meter).

Readiness is driven through the worker's own surface — _mark_poll_started /
_mark_poll_ok to record a cycle, settings.POLL_INTERVAL to size the staleness
window — rather than by reaching into the freshness bookkeeping, so a further
change to how the worker judges an in-flight cycle does not have to rewrite
these assertions. _clear_poll_state() is the one place that touches those names.

No DB, no credentials, no network beyond 127.0.0.1.

Run:
    PYTHONPATH=packages/schemas-py:services/explainer:services/semantic \
        python -m pytest services/semantic/tests/test_worker_metrics.py -v
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
from unittest.mock import AsyncMock, MagicMock, patch

import semantic_svc.worker as worker  # import before patch() resolves "semantic_svc.worker.*"
from dunetrace_schemas import metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION
from semantic_svc.config import Settings

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


BACKLOG = "dunetrace_semantic_backlog"
PROCESSED = "dunetrace_semantic_processed_total"
FAILURES = "dunetrace_semantic_failures_total"
EXTERNAL_SECONDS_COUNT = "dunetrace_semantic_external_call_seconds_count"
EXTERNAL_CALLS = "dunetrace_semantic_external_calls_total"
POLL_SECONDS_COUNT = "dunetrace_semantic_poll_seconds_count"
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

# The module global(s) recording cycle timing. Only _clear_poll_state() knows
# these names: every test below drives freshness forward through the worker's
# own _mark_poll_started/_mark_poll_ok and reads it back through _ready_check(),
# so a further rewrite of the freshness rule touches this tuple and nothing
# else. _poll_finished_at is what alerts_svc calls its completion stamp.
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


def _result(fired=True, confidence=0.9, evaluator="hallucination"):
    """An evaluator EvalResult, structurally (the real one is a dataclass in
    semantic_svc.evaluators; only these attributes are read downstream)."""
    return types.SimpleNamespace(
        evaluator=evaluator,
        fired=fired,
        confidence=confidence,
        reasoning="because",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0001,
    )


# ── poll_once: backlog gauge and the sampled/skipped split ────────────────────


@_needs_prometheus
class TestPollOnceMetrics(unittest.IsolatedAsyncioTestCase):
    async def _poll(self, runs, results):
        with (
            patch("semantic_svc.worker.fetch_unevaluated_runs", AsyncMock(return_value=runs)),
            patch("semantic_svc.worker.process_run", AsyncMock(side_effect=results)),
        ):
            return await worker.poll_once()

    @staticmethod
    def _runs(n):
        return [
            {"run_id": f"r{i}", "agent_id": "a", "agent_version": "v", "org_id": "o"}
            for i in range(n)
        ]

    async def test_empty_poll_zeroes_backlog_and_counts_nothing(self):
        m = _Meter(
            (PROCESSED, {"result": "sampled"}),
            (PROCESSED, {"result": "skipped"}),
        )
        with patch("semantic_svc.worker.fetch_unevaluated_runs", AsyncMock(return_value=[])):
            self.assertEqual(await worker.poll_once(), (0, 0, 0))
        self.assertEqual(_sample(BACKLOG), 0.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "sampled"}), 0.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "skipped"}), 0.0)

    async def test_backlog_gauge_is_the_runs_this_poll_found(self):
        await self._poll(self._runs(3), [(False, 0)] * 3)
        self.assertEqual(_sample(BACKLOG), 3.0)

    async def test_backlog_drops_back_to_zero_on_the_next_empty_poll(self):
        await self._poll(self._runs(4), [(False, 0)] * 4)
        self.assertEqual(_sample(BACKLOG), 4.0)
        with patch("semantic_svc.worker.fetch_unevaluated_runs", AsyncMock(return_value=[])):
            await worker.poll_once()
        self.assertEqual(_sample(BACKLOG), 0.0)

    async def test_every_run_lands_in_exactly_one_result_bucket(self):
        m = _Meter(
            (PROCESSED, {"result": "sampled"}),
            (PROCESSED, {"result": "skipped"}),
        )
        seen, sampled, signals = await self._poll(self._runs(3), [(True, 1), (False, 0), (True, 2)])
        self.assertEqual((seen, sampled, signals), (3, 2, 3))
        self.assertEqual(m.delta(PROCESSED, {"result": "sampled"}), 2.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "skipped"}), 1.0)

    async def test_a_run_that_was_not_sampled_is_skipped_not_sampled(self):
        m = _Meter(
            (PROCESSED, {"result": "sampled"}),
            (PROCESSED, {"result": "skipped"}),
        )
        await self._poll(self._runs(2), [(False, 0), (False, 0)])
        self.assertEqual(m.delta(PROCESSED, {"result": "sampled"}), 0.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "skipped"}), 2.0)


# ── _observed_evaluate: the LLM-call histogram and counter ────────────────────


@_needs_prometheus
class TestExternalCallMetrics(unittest.IsolatedAsyncioTestCase):
    def _meter(self, provider: str) -> _Meter:
        return _Meter(
            (EXTERNAL_SECONDS_COUNT, {"provider": provider}),
            (EXTERNAL_CALLS, {"provider": provider, "status": "ok"}),
            (EXTERNAL_CALLS, {"provider": provider, "status": "error"}),
        )

    async def test_successful_call_observes_ok_once_for_its_provider(self):
        m = self._meter("anthropic")
        evaluator = MagicMock()
        evaluator.evaluate.return_value = _result()
        out = await worker._observed_evaluate(evaluator, {"input": "x"}, "anthropic")
        self.assertIs(out, evaluator.evaluate.return_value)
        self.assertEqual(m.delta(EXTERNAL_SECONDS_COUNT, {"provider": "anthropic"}), 1.0)
        self.assertEqual(m.delta(EXTERNAL_CALLS, {"provider": "anthropic", "status": "ok"}), 1.0)
        self.assertEqual(m.delta(EXTERNAL_CALLS, {"provider": "anthropic", "status": "error"}), 0.0)

    async def test_failed_call_observes_error_and_re_raises(self):
        m = self._meter("openai")
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("rate limited")
        with self.assertRaises(RuntimeError):
            await worker._observed_evaluate(evaluator, {"input": "x"}, "openai")
        # A call that failed still took time and still cost a slot.
        self.assertEqual(m.delta(EXTERNAL_SECONDS_COUNT, {"provider": "openai"}), 1.0)
        self.assertEqual(m.delta(EXTERNAL_CALLS, {"provider": "openai", "status": "error"}), 1.0)
        self.assertEqual(m.delta(EXTERNAL_CALLS, {"provider": "openai", "status": "ok"}), 0.0)

    async def test_each_provider_is_its_own_series(self):
        m = self._meter("mistral")
        evaluator = MagicMock()
        evaluator.evaluate.return_value = _result()
        await worker._observed_evaluate(evaluator, {}, "mistral")
        self.assertEqual(m.delta(EXTERNAL_CALLS, {"provider": "mistral", "status": "ok"}), 1.0)
        self.assertEqual(m.delta(EXTERNAL_SECONDS_COUNT, {"provider": "mistral"}), 1.0)

    async def test_missing_provider_is_labelled_unknown_not_none(self):
        """The label set must stay closed and stringly-typed — a None provider
        must not become a `None` time series."""
        m = self._meter("unknown")
        evaluator = MagicMock()
        evaluator.evaluate.return_value = _result()
        await worker._observed_evaluate(evaluator, {}, None)
        self.assertEqual(m.delta(EXTERNAL_CALLS, {"provider": "unknown", "status": "ok"}), 1.0)


# ── failures_total{where}: one label per containment boundary ─────────────────


@_needs_prometheus
class TestFailureCounters(unittest.IsolatedAsyncioTestCase):
    async def test_evaluator_failure_is_counted_and_contained(self):
        m = _Meter((FAILURES, {"where": "evaluator"}))
        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[])),
            patch("semantic_svc.worker.build_evaluation_input", return_value={"input": "x"}),
            patch.dict(
                "semantic_svc.worker._evaluators", {"hallucination": MagicMock()}, clear=True
            ),
            patch(
                "semantic_svc.worker._observed_evaluate",
                AsyncMock(side_effect=RuntimeError("provider down")),
            ),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()) as log_mock,
        ):
            written = await worker._run_evaluators("run-1", "agent-1", "v1", "org-1", None)
        self.assertEqual(written, 0)  # contained: the run is not abandoned
        log_mock.assert_not_awaited()
        self.assertEqual(m.delta(FAILURES, {"where": "evaluator"}), 1.0)

    async def test_second_opinion_failure_has_its_own_label(self):
        """A second opinion that fails must not be counted as an evaluator
        failure, and must not discard the primary HIGH finding."""
        m = _Meter((FAILURES, {"where": "second_opinion"}), (FAILURES, {"where": "evaluator"}))
        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[])),
            patch("semantic_svc.worker.build_evaluation_input", return_value={"input": "x"}),
            patch.dict(
                "semantic_svc.worker._evaluators", {"hallucination": MagicMock()}, clear=True
            ),
            patch.dict(
                "semantic_svc.worker._second_opinion_evaluators",
                {"hallucination": MagicMock()},
                clear=True,
            ),
            patch(
                "semantic_svc.worker._observed_evaluate",
                AsyncMock(side_effect=[_result(confidence=0.8), RuntimeError("boom")]),
            ),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()),
            patch("semantic_svc.worker.root_cause_hash", return_value="h"),
            patch("semantic_svc.worker.fetch_signal_group_fp_count", AsyncMock(return_value=0)),
            patch("semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=1)),
            patch("semantic_svc.worker.record_signal_group_membership", AsyncMock()),
        ):
            written = await worker._run_evaluators("run-1", "agent-1", "v1", "org-1", None)
        self.assertEqual(written, 1)
        self.assertEqual(m.delta(FAILURES, {"where": "second_opinion"}), 1.0)
        self.assertEqual(m.delta(FAILURES, {"where": "evaluator"}), 0.0)

    async def test_conversation_evaluator_failure_has_its_own_label(self):
        m = _Meter((FAILURES, {"where": "conversation_evaluator"}))
        with (
            patch.dict(
                "semantic_svc.worker._conversation_evaluators",
                {"user_frustration": MagicMock()},
                clear=True,
            ),
            patch("semantic_svc.worker.fetch_run_conversation_id", AsyncMock(return_value="c-1")),
            patch(
                "semantic_svc.worker.fetch_conversation_run_ids",
                AsyncMock(return_value=["r1", "r2", "r3"]),
            ),
            patch(
                "semantic_svc.worker.decide_conversation_sampling",
                return_value=(True, "sampled"),
            ),
            patch(
                "semantic_svc.worker.fetch_org_conversation_quota_settings",
                AsyncMock(return_value={"quota": 100, "allow_overage": False}),
            ),
            patch(
                "semantic_svc.worker.consume_org_conversation_quota", AsyncMock(return_value=True)
            ),
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[])),
            patch(
                "semantic_svc.worker.build_conversation_evaluation_input",
                return_value={"turns": []},
            ),
            patch(
                "semantic_svc.worker._observed_evaluate",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            written = await worker._maybe_run_conversation_evaluator("r1", "a1", "v1", "org-1")
        self.assertEqual(written, 0)
        self.assertEqual(m.delta(FAILURES, {"where": "conversation_evaluator"}), 1.0)

    async def test_conversation_step_failure_never_blocks_the_sampling_decision(self):
        m = _Meter((FAILURES, {"where": "conversation"}))
        with (
            patch("semantic_svc.worker.has_structural_signal", AsyncMock(return_value=False)),
            patch("semantic_svc.worker.has_retrieval_event", AsyncMock(return_value=False)),
            patch("semantic_svc.worker.fetch_agent_semantic_config", AsyncMock(return_value=None)),
            patch("semantic_svc.worker.decide_sampling", return_value=(False, "not_sampled", 0.0)),
            patch(
                "semantic_svc.worker._maybe_run_conversation_evaluator",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch("semantic_svc.worker.mark_run_processed", AsyncMock()) as mark_mock,
        ):
            sampled, signals = await worker.process_run("run-1", "agent-1", "v1", "org-1")
        self.assertEqual((sampled, signals), (False, 0))
        mark_mock.assert_awaited_once()
        self.assertEqual(m.delta(FAILURES, {"where": "conversation"}), 1.0)


# ── The loop in run_worker: poll timing, poll failures, schema gauge ──────────


@_needs_prometheus
class TestRunWorkerLoopMetrics(unittest.IsolatedAsyncioTestCase):
    """One or two turns of run_worker's loop, then out via CancelledError —
    which the inner `except Exception` deliberately does not catch."""

    def setUp(self):
        self._saved = {
            name: getattr(worker, name)
            for name in (
                "_evaluators",
                "_second_opinion_evaluators",
                "_conversation_evaluators",
            )
        }
        self.addCleanup(lambda: [setattr(worker, k, v) for k, v in self._saved.items()])
        self.addCleanup(_clear_poll_state)
        for name, value in (
            ("SEMANTIC_WORKER_ENABLED", True),
            ("POLL_INTERVAL", 0),
            ("METRICS_PORT", 0),
        ):
            p = patch.object(worker.settings, name, value)
            p.start()
            self.addCleanup(p.stop)
        for name in (
            "_build_evaluators",
            "_build_second_opinion_evaluators",
            "_build_conversation_evaluators",
        ):
            p = patch.object(worker, name, MagicMock(return_value={}))
            p.start()
            self.addCleanup(p.stop)

    async def _run_loop(self, poll_side_effect):
        close_mock = AsyncMock()
        with (
            patch("semantic_svc.worker.init_pool", AsyncMock()),
            patch("semantic_svc.worker.ensure_semantic_schema", AsyncMock()),
            patch("semantic_svc.worker.close_pool", close_mock),
            patch("semantic_svc.worker.poll_once", AsyncMock(side_effect=poll_side_effect)),
        ):
            await worker.run_worker()
        return close_mock

    async def test_a_completed_cycle_is_timed_and_the_schema_gauge_is_set(self):
        m = _Meter((POLL_SECONDS_COUNT, None))
        close_mock = await self._run_loop([(0, 0, 0), asyncio.CancelledError()])
        self.assertEqual(m.delta(POLL_SECONDS_COUNT), 1.0)
        self.assertEqual(
            _sample(SCHEMA_GAUGE, {"service": "semantic"}), float(CURRENT_SCHEMA_VERSION)
        )
        close_mock.assert_awaited_once()

    async def test_a_failing_cycle_counts_where_poll_and_keeps_the_loop_alive(self):
        m = _Meter((FAILURES, {"where": "poll"}), (POLL_SECONDS_COUNT, None))
        await self._run_loop([RuntimeError("db gone"), (0, 0, 0), asyncio.CancelledError()])
        self.assertEqual(m.delta(FAILURES, {"where": "poll"}), 1.0)
        # Both cycles were timed — a failed cycle still took wall-clock time.
        self.assertEqual(m.delta(POLL_SECONDS_COUNT), 2.0)

    async def test_a_completed_cycle_makes_ready_report_a_fresh_poll(self):
        _clear_poll_state()
        await self._run_loop([(0, 0, 0), asyncio.CancelledError()])
        with (
            patch.object(worker.settings, "POLL_INTERVAL", 10),
            patch.object(worker._db, "_pool", _FakePool(version=CURRENT_SCHEMA_VERSION)),
        ):
            ok, info = await worker._ready_check()
        self.assertTrue(ok)
        self.assertEqual(info["poll"], "ok")


@_needs_prometheus
class TestDisabledWorkerRegistersNothing(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_worker_never_starts_a_metrics_server(self):
        with (
            patch.object(worker.settings, "SEMANTIC_WORKER_ENABLED", False),
            patch("semantic_svc.worker.init_pool", AsyncMock()) as init_mock,
            patch.object(worker._metrics, "start_metrics_server") as start_mock,
        ):
            await worker.run_worker()
        init_mock.assert_not_awaited()
        start_mock.assert_not_called()


# ── _ready_check: the four states a compose healthcheck sees ──────────────────


class TestReadyCheck(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _clear_poll_state()
        self.addCleanup(_clear_poll_state)
        p = patch.object(worker.settings, "POLL_INTERVAL", 10)
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
        with patch.object(worker.settings, "POLL_INTERVAL", 0.001):
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
        """A busy cycle (a batch of runs × several LLM evaluators) outlives
        3 × POLL_INTERVAL routinely; reporting unhealthy then invites an
        orchestrator to restart the busiest worker mid-batch."""
        with patch.object(worker.settings, "POLL_INTERVAL", 0.001):
            _complete_cycle()
            _start_cycle()  # ... and this one has not finished
            await asyncio.sleep(0.05)  # well past 3 × POLL_INTERVAL
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
            patch.object(worker.settings, "POLL_INTERVAL", 0.001),
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
        """A port already in use (a second replica in the same network
        namespace, a leftover process) must cost the scrape endpoint, not the
        worker — start_metrics_server returns None and logs."""
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
        poll_mock = AsyncMock(side_effect=[(0, 0, 0), asyncio.CancelledError()])
        close_mock = AsyncMock()
        with (
            patch.object(worker.settings, "SEMANTIC_WORKER_ENABLED", True),
            patch.object(worker.settings, "POLL_INTERVAL", 0),
            patch.object(worker, "_build_evaluators", MagicMock(return_value={})),
            patch.object(worker, "_build_second_opinion_evaluators", MagicMock(return_value={})),
            patch.object(worker, "_build_conversation_evaluators", MagicMock(return_value={})),
            patch.object(worker._metrics, "start_metrics_server", MagicMock(return_value=None)),
            patch("semantic_svc.worker.init_pool", AsyncMock()),
            patch("semantic_svc.worker.ensure_semantic_schema", AsyncMock()),
            patch("semantic_svc.worker.close_pool", close_mock),
            patch("semantic_svc.worker.poll_once", poll_mock),
        ):
            await worker.run_worker()  # must not raise
        self.assertEqual(poll_mock.await_count, 2)
        close_mock.assert_awaited_once()

    @_needs_prometheus
    async def test_server_serves_metrics_ready_and_health(self):
        port = _free_port()
        metrics.register_standard("semantic", worker.settings.APP_VERSION)
        metrics.set_schema_version(CURRENT_SCHEMA_VERSION)
        with patch.object(worker.settings, "POLL_INTERVAL", 10):
            _clear_poll_state()
            server = metrics.start_metrics_server(
                port, worker._ready_check, asyncio.get_running_loop(), host="127.0.0.1"
            )
            self.assertIsNotNone(server)
            try:
                base = f"http://127.0.0.1:{server.port}"
                status, _ = await asyncio.to_thread(_get, base + "/health")
                self.assertEqual(status, 200)

                # No cycle has completed yet → 503 with the reason.
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
                self.assertIn('dunetrace_build_info{service="semantic"', text)
                self.assertIn(
                    f'dunetrace_schema_version{{service="semantic"}} '
                    f"{float(CURRENT_SCHEMA_VERSION)}",
                    text,
                )
                for name in (
                    BACKLOG,
                    PROCESSED,
                    FAILURES,
                    "dunetrace_semantic_external_call_seconds",
                    EXTERNAL_CALLS,
                    "dunetrace_semantic_poll_seconds",
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
        self.assertEqual(Settings.METRICS_PORT, int(os.getenv("METRICS_PORT", "9103")))

    def test_app_version_present(self):
        self.assertTrue(Settings.APP_VERSION)


if __name__ == "__main__":
    unittest.main()
