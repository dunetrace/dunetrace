"""
Metrics + readiness for the alerts worker (hardening Phase 3.1/3.2).

Covers the poll_once outcome counters and backlog gauge (with the same DB/
sender mocks the other worker tests use), the per-call delivery histogram and
result counter in sender.py / linear_client.py (HTTP client mocked, success
and failure), the poll-freshness readiness logic, the /metrics + /ready server
on a free localhost port, and METRICS_PORT=0 disabling it.

No DB, no network beyond 127.0.0.1.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/alerts \
        python -m pytest services/alerts/tests/test_metrics.py -v
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import socket
import sys
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    _ROOT,
    os.path.join(_ROOT, "packages/schemas-py"),
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dunetrace_schemas import metrics  # noqa: E402
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION  # noqa: E402
from scripts.require_test_deps import require_prometheus_client  # noqa: E402

import alerts_svc.linear_client as linear_module  # noqa: E402
import alerts_svc.sender as sender_module  # noqa: E402
import alerts_svc.worker as worker_module  # noqa: E402
from alerts_svc.config import Settings  # noqa: E402
from alerts_svc.sender import SendResult, send_with_retry  # noqa: E402

# Without prometheus_client every metric in dunetrace_schemas.metrics is a no-op
# and each assertion below reads 0.0 from nothing. This used to be a
# `skipUnless(metrics.AVAILABLE, ...)` on each class, i.e. a silent pass with
# exit code 0 the moment the dependency was dropped from requirements.txt. It is
# now a hard error unless DUNETRACE_ALLOW_MISSING_TEST_DEPS is set.
require_prometheus_client("services/alerts/requirements.txt")
assert metrics.AVAILABLE, "prometheus_client imports but metrics degraded to no-ops"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sample(name: str, labels: dict | None = None) -> float:
    value = metrics.REGISTRY.get_sample_value(name, labels or {})
    return float(value or 0.0)


class _Meter:
    """Snapshot metric samples before an action, read the deltas after."""

    def __init__(self, *specs: tuple):
        self._specs = [(n, dict(l) if l else {}) for n, l in specs]
        self._before = {self._key(n, l): _sample(n, l) for n, l in self._specs}

    @staticmethod
    def _key(name: str, labels: dict) -> tuple:
        return (name, tuple(sorted(labels.items())))

    def delta(self, name: str, labels: dict | None = None) -> float:
        labels = labels or {}
        return _sample(name, labels) - self._before[self._key(name, labels)]


PROCESSED = "dunetrace_alerts_processed_total"
EXCEPTIONS = "dunetrace_alerts_exceptions_total"
DELIVERY_TOTAL = "dunetrace_alerts_delivery_total"
DELIVERY_COUNT = "dunetrace_alerts_delivery_seconds_count"
POLL_COUNT = "dunetrace_alerts_poll_seconds_count"
BACKLOG = "dunetrace_alerts_backlog_signals"

_RESULTS = ("delivered", "suppressed", "failed", "skipped")


def _poll_meter() -> _Meter:
    return _Meter(
        *[(PROCESSED, {"result": r}) for r in _RESULTS],
        (POLL_COUNT, None),
        *[
            (EXCEPTIONS, {"where": w})
            for w in ("reconstruct", "explain", "deliver", "poll", "approvals", "digest")
        ],
    )


def _row(
    signal_id: int,
    agent_id: str = "agent-1",
    failure_type: str = "TOOL_LOOP",
    confidence: float = 0.9,
    severity: str = "HIGH",
) -> dict:
    return {
        "id": signal_id,
        "failure_type": failure_type,
        "severity": severity,
        "run_id": f"run-{signal_id}",
        "agent_id": agent_id,
        "org_id": "org-1",
        "agent_version": "v1",
        "step_index": 3,
        "confidence": confidence,
        "evidence": {"tool": "web_search", "count": 5, "window": 5},
        "detected_at": time.time(),
    }


def _ok_slack() -> dict:
    return {"slack": SendResult(True, "slack", 1, 200)}


def _failed_slack() -> dict:
    return {"slack": SendResult(False, "slack", 3, 503, "err")}


def _reset_freshness() -> None:
    worker_module._poll_started_at = None
    worker_module._poll_finished_at = None


# ── poll_once outcome metrics ──────────────────────────────────────────────────


class TestPollOnceMetrics(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _reset_freshness()
        self._dedup = patch.object(worker_module.settings, "ALERT_DEDUP_WINDOW", 0)
        self._dedup.start()
        self.addCleanup(self._dedup.stop)

    async def test_empty_poll_sets_backlog_zero_and_times_cycle(self):
        m = _poll_meter()
        with patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[])):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (0, 0))
        self.assertEqual(_sample(BACKLOG), 0.0)
        self.assertEqual(m.delta(POLL_COUNT), 1.0)
        for r in _RESULTS:
            self.assertEqual(m.delta(PROCESSED, {"result": r}), 0.0, r)
        self.assertIsNotNone(worker_module._poll_finished_at)

    async def test_delivered_signal_counts_delivered_and_backlog(self):
        m = _poll_meter()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[_row(1)])),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver", return_value=_ok_slack()),
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (1, 1))
        self.assertEqual(_sample(BACKLOG), 1.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "delivered"}), 1.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "failed"}), 0.0)

    async def test_all_destinations_failed_counts_failed(self):
        m = _poll_meter()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[_row(2)])),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver", return_value=_failed_slack()),
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (1, 0))
        self.assertEqual(m.delta(PROCESSED, {"result": "failed"}), 1.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "delivered"}), 0.0)

    async def test_no_destinations_counts_skipped_not_delivered(self):
        """deliver() returning {} marks the row alerted (nothing to send it
        to) — that is a skip in the metric, even though poll_once's return
        value counts it as handled."""
        m = _poll_meter()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[_row(3)])),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver", return_value={}),
        ):
            await worker_module.poll_once()
        self.assertEqual(m.delta(PROCESSED, {"result": "skipped"}), 1.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "delivered"}), 0.0)

    async def test_lower_confidence_duplicate_in_group_counts_skipped(self):
        m = _poll_meter()
        rows = [_row(10, confidence=0.9), _row(11, confidence=0.5)]
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=rows)),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver", return_value=_ok_slack()),
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (2, 1))
        self.assertEqual(m.delta(PROCESSED, {"result": "delivered"}), 1.0)
        self.assertEqual(m.delta(PROCESSED, {"result": "skipped"}), 1.0)

    async def test_policy_pending_counts_suppressed(self):
        m = _poll_meter()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[_row(20)])),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch(
                "alerts_svc.worker.evaluate_alert_policy",
                AsyncMock(return_value=(False, "1/3 runs")),
            ),
            patch("alerts_svc.worker.deliver") as mock_deliver,
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (1, 0))
        mock_deliver.assert_not_called()
        self.assertEqual(m.delta(PROCESSED, {"result": "suppressed"}), 1.0)

    async def test_dedup_window_counts_suppressed(self):
        self._dedup.stop()
        m = _poll_meter()
        state = {
            ("org-1", "agent-1", "TOOL_LOOP"): {
                "last_alerted_at": datetime.now(timezone.utc),
                "suppressed_count": 0,
            }
        }
        with (
            patch.object(worker_module.settings, "ALERT_DEDUP_WINDOW", 3600),
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[_row(30)])),
            patch("alerts_svc.worker.fetch_dedup_states", AsyncMock(return_value=state)),
            patch("alerts_svc.worker.increment_suppressed_count", AsyncMock()),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver") as mock_deliver,
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (1, 0))
        mock_deliver.assert_not_called()
        self.assertEqual(m.delta(PROCESSED, {"result": "suppressed"}), 1.0)

    async def test_delivery_exception_counts_failed_and_exception(self):
        m = _poll_meter()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[_row(40)])),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver", side_effect=RuntimeError("boom")),
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (1, 0))
        self.assertEqual(m.delta(PROCESSED, {"result": "failed"}), 1.0)
        self.assertEqual(m.delta(EXCEPTIONS, {"where": "deliver"}), 1.0)

    async def test_unreconstructable_row_counts_failed_and_exception(self):
        m = _poll_meter()
        bad = _row(50, severity="NOT_A_SEVERITY")
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[bad])),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver") as mock_deliver,
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (1, 0))
        mock_deliver.assert_not_called()
        self.assertEqual(m.delta(PROCESSED, {"result": "failed"}), 1.0)
        self.assertEqual(m.delta(EXCEPTIONS, {"where": "reconstruct"}), 1.0)

    async def test_outcomes_sum_to_backlog(self):
        """Every claimed row lands in exactly one result bucket."""
        m = _poll_meter()
        rows = [
            _row(60, agent_id="a", confidence=0.9),  # delivered
            _row(61, agent_id="a", confidence=0.4),  # duplicate → skipped
            _row(62, agent_id="b"),  # all destinations fail → failed
            _row(63, agent_id="c", failure_type="RETRY_STORM"),  # policy pending
        ]

        async def _policy(org_id, agent_id, failure_type, **kw):
            return (failure_type != "RETRY_STORM", "reason")

        def _deliver(explanation, *a, **kw):
            return _ok_slack() if explanation.agent_id == "a" else _failed_slack()

        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=rows)),
            patch("alerts_svc.worker.evaluate_alert_policy", AsyncMock(side_effect=_policy)),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.deliver", side_effect=_deliver),
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (4, 1))
        deltas = {r: m.delta(PROCESSED, {"result": r}) for r in _RESULTS}
        self.assertEqual(
            deltas, {"delivered": 1.0, "skipped": 1.0, "failed": 1.0, "suppressed": 1.0}
        )
        self.assertEqual(sum(deltas.values()), _sample(BACKLOG))

    async def test_raising_poll_still_stamps_freshness_and_timing(self):
        m = _poll_meter()
        with patch(
            "alerts_svc.worker.claim_unalerted_signals",
            AsyncMock(side_effect=ConnectionError("db down")),
        ):
            with self.assertRaises(ConnectionError):
                await worker_module.poll_once()
        self.assertEqual(m.delta(POLL_COUNT), 1.0)
        self.assertIsNotNone(worker_module._poll_finished_at)
        self.assertGreaterEqual(worker_module._poll_finished_at, worker_module._poll_started_at)


# ── Delivery histogram / counter around each outbound call ─────────────────────


class TestDeliveryMetrics(unittest.TestCase):
    def _meter(self, dest: str, *classes: str) -> _Meter:
        return _Meter(
            (DELIVERY_COUNT, {"destination": dest}),
            *[(DELIVERY_TOTAL, {"destination": dest, "status": c}) for c in classes],
        )

    def test_success_observes_2xx_once(self):
        m = self._meter("slack", "2xx", "error")
        with patch("alerts_svc.sender._post", return_value=(200, "ok")):
            result = send_with_retry("https://x", b"{}", {}, "slack", max_retries=2)
        self.assertTrue(result.success)
        self.assertEqual(m.delta(DELIVERY_COUNT, {"destination": "slack"}), 1.0)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "slack", "status": "2xx"}), 1.0)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "slack", "status": "error"}), 0.0)

    def test_each_retry_attempt_is_observed(self):
        m = self._meter("slack", "5xx")
        with (
            patch("alerts_svc.sender._post", return_value=(500, "err")),
            patch("alerts_svc.sender.time.sleep"),
        ):
            result = send_with_retry("https://x", b"{}", {}, "slack", max_retries=1)
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(m.delta(DELIVERY_COUNT, {"destination": "slack"}), 2.0)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "slack", "status": "5xx"}), 2.0)

    def test_http_error_records_status_class(self):
        m = self._meter("webhook", "4xx")
        err = urllib.error.HTTPError("https://x", 404, "nope", {}, None)
        with (
            patch("alerts_svc.sender._post", side_effect=err),
            patch("alerts_svc.sender.time.sleep"),
        ):
            send_with_retry("https://x", b"{}", {}, "webhook", max_retries=0)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "webhook", "status": "4xx"}), 1.0)
        self.assertEqual(m.delta(DELIVERY_COUNT, {"destination": "webhook"}), 1.0)

    def test_connection_failure_records_error(self):
        m = self._meter("webhook", "error")
        with (
            patch("alerts_svc.sender._post", side_effect=urllib.error.URLError("refused")),
            patch("alerts_svc.sender.time.sleep"),
        ):
            send_with_retry("https://x", b"{}", {}, "webhook", max_retries=0)
        self.assertEqual(
            m.delta(DELIVERY_TOTAL, {"destination": "webhook", "status": "error"}), 1.0
        )
        self.assertEqual(m.delta(DELIVERY_COUNT, {"destination": "webhook"}), 1.0)

    def test_unexpected_exception_records_error(self):
        m = self._meter("slack", "error")
        with (
            patch("alerts_svc.sender._post", side_effect=ValueError("bad json")),
            patch("alerts_svc.sender.time.sleep"),
        ):
            send_with_retry("https://x", b"{}", {}, "slack", max_retries=0)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "slack", "status": "error"}), 1.0)

    def test_unknown_destination_is_bucketed_as_other(self):
        """The label set is closed — a URL passed as destination must not
        become a new time series."""
        m = _Meter((DELIVERY_TOTAL, {"destination": "other", "status": "2xx"}))
        with patch("alerts_svc.sender._post", return_value=(200, "ok")):
            send_with_retry("https://x", b"{}", {}, "https://example.com/hook", max_retries=0)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "other", "status": "2xx"}), 1.0)

    def test_status_class(self):
        sc = sender_module._status_class
        self.assertEqual(sc(200), "2xx")
        self.assertEqual(sc(299), "2xx")
        self.assertEqual(sc(404), "4xx")
        self.assertEqual(sc(503), "5xx")
        self.assertEqual(sc(None), "error")
        self.assertEqual(sc(MagicMock()), "error")
        self.assertEqual(sc(True), "error")

    def _linear_resp(self, status: int, body: dict | None = None, raise_status: bool = False):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = body or {
            "data": {"issueCreate": {"success": True, "issue": {"id": "iss-1"}}}
        }
        if raise_status:
            resp.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
        return resp

    def test_linear_success_observes_2xx(self):
        m = self._meter("linear", "2xx")
        with patch("httpx.post", return_value=self._linear_resp(200)):
            issue_id = linear_module.create_issue("key", "team", "t", "d")
        self.assertEqual(issue_id, "iss-1")
        self.assertEqual(m.delta(DELIVERY_COUNT, {"destination": "linear"}), 1.0)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "linear", "status": "2xx"}), 1.0)

    def test_linear_http_failure_observes_status_class(self):
        m = self._meter("linear", "4xx")
        with patch("httpx.post", return_value=self._linear_resp(401, raise_status=True)):
            issue_id = linear_module.create_issue("key", "team", "t", "d")
        self.assertIsNone(issue_id)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "linear", "status": "4xx"}), 1.0)
        self.assertEqual(m.delta(DELIVERY_COUNT, {"destination": "linear"}), 1.0)

    def test_linear_transport_failure_observes_error(self):
        m = self._meter("linear", "error")
        with patch("httpx.post", side_effect=ConnectionError("boom")):
            issue_id = linear_module.create_issue("key", "team", "t", "d")
        self.assertIsNone(issue_id)
        self.assertEqual(m.delta(DELIVERY_TOTAL, {"destination": "linear", "status": "error"}), 1.0)
        self.assertEqual(m.delta(DELIVERY_COUNT, {"destination": "linear"}), 1.0)


# ── Readiness: poll freshness + db_ready ───────────────────────────────────────


class TestPollFreshness(unittest.TestCase):
    def setUp(self):
        _reset_freshness()
        self.addCleanup(_reset_freshness)
        for name, value in (("POLL_INTERVAL", 10.0), ("CLAIM_TIMEOUT_SECS", 300.0)):
            p = patch.object(worker_module.settings, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_not_ready_before_first_poll(self):
        ok, info = worker_module.poll_freshness(now=1000.0)
        self.assertFalse(ok)
        self.assertIn("no poll", info["reason"])
        self.assertIsNone(info["last_poll_age_seconds"])

    def test_fresh_within_three_intervals(self):
        worker_module._poll_started_at = 970.0
        worker_module._poll_finished_at = 971.0
        ok, info = worker_module.poll_freshness(now=1000.0)
        self.assertTrue(ok)
        self.assertEqual(info["max_age_seconds"], 30.0)
        self.assertAlmostEqual(info["last_poll_age_seconds"], 29.0)
        self.assertFalse(info["in_flight"])

    def test_stale_after_three_intervals(self):
        worker_module._poll_started_at = 960.0
        worker_module._poll_finished_at = 961.0
        ok, info = worker_module.poll_freshness(now=1000.0)
        self.assertFalse(ok)
        self.assertEqual(info["reason"], "poll loop stale")
        self.assertAlmostEqual(info["last_poll_age_seconds"], 39.0)

    def test_in_flight_poll_is_allowed_the_claim_timeout(self):
        """A batch mid-delivery (three destinations × full backoff) can run
        past 3×POLL_INTERVAL; it is not a dead worker until CLAIM_TIMEOUT_SECS."""
        worker_module._poll_finished_at = 800.0
        worker_module._poll_started_at = 810.0  # started after the last finish
        ok, info = worker_module.poll_freshness(now=1000.0)
        self.assertTrue(ok)
        self.assertTrue(info["in_flight"])
        ok, info = worker_module.poll_freshness(now=810.0 + 300.0 + 1.0)
        self.assertFalse(ok)
        self.assertTrue(info["in_flight"])

    def test_first_poll_in_flight_counts_as_in_flight(self):
        worker_module._poll_started_at = 1000.0
        ok, info = worker_module.poll_freshness(now=1005.0)
        self.assertTrue(ok)
        self.assertTrue(info["in_flight"])


class TestReadiness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _reset_freshness()
        self.addCleanup(_reset_freshness)
        p = patch.object(worker_module.settings, "POLL_INTERVAL", 10.0)
        p.start()
        self.addCleanup(p.stop)

    async def test_ready_when_db_ok_and_poll_fresh(self):
        now = time.monotonic()
        worker_module._poll_started_at = now - 2
        worker_module._poll_finished_at = now - 1
        db_info = {"db": "ok", "schema_version": CURRENT_SCHEMA_VERSION, "required": 1}
        with patch(
            "dunetrace_schemas.metrics.db_ready", AsyncMock(return_value=(True, db_info))
        ) as mock_db:
            ok, info = await worker_module.readiness()
        self.assertTrue(ok)
        self.assertEqual(info["db"], "ok")
        self.assertTrue(info["poll"]["fresh"])
        mock_db.assert_awaited_once_with(worker_module._db._pool, CURRENT_SCHEMA_VERSION)

    async def test_not_ready_when_poll_stale_even_if_db_ok(self):
        now = time.monotonic()
        worker_module._poll_started_at = now - 100
        worker_module._poll_finished_at = now - 99
        db_info = {"db": "ok", "schema_version": CURRENT_SCHEMA_VERSION, "required": 1}
        with patch("dunetrace_schemas.metrics.db_ready", AsyncMock(return_value=(True, db_info))):
            ok, info = await worker_module.readiness()
        self.assertFalse(ok)
        self.assertEqual(info["db"], "ok")
        self.assertFalse(info["poll"]["fresh"])

    async def test_not_ready_when_db_down_even_if_poll_fresh(self):
        now = time.monotonic()
        worker_module._poll_started_at = now - 2
        worker_module._poll_finished_at = now - 1
        db_info = {"db": "ConnectionRefusedError", "schema_version": None, "required": 1}
        with patch("dunetrace_schemas.metrics.db_ready", AsyncMock(return_value=(False, db_info))):
            ok, info = await worker_module.readiness()
        self.assertFalse(ok)
        self.assertEqual(info["db"], "ConnectionRefusedError")
        self.assertTrue(info["poll"]["fresh"])


# ── start_observability: server on/off, never fatal ────────────────────────────


def _free_port() -> int:
    """A port that was free a moment ago — only safe inside a retry loop.

    Probing and binding are two steps, so the port can be taken in between;
    start_metrics_server swallows the resulting bind failure and returns None.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# start_metrics_server defaults to host="0.0.0.0" and start_observability does
# not override it, so without this the port is probed on loopback and bound on
# every interface. The suite's docstring promises "no network beyond 127.0.0.1".
_loopback_metrics_server = functools.partial(metrics.start_metrics_server, host="127.0.0.1")


def _close(server) -> None:
    """shutdown() stops the serve_forever thread but leaves the listening
    socket open — every run used to leak one fd per server test."""
    server.shutdown()
    server.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class TestStartObservability(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _reset_freshness()
        self.addCleanup(_reset_freshness)

    async def _serve_on_free_port(self, attempts: int = 8):
        """Start the observability server on a loopback port, retrying a lost
        race, and register the shutdown+close that stops the fd leak."""
        for _ in range(attempts):
            with (
                patch.object(worker_module.settings, "METRICS_PORT", _free_port()),
                patch("dunetrace_schemas.metrics.start_metrics_server", _loopback_metrics_server),
            ):
                server = await worker_module.start_observability(asyncio.get_running_loop())
            if server is not None:
                self.addCleanup(_close, server)
                return server
        self.fail(f"could not bind a free loopback port in {attempts} attempts")

    async def test_metrics_port_zero_disables_server(self):
        db_info = {"db": "ok", "schema_version": CURRENT_SCHEMA_VERSION, "required": 1}
        with (
            patch.object(worker_module.settings, "METRICS_PORT", 0),
            patch("dunetrace_schemas.metrics.db_ready", AsyncMock(return_value=(True, db_info))),
            patch(
                "dunetrace_schemas.metrics.start_metrics_server", wraps=metrics.start_metrics_server
            ) as mock_start,
        ):
            server = await worker_module.start_observability(asyncio.get_running_loop())
        self.assertIsNone(server)
        mock_start.assert_called_once()
        self.assertEqual(mock_start.call_args.args[0], 0)

    async def test_server_failure_is_logged_not_raised(self):
        m = _Meter((EXCEPTIONS, {"where": "metrics_server"}))
        with (
            patch.object(worker_module.settings, "METRICS_PORT", 1),
            patch("dunetrace_schemas.metrics.db_ready", AsyncMock(side_effect=RuntimeError("x"))),
        ):
            with self.assertLogs("dunetrace.alerts", level="WARNING"):
                server = await worker_module.start_observability(asyncio.get_running_loop())
        self.assertIsNone(server)
        self.assertEqual(m.delta(EXCEPTIONS, {"where": "metrics_server"}), 1.0)

    async def test_db_ready_failure_at_startup_still_starts_server(self):
        """A DB that is down when the server starts must not stop /ready
        from existing — that is precisely when a probe needs an answer."""
        db_info = {"db": "ConnectionRefusedError", "schema_version": None, "required": 1}
        with patch("dunetrace_schemas.metrics.db_ready", AsyncMock(return_value=(False, db_info))):
            server = await self._serve_on_free_port()
            status, body = await asyncio.to_thread(_get, f"http://127.0.0.1:{server.port}/ready")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["db"], "ConnectionRefusedError")

    async def test_server_serves_metrics_ready_and_health(self):
        db_info = {"db": "ok", "schema_version": CURRENT_SCHEMA_VERSION, "required": 1}
        with (
            patch.object(worker_module.settings, "POLL_INTERVAL", 10.0),
            patch("dunetrace_schemas.metrics.db_ready", AsyncMock(return_value=(True, db_info))),
        ):
            server = await self._serve_on_free_port()
            base = f"http://127.0.0.1:{server.port}"
            status, body = await asyncio.to_thread(_get, base + "/health")
            self.assertEqual(status, 200)

            # Before the first poll completes: 503 with the reason.
            status, body = await asyncio.to_thread(_get, base + "/ready")
            self.assertEqual(status, 503)
            payload = json.loads(body)
            self.assertEqual(payload["status"], "not_ready")
            self.assertEqual(payload["db"], "ok")
            self.assertFalse(payload["poll"]["fresh"])

            # After a poll: 200.
            now = time.monotonic()
            worker_module._poll_started_at = now - 1
            worker_module._poll_finished_at = now
            status, body = await asyncio.to_thread(_get, base + "/ready")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["status"], "ok")

            status, body = await asyncio.to_thread(_get, base + "/metrics")
            self.assertEqual(status, 200)
            text = body.decode()
            self.assertIn('dunetrace_build_info{service="alerts"', text)
            self.assertIn(
                f'dunetrace_schema_version{{service="alerts"}} {float(CURRENT_SCHEMA_VERSION)}',
                text,
            )
            for name in (
                "dunetrace_alerts_backlog_signals",
                "dunetrace_alerts_processed_total",
                "dunetrace_alerts_poll_seconds",
                "dunetrace_alerts_exceptions_total",
                "dunetrace_alerts_approvals_delivered_total",
                "dunetrace_alerts_digests_sent_total",
                "dunetrace_alerts_delivery_seconds",
                "dunetrace_alerts_delivery_total",
            ):
                self.assertIn(f"# TYPE {name}", text, name)


# ── Main-loop cycle: exceptions_total{where}, digests, approvals ───────────────


class TestCycleMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_each_stage_failure_is_counted_and_isolated(self):
        m = _poll_meter()
        with (
            patch("alerts_svc.worker.poll_once", AsyncMock(side_effect=RuntimeError("p"))),
            patch(
                "alerts_svc.worker.deliver_pending_approvals",
                AsyncMock(side_effect=RuntimeError("a")),
            ) as approvals,
            patch(
                "alerts_svc.worker.send_weekly_digest", AsyncMock(side_effect=RuntimeError("d"))
            ) as digest,
        ):
            await worker_module._cycle()
        approvals.assert_awaited_once()
        digest.assert_awaited_once()
        for where in ("poll", "approvals", "digest"):
            self.assertEqual(m.delta(EXCEPTIONS, {"where": where}), 1.0, where)

    async def test_digests_sent_counter(self):
        m = _Meter(("dunetrace_alerts_digests_sent_total", None))
        with (
            patch("alerts_svc.worker.poll_once", AsyncMock(return_value=(0, 0))),
            patch("alerts_svc.worker.deliver_pending_approvals", AsyncMock(return_value=0)),
            patch("alerts_svc.worker.send_weekly_digest", AsyncMock(return_value=3)),
        ):
            await worker_module._cycle()
        self.assertEqual(m.delta("dunetrace_alerts_digests_sent_total"), 3.0)


class TestApprovalMetrics(unittest.IsolatedAsyncioTestCase):
    _APPROVAL = {
        "id": 7,
        "org_id": "org-1",
        "run_id": "run-1",
        "agent_id": "payments-agent",
        "tool_name": "wire_money",
        "tool_args": "{}",
    }

    async def test_delivered_approval_is_counted(self):
        m = _Meter(("dunetrace_alerts_approvals_delivered_total", None))
        with (
            patch(
                "alerts_svc.worker.fetch_undelivered_approvals",
                AsyncMock(return_value=[self._APPROVAL]),
            ),
            patch(
                "alerts_svc.worker._resolve_slack_destination",
                AsyncMock(return_value="https://hook"),
            ),
            patch("alerts_svc.worker.send_slack", return_value=SendResult(True, "slack", 1, 200)),
            patch("alerts_svc.worker.mark_approval_delivered", AsyncMock()),
            patch.object(worker_module.settings, "WEBHOOK_URL", ""),
        ):
            handled = await worker_module.deliver_pending_approvals()
        self.assertEqual(handled, 1)
        self.assertEqual(m.delta("dunetrace_alerts_approvals_delivered_total"), 1.0)

    async def test_no_channel_approval_is_not_counted_as_delivered(self):
        m = _Meter(("dunetrace_alerts_approvals_delivered_total", None))
        with (
            patch(
                "alerts_svc.worker.fetch_undelivered_approvals",
                AsyncMock(return_value=[self._APPROVAL]),
            ),
            patch("alerts_svc.worker._resolve_slack_destination", AsyncMock(return_value=None)),
            patch("alerts_svc.worker.mark_approval_delivered", AsyncMock()) as mark,
            patch.object(worker_module.settings, "SLACK_WEBHOOK_URL", ""),
            patch.object(worker_module.settings, "WEBHOOK_URL", ""),
        ):
            handled = await worker_module.deliver_pending_approvals()
        self.assertEqual(handled, 1)
        mark.assert_awaited_once_with(7)
        self.assertEqual(m.delta("dunetrace_alerts_approvals_delivered_total"), 0.0)


# ── Config ─────────────────────────────────────────────────────────────────────


class TestConfig(unittest.TestCase):
    def test_metrics_port_default(self):
        self.assertEqual(Settings.METRICS_PORT, int(os.getenv("METRICS_PORT", "9102")))

    def test_app_version_present(self):
        self.assertTrue(Settings.APP_VERSION)


if __name__ == "__main__":
    unittest.main()
