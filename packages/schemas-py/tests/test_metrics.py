"""
dunetrace_schemas.metrics — the shared /metrics + /ready helper.

Run: PYTHONPATH=packages/schemas-py python -m pytest packages/schemas-py/tests/test_metrics.py -v
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import json
import pathlib
import re
import socket
import threading
import unittest
import weakref
import urllib.request
from unittest.mock import MagicMock, patch

from dunetrace_schemas import metrics


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


class TestFactories(unittest.TestCase):
    def setUp(self):
        self.registry = metrics.CollectorRegistry()

    def test_requires_dunetrace_prefix(self):
        with self.assertRaises(ValueError):
            metrics.counter("requests_total", "doc", registry=self.registry)

    def test_counter_gauge_histogram_register_and_render(self):
        c = metrics.counter(
            "dunetrace_t_requests_total", "doc", ("status",), registry=self.registry
        )
        g = metrics.gauge("dunetrace_t_backlog", "doc", registry=self.registry)
        h = metrics.histogram(
            "dunetrace_t_seconds", "doc", buckets=(0.1, 1.0), registry=self.registry
        )
        c.labels(status="200").inc()
        g.set(7)
        h.observe(0.5)
        body, ctype = metrics.render(registry=self.registry)
        text = body.decode()
        self.assertIn('dunetrace_t_requests_total{status="200"} 1.0', text)
        self.assertIn("dunetrace_t_backlog 7.0", text)
        self.assertIn('dunetrace_t_seconds_bucket{le="1.0"} 1.0', text)
        self.assertIn("text/plain", ctype)

    def test_double_registration_returns_the_same_metric(self):
        a = metrics.counter("dunetrace_t_twice_total", "doc", registry=self.registry)
        b = metrics.counter("dunetrace_t_twice_total", "doc", registry=self.registry)
        self.assertIs(a, b)

    def test_register_standard_and_schema_version(self):
        metrics.register_standard("testsvc", "1.2.3", registry=self.registry)
        metrics.set_schema_version(12)
        text = metrics.render(registry=self.registry)[0].decode()
        self.assertIn('dunetrace_build_info{service="testsvc",version="1.2.3"} 1.0', text)
        self.assertIn('dunetrace_schema_version{service="testsvc"} 12.0', text)


class TestWithoutPrometheusClient(unittest.TestCase):
    def test_factories_return_noops_and_render_explains(self):
        with (
            patch.object(metrics, "AVAILABLE", False),
            patch.object(metrics, "_warned_unavailable", False),
        ):
            with self.assertLogs("dunetrace.metrics", level="WARNING") as logs:
                c = metrics.counter("dunetrace_t_absent_total", "doc", ("a",))
            self.assertIsInstance(c, metrics.NoopMetric)
            self.assertIs(c.labels(a="x"), c)
            c.labels(a="x").inc()  # no raise
            metrics.gauge("dunetrace_t_absent", "doc").set(1)
            metrics.histogram("dunetrace_t_absent_seconds", "doc").observe(0.1)
            body, ctype = metrics.render()
            self.assertIn(b"prometheus_client is not installed", body)
            self.assertIn("text/plain", ctype)
            self.assertEqual(len(logs.output), 1)
            # warned once per process, not once per metric
            metrics.counter("dunetrace_t_absent2_total", "doc")

    def test_set_schema_version_before_register_is_a_noop(self):
        with (
            patch.object(metrics, "_schema_gauge", None),
            patch.object(metrics, "_service_name", None),
        ):
            metrics.set_schema_version(3)  # no raise


class _FakeConn:
    def __init__(self, version=12, fail=None):
        self.version = version
        self.fail = fail

    async def fetchval(self, sql, *args):
        if self.fail is not None:
            raise self.fail
        if "schema_version" in sql:
            return self.version
        return 1


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()

    def get_size(self):
        return 3

    def get_idle_size(self):
        return 2


class TestDbReady(unittest.IsolatedAsyncioTestCase):
    async def test_ok_when_current(self):
        ok, info = await metrics.db_ready(_FakePool(_FakeConn(version=12)), 12)
        self.assertTrue(ok)
        self.assertEqual(info["db"], "ok")
        self.assertEqual(info["schema_version"], 12)
        self.assertEqual(info["required"], 12)
        self.assertEqual(info["pool"], {"size": 3, "idle": 2})

    async def test_not_ready_when_schema_is_old(self):
        ok, info = await metrics.db_ready(_FakePool(_FakeConn(version=5)), 12)
        self.assertFalse(ok)
        self.assertEqual(info["schema_version"], 5)

    async def test_not_ready_on_connection_error(self):
        ok, info = await metrics.db_ready(_FakePool(_FakeConn(fail=ConnectionError("down"))), 1)
        self.assertFalse(ok)
        self.assertEqual(info["db"], "ConnectionError")
        self.assertIsNone(info["schema_version"])

    async def test_not_ready_without_pool(self):
        ok, info = await metrics.db_ready(None, 1)
        self.assertFalse(ok)
        self.assertEqual(info["db"], "no_pool")


class TestMetricsServer(unittest.TestCase):
    def setUp(self):
        self.registry = metrics.CollectorRegistry()
        metrics.counter("dunetrace_t_srv_total", "doc", registry=self.registry).inc(4)
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()

    def _start(self, check, loop=None):
        server = metrics.start_metrics_server(
            _free_port(), check, loop, host="127.0.0.1", registry=self.registry
        )
        self.assertIsNotNone(server)
        self.servers.append(server)
        return f"http://127.0.0.1:{server.port}"

    def test_port_zero_disables(self):
        self.assertIsNone(metrics.start_metrics_server(0, lambda: (True, {})))

    def test_serves_metrics_ready_and_health_sync(self):
        base = self._start(lambda: (True, {"db": "ok"}))
        status, body, ctype = _get(base + "/metrics")
        self.assertEqual(status, 200)
        self.assertIn(b"dunetrace_t_srv_total 4.0", body)
        status, body, ctype = _get(base + "/ready")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok", "db": "ok"})
        self.assertIn("application/json", ctype)
        status, body, _ = _get(base + "/health")
        self.assertEqual(status, 200)
        status, _, _ = _get(base + "/nope")
        self.assertEqual(status, 404)

    def test_ready_503_when_check_fails_or_raises(self):
        base = self._start(lambda: (False, {"db": "ConnectionError"}))
        status, body, _ = _get(base + "/ready")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["status"], "not_ready")

        def boom():
            raise RuntimeError("kaput")

        base2 = self._start(boom)
        status, body, _ = _get(base2 + "/ready")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"], "RuntimeError")

    def test_async_check_runs_on_the_given_loop(self):
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            seen = {}

            async def check():
                seen["loop"] = asyncio.get_running_loop()
                return True, {"schema_version": 12}

            base = self._start(check, loop)
            status, body, _ = _get(base + "/ready")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["schema_version"], 12)
            self.assertIs(seen["loop"], loop)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            loop.close()

    def test_async_check_without_a_loop_is_not_ready(self):
        async def check():
            return True, {}

        base = self._start(check, None)
        status, body, _ = _get(base + "/ready")
        self.assertEqual(status, 503)
        self.assertIn("no event loop", json.loads(body)["error"])

    def test_bind_failure_is_logged_not_raised(self):
        base = self._start(lambda: (True, {}))
        port = int(base.rsplit(":", 1)[1])
        with self.assertLogs("dunetrace.metrics", level="WARNING"):
            self.assertIsNone(
                metrics.start_metrics_server(port, lambda: (True, {}), host="127.0.0.1")
            )


if __name__ == "__main__":
    unittest.main()


class TestMetricCatalogueDocumented(unittest.TestCase):
    """Every metric a service registers is in docs/operations.md's catalogue.

    The counts and names in the docs have drifted before (detector counts,
    tool lists); a metric nobody can find in the catalogue is one nobody
    alerts on. The scan is textual — the first string argument of every
    counter(/gauge(/histogram( call under services/ and in this module — so
    it needs no service on the path.
    """

    _CALL = re.compile(
        r"\b(?:_metrics\.|dt_metrics\.)?(?:counter|gauge|histogram)\(\s*\n?\s*\"(dunetrace_[a-z0-9_]+)\""
    )

    def test_every_registered_metric_is_in_the_catalogue(self):
        root = pathlib.Path(__file__).resolve().parents[3]
        doc = (root / "docs" / "operations.md").read_text()
        sources = list((root / "services").rglob("*.py")) + [
            root / "packages" / "schemas-py" / "dunetrace_schemas" / "metrics.py"
        ]
        registered = set()
        for path in sources:
            if "tests" in path.parts:
                continue
            registered.update(self._CALL.findall(path.read_text()))
        self.assertGreater(len(registered), 30, "the scan found suspiciously few metrics")
        missing = sorted(name for name in registered if f"`{name}`" not in doc)
        self.assertEqual(
            missing, [], f"registered but not in docs/operations.md's catalogue: {missing}"
        )


class TestRegistryCacheLifetime(unittest.TestCase):
    """The dedupe cache must not outlive the registry it is keyed on.

    It used to be a plain dict keyed on ``id(registry)``, holding no reference
    to the registry itself. Once one was collected CPython could reuse its
    address, and the next registry with that address got back a metric bound to
    the dead one — registered against nothing, and missing from its own
    ``render()``. Measured at 3 stale hits in 200 iterations.
    """

    def test_a_fresh_registry_never_gets_a_dead_registrys_metric(self):
        stale = 0
        for _ in range(400):
            reg = metrics.CollectorRegistry()
            metrics.counter("dunetrace_lifetime_probe_total", "doc", registry=reg).inc()
            if b"dunetrace_lifetime_probe_total 1.0" not in metrics.render(registry=reg)[0]:
                stale += 1
            del reg
        self.assertEqual(stale, 0)

    def test_the_cache_does_not_keep_registries_alive(self):
        reg = metrics.CollectorRegistry()
        metrics.counter("dunetrace_weak_probe_total", "doc", registry=reg)
        ref = weakref.ref(reg)
        del reg
        gc.collect()
        self.assertIsNone(ref(), "the metric cache is pinning collected registries")

    def test_same_registry_still_dedupes(self):
        reg = metrics.CollectorRegistry()
        a = metrics.counter("dunetrace_dedupe_probe_total", "doc", registry=reg)
        b = metrics.counter("dunetrace_dedupe_probe_total", "doc", registry=reg)
        self.assertIs(a, b)


class TestReadinessFutureIsCancelled(unittest.TestCase):
    """A probe that gives up must not leave its coroutine queued.

    concurrent.futures.Future.result(timeout) raises but leaves the future
    pending and the coroutine scheduled. On a blocked loop every timed-out
    probe piled one more db_ready() call onto the queue; when the loop resumed
    they all ran at once, each taking one of the worker's five pool
    connections, with nothing bounding how many had accumulated.
    """

    class _StuckFuture:
        def __init__(self):
            self.cancelled_called = False

        def result(self, timeout=None):
            raise concurrent.futures.TimeoutError("still running")

        def cancel(self):
            self.cancelled_called = True
            return True

    def _run_with(self, future):
        async def check():
            return True, {}

        coro = check()
        loop = MagicMock()
        loop.is_closed.return_value = False
        try:
            with patch.object(metrics.asyncio, "run_coroutine_threadsafe", return_value=future):
                return metrics._run_ready_check(lambda: coro, loop)
        finally:
            coro.close()

    def test_timed_out_check_cancels_its_future(self):
        future = self._StuckFuture()
        ok, info = self._run_with(future)
        self.assertFalse(ok)
        self.assertTrue(future.cancelled_called, "the timed-out future was left pending")
        self.assertIn("error", info)

    def test_a_check_that_answers_in_time_is_not_cancelled(self):
        class _Done(self._StuckFuture):
            def result(self, timeout=None):
                return True, {"db": "ok"}

        future = _Done()
        ok, info = self._run_with(future)
        self.assertTrue(ok)
        self.assertFalse(future.cancelled_called)
        self.assertEqual(info["db"], "ok")
