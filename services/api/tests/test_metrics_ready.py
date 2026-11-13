"""
/metrics, /ready and the request/LLM instrumentation on the Customer API.

Every metric here goes through dunetrace_schemas.metrics, whose factories
dedupe by name against prometheus_client's process-wide default registry —
so counts accumulate across tests in one process. Assertions therefore read a
sample before and after the action and check the delta, never an absolute.

The app is built with create_app() and the DB layer patched, as the rest of
this suite does; no lifespan (no init_pool), no network, no credentials.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api \
      python -m pytest services/api/tests/test_metrics_ready.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api_svc import llm_provider
from api_svc.config import settings
from api_svc.main import create_app
from dunetrace_schemas import metrics as dt_metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.require_test_deps import require_prometheus_client  # noqa: E402

# The API image pins prometheus-client (services/api/requirements.txt), so a
# missing package here is an environment problem, not a legitimate no-op path
# — that path is covered by packages/schemas-py/tests/test_metrics.py.
# `pytest.importorskip` used to turn "the pin was dropped" into a silent skip
# with exit code 0; it is now a hard error unless the developer opts out with
# DUNETRACE_ALLOW_MISSING_TEST_DEPS.
prometheus_client = require_prometheus_client("services/api/requirements.txt")


def _sample(name: str, labels: dict) -> float:
    return prometheus_client.REGISTRY.get_sample_value(name, labels) or 0.0


# ── Fake pool (same shape as test_schema_version_gate's) ─────────────────────


class _Conn:
    def __init__(self, version, fail=False):
        self._version = version
        self._fail = fail

    async def fetchval(self, sql, *args):
        if self._fail:
            raise ConnectionRefusedError("db down")
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


class _Pool:
    def __init__(self, version, fail=False):
        self._conn = _Conn(version, fail)

    def acquire(self):
        return _Acquire(self._conn)

    def get_size(self):
        return 3

    def get_idle_size(self):
        return 2


# ── /metrics ─────────────────────────────────────────────────────────────────


class TestMetricsEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_metrics_is_prometheus_text_with_build_and_schema_gauges(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        body = r.text
        self.assertIn(
            'dunetrace_build_info{service="api",version="%s"} 1.0' % settings.APP_VERSION, body
        )
        self.assertIn('dunetrace_schema_version{service="api"}', body)
        self.assertIn("# TYPE dunetrace_api_requests_total counter", body)
        self.assertIn("# TYPE dunetrace_api_request_seconds histogram", body)

    def test_metrics_is_unauthenticated(self):
        """Scrapers and compose healthchecks carry no API key. Prove it in prod
        auth mode rather than relying on the dev default."""
        with patch.object(settings, "AUTH_MODE", "prod"):
            self.assertEqual(TestClient(create_app()).get("/metrics").status_code, 200)

    def test_schema_version_gauge_is_set_from_the_pool_at_startup(self):
        """The lifespan reads the applied migration version after init_pool and
        publishes it; a replica lagging a deploy shows up as a lower gauge."""
        with (
            patch("api_svc.main.init_pool", AsyncMock()),
            patch("api_svc.main.close_pool", AsyncMock()),
            patch("api_svc.main.get_pool", return_value=_Pool(CURRENT_SCHEMA_VERSION)),
            patch("api_svc.voice_pricing.check_pricing_staleness", MagicMock()),
        ):
            with TestClient(create_app()) as client:
                body = client.get("/metrics").text
        self.assertIn('dunetrace_schema_version{service="api"} %s.0' % CURRENT_SCHEMA_VERSION, body)


# ── request counter / histogram ──────────────────────────────────────────────


class TestRequestMetrics(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def _count(self, path, method="GET", status="200"):
        return _sample(
            "dunetrace_api_requests_total",
            {"path": path, "method": method, "status": status},
        )

    def _observed(self, path, method="GET"):
        return _sample("dunetrace_api_request_seconds_count", {"path": path, "method": method})

    def test_counter_and_histogram_increment_per_request(self):
        before, before_obs = self._count("/health"), self._observed("/health")
        self.client.get("/health")
        self.client.get("/health")
        self.assertEqual(self._count("/health") - before, 2)
        self.assertEqual(self._observed("/health") - before_obs, 2)

    def test_path_label_is_the_route_template_not_the_url(self):
        """One label per route, however many ids flow through it."""

        @self.app.get("/_t/{item_id}")
        async def _item(item_id: str):
            return {"id": item_id}

        before = self._count("/_t/{item_id}")
        self.client.get("/_t/abc")
        self.client.get("/_t/def")
        self.assertEqual(self._count("/_t/{item_id}") - before, 2)
        self.assertEqual(self._count("/_t/abc"), 0.0)

    def test_unmatched_paths_fold_into_other(self):
        """A scanner probing random URLs must not be able to inflate the label set."""
        before = self._count("other", status="404")
        self.client.get("/no/such/route-1")
        self.client.get("/no/such/route-2")
        self.assertEqual(self._count("other", status="404") - before, 2)
        self.assertEqual(self._count("/no/such/route-1", status="404"), 0.0)

    def test_status_label_carries_the_response_code(self):
        before_503 = self._count("/ready", status="503")
        with patch("api_svc.main.get_pool", return_value=None):
            self.client.get("/ready")
        self.assertEqual(self._count("/ready", status="503") - before_503, 1)

    def test_a_handler_exception_is_counted_as_500(self):
        @self.app.get("/_boom")
        async def _boom():
            raise RuntimeError("boom")

        before = self._count("/_boom", status="500")
        with self.assertRaises(RuntimeError):
            TestClient(self.app, raise_server_exceptions=True).get("/_boom")
        self.assertEqual(self._count("/_boom", status="500") - before, 1)


# ── /ready ───────────────────────────────────────────────────────────────────


class TestReadyEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_ready_200_when_db_answers_at_current_schema(self):
        with patch("api_svc.main.get_pool", return_value=_Pool(CURRENT_SCHEMA_VERSION)):
            r = self.client.get("/ready")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["db"], "ok")
        self.assertEqual(body["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(body["required"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(body["pool"], {"size": 3, "idle": 2})
        self.assertEqual(body["version"], settings.APP_VERSION)

    def test_ready_503_before_the_pool_exists(self):
        with patch("api_svc.main.get_pool", return_value=None):
            r = self.client.get("/ready")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["status"], "not_ready")
        self.assertEqual(r.json()["db"], "no_pool")

    def test_ready_503_when_the_connection_fails(self):
        with patch("api_svc.main.get_pool", return_value=_Pool(CURRENT_SCHEMA_VERSION, fail=True)):
            r = self.client.get("/ready")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["db"], "ConnectionRefusedError")
        self.assertIsNone(r.json()["schema_version"])

    def test_ready_503_when_the_schema_is_older_than_this_build_needs(self):
        with patch("api_svc.main.get_pool", return_value=_Pool(CURRENT_SCHEMA_VERSION - 1)):
            r = self.client.get("/ready")
        self.assertEqual(r.status_code, 503)
        body = r.json()
        self.assertEqual(body["db"], "ok")
        self.assertEqual(body["schema_version"], CURRENT_SCHEMA_VERSION - 1)
        self.assertEqual(body["required"], CURRENT_SCHEMA_VERSION)

    def test_health_stays_liveness_and_never_fails(self):
        """/health answers 200 even with no pool — that is /ready's job now."""
        with patch("api_svc.main.get_pool", return_value=None):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("db", r.json())

    def test_health_never_touches_the_pool(self):
        """Liveness must not acquire a connection: a saturated pool would make
        the probe hang and a working process read as dead."""
        pool = MagicMock()
        pool.acquire.side_effect = AssertionError("liveness must not acquire a connection")
        with patch("api_svc.db.queries._pool", pool):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        pool.acquire.assert_not_called()

    def test_ready_is_unauthenticated(self):
        with patch.object(settings, "AUTH_MODE", "prod"):
            with patch("api_svc.main.get_pool", return_value=_Pool(CURRENT_SCHEMA_VERSION)):
                self.assertEqual(TestClient(create_app()).get("/ready").status_code, 200)


# ── LLM call metrics ─────────────────────────────────────────────────────────


class TestLLMCallMetrics(unittest.IsolatedAsyncioTestCase):
    """Around llm_provider.complete — the one place the API's four LLM features
    talk to a vendor — with the vendor client mocked (llm_test_utils pattern)."""

    @staticmethod
    def _calls(provider, status):
        return _sample("dunetrace_api_llm_calls_total", {"provider": provider, "status": status})

    @staticmethod
    def _observed(provider):
        return _sample("dunetrace_api_llm_call_seconds_count", {"provider": provider})

    async def test_successful_call_is_counted_and_timed(self):
        from llm_test_utils import configured_llm

        msg = MagicMock()
        msg.content = [MagicMock(text="answer")]
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=msg)

        ok_before, obs_before = self._calls("anthropic", "ok"), self._observed("anthropic")
        with (
            configured_llm("anthropic"),
            patch("anthropic.AsyncAnthropic", MagicMock(return_value=client)),
        ):
            out = await llm_provider.complete("sys", "user", max_tokens=8)

        self.assertEqual(out, "answer")
        self.assertEqual(self._calls("anthropic", "ok") - ok_before, 1)
        self.assertEqual(self._observed("anthropic") - obs_before, 1)

    async def test_failed_call_is_counted_as_error_and_still_raises(self):
        from llm_test_utils import configured_llm

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=TimeoutError("slow vendor"))

        err_before, obs_before = self._calls("openai", "error"), self._observed("openai")
        ok_before = self._calls("openai", "ok")
        with (
            configured_llm("openai"),
            patch("openai.AsyncOpenAI", MagicMock(return_value=client)),
        ):
            with self.assertRaises(TimeoutError):
                await llm_provider.complete("sys", "user", max_tokens=8)

        self.assertEqual(self._calls("openai", "error") - err_before, 1)
        self.assertEqual(self._calls("openai", "ok"), ok_before)
        self.assertEqual(self._observed("openai") - obs_before, 1)

    async def test_unconfigured_provider_is_not_a_call(self):
        """No vendor was contacted, so nothing is observed under any provider."""
        from llm_test_utils import no_llm

        totals_before = {
            p: (self._calls(p, "ok"), self._calls(p, "error"))
            for p in llm_provider.SUPPORTED_PROVIDERS
        }
        with no_llm():
            with self.assertRaises(ValueError):
                await llm_provider.complete("sys", "user", max_tokens=8)
        for p, (ok, err) in totals_before.items():
            self.assertEqual((self._calls(p, "ok"), self._calls(p, "error")), (ok, err))

    def test_metric_names_follow_prometheus_conventions(self):
        names = {
            m.name
            for m in prometheus_client.REGISTRY.collect()
            if m.name.startswith("dunetrace_api_")
        }
        # prometheus_client strips _total from a counter's family name and
        # exposes the histogram family under its base name.
        self.assertIn("dunetrace_api_requests", names)
        self.assertIn("dunetrace_api_request_seconds", names)
        self.assertIn("dunetrace_api_llm_calls", names)
        self.assertIn("dunetrace_api_llm_call_seconds", names)
        self.assertFalse(dt_metrics.AVAILABLE is False)
