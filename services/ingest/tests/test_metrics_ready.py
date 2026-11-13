"""
GET /metrics, GET /ready and the ingest service's Prometheus counters.

The registry is process-global and every test in this directory builds the app
afresh, so nothing here asserts an absolute value: each test scrapes before and
after and checks the delta.

DB is mocked — no Postgres needed. Reuses test_ingest.py's fixtures.

Run:
    PYTHONPATH=packages/sdk-py:packages/schemas-py:services/ingest \
      python -m pytest services/ingest/tests/test_metrics_ready.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client.parser import text_string_to_metric_families

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_ingest import client, fresh_limiter, make_batch, make_event, mock_db  # noqa: F401

from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION
from ingest_svc.config import settings

pytestmark = pytest.mark.asyncio


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _scrape(client) -> str:
    r = await client.get("/metrics")
    assert r.status_code == 200
    return r.text


def _sample(text: str, name: str, **labels) -> float:
    """Value of the sample called ``name`` whose labels include ``labels``; 0
    when absent (a counter that has never been incremented for that label set
    is simply not in the output yet)."""
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return 0.0


def _has_sample(text: str, name: str, **labels) -> bool:
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return True
    return False


def _ready_pool(schema_version: int, *, select_error: Exception | None = None):
    """A duck-typed asyncpg pool for db_ready: acquire() yields a conn whose
    fetchval answers SELECT 1 and the schema_version read."""
    conn = AsyncMock()

    async def fetchval(sql, *args):
        if "schema_version" in sql:
            return schema_version
        if select_error is not None:
            raise select_error
        return 1

    conn.fetchval = AsyncMock(side_effect=fetchval)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool.get_size.return_value = 3
    pool.get_min_size.return_value = 1
    pool.get_max_size.return_value = 5
    pool.get_idle_size.return_value = 2
    return pool


# ── /metrics ───────────────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    async def test_serves_prometheus_text(self, client):
        r = await client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert "# HELP dunetrace_" in r.text

    async def test_standard_metrics_present(self, client):
        text = await _scrape(client)
        assert (
            _sample(text, "dunetrace_build_info", service="ingest", version=settings.APP_VERSION)
            == 1.0
        )
        assert _has_sample(text, "dunetrace_schema_version", service="ingest")

    async def test_every_ingest_metric_is_registered(self, client):
        text = await _scrape(client)
        families = {f.name for f in text_string_to_metric_families(text)}
        # Counters are exposed under their family name (no _total suffix).
        for name in (
            "dunetrace_ingest_requests",
            "dunetrace_ingest_persist_failures",
            "dunetrace_ingest_persist_seconds",
            "dunetrace_ingest_rate_limited",
            "dunetrace_ingest_body_too_large",
            "dunetrace_ingest_events_accepted",
        ):
            assert name in families, name

    async def test_hidden_from_openapi(self, client):
        paths = (await client.get("/openapi.json")).json()["paths"]
        assert "/metrics" not in paths
        assert "/ready" not in paths
        assert "/health" not in paths


# ── dunetrace_ingest_requests_total ────────────────────────────────────────────


class TestRequestsTotal:
    _N = "dunetrace_ingest_requests_total"

    async def test_counts_a_202_under_the_route_path(self, client):
        before = _sample(
            await _scrape(client), self._N, path="/v1/ingest", method="POST", status="202"
        )
        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 202
        after = _sample(
            await _scrape(client), self._N, path="/v1/ingest", method="POST", status="202"
        )
        assert after == before + 1

    async def test_templated_route_is_labelled_by_template_not_literal(self, client, monkeypatch):
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        template = "/admin/keys/{key_id}/agents/{agent_id}/quota"
        before = _sample(await _scrape(client), self._N, path=template, method="GET", status="403")
        r = await client.get("/admin/keys/7/agents/agent-x/quota", params={"admin_key": "nope"})
        assert r.status_code == 403
        text = await _scrape(client)
        assert _sample(text, self._N, path=template, method="GET", status="403") == before + 1
        assert "/admin/keys/7/" not in text  # the literal id never becomes a label

    async def test_unknown_path_is_other(self, client):
        before = _sample(await _scrape(client), self._N, path="other", method="GET", status="404")
        r = await client.get("/no-such-endpoint-9f2c")
        assert r.status_code == 404
        text = await _scrape(client)
        assert _sample(text, self._N, path="other", method="GET", status="404") == before + 1
        assert "no-such-endpoint" not in text

    async def test_counts_a_validation_422(self, client):
        before = _sample(
            await _scrape(client), self._N, path="/v1/ingest", method="POST", status="422"
        )
        r = await client.post("/v1/ingest", json={"agent_id": "a", "events": []})
        assert r.status_code == 422
        after = _sample(
            await _scrape(client), self._N, path="/v1/ingest", method="POST", status="422"
        )
        assert after == before + 1


# ── Persist path ───────────────────────────────────────────────────────────────


class TestPersistMetrics:
    _F = "dunetrace_ingest_persist_failures_total"
    _H = "dunetrace_ingest_persist_seconds_count"
    _A = "dunetrace_ingest_events_accepted_total"

    async def test_store_exception_counts_a_failure_and_a_503(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=RuntimeError("db down")),
        )
        text = await _scrape(client)
        fail_before = _sample(text, self._F, reason="exception")
        hist_before = _sample(text, self._H)
        req_before = _sample(
            text, "dunetrace_ingest_requests_total", path="/v1/ingest", method="POST", status="503"
        )
        accepted_before = _sample(text, self._A)

        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 503

        text = await _scrape(client)
        assert _sample(text, self._F, reason="exception") == fail_before + 1
        assert _sample(text, self._H) == hist_before + 1  # timed even on failure
        assert (
            _sample(
                text,
                "dunetrace_ingest_requests_total",
                path="/v1/ingest",
                method="POST",
                status="503",
            )
            == req_before + 1
        )
        assert _sample(text, self._A) == accepted_before  # nothing was accepted

    async def test_insert_shortfall_is_its_own_reason(self, client, monkeypatch):
        # Two events, one row written, and the batch cannot be shown durable
        # (mock_db's pool is a bare object(), so the presence check fails closed).
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=lambda events, batch_id, org_id: len(events) - 1),
        )
        text = await _scrape(client)
        short_before = _sample(text, self._F, reason="shortfall")
        exc_before = _sample(text, self._F, reason="exception")

        events = [make_event(step_index=i, event_id=f"ev-{i}") for i in range(2)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 503

        text = await _scrape(client)
        assert _sample(text, self._F, reason="shortfall") == short_before + 1
        assert _sample(text, self._F, reason="exception") == exc_before

    async def test_accepted_events_counted_per_event_not_per_batch(self, client):
        text = await _scrape(client)
        accepted_before = _sample(text, self._A)
        hist_before = _sample(text, self._H)

        events = [make_event(step_index=i) for i in range(4)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 202

        text = await _scrape(client)
        assert _sample(text, self._A) == accepted_before + 4
        assert _sample(text, self._H) == hist_before + 1
        assert _sample(text, "dunetrace_ingest_persist_seconds_sum") >= 0.0


# ── Middleware short-circuits ──────────────────────────────────────────────────


class TestShortCircuitCounters:
    async def test_413_increments_body_too_large_and_requests(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INGEST_MAX_BODY_BYTES", 128)
        text = await _scrape(client)
        big_before = _sample(text, "dunetrace_ingest_body_too_large_total")
        req_before = _sample(
            text, "dunetrace_ingest_requests_total", path="/v1/ingest", method="POST", status="413"
        )

        events = [make_event(step_index=i) for i in range(5)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 413

        text = await _scrape(client)
        assert _sample(text, "dunetrace_ingest_body_too_large_total") == big_before + 1
        assert (
            _sample(
                text,
                "dunetrace_ingest_requests_total",
                path="/v1/ingest",
                method="POST",
                status="413",
            )
            == req_before + 1
        )

    async def test_429_increments_rate_limited_and_requests(self, client, fresh_limiter):
        text = await _scrape(client)
        rl_before = _sample(text, "dunetrace_ingest_rate_limited_total")
        req_before = _sample(
            text, "dunetrace_ingest_requests_total", path="/v1/ingest", method="POST", status="429"
        )

        # Distinct agent_id per warm-up post: at rpm=3 the 20% per-agent
        # sub-quota is 1 rpm, so a shared agent would 429 from the second post
        # on and the delta below would be 3. This exhausts the key-level window
        # exactly once.
        for i in range(3):
            r = await client.post(
                "/v1/ingest", json=make_batch(api_key="dt_live_metrics_429", agent_id=f"agent-{i}")
            )
            assert r.status_code == 202
        r = await client.post(
            "/v1/ingest", json=make_batch(api_key="dt_live_metrics_429", agent_id="agent-3")
        )
        assert r.status_code == 429

        text = await _scrape(client)
        assert _sample(text, "dunetrace_ingest_rate_limited_total") == rl_before + 1
        assert (
            _sample(
                text,
                "dunetrace_ingest_requests_total",
                path="/v1/ingest",
                method="POST",
                status="429",
            )
            == req_before + 1
        )

    async def test_metrics_and_ready_are_not_rate_limited(self, client, fresh_limiter):
        # rpm=3 — well past it on both probes, none of them a 429.
        for _ in range(6):
            assert (await client.get("/metrics")).status_code == 200
            assert (await client.get("/health")).status_code == 200


# ── /ready ─────────────────────────────────────────────────────────────────────


class TestReady:
    async def test_200_with_the_full_verdict(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", _ready_pool(CURRENT_SCHEMA_VERSION))
        r = await client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["db"] == "ok"
        assert body["schema_version"] == CURRENT_SCHEMA_VERSION
        assert body["required"] == CURRENT_SCHEMA_VERSION
        assert body["version"] == settings.APP_VERSION
        assert body["pool"] == {"size": 3, "min": 1, "max": 5, "idle": 2}

    async def test_newer_schema_is_still_ready(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", _ready_pool(CURRENT_SCHEMA_VERSION + 3))
        assert (await client.get("/ready")).status_code == 200

    async def test_503_when_select_1_fails(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.db.postgres._pool",
            _ready_pool(CURRENT_SCHEMA_VERSION, select_error=ConnectionResetError("gone")),
        )
        r = await client.get("/ready")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "not_ready"
        assert body["db"] == "ConnectionResetError"
        assert body["schema_version"] is None
        assert body["required"] == CURRENT_SCHEMA_VERSION
        assert "gone" not in r.text  # class name only, never the message

    async def test_503_when_schema_is_behind(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", _ready_pool(CURRENT_SCHEMA_VERSION - 1))
        r = await client.get("/ready")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "not_ready"
        assert body["db"] == "ok"  # the DB answered; the schema is what's wrong
        assert body["schema_version"] == CURRENT_SCHEMA_VERSION - 1
        assert body["required"] == CURRENT_SCHEMA_VERSION

    async def test_503_without_a_pool(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", None)
        r = await client.get("/ready")
        assert r.status_code == 503
        assert r.json()["db"] == "no_pool"

    async def test_ready_probes_are_counted(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", None)
        before = _sample(
            await _scrape(client),
            "dunetrace_ingest_requests_total",
            path="/ready",
            method="GET",
            status="503",
        )
        await client.get("/ready")
        after = _sample(
            await _scrape(client),
            "dunetrace_ingest_requests_total",
            path="/ready",
            method="GET",
            status="503",
        )
        assert after == before + 1


# ── /health stays liveness-only ────────────────────────────────────────────────


class TestHealthIsLivenessOnly:
    async def test_never_acquires_a_connection(self, client, monkeypatch):
        pool = MagicMock()
        pool.acquire.side_effect = AssertionError("liveness must not touch the pool")
        monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        pool.acquire.assert_not_called()

    async def test_200_even_with_no_pool(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", None)
        r = await client.get("/health")
        assert r.status_code == 200
        assert "db" not in r.json()
