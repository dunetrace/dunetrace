"""
Phase 2 route-level tests for POST /v1/otlp/traces: auth methods, org
attribution, rejection, per-org enablement, and the admin toggle. The OTLP
route had no route-level tests before this. DB is mocked, no Postgres needed.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = pytest.mark.asyncio


# ── Fixtures ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def store():
    """A mock event store so tests can assert whether persistence happened."""
    s = MagicMock()
    s.insert_events = AsyncMock(return_value=1)
    return s


@pytest.fixture
def mock_db(monkeypatch, store):
    monkeypatch.setattr("ingest_svc.db.postgres._pool", object())
    monkeypatch.setattr("ingest_svc.db.postgres.init_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.close_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.ensure_schema", AsyncMock())
    monkeypatch.setattr(
        "ingest_svc.db.postgres.retention_looks_stale", AsyncMock(return_value=False)
    )
    # Persistence flows through otlp_limits -> _default_insert -> get_event_store,
    # which is imported from ingest_svc.db at call time.
    monkeypatch.setattr("ingest_svc.db.get_event_store", lambda: store)
    monkeypatch.setattr(
        "ingest_svc.routers.otlp.fetch_otel_ingestion_enabled", AsyncMock(return_value=True)
    )
    # Enablement is cached in-module; clear it so each test starts clean.
    import ingest_svc.routers.otlp as otlp_mod

    otlp_mod._enable_cache.clear()

    # Reset the per-process OTLP limit singletons so rate/backpressure/retry
    # state can't leak between tests.
    import ingest_svc.otlp_limits as lim

    lim._span_limiter = None
    lim._inflight = None
    lim._persist_retry = None


@pytest.fixture
async def client(mock_db):
    from ingest_svc.main import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Body helper ──────────────────────────────────────────────────────────────────


def _otlp_body(resource_attrs=None):
    root = {
        "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
        "spanId": "aaaa",
        "parentSpanId": "",
        "name": "root",
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "2000000000",
        "attributes": [],
        "status": {"code": 0},
    }
    resource = {
        "attributes": resource_attrs
        or [{"key": "service.name", "value": {"stringValue": "my-agent"}}]
    }
    return {"resourceSpans": [{"resource": resource, "scopeSpans": [{"spans": [root]}]}]}


# ── Auth methods ─────────────────────────────────────────────────────────────────


class TestAuthAccepted:
    async def test_bearer_key(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"Authorization": "Bearer good-key"}
        )
        assert r.status_code == 200

    async def test_x_dunetrace_api_key_header(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"X-Dunetrace-API-Key": "good-key"}
        )
        assert r.status_code == 200

    async def test_resource_attribute_key(self, client, monkeypatch):
        # Header key resolves to None; the dunetrace.api_key resource attribute
        # then resolves to a valid org.
        async def verify(key):
            return "org-1" if key == "rkey" else None

        monkeypatch.setattr("ingest_svc.routers.otlp.verify_api_key", AsyncMock(side_effect=verify))
        body = _otlp_body(
            resource_attrs=[
                {"key": "service.name", "value": {"stringValue": "a"}},
                {"key": "dunetrace.api_key", "value": {"stringValue": "rkey"}},
            ]
        )
        r = await client.post("/v1/otlp/traces", json=body)
        assert r.status_code == 200


class TestAuthRejected:
    async def test_invalid_key_rejected_401(self, client, monkeypatch, store):
        monkeypatch.setattr("ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value=None))
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"Authorization": "Bearer bad"}
        )
        assert r.status_code == 401
        # Rejected spans must not reach storage.
        store.insert_events.assert_not_called()

    async def test_missing_auth_rejected_when_key_invalid(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value=None))
        r = await client.post("/v1/otlp/traces", json=_otlp_body())
        assert r.status_code == 401

    async def test_trusted_missing_org_rejected(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "secret-token")
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"x-internal-token": "secret-token"}
        )
        assert r.status_code == 401


class TestTrusted:
    async def test_trusted_with_org_accepted(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "secret-token")
        r = await client.post(
            "/v1/otlp/traces",
            json=_otlp_body(),
            headers={"x-internal-token": "secret-token", "x-org-id": "org-9"},
        )
        assert r.status_code == 200


# ── Malformed body ───────────────────────────────────────────────────────────────


class TestMalformed:
    async def test_malformed_json_400(self, client, monkeypatch, store):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )
        r = await client.post(
            "/v1/otlp/traces",
            content=b"{not json",
            headers={"Authorization": "Bearer good", "content-type": "application/json"},
        )
        assert r.status_code == 400
        store.insert_events.assert_not_called()


# ── Per-org enablement ───────────────────────────────────────────────────────────


class TestEnablement:
    async def test_disabled_org_rejected_403(self, client, monkeypatch, store):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.fetch_otel_ingestion_enabled", AsyncMock(return_value=False)
        )
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"Authorization": "Bearer good"}
        )
        assert r.status_code == 403
        store.insert_events.assert_not_called()

    async def test_admin_toggle_requires_admin_key(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.routers.otlp.set_otel_ingestion_enabled", AsyncMock())
        monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
        # Wrong admin key rejected. org_id is a body field, not a path param.
        bad = await client.put(
            "/admin/otel-ingestion",
            json={"admin_key": "nope", "org_id": "org-1", "enabled": False},
        )
        assert bad.status_code == 403
        # Correct admin key accepted.
        ok = await client.put(
            "/admin/otel-ingestion",
            json={"admin_key": "admin-secret", "org_id": "org-1", "enabled": False},
        )
        assert ok.status_code == 200
        assert ok.json()["otel_ingestion_enabled"] is False


# ── Auth-failure log throttle ────────────────────────────────────────────────────


async def test_auth_failure_logging_is_throttled(caplog):
    import ingest_svc.routers.otlp as otlp_mod

    otlp_mod._last_auth_fail_log = 0.0
    otlp_mod._auth_fail_count = 0
    req = SimpleNamespace(client=SimpleNamespace(host="1.2.3.4"))

    with caplog.at_level("WARNING", logger="dunetrace.ingest.otlp"):
        for _ in range(50):
            otlp_mod._record_auth_failure(req)

    hits = [r for r in caplog.records if "OTLP auth rejected" in r.getMessage()]
    assert len(hits) == 1  # 50 failures collapsed to a single log line in the window


# ── Phase 3: failure isolation at the route ──────────────────────────────────────


class TestFailureIsolation:
    async def test_unsupported_content_type_415(self, client):
        r = await client.post(
            "/v1/otlp/traces",
            content=b"<xml/>",
            headers={"content-type": "application/xml"},
        )
        assert r.status_code == 415

    async def test_oversized_body_413(self, client, monkeypatch, store):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )
        monkeypatch.setattr("ingest_svc.config.settings.OTLP_MAX_BODY_BYTES", 50)
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"Authorization": "Bearer good"}
        )
        assert r.status_code == 413
        store.insert_events.assert_not_called()

    async def test_gzip_bomb_413(self, client, monkeypatch, store):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )
        monkeypatch.setattr("ingest_svc.config.settings.OTLP_MAX_DECOMPRESSED_BYTES", 10)
        payload = gzip.compress(json.dumps(_otlp_body()).encode())  # expands past 10 bytes
        r = await client.post(
            "/v1/otlp/traces",
            content=payload,
            headers={
                "Authorization": "Bearer good",
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
        )
        assert r.status_code == 413
        store.insert_events.assert_not_called()

    async def test_span_rate_limit_429_with_retry_after(self, client, monkeypatch, store):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )

        class _DeniedLimiter:
            def allow(self, org_id, span_count):
                return False, 7

        monkeypatch.setattr("ingest_svc.routers.otlp.get_span_limiter", lambda: _DeniedLimiter())
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"Authorization": "Bearer good"}
        )
        assert r.status_code == 429
        assert r.headers["Retry-After"] == "7"
        store.insert_events.assert_not_called()

    async def test_backpressure_429(self, client, monkeypatch, store):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )

        class _FullGuard:
            def try_reserve(self):
                return False

            def release(self):
                pass

        monkeypatch.setattr("ingest_svc.routers.otlp.get_inflight_guard", lambda: _FullGuard())
        r = await client.post(
            "/v1/otlp/traces", json=_otlp_body(), headers={"Authorization": "Bearer good"}
        )
        assert r.status_code == 429
        store.insert_events.assert_not_called()

    async def test_rate_limiting_under_concurrent_load(self, client, monkeypatch):
        """End-to-end: a burst of concurrent requests against a low per-org rate
        gets partly admitted and partly shed with 429 — one org can't flood."""
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-load")
        )
        monkeypatch.setattr("ingest_svc.config.settings.OTLP_MAX_SPANS_PER_SEC", 10)  # burst 20
        body = _otlp_body()  # one span per request

        async def one():
            r = await client.post(
                "/v1/otlp/traces", json=body, headers={"Authorization": "Bearer good"}
            )
            return r.status_code

        results = await asyncio.gather(*[one() for _ in range(60)])
        assert results.count(429) > 0  # load shed
        assert results.count(200) > 0  # some admitted
        assert results.count(200) < 60  # capping actually happened
