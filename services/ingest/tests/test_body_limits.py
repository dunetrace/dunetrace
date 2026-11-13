"""
INGEST_MAX_BODY_BYTES — POST /v1/ingest and /v1/deploy reject an oversized body
with 413 before it is buffered, parsed, authenticated or persisted.

Before this cap the only limit on /v1/ingest was MAX_BATCH_SIZE (an event
count), and both middlewares in main.py buffered the whole body with
request.body() before anything could look at its size. The check now lives in
the outermost middleware and reads through ingest_svc.body_limits — the same
bounded reader the OTLP route used to keep a private copy of.

DB is mocked — no Postgres needed. Reuses test_ingest.py's fixtures.

Run:
    PYTHONPATH=packages/sdk-py:packages/schemas-py:services/ingest \
      python -m pytest services/ingest/tests/test_body_limits.py -v
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_ingest import client, fresh_limiter, make_batch, make_event, mock_db  # noqa: F401

from ingest_svc.body_limits import BodyTooLarge, read_bounded_body

pytestmark = pytest.mark.asyncio

_CAP = 256
_SENTINEL = "BODY-CONTENT-SENTINEL-9f2c"
_JSON = {"Content-Type": "application/json"}


# ── Helpers / fixtures ─────────────────────────────────────────────────────────


def _batch_bytes(**overrides) -> bytes:
    return json.dumps(make_batch(**overrides)).encode()


def _oversized_batch() -> dict:
    """A syntactically valid batch whose JSON is comfortably over _CAP, carrying
    a sentinel string so a test can prove the body never reached a log line."""
    events = [
        make_event(step_index=i, payload={"tool_name": "t", "args": _SENTINEL * 4})
        for i in range(8)
    ]
    body = make_batch(events=events)
    assert len(json.dumps(body)) > _CAP * 2
    return body


def _hundred_byte_chunks(n: int):
    """An async body generator with no Content-Length (httpx sends it chunked).
    Counts how many chunks the server actually pulled."""
    pulled = SimpleNamespace(n=0)
    chunk = b"{" + b" " * 99

    async def gen():
        for _ in range(n):
            pulled.n += 1
            yield chunk

    return gen(), pulled


@pytest.fixture
def small_cap(monkeypatch):
    monkeypatch.setattr("ingest_svc.config.settings.INGEST_MAX_BODY_BYTES", _CAP)
    return _CAP


@pytest.fixture
def spies(mock_db, monkeypatch):
    """Everything a rejected request must never reach: the two stores, and both
    api_key lookups (set_org_context's in main.py, and the route's). Depends on
    mock_db so these patches land on top of its defaults."""
    insert_events = AsyncMock(side_effect=lambda events, batch_id, org_id: len(events))
    monkeypatch.setattr("ingest_svc.db.postgres.insert_events", insert_events)
    insert_deploy = AsyncMock(return_value=7)
    monkeypatch.setattr("ingest_svc.routers.ingest.insert_deploy_event", insert_deploy)
    verify_route = AsyncMock(return_value="org-test")
    monkeypatch.setattr("ingest_svc.routers.ingest.verify_api_key", verify_route)
    verify_mw = AsyncMock(return_value="org-test")
    monkeypatch.setattr("ingest_svc.main.verify_api_key", verify_mw)
    return SimpleNamespace(
        insert_events=insert_events,
        insert_deploy=insert_deploy,
        verify_route=verify_route,
        verify_mw=verify_mw,
    )


def _assert_nothing_reached(spies) -> None:
    spies.insert_events.assert_not_awaited()
    spies.insert_deploy.assert_not_awaited()
    spies.verify_route.assert_not_awaited()
    spies.verify_mw.assert_not_awaited()


def _too_large_records(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "rejected with 413" in r.getMessage()
    ]


# ── Content-Length over the cap: refused before a byte is read ─────────────────


class TestContentLengthOverCap:
    async def test_ingest_413_and_nothing_downstream_runs(self, client, small_cap, spies):
        r = await client.post("/v1/ingest", json=_oversized_batch())
        assert r.status_code == 413
        assert str(small_cap) in r.json()["detail"]
        _assert_nothing_reached(spies)

    async def test_deploy_413_and_nothing_downstream_runs(self, client, small_cap, spies):
        body = {
            "api_key": "dt_dev_test",
            "agent_id": "agent-xyz",
            "version": "v1.0.0",
            "meta": {"notes": _SENTINEL * 20},
        }
        r = await client.post("/v1/deploy", json=body)
        assert r.status_code == 413
        assert "detail" in r.json()
        _assert_nothing_reached(spies)

    async def test_rejected_before_json_parsing(self, client, small_cap, spies):
        # Not JSON at all — parsing it would be a 422. The cap answers first.
        r = await client.post("/v1/ingest", content=b"x" * (small_cap + 1), headers=_JSON)
        assert r.status_code == 413
        _assert_nothing_reached(spies)

    async def test_trusted_gateway_path_is_capped_too(self, client, small_cap, spies, monkeypatch):
        # The gateway skips rate limiting, not the body cap: this process is
        # the one that buffers the body, whatever upstream promised.
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")
        r = await client.post(
            "/v1/ingest",
            json=_oversized_batch(),
            headers={"x-internal-token": "shared-secret", "x-org-id": "org-trusted"},
        )
        assert r.status_code == 413
        _assert_nothing_reached(spies)

    async def test_413_does_not_consume_rate_limit_budget(
        self, client, spies, fresh_limiter, monkeypatch
    ):
        # fresh_limiter allows 3 rpm. Three oversized requests are refused
        # before the bucket is even known (the api_key is inside the body that
        # was never read), so a fourth, well-formed request still gets through.
        # The cap here admits a one-event batch (~311 bytes) but not the
        # oversized one (~2.8 KB).
        monkeypatch.setattr("ingest_svc.config.settings.INGEST_MAX_BODY_BYTES", 512)
        for _ in range(3):
            r = await client.post("/v1/ingest", json=_oversized_batch())
            assert r.status_code == 413
        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 202


# ── Streaming cap: Content-Length missing or lying ─────────────────────────────


class TestStreamingCap:
    async def test_lying_content_length_413(self, client, small_cap, spies):
        payload = json.dumps(_oversized_batch()).encode()
        r = await client.post(
            "/v1/ingest", content=payload, headers={**_JSON, "Content-Length": "10"}
        )
        assert r.status_code == 413
        _assert_nothing_reached(spies)

    async def test_missing_content_length_stops_reading_at_cap(self, client, small_cap, spies):
        body, pulled = _hundred_byte_chunks(50)
        r = await client.post("/v1/ingest", content=body, headers=_JSON)
        assert r.status_code == 413
        # 256-byte cap, 100-byte chunks: the third chunk crosses it. The other
        # 47 were never pulled — the body was not buffered and then measured.
        assert pulled.n == 3
        _assert_nothing_reached(spies)

    async def test_body_exactly_at_cap_is_accepted(self, client, spies, monkeypatch):
        payload = _batch_bytes()
        monkeypatch.setattr("ingest_svc.config.settings.INGEST_MAX_BODY_BYTES", len(payload))
        r = await client.post("/v1/ingest", content=payload, headers=_JSON)
        assert r.status_code == 202
        assert r.json()["accepted"] == 1
        spies.insert_events.assert_awaited_once()

    async def test_one_byte_over_cap_is_rejected(self, client, spies, monkeypatch):
        payload = _batch_bytes() + b" "  # still valid JSON, one byte longer
        monkeypatch.setattr("ingest_svc.config.settings.INGEST_MAX_BODY_BYTES", len(payload) - 1)
        r = await client.post("/v1/ingest", content=payload, headers=_JSON)
        assert r.status_code == 413
        _assert_nothing_reached(spies)

    async def test_chunked_body_under_cap_is_reassembled_for_the_route(self, client, spies):
        # No Content-Length, several chunks, default cap: the bounded reader
        # joins them and the route sees the whole batch.
        payload = _batch_bytes(events=[make_event(step_index=i) for i in range(3)])

        async def gen():
            for i in range(0, len(payload), 64):
                yield payload[i : i + 64]

        r = await client.post("/v1/ingest", content=gen(), headers=_JSON)
        assert r.status_code == 202
        assert r.json()["accepted"] == 3


# ── Everything below the cap still sees the body ───────────────────────────────


class TestDownstreamStillSeesBody:
    async def test_org_context_middleware_and_route_read_the_cached_body(self, client, spies):
        events = [make_event(step_index=i) for i in range(4)]
        r = await client.post(
            "/v1/ingest", json=make_batch(api_key="dt_live_visible", events=events)
        )
        assert r.status_code == 202
        assert r.json()["accepted"] == 4
        # set_org_context parsed the api_key out of the body the outer
        # middleware had already read; so did the route. The route verifies
        # independently on purpose — see _resolve_org_id for why its auth
        # decision must not depend on middleware state.
        spies.verify_mw.assert_awaited_once_with("dt_live_visible")
        spies.verify_route.assert_awaited_once_with("dt_live_visible")
        spies.insert_events.assert_awaited_once()
        assert len(spies.insert_events.await_args.args[0]) == 4

    async def test_deploy_still_202(self, client, spies):
        r = await client.post(
            "/v1/deploy",
            json={"api_key": "dt_dev_test", "agent_id": "agent-xyz", "version": "v1.0.0"},
        )
        assert r.status_code == 202
        assert r.json()["id"] == 7
        spies.insert_deploy.assert_awaited_once()

    async def test_default_cap_is_10_mib(self):
        from ingest_svc.config import settings

        assert settings.INGEST_MAX_BODY_BYTES == 10 * 1024 * 1024


# ── Logging: sizes and client, never the body ──────────────────────────────────


class TestLogging:
    async def test_header_rejection_logged_at_warning_without_body(
        self, client, small_cap, spies, caplog
    ):
        payload = json.dumps(_oversized_batch()).encode()
        with caplog.at_level(logging.WARNING, logger="dunetrace.ingest"):
            r = await client.post("/v1/ingest", content=payload, headers=_JSON)
        assert r.status_code == 413
        msgs = _too_large_records(caplog)
        assert len(msgs) == 1
        msg = msgs[0]
        assert "path=/v1/ingest" in msg
        assert f"declared={len(payload)}" in msg
        assert "received=0" in msg  # refused on the header alone
        assert f"limit={small_cap}" in msg
        assert "client=" in msg
        assert _SENTINEL not in msg
        assert "dt_dev_test" not in msg

    async def test_streamed_rejection_logs_bytes_received(self, client, small_cap, spies, caplog):
        body, _ = _hundred_byte_chunks(10)
        with caplog.at_level(logging.WARNING, logger="dunetrace.ingest"):
            r = await client.post("/v1/ingest", content=body, headers=_JSON)
        assert r.status_code == 413
        msgs = _too_large_records(caplog)
        assert len(msgs) == 1
        assert "declared=absent" in msgs[0]
        received = int(re.search(r"received=(\d+)", msgs[0]).group(1))
        assert received > small_cap


# ── OTLP keeps its own cap ─────────────────────────────────────────────────────


class TestOtlpIsCappedSeparately:
    async def test_ingest_cap_does_not_gate_otlp(self, client, small_cap, spies, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-1")
        )
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.fetch_otel_ingestion_enabled", AsyncMock(return_value=True)
        )
        import ingest_svc.routers.otlp as otlp_mod

        otlp_mod._enable_cache.clear()
        # Well over the 256-byte ingest cap, well under OTLP_MAX_BODY_BYTES.
        body = {"resourceSpans": [], "padding": "x" * 2048}
        r = await client.post(
            "/v1/otlp/traces", json=body, headers={"Authorization": "Bearer good"}
        )
        assert r.status_code == 200


# ── The reader itself ──────────────────────────────────────────────────────────


def _request(headers: dict[str, str], chunks: list[bytes]):
    """A bare starlette Request over a scripted ASGI receive, counting calls."""
    msgs = [{"type": "http.request", "body": c, "more_body": True} for c in chunks]
    msgs.append({"type": "http.request", "body": b"", "more_body": False})
    it = iter(msgs)
    calls = SimpleNamespace(n=0)

    async def receive():
        calls.n += 1
        return next(it)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/ingest",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "query_string": b"",
    }
    return Request(scope, receive), calls


class TestReadBoundedBody:
    async def test_content_length_over_limit_reads_nothing(self):
        req, calls = _request({"content-length": "1000"}, [b"x" * 1000])
        with pytest.raises(BodyTooLarge) as ei:
            await read_bounded_body(req, 100)
        assert (ei.value.limit, ei.value.declared, ei.value.received) == (100, 1000, 0)
        assert calls.n == 0

    async def test_stream_over_limit_stops_at_the_crossing_chunk(self):
        req, calls = _request({}, [b"x" * 60] * 10)
        with pytest.raises(BodyTooLarge) as ei:
            await read_bounded_body(req, 100)
        assert ei.value.declared is None
        assert ei.value.received == 120
        assert calls.n == 2

    async def test_lying_content_length_is_caught_by_the_stream_cap(self):
        req, _ = _request({"content-length": "10"}, [b"x" * 500])
        with pytest.raises(BodyTooLarge) as ei:
            await read_bounded_body(req, 100)
        assert (ei.value.declared, ei.value.received) == (10, 500)

    async def test_malformed_content_length_falls_back_to_the_stream_cap(self):
        req, _ = _request({"content-length": "lots"}, [b"x" * 500])
        with pytest.raises(BodyTooLarge) as ei:
            await read_bounded_body(req, 100)
        assert ei.value.declared is None

    async def test_at_limit_returns_body_and_caches_it_on_the_request(self):
        req, _ = _request({"content-length": "100"}, [b"a" * 40, b"b" * 60])
        body = await read_bounded_body(req, 100)
        assert body == b"a" * 40 + b"b" * 60
        # Cached like Request.body() would: neither call raises "Stream consumed",
        # which is what lets a downstream app see the body after a middleware read.
        assert await req.body() == body
        assert b"".join([c async for c in req.stream()]) == body

    async def test_empty_body(self):
        req, _ = _request({"content-length": "0"}, [])
        assert await read_bounded_body(req, 100) == b""
        assert await req.body() == b""

    async def test_exception_carries_sizes_not_content(self):
        exc = BodyTooLarge(limit=5, declared=9, received=0)
        assert str(exc) == "request body exceeds 5 bytes"
        assert (exc.limit, exc.declared, exc.received) == (5, 9, 0)
