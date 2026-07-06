"""
Tests for the ingest API. DB is mocked — no Postgres needed.

Run:
    cd services/ingest
    pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = pytest.mark.asyncio


# ── App fixture ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_db(monkeypatch):
    """Patch all DB calls so tests run without Postgres."""
    monkeypatch.setattr("ingest_svc.db.postgres._pool", object())  # truthy non-None
    monkeypatch.setattr("ingest_svc.db.postgres.init_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.close_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.ensure_schema", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.check_db", AsyncMock(return_value="ok"))
    monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=1))
    # Patched where routers/ingest.py actually looks it up (`from ingest_svc.db
    # import verify_api_key` binds a local name there) — patching
    # ingest_svc.db.postgres.verify_api_key instead is a no-op, since that
    # module-level binding was already copied before this fixture runs.
    # verify_api_key returns org_id (not agent_id) since v0.5.0 — keys are
    # org-scoped, agent_id is just per-event data, unrelated to which key sent it.
    monkeypatch.setattr(
        "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value="org-test")
    )


@pytest.fixture
async def client(mock_db):
    from ingest_svc.main import create_app

    application = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as c:
        yield c


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_event(**overrides) -> dict:
    e = {
        "event_type": "tool.called",
        "run_id": "run-abc123",
        "agent_id": "agent-xyz",
        "agent_version": "9a3f1b2c",
        "step_index": 1,
        "timestamp": 1708934400.0,
        "payload": {"tool_name": "web_search", "args": "aabb"},
        "parent_run_id": None,
    }
    e.update(overrides)
    return e


def make_batch(events=None, api_key="dt_dev_test", agent_id="agent-xyz") -> dict:
    return {
        "api_key": api_key,
        "agent_id": agent_id,
        "events": [make_event()] if events is None else events,
    }


# ── Happy path ─────────────────────────────────────────────────────────────────


class TestHappyPath:
    async def test_returns_202(self, client):
        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 202

    async def test_response_shape(self, client):
        body = (await client.post("/v1/ingest", json=make_batch())).json()
        assert "accepted" in body
        assert "batch_id" in body
        assert "queued_at" in body

    async def test_accepted_count_matches_batch_size(self, client):
        events = [make_event(step_index=i) for i in range(7)]
        body = (await client.post("/v1/ingest", json=make_batch(events=events))).json()
        assert body["accepted"] == 7

    async def test_batch_ids_are_unique(self, client):
        id1 = (await client.post("/v1/ingest", json=make_batch())).json()["batch_id"]
        id2 = (await client.post("/v1/ingest", json=make_batch())).json()["batch_id"]
        assert id1 != id2

    async def test_all_event_types_accepted(self, client):
        event_types = [
            "run.started",
            "run.completed",
            "run.errored",
            "llm.called",
            "llm.responded",
            "tool.called",
            "tool.responded",
            "retrieval.called",
            "retrieval.responded",
        ]
        events = [make_event(event_type=t, step_index=i) for i, t in enumerate(event_types)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 202
        assert r.json()["accepted"] == len(event_types)

    async def test_event_with_parent_run_id(self, client):
        r = await client.post(
            "/v1/ingest",
            json=make_batch(events=[make_event(parent_run_id="parent-abc")]),
        )
        assert r.status_code == 202

    async def test_empty_payload_accepted(self, client):
        r = await client.post("/v1/ingest", json=make_batch(events=[make_event(payload={})]))
        assert r.status_code == 202

    async def test_missing_timestamp_uses_default(self, client):
        event = make_event()
        del event["timestamp"]
        r = await client.post("/v1/ingest", json=make_batch(events=[event]))
        assert r.status_code == 202

    async def test_max_batch_of_500_accepted(self, client):
        events = [make_event(step_index=i) for i in range(500)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 202
        assert r.json()["accepted"] == 500

    async def test_rag_retrieval_event(self, client):
        event = make_event(
            event_type="retrieval.responded",
            payload={"index_name": "docs", "result_count": 0, "top_score": None},
        )
        r = await client.post("/v1/ingest", json=make_batch(events=[event]))
        assert r.status_code == 202

    async def test_run_started_event(self, client):
        event = make_event(
            event_type="run.started",
            step_index=0,
            payload={"input_text": "abc", "model": "gpt-4o", "tools": ["web_search"]},
        )
        r = await client.post("/v1/ingest", json=make_batch(events=[event]))
        assert r.status_code == 202


# ── Validation — FastAPI returns 422 with detail array ─────────────────────────


class TestValidation:
    async def test_empty_events_rejected_422(self, client):
        r = await client.post("/v1/ingest", json=make_batch(events=[]))
        assert r.status_code == 422

    async def test_unknown_event_type_rejected_422(self, client):
        r = await client.post(
            "/v1/ingest", json=make_batch(events=[make_event(event_type="not.valid")])
        )
        assert r.status_code == 422

    async def test_missing_run_id_rejected_422(self, client):
        event = make_event()
        del event["run_id"]
        r = await client.post("/v1/ingest", json=make_batch(events=[event]))
        assert r.status_code == 422

    async def test_empty_run_id_rejected_422(self, client):
        r = await client.post("/v1/ingest", json=make_batch(events=[make_event(run_id="")]))
        assert r.status_code == 422

    async def test_empty_agent_id_rejected_422(self, client):
        r = await client.post("/v1/ingest", json=make_batch(events=[make_event(agent_id="")]))
        assert r.status_code == 422

    async def test_negative_step_index_rejected_422(self, client):
        r = await client.post("/v1/ingest", json=make_batch(events=[make_event(step_index=-1)]))
        assert r.status_code == 422

    async def test_missing_api_key_accepted(self, client):
        # api_key has a default of "" so omitting it is valid schema-wise;
        # auth is enforced separately (401) not at validation (422)
        body = make_batch()
        del body["api_key"]
        r = await client.post("/v1/ingest", json=body)
        assert r.status_code in (202, 401)

    async def test_empty_api_key_accepted(self, client):
        r = await client.post("/v1/ingest", json=make_batch(api_key=""))
        assert r.status_code in (202, 401)

    async def test_batch_over_500_rejected_422(self, client):
        events = [make_event(step_index=i) for i in range(501)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 422

    async def test_non_json_body_rejected(self, client):
        r = await client.post(
            "/v1/ingest",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    async def test_422_has_detail_field(self, client):
        r = await client.post("/v1/ingest", json=make_batch(events=[make_event(event_type="bad")]))
        assert "detail" in r.json()

    async def test_non_dict_payload_rejected_422(self, client):
        r = await client.post("/v1/ingest", json=make_batch(events=[make_event(payload="string")]))
        assert r.status_code == 422


# ── Auth ───────────────────────────────────────────────────────────────────────


class TestAuth:
    async def test_invalid_key_returns_401(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value=None)
        )
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_bad"))
        assert r.status_code == 401

    async def test_401_has_detail(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value=None)
        )
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_bad"))
        assert "detail" in r.json()

    async def test_valid_key_accepted(self, client):
        # mock_db fixture patches verify_api_key to return "agent-xyz",
        # matching make_batch()'s default agent_id
        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 202

    async def test_org_scoped_key_accepts_any_agent_id(self, client):
        # v0.5.0 multi-tenancy: keys are org-scoped, not agent-scoped. A valid
        # key may submit events for any agent_id under its org — agents are
        # discovered on first ingest, not fixed at key-creation time. See
        # docs/migrations/multi-tenancy-v0.5.0.md ("Security posture change").
        r = await client.post("/v1/ingest", json=make_batch(agent_id="a-brand-new-agent"))
        assert r.status_code == 202

    async def test_org_scoped_key_accepts_mixed_agent_ids_within_batch(self, client):
        # One key, one batch, two different agents — allowed under the org-scoped
        # model as long as both belong to the org the key resolves to.
        events = [make_event(), make_event(agent_id="a-different-agent")]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 202


# ── Trusted gateway (dunetrace-cloud) ───────────────────────────────────────────
#
# dunetrace-cloud's tenancy middleware validates the caller against its own
# org_api_keys table, then proxies here with x-internal-token set plus an
# x-org-id header carrying the already-authenticated org identity (x-customer-id
# is accepted as a fallback for older gateway builds — see
# docs/migrations/multi-tenancy-v0.5.0.md). These routes must accept that trust
# signal and skip the OSS-only api_keys lookup entirely — that table has no
# rows in production (org auth lives in dunetrace-cloud now), so any request
# that ran the DB check would 401 regardless of key validity. See
# ingest_svc/auth.py::is_trusted.
#
# mock_db's default verify_api_key mock unconditionally returns "agent-xyz"
# regardless of the api_key passed in, so "still enforces" tests below must
# explicitly override it to return None to simulate a bad/unresolvable key —
# they can no longer rely on an unmocked lookup naturally failing.


class TestTrustedGateway:
    async def test_trusted_header_bypasses_api_keys_check(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")

        r = await client.post(
            "/v1/ingest",
            json=make_batch(api_key="not-a-dev-key"),
            headers={"x-internal-token": "shared-secret", "x-org-id": "org-trusted"},
        )
        assert r.status_code == 202

    async def test_trusted_request_missing_org_id_header_401s(self, client, monkeypatch):
        # Trusted path still requires an org identity — dunetrace-cloud must send
        # x-org-id (or the legacy x-customer-id fallback). Without either, there's
        # nothing to stamp on the persisted events, so the request is rejected
        # rather than silently falling back to some default.
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")

        r = await client.post(
            "/v1/ingest",
            json=make_batch(api_key="not-a-dev-key"),
            headers={"x-internal-token": "shared-secret"},
        )
        assert r.status_code == 401

    async def test_trusted_header_accepts_legacy_customer_id_fallback(self, client, monkeypatch):
        # Pre-v0.5.0 cloud gateway builds send x-customer-id instead of x-org-id.
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")

        r = await client.post(
            "/v1/ingest",
            json=make_batch(api_key="not-a-dev-key"),
            headers={"x-internal-token": "shared-secret", "x-customer-id": "org-legacy"},
        )
        assert r.status_code == 202

    async def test_missing_trusted_header_still_enforces_api_keys_check(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value=None)
        )

        r = await client.post("/v1/ingest", json=make_batch(api_key="not-a-dev-key"))
        assert r.status_code == 401

    async def test_wrong_trusted_token_still_enforces_api_keys_check(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value=None)
        )

        r = await client.post(
            "/v1/ingest",
            json=make_batch(api_key="not-a-dev-key"),
            headers={"x-internal-token": "wrong-token"},
        )
        assert r.status_code == 401

    async def test_empty_internal_token_setting_never_trusts(self, client, monkeypatch):
        # INTERNAL_TOKEN unset (dev default "") — is_trusted() must return
        # False even if a client sends an empty x-internal-token header.
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value=None)
        )

        r = await client.post(
            "/v1/ingest",
            json=make_batch(api_key="not-a-dev-key"),
            headers={"x-internal-token": ""},
        )
        assert r.status_code == 401

    async def test_deploy_trusted_header_bypasses_api_keys_check(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")

        r = await client.post(
            "/v1/deploy",
            json={"api_key": "not-a-dev-key", "agent_id": "agent-xyz", "version": "v1.0.0"},
            headers={"x-internal-token": "shared-secret", "x-org-id": "org-trusted"},
        )
        assert r.status_code == 202

    async def test_deploy_missing_trusted_header_still_enforces_api_keys_check(
        self, client, monkeypatch
    ):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value=None)
        )

        r = await client.post(
            "/v1/deploy",
            json={"api_key": "not-a-dev-key", "agent_id": "agent-xyz", "version": "v1.0.0"},
        )
        assert r.status_code == 401

    async def test_deploy_trusted_response_uses_body_agent_id(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")

        r = await client.post(
            "/v1/deploy",
            json={"api_key": "not-a-dev-key", "agent_id": "trusted-agent", "version": "v1.0.0"},
            headers={"x-internal-token": "shared-secret", "x-org-id": "org-trusted"},
        )
        assert r.json()["agent_id"] == "trusted-agent"


# ── Rate limiting middleware (main.py::rate_limit_and_log) ─────────────────────
#
# C8 audit finding: the RateLimiter class itself was already well unit-tested
# (test_rate_limiter.py), but the HTTP-level wiring — which bucket a request
# lands in (api_key vs. IP), the dev-key IP-fallback, the trusted-path bypass,
# and the actual 429/Retry-After response shape — had zero coverage. These
# tests exercise that wiring through real HTTP requests against the app.


@pytest.fixture
def fresh_limiter(monkeypatch):
    """A small-rpm RateLimiter so tests trip the limit in a handful of
    requests instead of needing hundreds. Replaces the module-level
    singleton, which otherwise persists rate-limit windows across tests."""
    from ingest_svc.rate_limiter import RateLimiter

    limiter = RateLimiter(default_rpm=3)
    monkeypatch.setattr("ingest_svc.rate_limiter._limiter", limiter)
    monkeypatch.setattr("ingest_svc.main.get_limiter", lambda: limiter)
    return limiter


class TestRateLimitMiddleware:
    async def test_requests_within_limit_succeed(self, client, fresh_limiter):
        for _ in range(3):
            r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_a"))
            assert r.status_code == 202

    async def test_request_exceeding_limit_returns_429(self, client, fresh_limiter):
        for _ in range(3):
            await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_b"))
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_b"))
        assert r.status_code == 429

    async def test_429_has_retry_after_header(self, client, fresh_limiter):
        for _ in range(3):
            await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_c"))
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_c"))
        assert "retry-after" in {k.lower() for k in r.headers.keys()}
        assert int(r.headers["retry-after"]) >= 1

    async def test_429_has_detail_message(self, client, fresh_limiter):
        for _ in range(3):
            await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_d"))
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_d"))
        assert "detail" in r.json()

    async def test_real_api_keys_are_bucketed_independently(self, client, fresh_limiter):
        """A different (non-dev) key must get its own quota — exhausting one
        real key's limit must not affect another."""
        for _ in range(3):
            await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_e1"))
        exhausted = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_e1"))
        assert exhausted.status_code == 429

        other_key = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_e2"))
        assert other_key.status_code == 202

    async def test_dev_keys_are_bucketed_by_ip_not_by_key(self, client, fresh_limiter):
        """dt_dev_* keys share the IP-address bucket rather than a per-key
        one — two *different* dev keys from the same client must share one
        quota, unlike two different real keys (see test above)."""
        for _ in range(3):
            r = await client.post("/v1/ingest", json=make_batch(api_key="dt_dev_test_1"))
            assert r.status_code == 202
        # A *different* dev key, same client — must already be exhausted,
        # because dev keys are bucketed by IP, not by the key string.
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_dev_test_2"))
        assert r.status_code == 429

    async def test_missing_api_key_falls_back_to_ip_bucket(self, client, fresh_limiter):
        """An unparseable/missing api_key must not crash the middleware or
        skip rate limiting — it falls back to the IP bucket, same as dev keys."""
        body = {"agent_id": "agent-xyz", "events": [make_event()]}  # no api_key at all
        for _ in range(3):
            await client.post("/v1/ingest", json=body)
        r = await client.post("/v1/ingest", json=body)
        assert r.status_code == 429

    async def test_trusted_path_bypasses_rate_limiting_entirely(
        self, client, fresh_limiter, monkeypatch
    ):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")
        headers = {"x-internal-token": "shared-secret", "x-org-id": "org-trusted"}
        # 3 is the configured limit — a non-trusted caller would 429 on the 4th.
        for _ in range(6):
            r = await client.post(
                "/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_f"), headers=headers
            )
            assert r.status_code == 202

    async def test_deploy_endpoint_is_also_rate_limited(self, client, fresh_limiter):
        body = {"api_key": "dt_live_ratelimit_g", "agent_id": "agent-xyz", "version": "v1.0.0"}
        for _ in range(3):
            r = await client.post("/v1/deploy", json=body)
            assert r.status_code == 202
        r = await client.post("/v1/deploy", json=body)
        assert r.status_code == 429

    async def test_other_endpoints_are_not_rate_limited(self, client, fresh_limiter):
        """Only /v1/ingest and /v1/deploy are rate limited — e.g. /v1/policies
        must not be affected by an exhausted ingest quota."""
        for _ in range(3):
            await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_h"))
        exhausted = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_ratelimit_h"))
        assert exhausted.status_code == 429

        r = await client.get(
            "/v1/policies", params={"agent_id": "agent-xyz", "api_key": "dt_live_ratelimit_h"}
        )
        assert r.status_code != 429


# ── verify_api_key (real implementation, not the router-level mock) ────────────


class TestVerifyApiKeyDevMode:
    async def test_dev_mode_dt_dev_key_resolves_to_default_org(self, monkeypatch):
        from ingest_svc.db.postgres import verify_api_key

        monkeypatch.setattr("ingest_svc.db.postgres.settings.ENV", "dev")
        assert await verify_api_key("dt_dev_anything") == "default"

    async def test_dev_mode_empty_key_resolves_to_default_org(self, monkeypatch):
        from ingest_svc.db.postgres import verify_api_key

        monkeypatch.setattr("ingest_svc.db.postgres.settings.ENV", "dev")
        assert await verify_api_key("") == "default"

    async def test_non_dev_mode_dt_dev_key_is_not_special_cased(self, monkeypatch):
        # dt_dev_* is only a wildcard in dev mode (is_dev checks settings.ENV,
        # not AUTH_MODE). In prod, it's just a string that won't match any row
        # and correctly resolves to no org.
        from ingest_svc.db.postgres import verify_api_key

        monkeypatch.setattr("ingest_svc.db.postgres.settings.ENV", "production")
        monkeypatch.setattr("ingest_svc.db.postgres._pool", None)
        assert await verify_api_key("dt_dev_anything") is None


# ── Health ─────────────────────────────────────────────────────────────────────


class TestHealth:
    async def test_returns_200(self, client):
        r = await client.get("/health")
        assert r.status_code == 200

    async def test_shape(self, client):
        body = (await client.get("/health")).json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "db" in body

    async def test_db_status_reported(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.routers.health.check_db", AsyncMock(return_value="no_pool"))
        body = (await client.get("/health")).json()
        assert body["db"] == "no_pool"


# ── Partition management tests ─────────────────────────────────────────────────


class _FakeRecord:
    """Minimal asyncpg Record stand-in that supports dict-style access."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]


def _make_pool(fetchval_return, fetch_return=None, execute_side_effect=None):
    """Build a mock asyncpg pool whose acquire() context manager yields a mock conn."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=fetchval_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    if execute_side_effect:
        conn.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        conn.execute = AsyncMock()

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


@pytest.mark.asyncio
class TestPruneOldEvents:
    async def test_returns_zero_when_no_pool(self, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", None)
        from ingest_svc.db.postgres import prune_old_events

        result = await prune_old_events(retention_days=90)
        assert result == 0

    async def test_returns_zero_on_non_partitioned_table(self, monkeypatch):
        pool, conn = _make_pool(fetchval_return=0)  # relkind != 'p'
        monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
        from ingest_svc.db.postgres import prune_old_events

        result = await prune_old_events(retention_days=90)
        assert result == 0
        conn.execute.assert_not_called()

    async def test_drops_old_partition(self, monkeypatch):
        # events_202001 ends 2020-02-01 — always older than any reasonable retention
        pool, conn = _make_pool(
            fetchval_return=1,
            fetch_return=[_FakeRecord(relname="events_202001")],
        )
        monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
        from ingest_svc.db.postgres import prune_old_events

        dropped = await prune_old_events(retention_days=30)
        assert dropped == 1
        conn.execute.assert_called_once()
        call_sql = conn.execute.call_args[0][0]
        assert "events_202001" in call_sql

    async def test_keeps_recent_partition(self, monkeypatch):
        # events_209912 ends 2100-01-01 — always in the future
        pool, conn = _make_pool(
            fetchval_return=1,
            fetch_return=[_FakeRecord(relname="events_209912")],
        )
        monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
        from ingest_svc.db.postgres import prune_old_events

        dropped = await prune_old_events(retention_days=30)
        assert dropped == 0
        conn.execute.assert_not_called()

    async def test_drops_old_keeps_recent_mixed(self, monkeypatch):
        pool, conn = _make_pool(
            fetchval_return=1,
            fetch_return=[
                _FakeRecord(relname="events_202001"),  # old → drop
                _FakeRecord(relname="events_209901"),  # future → keep
            ],
        )
        monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
        from ingest_svc.db.postgres import prune_old_events

        dropped = await prune_old_events(retention_days=30)
        assert dropped == 1

    async def test_december_partition_end_wraps_to_january(self, monkeypatch):
        # events_202012 ends 2021-01-01 — should be treated as old
        pool, conn = _make_pool(
            fetchval_return=1,
            fetch_return=[_FakeRecord(relname="events_202012")],
        )
        monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
        from ingest_svc.db.postgres import prune_old_events

        dropped = await prune_old_events(retention_days=30)
        assert dropped == 1

    async def test_malformed_partition_name_skipped(self, monkeypatch):
        pool, conn = _make_pool(
            fetchval_return=1,
            fetch_return=[_FakeRecord(relname="events_badname")],
        )
        monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
        from ingest_svc.db.postgres import prune_old_events

        dropped = await prune_old_events(retention_days=30)
        assert dropped == 0
        conn.execute.assert_not_called()


# ── Retention scheduling (main.py::_run_prune_once) ────────────────────────────
#
# C9 audit finding: prune_old_events() existed and was tested in isolation
# (above) but was never called from anywhere in the running service — no
# cron loop, no scheduled task. These tests cover the scheduling glue itself:
# does it call through with the configured retention window, and does a
# failure get swallowed (logged, not crashed) so one bad tick doesn't kill
# the whole background loop.


class TestPruneScheduling:
    @pytest.fixture(autouse=True)
    def _reset_event_store(self):
        # These tests install throwaway custom stores via set_event_store();
        # reset to the default afterward so no other test in this file (or a
        # later-collected file) accidentally exercises a stale one.
        yield
        import ingest_svc.db.event_store as es_mod

        es_mod._store = None

    async def test_calls_event_store_with_configured_retention_days(self, monkeypatch):
        from ingest_svc.db.event_store import InMemoryEventStore, set_event_store
        from ingest_svc.main import _run_prune_once

        store = InMemoryEventStore()
        set_event_store(store)
        monkeypatch.setattr("ingest_svc.config.settings.EVENT_RETENTION_DAYS", 45)

        calls = []
        original_prune = store.prune_old_events

        async def _spy(retention_days):
            calls.append(retention_days)
            return await original_prune(retention_days)

        monkeypatch.setattr(store, "prune_old_events", _spy)

        await _run_prune_once()

        assert calls == [45]

    async def test_failure_is_logged_not_raised(self, monkeypatch, caplog):
        from ingest_svc.db.event_store import EventStore, set_event_store
        from ingest_svc.main import _run_prune_once

        class _FailingStore(EventStore):
            async def insert_events(self, events, batch_id, org_id):
                raise NotImplementedError

            async def prune_old_events(self, retention_days):
                raise RuntimeError("db unreachable")

        set_event_store(_FailingStore())

        import logging

        with caplog.at_level(logging.WARNING, logger="dunetrace.ingest"):
            await _run_prune_once()  # must not raise

        assert any("prune_old_events failed" in r.message for r in caplog.records)

    async def test_logs_info_when_partitions_are_dropped(self, monkeypatch, caplog):
        from ingest_svc.db.event_store import EventStore, set_event_store
        from ingest_svc.main import _run_prune_once

        class _DroppingStore(EventStore):
            async def insert_events(self, events, batch_id, org_id):
                raise NotImplementedError

            async def prune_old_events(self, retention_days):
                return 3

        set_event_store(_DroppingStore())

        import logging

        with caplog.at_level(logging.INFO, logger="dunetrace.ingest"):
            await _run_prune_once()

        assert any("Pruned 3 stale event partition" in r.message for r in caplog.records)


# ── OTLP receiver (D11 — production-readiness gaps) ────────────────────────────
#
# Prior coverage (test_otlp.py) was entirely the otel.py mapper in isolation —
# zero tests exercised the actual HTTP endpoint (auth, body parsing, content
# negotiation, rate limiting). These do.


def _protobuf_export_request_bytes(agent_id: str = "otlp-agent", span_name: str = "root") -> bytes:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue

    req = ExportTraceServiceRequest()
    rs = req.resource_spans.add()
    rs.resource.attributes.add(key="service.name", value=AnyValue(string_value=agent_id))
    ss = rs.scope_spans.add()
    span = ss.spans.add()
    span.trace_id = bytes.fromhex("0123456789abcdef0123456789abcdef")
    span.span_id = bytes.fromhex("0123456789abcdef")
    span.name = span_name
    now_ns = int(time.time() * 1e9)
    span.start_time_unix_nano = now_ns
    span.end_time_unix_nano = now_ns + 1_000_000
    return req.SerializeToString()


def _json_otlp_body(agent_id: str = "otlp-agent") -> dict:
    now_ns = str(int(time.time() * 1_000_000_000))
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": agent_id}}]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "0123456789abcdef0123456789abcdef",
                                "spanId": "0123456789abcdef",
                                "name": "root",
                                "startTimeUnixNano": now_ns,
                                "endTimeUnixNano": str(int(now_ns) + 1_000_000),
                            }
                        ]
                    }
                ],
            }
        ]
    }


class TestOtlpEndpoint:
    async def test_json_body_accepted(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        r = await client.post(
            "/v1/otlp/traces",
            json=_json_otlp_body(),
            headers={"Authorization": "Bearer dt_live_test"},
        )
        assert r.status_code == 200
        assert r.json() == {}

    async def test_protobuf_body_accepted(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        r = await client.post(
            "/v1/otlp/traces",
            content=_protobuf_export_request_bytes(),
            headers={
                "Authorization": "Bearer dt_live_test",
                "Content-Type": "application/x-protobuf",
            },
        )
        assert r.status_code == 200
        assert r.json() == {}

    async def test_gzip_json_body_accepted(self, client, monkeypatch):
        import gzip as _gzip
        import json as _json

        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        compressed = _gzip.compress(_json.dumps(_json_otlp_body()).encode())
        r = await client.post(
            "/v1/otlp/traces",
            content=compressed,
            headers={
                "Authorization": "Bearer dt_live_test",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        assert r.status_code == 200

    async def test_gzip_protobuf_body_accepted(self, client, monkeypatch):
        import gzip as _gzip

        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        compressed = _gzip.compress(_protobuf_export_request_bytes())
        r = await client.post(
            "/v1/otlp/traces",
            content=compressed,
            headers={
                "Authorization": "Bearer dt_live_test",
                "Content-Type": "application/x-protobuf",
                "Content-Encoding": "gzip",
            },
        )
        assert r.status_code == 200

    async def test_malformed_json_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        r = await client.post(
            "/v1/otlp/traces",
            content=b"{not valid json",
            headers={"Authorization": "Bearer dt_live_test", "Content-Type": "application/json"},
        )
        assert r.status_code == 400

    async def test_malformed_protobuf_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        r = await client.post(
            "/v1/otlp/traces",
            content=b"\xff\xff\xff not-a-real-protobuf-message",
            headers={
                "Authorization": "Bearer dt_live_test",
                "Content-Type": "application/x-protobuf",
            },
        )
        assert r.status_code == 400

    async def test_malformed_gzip_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        r = await client.post(
            "/v1/otlp/traces",
            content=b"not actually gzipped",
            headers={
                "Authorization": "Bearer dt_live_test",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        assert r.status_code == 400

    async def test_invalid_api_key_returns_401(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value=None))
        r = await client.post(
            "/v1/otlp/traces",
            json=_json_otlp_body(),
            headers={"Authorization": "Bearer dt_live_bad"},
        )
        assert r.status_code == 401

    async def test_trusted_gateway_path_accepted(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")
        r = await client.post(
            "/v1/otlp/traces",
            json=_json_otlp_body(),
            headers={"x-internal-token": "shared-secret", "x-org-id": "org-trusted"},
        )
        assert r.status_code == 200

    async def test_trusted_gateway_missing_org_id_returns_401(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.config.settings.INTERNAL_TOKEN", "shared-secret")
        r = await client.post(
            "/v1/otlp/traces",
            json=_json_otlp_body(),
            headers={"x-internal-token": "shared-secret"},
        )
        assert r.status_code == 401

    async def test_empty_resource_spans_returns_200_without_persisting(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        r = await client.post(
            "/v1/otlp/traces",
            json={"resourceSpans": []},
            headers={"Authorization": "Bearer dt_live_test"},
        )
        assert r.status_code == 200

    async def test_uses_event_store_abstraction(self, client, monkeypatch):
        """D11 fixed a C9 consistency gap: otlp.py previously called the free
        insert_events() function directly instead of going through
        get_event_store()."""
        from ingest_svc.db.event_store import InMemoryEventStore, set_event_store

        store = InMemoryEventStore()
        set_event_store(store)
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        try:
            r = await client.post(
                "/v1/otlp/traces",
                json=_json_otlp_body(),
                headers={"Authorization": "Bearer dt_live_test"},
            )
            assert r.status_code == 200
            # _persist_otlp runs as a BackgroundTask; httpx's ASGITransport
            # awaits it before the response context manager exits.
            assert len(store.batches) == 1
        finally:
            import ingest_svc.db.event_store as es_mod

            es_mod._store = None


class TestOtlpRateLimiting:
    @pytest.fixture
    def fresh_limiter(self, monkeypatch):
        from ingest_svc.rate_limiter import RateLimiter

        limiter = RateLimiter(default_rpm=3)
        monkeypatch.setattr("ingest_svc.rate_limiter._limiter", limiter)
        monkeypatch.setattr("ingest_svc.main.get_limiter", lambda: limiter)
        return limiter

    async def test_otlp_traces_is_rate_limited(self, client, fresh_limiter, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        for _ in range(3):
            r = await client.post(
                "/v1/otlp/traces",
                json=_json_otlp_body(),
                headers={"Authorization": "Bearer dt_live_otlp_a"},
            )
            assert r.status_code == 200
        r = await client.post(
            "/v1/otlp/traces",
            json=_json_otlp_body(),
            headers={"Authorization": "Bearer dt_live_otlp_a"},
        )
        assert r.status_code == 429

    async def test_otlp_bucket_key_comes_from_bearer_header_not_body(
        self, client, fresh_limiter, monkeypatch
    ):
        """OTLP auth is a header, not a JSON body field — the rate limiter
        must bucket by it correctly rather than falling back to IP for
        every request (which would make two different real keys share
        one quota)."""
        monkeypatch.setattr(
            "ingest_svc.routers.otlp.verify_api_key", AsyncMock(return_value="org-test")
        )
        for _ in range(3):
            await client.post(
                "/v1/otlp/traces",
                json=_json_otlp_body(),
                headers={"Authorization": "Bearer dt_live_otlp_b1"},
            )
        exhausted = await client.post(
            "/v1/otlp/traces",
            json=_json_otlp_body(),
            headers={"Authorization": "Bearer dt_live_otlp_b1"},
        )
        assert exhausted.status_code == 429

        other_key = await client.post(
            "/v1/otlp/traces",
            json=_json_otlp_body(),
            headers={"Authorization": "Bearer dt_live_otlp_b2"},
        )
        assert other_key.status_code == 200
