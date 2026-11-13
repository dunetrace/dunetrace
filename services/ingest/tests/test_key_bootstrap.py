"""
POST /v1/keys (ingest) — the bootstrap key can carry the admin scope.

The Customer API enforces scopes: every key-management, policy-write,
integration and org-settings route needs ``admin``, and its own POST /v1/keys
refuses to mint a scope the caller does not hold. On a fresh self-hosted
install the only key-minting path that does not itself need a key is this
service's POST /v1/keys, gated on ADMIN_API_KEY. Its INSERT used to omit the
``scopes`` column, so the column default (ARRAY['ingest']) always applied and
no admin key could ever exist.

Two layers here:

* Route tests — ``create_api_key`` is patched where the router binds it
  (``ingest_svc.routers.ingest.create_api_key``) with a fake that records the
  scopes it was handed and normalises them exactly as the real function does.
* DB tests — the real ``create_api_key`` against a mock asyncpg pool, asserting
  the INSERT carries the normalised scopes array.

DB is mocked — no Postgres needed. Reuses test_ingest.py's mock_db/client
fixtures and _make_pool helper.

Run:
    PYTHONPATH=packages/sdk-py:packages/schemas-py:services/ingest \
      python -m pytest services/ingest/tests/test_key_bootstrap.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_ingest import _make_pool, client, mock_db  # noqa: F401

from dunetrace_schemas.keys import KEY_PREFIX_LENGTH
from dunetrace_schemas.scopes import normalise

pytestmark = pytest.mark.asyncio

_ADMIN_KEY = "correct-admin-key"
_FAKE_KEY = "dt_FAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKE"


@pytest.fixture
def admin_env(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", _ADMIN_KEY)


@pytest.fixture
def fake_create(monkeypatch):
    """Stand-in for ingest_svc.db.postgres.create_api_key, patched where
    routers/ingest.py looks it up. Records every call and returns the same
    shape the real function does, normalising scopes the same way — so a route
    test asserts both what the route *asks for* and what the caller *sees*."""
    calls: list[dict] = []

    async def _fake(org_id, org_name=None, rate_limit_rpm=600, scopes=None):
        calls.append(
            {
                "org_id": org_id,
                "org_name": org_name,
                "rate_limit_rpm": rate_limit_rpm,
                "scopes": scopes,
            }
        )
        return {
            "key": _FAKE_KEY,
            "key_prefix": _FAKE_KEY[:KEY_PREFIX_LENGTH],
            "org_id": org_id,
            "org_name": org_name or org_id,
            "rate_limit_rpm": rate_limit_rpm,
            "scopes": list(normalise(scopes)),
        }

    monkeypatch.setattr("ingest_svc.routers.ingest.create_api_key", _fake)
    return calls


def _body(**overrides) -> dict:
    body = {"org_id": "org-fresh", "admin_key": _ADMIN_KEY}
    body.update(overrides)
    return body


# ── Route: default and explicit scopes ─────────────────────────────────────────


class TestBootstrapKeyScopes:
    async def test_omitted_scopes_mints_admin(self, client, admin_env, fake_create):
        """The whole point: with no scopes in the body, the bootstrap key is
        admin — the one credential a fresh install cannot get any other way."""
        resp = await client.post("/v1/keys", json=_body())
        assert resp.status_code == 201, resp.text
        assert resp.json()["scopes"] == ["admin"]
        assert fake_create[0]["scopes"] == ["admin"]

    async def test_explicit_ingest_mints_ingest_only(self, client, admin_env, fake_create):
        """An operator can still hand out a narrower key from here."""
        resp = await client.post("/v1/keys", json=_body(scopes=["ingest"]))
        assert resp.status_code == 201, resp.text
        assert resp.json()["scopes"] == ["ingest"]
        assert fake_create[0]["scopes"] == ["ingest"]

    async def test_explicit_admin_is_honoured(self, client, admin_env, fake_create):
        resp = await client.post("/v1/keys", json=_body(scopes=["admin"]))
        assert resp.status_code == 201, resp.text
        assert resp.json()["scopes"] == ["admin"]

    async def test_unknown_scope_names_are_dropped(self, client, admin_env, fake_create):
        """normalise() drops names it does not know; the route passes the
        caller's list through untouched and reports what survived."""
        resp = await client.post("/v1/keys", json=_body(scopes=["admin", "superuser"]))
        assert resp.status_code == 201, resp.text
        assert resp.json()["scopes"] == ["admin"]
        # The route does not pre-filter — normalisation is the DB layer's job,
        # so there is exactly one place that decides what a scope name means.
        assert fake_create[0]["scopes"] == ["admin", "superuser"]

    async def test_all_unknown_list_falls_back_to_ingest(self, client, admin_env, fake_create):
        """Per normalise(): nothing recognised -> DEFAULT_SCOPES, which is
        ingest — not admin. Asking for garbage does not escalate."""
        resp = await client.post("/v1/keys", json=_body(scopes=["superuser", "root"]))
        assert resp.status_code == 201, resp.text
        assert resp.json()["scopes"] == ["ingest"]

    async def test_empty_list_is_not_omitted(self, client, admin_env, fake_create):
        """``[]`` is an explicit (if odd) request, not an omission: it goes
        through normalise() and lands on ingest. Only a *missing* field means
        'give me the bootstrap admin key'."""
        resp = await client.post("/v1/keys", json=_body(scopes=[]))
        assert resp.status_code == 201, resp.text
        assert resp.json()["scopes"] == ["ingest"]
        assert fake_create[0]["scopes"] == []

    async def test_scopes_must_be_a_list(self, client, admin_env, fake_create):
        resp = await client.post("/v1/keys", json=_body(scopes="admin"))
        assert resp.status_code == 422
        assert fake_create == []


# ── Route: response shape and logging ──────────────────────────────────────────


class TestBootstrapKeyResponse:
    async def test_response_carries_key_prefix_and_scopes(self, client, admin_env, fake_create):
        resp = await client.post("/v1/keys", json=_body(org_name="Fresh Org", rate_limit_rpm=1200))
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["key"] == _FAKE_KEY
        assert data["key_prefix"] == _FAKE_KEY[:KEY_PREFIX_LENGTH]
        assert data["scopes"] == ["admin"]
        assert data["org_id"] == "org-fresh"
        assert data["org_name"] == "Fresh Org"
        assert "created_at" in data
        # org_name and rpm still reach the DB layer as before.
        assert fake_create[0]["org_name"] == "Fresh Org"
        assert fake_create[0]["rate_limit_rpm"] == 1200

    async def test_logs_scopes_and_prefix_never_the_key(
        self, client, admin_env, fake_create, caplog
    ):
        caplog.set_level(logging.INFO, logger="dunetrace.ingest")
        resp = await client.post("/v1/keys", json=_body(scopes=["ingest", "admin"]))
        assert resp.status_code == 201, resp.text

        created = [r for r in caplog.records if "API key created" in r.getMessage()]
        assert len(created) == 1, [r.getMessage() for r in caplog.records]
        msg = created[0].getMessage()
        assert "scopes=ingest,admin" in msg
        assert f"key_prefix={_FAKE_KEY[:KEY_PREFIX_LENGTH]}" in msg
        # The plaintext key is shown to the caller once and goes nowhere else.
        assert _FAKE_KEY not in msg
        assert all(_FAKE_KEY not in r.getMessage() for r in caplog.records)


# ── Route: admin gate is unchanged ─────────────────────────────────────────────


class TestBootstrapKeyAdminGate:
    async def test_wrong_admin_key_rejected(self, client, admin_env, fake_create):
        resp = await client.post("/v1/keys", json=_body(admin_key="wrong-key"))
        assert resp.status_code == 403
        assert fake_create == []

    async def test_no_admin_key_configured_rejects_everything(
        self, client, monkeypatch, fake_create
    ):
        """An unset ADMIN_API_KEY never matches — the endpoint that mints admin
        credentials is closed by default, not open by default."""
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        resp = await client.post("/v1/keys", json=_body(admin_key="anything"))
        assert resp.status_code == 403
        assert fake_create == []


# ── DB layer: postgres.py::create_api_key writes scopes ────────────────────────


def _txn_pool(monkeypatch):
    """_make_pool's conn plus a working ``async with conn.transaction():`` —
    an AsyncMock's method call returns a coroutine, which is not an async
    context manager."""
    pool, conn = _make_pool(fetchval_return=None)
    conn.transaction = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("ingest_svc.db.postgres._pool", pool)
    return conn


def _api_keys_insert(conn):
    """The (sql, *params) of the INSERT INTO api_keys call on a mock conn."""
    inserts = [c.args for c in conn.execute.call_args_list if "INSERT INTO api_keys" in c.args[0]]
    assert len(inserts) == 1, [c.args[0] for c in conn.execute.call_args_list]
    return inserts[0]


class TestCreateApiKeyWritesScopes:
    async def test_insert_carries_normalised_scopes(self, monkeypatch):
        """The bug: this INSERT omitted the scopes column, so the DB default
        (ARRAY['ingest']) always won."""
        conn = _txn_pool(monkeypatch)
        from ingest_svc.db.postgres import create_api_key

        created = await create_api_key("org-a", scopes=["Admin", "bogus", "admin"])

        sql, *params = _api_keys_insert(conn)
        assert "scopes" in sql
        assert params[-1] == ["admin"]  # lower-cased, de-duplicated, unknown dropped
        assert created["scopes"] == ["admin"]

    async def test_none_scopes_is_ingest_only_at_the_db_layer(self, monkeypatch):
        """The DB function is neutral — None means normalise()'s default, the
        same ingest-only default the Customer API's create_api_key has. Asking
        for admin is the bootstrap *route's* decision, not this function's."""
        conn = _txn_pool(monkeypatch)
        from ingest_svc.db.postgres import create_api_key

        created = await create_api_key("org-a")

        sql, *params = _api_keys_insert(conn)
        assert params[-1] == ["ingest"]
        assert created["scopes"] == ["ingest"]

    async def test_all_unknown_falls_back_to_ingest(self, monkeypatch):
        conn = _txn_pool(monkeypatch)
        from ingest_svc.db.postgres import create_api_key

        created = await create_api_key("org-a", scopes=["superuser"])

        _sql, *params = _api_keys_insert(conn)
        assert params[-1] == ["ingest"]
        assert created["scopes"] == ["ingest"]

    async def test_returns_plaintext_once_and_stores_only_the_hash(self, monkeypatch):
        from dunetrace_schemas.keys import hash_api_key, key_prefix

        conn = _txn_pool(monkeypatch)
        from ingest_svc.db.postgres import create_api_key

        created = await create_api_key("org-a", org_name="Org A", rate_limit_rpm=900, scopes=None)

        assert created["key"].startswith("dt_")
        assert created["key_prefix"] == key_prefix(created["key"])
        assert created["org_id"] == "org-a"
        assert created["org_name"] == "Org A"
        assert created["rate_limit_rpm"] == 900

        _sql, *params = _api_keys_insert(conn)
        assert params[0] == hash_api_key(created["key"])
        assert created["key"] not in params  # plaintext never reaches the DB
        assert params[1] == created["key_prefix"]
        assert params[2:4] == ["org-a", 900]

    async def test_raises_when_no_pool(self, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres._pool", None)
        from ingest_svc.db.postgres import create_api_key

        with pytest.raises(RuntimeError, match="pool not ready"):
            await create_api_key("org-a", scopes=["admin"])
