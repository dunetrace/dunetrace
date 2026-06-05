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
    monkeypatch.setattr(
        "ingest_svc.db.postgres.verify_api_key", AsyncMock(return_value="agent-123")
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
        "payload": {"tool_name": "web_search", "args_hash": "aabb"},
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
            "/v1/ingest", json=make_batch(events=[make_event(parent_run_id="parent-abc")])
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
            payload={"input_hash": "abc", "model": "gpt-4o", "tools": ["web_search"]},
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
        monkeypatch.setattr("ingest_svc.db.postgres.verify_api_key", AsyncMock(return_value=None))
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_bad"))
        assert r.status_code == 401

    async def test_401_has_detail(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres.verify_api_key", AsyncMock(return_value=None))
        r = await client.post("/v1/ingest", json=make_batch(api_key="dt_live_bad"))
        assert "detail" in r.json()

    async def test_valid_key_accepted(self, client):
        # mock_db fixture patches verify_api_key to return "agent-123"
        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 202


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
