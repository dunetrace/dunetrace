"""
POST /v1/ingest durability — a 202 is a promise.

Persistence used to run in a BackgroundTask after the 202 and _persist swallowed
every failure, so a DB outage looked like success to the SDK, which then
discarded its only copy of the events. Now the batch is committed before the
response is sent, and anything short of that is a 503 + Retry-After the SDK can
act on.

DB is mocked — no Postgres needed. Reuses test_ingest.py's mock_db/client
fixtures and payload helpers.

Run:
    PYTHONPATH=packages/sdk-py:packages/schemas-py:services/ingest \
      python -m pytest services/ingest/tests/test_ingest_durability.py -v
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_ingest import _make_pool, client, make_batch, make_event, mock_db  # noqa: F401

pytestmark = pytest.mark.asyncio

_FIXED_BATCH_ID = "batch-fixed-0000"


@pytest.fixture
def fixed_batch_id(monkeypatch):
    """Pin the route's uuid4 so a test can find its batch_id in the log — a 503
    carries no batch_id in the body (nothing was accepted), so the log line is
    the only place it appears."""
    monkeypatch.setattr("ingest_svc.routers.ingest.uuid.uuid4", lambda: _FIXED_BATCH_ID)
    return _FIXED_BATCH_ID


def _policy_eval_event(**overrides) -> dict:
    return make_event(
        event_type="policy.evaluated",
        payload={"policy_name": "refund-guard", "policy_id": 7, "fired": False, "conditions": []},
        **overrides,
    )


def _error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


# ── Store failures become 503, not a silent 202 ────────────────────────────────


class TestPersistFailureIs503:
    async def test_store_exception_returns_503_with_retry_after(
        self, client, monkeypatch, caplog, fixed_batch_id
    ):
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=RuntimeError("db down")),
        )
        with caplog.at_level(logging.ERROR, logger="dunetrace.ingest"):
            r = await client.post("/v1/ingest", json=make_batch())

        assert r.status_code == 503
        assert int(r.headers["Retry-After"]) >= 1
        assert "detail" in r.json()

    async def test_503_does_not_leak_the_exception_to_the_client(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=RuntimeError("password authentication failed for db-host-7")),
        )
        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 503
        assert "db-host-7" not in r.text
        assert "RuntimeError" not in r.text

    async def test_503_logs_error_with_batch_context(
        self, client, monkeypatch, caplog, fixed_batch_id
    ):
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=RuntimeError("db down")),
        )
        events = [make_event(step_index=i) for i in range(3)]
        with caplog.at_level(logging.ERROR, logger="dunetrace.ingest"):
            r = await client.post("/v1/ingest", json=make_batch(events=events))

        assert r.status_code == 503
        errors = _error_records(caplog)
        assert errors, "a rejected batch must be logged at ERROR"
        msg = errors[-1].getMessage()
        assert fixed_batch_id in msg
        assert "org_id=org-test" in msg
        assert "agent_id=agent-xyz" in msg
        assert "events=3" in msg
        assert "RuntimeError" in msg and "db down" in msg

    async def test_insert_shortfall_returns_503(self, client, monkeypatch, caplog, fixed_batch_id):
        # Three events sent, one row written, and nothing on disk to explain the
        # gap (the mock_db pool sentinel cannot answer the presence check).
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=1))
        events = [make_event(step_index=i) for i in range(3)]
        with caplog.at_level(logging.ERROR, logger="dunetrace.ingest"):
            r = await client.post("/v1/ingest", json=make_batch(events=events))

        assert r.status_code == 503
        assert int(r.headers["Retry-After"]) >= 1
        msg = _error_records(caplog)[-1].getMessage()
        assert fixed_batch_id in msg
        assert "inserted=1" in msg and "expected=3" in msg

    async def test_zero_rows_written_returns_503(self, client, monkeypatch):
        # PostgresEventStore returns 0 (not an exception) when the pool is gone
        # or the INSERT fails — that must not read as success.
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=0))
        r = await client.post("/v1/ingest", json=make_batch())
        assert r.status_code == 503

    async def test_insert_policy_evaluations_exception_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_policy_evaluations",
            AsyncMock(side_effect=RuntimeError("policy_evaluations unavailable")),
        )
        insert_events = AsyncMock(side_effect=lambda events, batch_id, org_id: len(events))
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", insert_events)

        r = await client.post(
            "/v1/ingest",
            json=make_batch(events=[_policy_eval_event(), make_event(step_index=2)]),
        )
        assert r.status_code == 503
        assert "Retry-After" in r.headers
        # Evaluations are written first; the failure stops the batch there.
        insert_events.assert_not_awaited()

    async def test_policy_evaluations_shortfall_is_logged_not_fatal(
        self, client, monkeypatch, caplog
    ):
        # The policy_evaluations sink is best-effort by design (its EventStore
        # default is a documented no-op returning 0): a row shortfall there is
        # visible at WARNING but never rejects the run events alongside it.
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_policy_evaluations", AsyncMock(return_value=0)
        )
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=lambda events, batch_id, org_id: len(events)),
        )
        with caplog.at_level(logging.WARNING, logger="dunetrace.ingest"):
            r = await client.post(
                "/v1/ingest",
                json=make_batch(events=[_policy_eval_event(), make_event(step_index=2)]),
            )
        assert r.status_code == 202
        assert any("policy_evaluations shortfall" in rec.getMessage() for rec in caplog.records)


# ── Happy path: schema unchanged, persistence in the request path ──────────────


class TestHappyPathUnchanged:
    async def test_202_with_unchanged_body(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=lambda events, batch_id, org_id: len(events)),
        )
        events = [make_event(step_index=i) for i in range(4)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))

        assert r.status_code == 202
        body = r.json()
        assert set(body) == {"accepted", "batch_id", "queued_at"}
        assert body["accepted"] == 4
        assert "Retry-After" not in r.headers

    async def test_persist_is_awaited_before_the_response(self, client, monkeypatch):
        persisted = AsyncMock(side_effect=lambda events, batch_id, org_id: len(events))
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", persisted)

        r = await client.post("/v1/ingest", json=make_batch())

        assert r.status_code == 202
        persisted.assert_awaited_once()
        # The batch_id the client got is the one the store was given: the
        # response was built from a completed write, not ahead of it.
        assert persisted.await_args.args[1] == r.json()["batch_id"]

    async def test_route_no_longer_schedules_a_background_task(self):
        """Structural guard: the old design took a BackgroundTasks dependency
        and persisted after the response. Persistence must be in-path."""
        from fastapi import BackgroundTasks

        from ingest_svc.routers.ingest import ingest

        params = inspect.signature(ingest).parameters.values()
        assert not any(p.annotation is BackgroundTasks for p in params)

    async def test_deploy_endpoint_unchanged(self, client, monkeypatch):
        # /v1/deploy has always written synchronously; it is outside this contract.
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.insert_deploy_event", AsyncMock(return_value=11)
        )
        r = await client.post(
            "/v1/deploy",
            json={"api_key": "dt_dev_test", "agent_id": "agent-xyz", "version": "v1"},
        )
        assert r.status_code == 202
        assert r.json()["id"] == 11


# ── A re-sent batch that is already stored is accepted, not 503'd forever ──────
#
# PostgresEventStore.insert_events drops events whose event_id is already in
# `events` and returns only the rows it wrote, so the re-send of a batch whose
# 202 was lost in transit comes back short (often 0). Both SDKs stamp event_id
# on every event and DurableRetryEmitter stops at its first failing batch —
# treating that re-send as a shortfall would wedge the whole queue behind it.


class TestResendOfStoredBatch:
    def _events_with_ids(self, n: int) -> list[dict]:
        return [make_event(step_index=i, event_id=f"evt-{i}") for i in range(n)]

    async def test_fully_deduplicated_resend_is_202(self, client, monkeypatch, caplog):
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=0))
        pool, conn = _make_pool(fetchval_return=3)
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: pool)

        with caplog.at_level(logging.INFO, logger="dunetrace.ingest"):
            r = await client.post("/v1/ingest", json=make_batch(events=self._events_with_ids(3)))

        assert r.status_code == 202
        assert r.json()["accepted"] == 3
        assert any("re-send" in rec.getMessage() for rec in caplog.records)
        # The presence check is scoped to the caller's org.
        sql, args = conn.fetchval.await_args.args[0], conn.fetchval.await_args.args[1:]
        assert "org_id = $1" in sql
        assert args[0] == "org-test"
        assert sorted(args[1]) == ["evt-0", "evt-1", "evt-2"]

    async def test_partially_deduplicated_resend_is_202(self, client, monkeypatch):
        # Two of three were stored last time; this attempt wrote the third.
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=1))
        pool, _ = _make_pool(fetchval_return=3)
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: pool)

        r = await client.post("/v1/ingest", json=make_batch(events=self._events_with_ids(3)))
        assert r.status_code == 202

    async def test_shortfall_with_rows_actually_missing_is_503(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=0))
        pool, _ = _make_pool(fetchval_return=2)  # only 2 of 3 on disk
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: pool)

        r = await client.post("/v1/ingest", json=make_batch(events=self._events_with_ids(3)))
        assert r.status_code == 503

    async def test_shortfall_with_an_event_lacking_event_id_fails_closed(self, client, monkeypatch):
        # An id-less event (older client) can never have been deduplicated, so
        # a short count with one in the batch is a real loss — even if the DB
        # says every *id-bearing* event is present.
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=0))
        pool, conn = _make_pool(fetchval_return=2)
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: pool)

        events = self._events_with_ids(2) + [make_event(step_index=9)]
        r = await client.post("/v1/ingest", json=make_batch(events=events))
        assert r.status_code == 503
        conn.fetchval.assert_not_awaited()

    async def test_presence_check_failure_fails_closed(self, client, monkeypatch):
        monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=0))
        pool, conn = _make_pool(fetchval_return=None)
        conn.fetchval = AsyncMock(side_effect=RuntimeError("db down"))
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: pool)

        r = await client.post("/v1/ingest", json=make_batch(events=self._events_with_ids(2)))
        assert r.status_code == 503

    async def test_presence_check_is_not_run_on_the_happy_path(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.db.postgres.insert_events",
            AsyncMock(side_effect=lambda events, batch_id, org_id: len(events)),
        )
        pool, conn = _make_pool(fetchval_return=3)
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: pool)

        r = await client.post("/v1/ingest", json=make_batch(events=self._events_with_ids(3)))
        assert r.status_code == 202
        conn.fetchval.assert_not_awaited()


# ── _persist() directly: raises PersistError instead of swallowing ─────────────


def _ns_event(event_type_value: str, event_id: str | None = None) -> SimpleNamespace:
    """Same stand-in test_policy_evaluations_routing.py uses."""
    return SimpleNamespace(
        event_type=SimpleNamespace(value=event_type_value),
        payload={},
        agent_id="billing",
        run_id="run-1",
        agent_version="v1",
        step_index=0,
        timestamp=1000.0,
        parent_run_id=None,
        trace_id=None,
        conversation_id=None,
        event_id=event_id,
    )


@pytest.fixture
def _reset_store():
    yield
    import ingest_svc.db.event_store as es_mod

    es_mod._store = None


class TestPersistRaises:
    async def test_store_exception_becomes_persist_error_with_cause(
        self, _reset_store, monkeypatch
    ):
        from ingest_svc.db.event_store import EventStore, set_event_store
        from ingest_svc.routers.ingest import PersistError, _persist

        class _Boom(EventStore):
            async def insert_events(self, events, batch_id, org_id):
                raise ConnectionResetError("socket closed")

            async def prune_old_events(self, retention_days):
                return 0

        set_event_store(_Boom())
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: None)

        with pytest.raises(PersistError) as info:
            await _persist([_ns_event("tool.called")], "b-1", "org-1")
        assert isinstance(info.value.__cause__, ConnectionResetError)
        assert "ConnectionResetError" in str(info.value)

    async def test_shortfall_becomes_persist_error(self, _reset_store, monkeypatch):
        from ingest_svc.db.event_store import EventStore, set_event_store
        from ingest_svc.routers.ingest import PersistError, _persist

        class _Short(EventStore):
            async def insert_events(self, events, batch_id, org_id):
                return len(events) - 1

            async def prune_old_events(self, retention_days):
                return 0

        set_event_store(_Short())
        monkeypatch.setattr("ingest_svc.routers.ingest.get_pool", lambda: None)

        with pytest.raises(PersistError, match="inserted=1 expected=2"):
            await _persist([_ns_event("tool.called"), _ns_event("tool.responded")], "b-2", "org-1")

    async def test_policy_evaluation_sink_exception_becomes_persist_error(
        self, _reset_store, monkeypatch
    ):
        from ingest_svc.db.event_store import InMemoryEventStore, set_event_store
        from ingest_svc.routers.ingest import PersistError, _persist

        store = InMemoryEventStore()

        async def _boom(events, batch_id, org_id):
            raise RuntimeError("policy_evaluations unavailable")

        monkeypatch.setattr(store, "insert_policy_evaluations", _boom)
        set_event_store(store)

        with pytest.raises(PersistError, match="policy_evaluations unavailable"):
            await _persist(
                [_ns_event("policy.evaluated"), _ns_event("tool.called")], "b-3", "org-1"
            )
        assert store.all_events == []  # stopped before the run events

    async def test_full_write_returns_normally(self, _reset_store):
        from ingest_svc.db.event_store import InMemoryEventStore, set_event_store
        from ingest_svc.routers.ingest import _persist

        store = InMemoryEventStore()
        set_event_store(store)

        await _persist([_ns_event("tool.called"), _ns_event("policy.evaluated")], "b-4", "org-1")
        assert len(store.all_events) == 1
        assert len(store.all_policy_evaluations) == 1
