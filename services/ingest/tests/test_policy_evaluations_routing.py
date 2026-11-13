"""
Ingest routing for policy.evaluated observability records (Phase 5).

_persist must split policy.evaluated events into the policy_evaluations sink and
keep every other event in the normal events stream — so run traces stay clean and
the dashboard endpoint has a dedicated table to read.

Run from services/ingest/ with:
  PYTHONPATH=../../packages/schemas-py:. python -m pytest tests/test_policy_evaluations_routing.py
"""

from __future__ import annotations

from types import SimpleNamespace

import json

import pytest


def _event(event_type_value, **payload):
    """Minimal event stand-in with the attributes _persist / the stores read."""
    return SimpleNamespace(
        event_type=SimpleNamespace(value=event_type_value),
        payload=payload,
        agent_id="billing",
        run_id="run-1",
        agent_version="v1",
        step_index=0,
        timestamp=1000.0,
        parent_run_id=None,
        trace_id=None,
        conversation_id=None,
        event_id=None,
    )


@pytest.fixture()
def _mem_store():
    from ingest_svc.db.event_store import InMemoryEventStore, set_event_store
    import ingest_svc.db.event_store as es_mod

    store = InMemoryEventStore()
    set_event_store(store)
    yield store
    es_mod._store = None


@pytest.mark.asyncio
async def test_policy_evaluated_routed_to_dedicated_sink(_mem_store):
    from ingest_svc.routers.ingest import _persist

    events = [
        _event("tool.called", tool_name="refund"),
        _event(
            "policy.evaluated",
            policy_name="refund-guard",
            policy_id=7,
            fired=False,
            conditions=[{"field_path": "args.amount", "result": False}],
        ),
        _event("tool.responded", success=True),
    ]
    await _persist(events, "batch-1", "org-1")

    # Trace events go to events; the policy.evaluated record does not.
    trace_types = [e.event_type.value for e in _mem_store.all_events]
    assert trace_types == ["tool.called", "tool.responded"]

    evals = _mem_store.all_policy_evaluations
    assert len(evals) == 1
    assert evals[0].payload["policy_name"] == "refund-guard"


@pytest.mark.asyncio
async def test_batch_with_only_evaluations_writes_no_events(_mem_store):
    from ingest_svc.routers.ingest import _persist

    await _persist([_event("policy.evaluated", policy_id=1)], "batch-2", "org-1")
    assert _mem_store.all_events == []
    assert len(_mem_store.all_policy_evaluations) == 1


@pytest.mark.asyncio
async def test_batch_without_evaluations_unchanged(_mem_store):
    from ingest_svc.routers.ingest import _persist

    await _persist([_event("tool.called")], "batch-3", "org-1")
    assert len(_mem_store.all_events) == 1
    assert _mem_store.all_policy_evaluations == []


# ── Postgres sink: policy_bundle_stale / policy_bundle_age_s ──────────────────
#
# The SDK has sent both fields on every policy.evaluated record since the
# policy-fetch resilience work (a stale bundle = the agent enforced policies
# older than its fetch TTL because the server was unreachable). Ingest used to
# drop them on the floor. These pin the row shape insert_policy_evaluations
# hands to asyncpg, without a database.


class _RecordingConn:
    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    async def executemany(self, sql, rows):
        self.calls.append((sql, list(rows)))


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


@pytest.fixture()
def _pg_sink(monkeypatch):
    from ingest_svc.db import postgres

    conn = _RecordingConn()
    monkeypatch.setattr(postgres, "_pool", _FakePool(conn))
    return conn


@pytest.mark.asyncio
async def test_bundle_fields_are_written_from_the_payload(_pg_sink):
    from ingest_svc.db import postgres

    n = await postgres.insert_policy_evaluations(
        [
            _event(
                "policy.evaluated",
                policy_id=7,
                policy_name="refund-guard",
                policy_bundle_stale=True,
                policy_bundle_age_s=42.5,
            )
        ],
        "batch-1",
        "org-1",
    )
    assert n == 1
    [(sql, rows)] = _pg_sink.calls
    assert "policy_bundle_stale" in sql and "policy_bundle_age_s" in sql
    assert "$13, $14" in sql
    (row,) = rows
    # 14 original columns + the 5 dry-run verdict columns from migration 14.
    assert len(row) == 19
    assert row[12] is True
    assert row[13] == 42.5


@pytest.mark.asyncio
async def test_dry_run_verdict_fields_are_written_from_the_payload(_pg_sink):
    """The dry-run verdict is the whole output of dry-run mode. If any of these
    five fails to reach the row, the dashboard cannot answer "what would this
    policy have done" and the feature is decorative."""
    from ingest_svc.db import postgres

    n = await postgres.insert_policy_evaluations(
        [
            _event(
                "policy.evaluated",
                policy_id=7,
                policy_name="cap-tools",
                fired=True,
                mode="dry_run",
                step_index=3,
                would_action_type="stop",
                would_action_params={"message": "too many tools"},
                matched_branch="tool_call_count",
            )
        ],
        "batch-1",
        "org-1",
    )
    assert n == 1
    [(sql, rows)] = _pg_sink.calls
    for col in ("mode", "step_index", "would_action_type", "would_action_params", "matched_branch"):
        assert col in sql, f"{col} missing from the INSERT"
    (row,) = rows
    assert row[14] == "dry_run"
    assert row[15] == 3
    assert row[16] == "stop"
    assert json.loads(row[17]) == {"message": "too many tools"}
    assert row[18] == "tool_call_count"


@pytest.mark.asyncio
async def test_dry_run_fields_missing_from_payload_write_null(_pg_sink):
    """An enforcing row, or one from an SDK predating dry run, knows none of
    these. NULL is the honest value; a default would make an old row look like
    one that really recorded them."""
    from ingest_svc.db import postgres

    n = await postgres.insert_policy_evaluations(
        [_event("policy.evaluated", policy_id=1, fired=True)], "batch-1", "org-1"
    )
    assert n == 1
    [(_sql, rows)] = _pg_sink.calls
    (row,) = rows
    assert row[14:19] == (None, None, None, None, None)


@pytest.mark.asyncio
async def test_bundle_fields_missing_from_payload_write_null(_pg_sink):
    """An SDK that predates the fields reports nothing — that must land as
    NULL, not as FALSE / 0.0, or an old SDK would read as 'bundle was fresh'."""
    from ingest_svc.db import postgres

    await postgres.insert_policy_evaluations(
        [_event("policy.evaluated", policy_id=1)], "batch-2", "org-1"
    )
    [(_, rows)] = _pg_sink.calls
    (row,) = rows
    assert row[12] is None
    assert row[13] is None


@pytest.mark.asyncio
async def test_bundle_fields_falsy_values_are_preserved(_pg_sink):
    """False and 0.0 are real reports (fresh bundle, fetched just now) and must
    not collapse into NULL via an `or` short-circuit."""
    from ingest_svc.db import postgres

    await postgres.insert_policy_evaluations(
        [
            _event(
                "policy.evaluated",
                policy_id=1,
                policy_bundle_stale=False,
                policy_bundle_age_s=0.0,
            ),
            # The SDK sends age None with stale=True when there has never been
            # a successful fetch: stale is known, age is not.
            _event(
                "policy.evaluated",
                policy_id=1,
                policy_bundle_stale=True,
                policy_bundle_age_s=None,
            ),
        ],
        "batch-3",
        "org-1",
    )
    [(_, rows)] = _pg_sink.calls
    assert rows[0][12] is False and rows[0][13] == 0.0
    assert rows[1][12] is True and rows[1][13] is None
