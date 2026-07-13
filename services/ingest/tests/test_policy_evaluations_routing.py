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
