"""
Cross-package schema conformance test.

Builds a real ingest payload using the actual SDK client (packages/sdk-py)
and sends it through the real FastAPI /v1/ingest endpoint, so it validates
against the real ingest_svc Pydantic model (ingest_svc.schemas.IngestRequest).
Only the DB layer is mocked — nothing about request parsing/validation is.

This is the regression test called out in PR #4: a payload/schema mismatch
between the SDK and ingest_svc reached production because nothing exercised
both sides together. Every event type the SDK's EventType enum can produce
must be accepted here.

Run:
    cd services/ingest
    pytest tests/test_sdk_schema_compat.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock

# Make the SDK importable without installing it — same trick conftest.py in
# packages/sdk-py uses for `dunetrace`, just from the other side of the repo.
_SDK_PY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "sdk-py")
sys.path.insert(0, os.path.abspath(_SDK_PY))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dunetrace.client import DunetraceClient  # noqa: E402
from dunetrace.models import EventType  # noqa: E402
from dunetrace.policies import ApprovalDenied  # noqa: E402

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db(monkeypatch):
    monkeypatch.setattr("ingest_svc.db.postgres._pool", object())
    monkeypatch.setattr("ingest_svc.db.postgres.init_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.close_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.ensure_schema", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.check_db", AsyncMock(return_value="ok"))
    monkeypatch.setattr("ingest_svc.db.postgres.insert_events", AsyncMock(return_value=1))
    # Patched where routers/ingest.py actually looks it up — see the identical
    # comment in tests/test_ingest.py's mock_db fixture. Return value matches
    # the agent_id _record_a_full_run() below uses for every event.
    monkeypatch.setattr(
        "ingest_svc.routers.ingest.verify_api_key",
        AsyncMock(return_value="schema-compat-agent"),
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


def _record_a_full_run() -> list:
    """Drive the real SDK through every kind of event it can emit — base events, a
    run that errors out (run.errored only fires on an exception), the voice-pack
    events, the full approval lifecycle (requested/granted/denied/timeout), and a
    shipped policy-evaluation record — and return the exact AgentEvent batch it
    would have shipped to /v1/ingest.

    The approval backend and its clock-bound polling are stubbed so the whole
    lifecycle runs offline and instantly; policy_evaluation_reporting is on so a
    policy.evaluated record is shipped alongside policy.triggered."""
    emitted = []
    sdk_client = DunetraceClient(api_key="dt_dev_test", policy_evaluation_reporting=True)
    sdk_client._ship = lambda batch: emitted.extend(batch)

    # Stub the approval backend so request_approval runs fully offline. A mutable
    # holder drives the outcome (granted → denied → timeout) without real polling.
    approval = {"status": "granted"}
    sdk_client._create_approval_request = lambda **kw: {"id": 1}
    sdk_client._get_approval = lambda approval_id: {"status": approval["status"]}
    sdk_client._decide_approval = lambda approval_id, status: {"id": approval_id, "status": status}

    sdk_client.add_policy(
        name="tool-call-log",
        condition={"trigger": "tool_call_count", "operator": "gt", "value": 0},
        action={"type": "log"},
    )

    with sdk_client.run("schema-compat-agent", user_input="hi", model="gpt-4o") as run:
        run.llm_called("gpt-4o", prompt_tokens=10)
        run.llm_responded(completion_tokens=5, output_length=20)
        run.tool_called("search", {"query": "x"})
        run.tool_responded("search", success=True, output_length=5)  # triggers the policy above
        run.retrieval_called("docs")
        run.retrieval_responded("docs", result_count=3, top_score=0.9)
        run.external_signal("rate_limit", source="openai")

        # Voice-pack events. tts_generated carries the Phase 4.1 provider
        # correlation fields so the real ingest schema is exercised with them
        # present, proving the wire format tolerates the new optional keys.
        run.transcription_received("hello", confidence=0.95)
        run.tts_generated(
            "hi there",
            latency_ms=40,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model="eleven_multilingual_v2",
            provider="elevenlabs",
            provider_generation_id="hist_abc123",
        )
        run.voice_activity_detected("speech_start", duration_ms=100)
        run.turn_taking("agent_speaking")
        run.recording_metadata("https://s3.example.com/call.wav", storage_provider="s3")

        # Memory channel events.
        run.memory_written("user_prefs", "prefers concise answers", source="user_input")
        run.memory_read("user_prefs")
        run.memory_cleared("user_prefs")

        # Full approval lifecycle: granted, then denied, then timeout (the last
        # two raise ApprovalDenied, which we swallow — we only want the events).
        run.request_approval("wire_money", {"amt": 1}, timeout_s=300)  # granted
        approval["status"] = "denied"
        try:
            run.request_approval("wire_money", {"amt": 1}, timeout_s=300)  # denied
        except ApprovalDenied:
            pass
        approval["status"] = "pending"
        try:
            run.request_approval("wire_money", {"amt": 1}, timeout_s=0)  # times out at once
        except ApprovalDenied:
            pass

        run.final_answer()

    try:
        with sdk_client.run("schema-compat-agent", user_input="hi", model="gpt-4o"):
            raise ValueError("boom")
    except ValueError:
        pass

    sdk_client.shutdown(timeout=2)
    return emitted


def _build_ingest_payload(events: list) -> dict:
    """Same shape DunetraceClient._ship() sends over the wire."""
    return {
        "api_key": "dt_dev_test",
        "agent_id": events[0].agent_id if events else "",
        "events": [e.to_dict() for e in events],
    }


class TestSdkPayloadValidatesAgainstRealSchema:
    async def test_full_run_payload_is_accepted(self, client):
        events = _record_a_full_run()
        payload = _build_ingest_payload(events)

        r = await client.post("/v1/ingest", json=payload)

        assert r.status_code == 202, r.text
        assert r.json()["accepted"] == len(events)

    async def test_every_sdk_event_type_is_individually_accepted(self, client):
        """Every member of the SDK's EventType enum must be a value ingest_svc
        accepts — this is what would have caught policy.triggered being
        missing from ingest_svc's VALID_EVENT_TYPES."""
        events = _record_a_full_run()
        seen_types = {e.event_type.value for e in events}

        # Sanity: the recorded run actually exercised every event type the
        # SDK can produce, otherwise this test isn't testing what it claims.
        assert seen_types == {t.value for t in EventType}

        for event in events:
            r = await client.post(
                "/v1/ingest",
                json=_build_ingest_payload([event]),
            )
            assert r.status_code == 202, f"{event.event_type.value} rejected: {r.text}"

    async def test_policy_triggered_event_is_accepted(self, client):
        events = _record_a_full_run()
        policy_events = [e for e in events if e.event_type == EventType.POLICY_TRIGGERED]
        assert policy_events, "test setup didn't actually trigger a policy"

        r = await client.post("/v1/ingest", json=_build_ingest_payload(policy_events))
        assert r.status_code == 202, r.text

    async def test_unknown_future_event_type_does_not_reject_batch(self, client):
        # Forward compat (Capability 1): a newer SDK emitting an event type this
        # ingest build has never heard of must NOT poison the batch. A mix of a
        # known event and a genuinely-unknown one is accepted whole, so a rolling
        # deploy (SDK ahead of ingest) never silently loses the known events too.
        known = {
            "event_type": "tool.called",
            "run_id": "r1",
            "agent_id": "a1",
            "agent_version": "v1",
            "step_index": 1,
            "timestamp": 1.0,
            "payload": {"tool_name": "search"},
        }
        future = dict(known, event_type="future.capability.kind", step_index=2)
        payload = {"api_key": "dt_dev_test", "agent_id": "a1", "events": [known, future]}

        r = await client.post("/v1/ingest", json=payload)
        assert r.status_code == 202, r.text
        assert r.json()["accepted"] == 2  # both events, including the unknown one
