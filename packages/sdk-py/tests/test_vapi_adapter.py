"""
Tests for the Vapi webhook adapter (dunetrace.integrations.vapi).

Two layers:
  1. Adapter-to-SDK integration (hermetic): a schema-accurate Vapi call is
     replayed through the real Dunetrace client (shipper captured, no network),
     asserting the four voice events land with the right payloads and that one
     Vapi call becomes one run keyed by conversation_id = call id.
  2. Live Vapi API connectivity (opt-in): hits the real Vapi API to prove the
     credential and auth path. Skipped unless DUNETRACE_VAPI_LIVE=1 and a key is
     available, because it needs network.

Run: python -m unittest tests.test_vapi_adapter -v
"""

from __future__ import annotations

import os
import unittest

from dunetrace.client import DunetraceClient
from dunetrace.integrations.vapi import DunetraceVapiAdapter
from dunetrace.models import EventType

CALL_ID = "call_abc123"


def _msg(mtype: str, **fields) -> dict:
    """A Vapi server-message envelope: {"message": {"type", "call", ...}}."""
    body = {"type": mtype, "call": {"id": CALL_ID}}
    body.update(fields)
    return {"message": body}


class _CaptureClient:
    def __init__(self):
        self.client = DunetraceClient(api_key="dt_test", debug=False)
        self.emitted: list = []
        self.client._ship = lambda batch: self.emitted.extend(batch)

    def of(self, event_type: EventType):
        return [e for e in self.emitted if e.event_type == event_type]


class TestVapiAdapterMapping(unittest.TestCase):
    def _run_a_call(self) -> _CaptureClient:
        cap = _CaptureClient()
        adapter = DunetraceVapiAdapter(cap.client, agent_id="support-line", model="gpt-4o")
        # A realistic ordered Vapi call.
        adapter.handle_message(_msg("speech-update", status="started", role="customer"))
        adapter.handle_message(
            _msg("transcript", role="customer", transcriptType="partial", transcript="I want")
        )
        adapter.handle_message(
            _msg(
                "transcript",
                role="customer",
                transcriptType="final",
                transcript="I want to check my balance",
            )
        )
        adapter.handle_message(_msg("speech-update", status="started", role="assistant"))
        adapter.handle_message(
            _msg(
                "transcript",
                role="assistant",
                transcriptType="final",
                transcript="Sure, one moment please.",
            )
        )
        adapter.handle_message(_msg("user-interrupted"))
        adapter.handle_message(
            _msg("status-update", status="ended", endedReason="customer-ended-call")
        )
        cap.client.shutdown(timeout=2)
        return cap

    def test_one_call_is_one_run(self):
        cap = self._run_a_call()
        self.assertEqual(len(cap.of(EventType.RUN_STARTED)), 1)
        self.assertEqual(len(cap.of(EventType.RUN_COMPLETED)), 1)

    def test_conversation_id_is_call_id(self):
        cap = self._run_a_call()
        started = cap.of(EventType.RUN_STARTED)[0]
        self.assertEqual(started.conversation_id, CALL_ID)

    def test_user_transcript_maps_to_transcription_received(self):
        cap = self._run_a_call()
        tr = cap.of(EventType.TRANSCRIPTION_RECEIVED)
        self.assertEqual(len(tr), 1)  # partial ignored, only the final
        self.assertEqual(tr[0].payload["text"], "I want to check my balance")

    def test_assistant_transcript_maps_to_tts_generated(self):
        cap = self._run_a_call()
        tts = cap.of(EventType.TTS_GENERATED)
        self.assertEqual(len(tts), 1)
        self.assertEqual(tts[0].payload["text"], "Sure, one moment please.")

    def test_speech_updates_map_to_turn_taking(self):
        cap = self._run_a_call()
        turns = cap.of(EventType.TURN_TAKING)
        actions = [e.payload["action"] for e in turns]
        self.assertIn("user_speaking", actions)
        self.assertIn("agent_speaking", actions)

    def test_user_interrupted_maps_to_barge_in(self):
        cap = self._run_a_call()
        vad = cap.of(EventType.VOICE_ACTIVITY_DETECTED)
        types = [e.payload["type"] for e in vad]
        self.assertIn("barge_in", types)

    def test_ended_reason_captured_as_external_signal(self):
        # Call-metrics derives completion status from this; the raw reason is
        # recorded, natural-vs-dropped mapping happens server-side.
        cap = self._run_a_call()
        sigs = [
            e
            for e in cap.of(EventType.EXTERNAL_SIGNAL)
            if e.payload.get("signal_name") == "call_ended"
        ]
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].payload["meta"]["reason"], "customer-ended-call")


class TestVapiAdapterRobustness(unittest.TestCase):
    def test_unknown_message_type_is_ignored(self):
        cap = _CaptureClient()
        adapter = DunetraceVapiAdapter(cap.client, agent_id="a")
        adapter.handle_message(_msg("knowledge-base-request"))  # must not raise
        adapter.handle_message({"message": {"type": "some-future-type", "call": {"id": CALL_ID}}})

    def test_missing_envelope_is_ignored(self):
        cap = _CaptureClient()
        adapter = DunetraceVapiAdapter(cap.client, agent_id="a")
        adapter.handle_message({})  # no "message"
        adapter.handle_message({"message": None})
        adapter.handle_message(None)  # type: ignore[arg-type]

    def test_terminal_only_call_emits_no_run(self):
        """A call that only ever produced a terminal message (nothing buffered)
        does not open an empty run."""
        cap = _CaptureClient()
        adapter = DunetraceVapiAdapter(cap.client, agent_id="a")
        adapter.handle_message(_msg("status-update", status="ended"))
        cap.client.shutdown(timeout=2)
        self.assertEqual(len(cap.of(EventType.RUN_STARTED)), 0)

    def test_end_of_call_report_finalizes(self):
        cap = _CaptureClient()
        adapter = DunetraceVapiAdapter(cap.client, agent_id="a", model="gpt-4o")
        adapter.handle_message(
            _msg("transcript", role="customer", transcriptType="final", transcript="hello")
        )
        adapter.handle_message(_msg("end-of-call-report", endedReason="assistant-ended-call"))
        cap.client.shutdown(timeout=2)
        self.assertEqual(len(cap.of(EventType.RUN_COMPLETED)), 1)
        self.assertEqual(len(cap.of(EventType.TRANSCRIPTION_RECEIVED)), 1)

    def test_recording_auto_captured_from_end_of_call_report(self):
        # Vapi's end-of-call-report artifact carries the recording URL (Phase 2.3).
        cap = _CaptureClient()
        adapter = DunetraceVapiAdapter(cap.client, agent_id="a")
        adapter.handle_message(
            _msg("transcript", role="customer", transcriptType="final", transcript="hi")
        )
        adapter.handle_message(
            _msg(
                "end-of-call-report",
                endedReason="customer-ended-call",
                artifact={"recordingUrl": "https://storage.vapi.ai/call.wav"},
            )
        )
        cap.client.shutdown(timeout=2)
        recs = cap.of(EventType.RECORDING_AVAILABLE)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].payload["url"], "https://storage.vapi.ai/call.wav")
        self.assertEqual(recs[0].payload["storage_provider"], "vapi")


@unittest.skipUnless(
    os.environ.get("DUNETRACE_VAPI_LIVE") == "1" and os.environ.get("VAPI_PRIVATE_API_KEY"),
    "live Vapi API check (set DUNETRACE_VAPI_LIVE=1 and VAPI_PRIVATE_API_KEY)",
)
class TestVapiLiveApi(unittest.TestCase):
    def test_auth_and_list_calls(self):
        import urllib.request

        key = os.environ["VAPI_PRIVATE_API_KEY"]
        # A User-Agent is required: Vapi sits behind a WAF that 403s the default
        # Python-urllib agent.
        req = urllib.request.Request(
            "https://api.vapi.ai/call?limit=1",
            headers={"Authorization": f"Bearer {key}", "User-Agent": "dunetrace-vapi-adapter/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.assertEqual(resp.status, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
