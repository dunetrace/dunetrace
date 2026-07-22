"""
Tests for Phase 1.1's voice event-emitting RunContext helpers
(transcription_received / tts_generated / voice_activity_detected /
turn_taking). No network — the client's shipper is replaced with a capturing
list.

The load-bearing assertion here isn't the payload shape (that's mechanical) —
it's the step-counter behavior. The always-on built-in detectors count steps
for every org regardless of whether the voice pack is active, so a voice hook
that advanced the counter on every VAD frame would false-fire
RunawayIterationDetector / StepCountInflationDetector on any real voice call.

Run: python -m unittest tests.test_voice_events -v
"""

from __future__ import annotations

import unittest

from dunetrace.client import DunetraceClient
from dunetrace.detectors import (
    RunawayIterationDetector,
    StepCountInflationDetector,
)
from dunetrace.models import EventType


def _make_client(**kwargs) -> DunetraceClient:
    defaults = dict(api_key="dt_test", debug=False)
    defaults.update(kwargs)
    return DunetraceClient(**defaults)


class _CapturingRun:
    """Context helper: opens a run against a client whose shipper is captured,
    exposing the emitted events after shutdown."""

    def __init__(self, **run_kwargs):
        self.client = _make_client()
        self.emitted = []
        self.client._ship = lambda batch: self.emitted.extend(batch)
        self._run_kwargs = run_kwargs

    def __enter__(self):
        self._cm = self.client.run("voice-agent", **self._run_kwargs)
        self.run = self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        self.client.shutdown(timeout=2)
        return False

    def events_of(self, event_type: EventType):
        return [e for e in self.emitted if e.event_type == event_type]


class TestTranscriptionReceived(unittest.TestCase):
    def test_emits_event_with_payload(self):
        with _CapturingRun() as h:
            h.run.transcription_received("hello there", confidence=0.87, latency_ms=120)
        evts = h.events_of(EventType.TRANSCRIPTION_RECEIVED)
        self.assertEqual(len(evts), 1)
        self.assertEqual(evts[0].payload["text"], "hello there")
        self.assertEqual(evts[0].payload["confidence"], 0.87)
        self.assertEqual(evts[0].payload["latency_ms"], 120)

    def test_advances_the_step_counter(self):
        with _CapturingRun() as h:
            before = h.run.step
            h.run.transcription_received("hi")
            after = h.run.step
        self.assertEqual(after, before + 1)

    def test_defaults(self):
        with _CapturingRun() as h:
            h.run.transcription_received("hi")
        p = h.events_of(EventType.TRANSCRIPTION_RECEIVED)[0].payload
        self.assertEqual(p["confidence"], 1.0)
        self.assertEqual(p["latency_ms"], 0)

    def test_audio_seconds_optional_and_omitted_when_zero(self):
        # Present when given (for STT cost), absent by default (backward compat).
        with _CapturingRun() as h:
            h.run.transcription_received("hi", audio_seconds=3.2)
            h.run.transcription_received("bye")
        evts = h.events_of(EventType.TRANSCRIPTION_RECEIVED)
        self.assertEqual(evts[0].payload["audio_seconds"], 3.2)
        self.assertNotIn("audio_seconds", evts[1].payload)


class TestTtsGenerated(unittest.TestCase):
    def test_emits_event_with_payload(self):
        with _CapturingRun() as h:
            h.run.tts_generated("your order shipped", latency_ms=90, truncated=True)
        evts = h.events_of(EventType.TTS_GENERATED)
        self.assertEqual(len(evts), 1)
        self.assertEqual(evts[0].payload["text"], "your order shipped")
        self.assertEqual(evts[0].payload["latency_ms"], 90)
        self.assertTrue(evts[0].payload["truncated"])

    def test_does_not_advance_the_step_counter(self):
        with _CapturingRun() as h:
            before = h.run.step
            h.run.tts_generated("done")
            after = h.run.step
        self.assertEqual(after, before)


class TestVoiceActivityDetected(unittest.TestCase):
    def test_emits_event_with_payload(self):
        with _CapturingRun() as h:
            h.run.voice_activity_detected("speech_start", duration_ms=0)
            h.run.voice_activity_detected("barge_in", duration_ms=340)
        evts = h.events_of(EventType.VOICE_ACTIVITY_DETECTED)
        self.assertEqual(len(evts), 2)
        self.assertEqual(evts[0].payload["type"], "speech_start")
        self.assertEqual(evts[1].payload["type"], "barge_in")
        self.assertEqual(evts[1].payload["duration_ms"], 340)

    def test_does_not_advance_the_step_counter(self):
        with _CapturingRun() as h:
            before = h.run.step
            for _ in range(50):
                h.run.voice_activity_detected("silence", duration_ms=20)
            after = h.run.step
        self.assertEqual(after, before)

    def test_all_valid_types_accepted(self):
        with _CapturingRun() as h:
            for t in ("speech_start", "speech_end", "silence", "barge_in"):
                h.run.voice_activity_detected(t)
        self.assertEqual(len(h.events_of(EventType.VOICE_ACTIVITY_DETECTED)), 4)

    def test_invalid_type_raises(self):
        with _CapturingRun() as h:
            with self.assertRaises(ValueError):
                h.run.voice_activity_detected("mumble")


class TestTurnTaking(unittest.TestCase):
    def test_emits_event_with_payload(self):
        with _CapturingRun() as h:
            h.run.turn_taking("agent_speaking", from_agent=True, to_user=False)
        evts = h.events_of(EventType.TURN_TAKING)
        self.assertEqual(len(evts), 1)
        self.assertEqual(evts[0].payload["action"], "agent_speaking")
        self.assertTrue(evts[0].payload["from_agent"])
        self.assertFalse(evts[0].payload["to_user"])

    def test_does_not_advance_the_step_counter(self):
        with _CapturingRun() as h:
            before = h.run.step
            h.run.turn_taking("both_speaking")
            after = h.run.step
        self.assertEqual(after, before)

    def test_all_valid_actions_accepted(self):
        with _CapturingRun() as h:
            for a in ("agent_speaking", "user_speaking", "both_speaking", "neither"):
                h.run.turn_taking(a)
        self.assertEqual(len(h.events_of(EventType.TURN_TAKING)), 4)

    def test_invalid_action_raises(self):
        with _CapturingRun() as h:
            with self.assertRaises(ValueError):
                h.run.turn_taking("whispering")


class TestBackwardCompat(unittest.TestCase):
    def test_existing_hooks_unaffected(self):
        """A run that mixes classic and voice hooks still emits the classic
        events with their existing step semantics unchanged."""
        with _CapturingRun() as h:
            h.run.llm_called("gpt-4o", prompt_tokens=100)
            h.run.llm_responded(finish_reason="stop", output="hi", output_length=2)
            h.run.tool_called("lookup", {"id": 1})
            h.run.tool_responded("lookup", success=True)
        types = [e.event_type for e in h.emitted]
        self.assertIn(EventType.LLM_CALLED, types)
        self.assertIn(EventType.TOOL_CALLED, types)
        self.assertIn(EventType.TOOL_RESPONDED, types)


class TestVoiceCallDoesNotFalseFireBuiltins(unittest.TestCase):
    """A long, healthy voice call — many VAD/turn-taking annotations, a normal
    number of actual turns — must not trip the always-on step/iteration
    detectors, which run for every org whether or not the voice pack is active.
    This is the whole reason the annotation hooks don't advance the counter."""

    def _build_voice_run_state(self):
        h = _CapturingRun()
        h.__enter__()
        run = h.run
        # 20 real turns; each turn: one transcription (advances), one llm
        # round-trip, one tts, and a burst of VAD/turn-taking annotations.
        for _ in range(20):
            run.transcription_received("user says something", confidence=0.9)
            run.turn_taking("user_speaking", to_user=True)
            for _ in range(8):
                run.voice_activity_detected("silence", duration_ms=25)
            run.llm_called("gpt-4o-realtime", prompt_tokens=200)
            run.llm_responded(finish_reason="stop", output="agent reply", output_length=11)
            run.turn_taking("agent_speaking", from_agent=True)
            run.tts_generated("agent reply", latency_ms=80)
        state = run.state
        state.current_step = run.step
        h.__exit__(None, None, None)
        return state

    def test_runaway_iteration_does_not_fire(self):
        state = self._build_voice_run_state()
        # 40 steps: only transcription_received and llm_called advance (2 per
        # turn x 20). The ~200 VAD/turn-taking annotation events emitted
        # alongside do NOT advance — so current_step stays under
        # STEP_THRESHOLD (50). Had annotations advanced, it would be ~260.
        self.assertEqual(state.current_step, 40)
        self.assertLess(state.current_step, RunawayIterationDetector.STEP_THRESHOLD)
        signal = RunawayIterationDetector().on_run_completion(state)
        self.assertIsNone(signal)

    def test_step_count_inflation_does_not_fire_against_a_normal_baseline(self):
        state = self._build_voice_run_state()
        # A baseline learned from prior similar voice calls (~40 real steps).
        # The annotation events must not inflate current_step past 2x it.
        state.baseline_p75_steps = 38.0
        signal = StepCountInflationDetector().on_run_completion(state)
        self.assertIsNone(signal)


class TestRecordingMetadata(unittest.TestCase):
    def test_emits_event_and_does_not_advance_step(self):
        with _CapturingRun() as h:
            before = h.run.step
            h.run.recording_metadata(
                "https://s3.example.com/call.wav",
                duration_seconds=42.5,
                format="wav",
                storage_provider="s3",
                start_offset_seconds=1.0,
            )
            after = h.run.step
        self.assertEqual(after, before)  # annotation, no step advance
        evts = h.events_of(EventType.RECORDING_AVAILABLE)
        self.assertEqual(len(evts), 1)
        p = evts[0].payload
        self.assertEqual(p["url"], "https://s3.example.com/call.wav")
        self.assertEqual(p["duration_seconds"], 42.5)
        self.assertEqual(p["storage_provider"], "s3")
        self.assertEqual(p["start_offset_seconds"], 1.0)

    def test_optional_fields_omitted_when_default(self):
        with _CapturingRun() as h:
            h.run.recording_metadata("https://x/y.mp3")
        p = h.events_of(EventType.RECORDING_AVAILABLE)[0].payload
        self.assertEqual(p["url"], "https://x/y.mp3")
        self.assertNotIn("duration_seconds", p)
        self.assertNotIn("storage_provider", p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
