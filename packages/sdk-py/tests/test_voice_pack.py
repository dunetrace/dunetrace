"""
Tests for the voice detector pack (Phase 1.2). Each detector reads voice
events out of state.events, so the tests build a RunState and append
AgentEvents directly — the same shape build_run_state and RunContext both
produce.

Run: python -m unittest tests.test_voice_pack -v
"""

from __future__ import annotations

import unittest

from dunetrace.detectors import CUSTOM_DETECTOR_REGISTRY
from dunetrace.models import AgentEvent, EventType, FailureType, RunState, Severity
from dunetrace.packs import PACK_REGISTRY
from dunetrace.packs.voice import (
    VoiceAudioQualityDegradationDetector,
    VoiceBargeInFailureDetector,
    VoiceLatencyInducedHangupDetector,
    VoiceSilenceTimeoutDetector,
    VoiceSpeakerConfusionDetector,
    VoiceTranscriptionConfidenceDropDetector,
    VoiceTtsTruncationDetector,
    VoiceTurnTakingCollisionDetector,
    VoiceVadFalseTriggerDetector,
)

_ALL_DETECTORS = [
    VoiceTranscriptionConfidenceDropDetector,
    VoiceSilenceTimeoutDetector,
    VoiceTurnTakingCollisionDetector,
    VoiceLatencyInducedHangupDetector,
    VoiceAudioQualityDegradationDetector,
    VoiceSpeakerConfusionDetector,
    VoiceBargeInFailureDetector,
    VoiceTtsTruncationDetector,
    VoiceVadFalseTriggerDetector,
]


def _state() -> RunState:
    return RunState(run_id="r1", agent_id="voice-agent", agent_version="v1")


def _evt(state: RunState, event_type: EventType, payload: dict) -> None:
    step = len(state.events)
    state.events.append(
        AgentEvent(
            event_type=event_type,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=step,
            payload=payload,
        )
    )
    state.current_step = step


def _transcription(state, text="hello", confidence=1.0, latency_ms=0):
    _evt(
        state,
        EventType.TRANSCRIPTION_RECEIVED,
        {"text": text, "confidence": confidence, "latency_ms": latency_ms},
    )


def _tts(state, text="ok", latency_ms=0, truncated=False):
    _evt(
        state,
        EventType.TTS_GENERATED,
        {"text": text, "latency_ms": latency_ms, "truncated": truncated},
    )


def _vad(state, type_, duration_ms=0):
    _evt(state, EventType.VOICE_ACTIVITY_DETECTED, {"type": type_, "duration_ms": duration_ms})


def _turn(state, action, from_agent=False, to_user=False):
    _evt(
        state,
        EventType.TURN_TAKING,
        {"action": action, "from_agent": from_agent, "to_user": to_user},
    )


# ── Pack-wide invariants ──────────────────────────────────────────────────────


class TestVoicePackShape(unittest.TestCase):
    def test_registered_with_nine_detectors(self):
        self.assertIn("voice", PACK_REGISTRY)
        self.assertEqual(len(PACK_REGISTRY["voice"].detectors), 9)

    def test_every_detector_marks_pack_voice(self):
        for d in _ALL_DETECTORS:
            self.assertEqual(d.pack, "voice")

    def test_no_detector_leaked_into_custom_registry(self):
        for d in _ALL_DETECTORS:
            self.assertNotIn(d.__name__, CUSTOM_DETECTOR_REGISTRY)

    def test_all_names_unique(self):
        names = [d.name for d in _ALL_DETECTORS]
        self.assertEqual(len(set(names)), len(names))

    def test_signals_use_custom_type_and_carry_detector_name(self):
        # Drive one detector to fire and inspect the signal envelope.
        state = _state()
        _transcription(state, confidence=0.1)
        _transcription(state, confidence=0.2)
        sig = VoiceTranscriptionConfidenceDropDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.failure_type, FailureType.CUSTOM)
        self.assertEqual(sig.evidence["detector_name"], "VOICE_TRANSCRIPTION_CONFIDENCE_DROP")

    def test_all_detectors_silent_on_empty_run(self):
        state = _state()
        for d in _ALL_DETECTORS:
            self.assertIsNone(d().on_run_completion(state), d.name)

    def test_all_detectors_silent_on_non_voice_run(self):
        # A normal LLM/tool run that happens to have the pack activated.
        state = _state()
        _evt(state, EventType.LLM_CALLED, {"model": "gpt-4o"})
        _evt(state, EventType.LLM_RESPONDED, {"output": "hi"})
        for d in _ALL_DETECTORS:
            self.assertIsNone(d().on_run_completion(state), d.name)


# ── 1. Confidence drop ────────────────────────────────────────────────────────


class TestConfidenceDrop(unittest.TestCase):
    def test_fires_on_repeated_low_confidence(self):
        state = _state()
        _transcription(state, confidence=0.3)
        _transcription(state, confidence=0.4)
        sig = VoiceTranscriptionConfidenceDropDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.evidence["low_confidence_count"], 2)

    def test_silent_below_count(self):
        state = _state()
        _transcription(state, confidence=0.3)
        _transcription(state, confidence=0.95)
        self.assertIsNone(VoiceTranscriptionConfidenceDropDetector().on_run_completion(state))

    def test_silent_when_all_high_confidence(self):
        state = _state()
        _transcription(state, confidence=0.99)
        _transcription(state, confidence=0.98)
        self.assertIsNone(VoiceTranscriptionConfidenceDropDetector().on_run_completion(state))


# ── 2. Silence timeout ────────────────────────────────────────────────────────


class TestSilenceTimeout(unittest.TestCase):
    def test_fires_on_long_silence(self):
        state = _state()
        _vad(state, "silence", duration_ms=12000)
        sig = VoiceSilenceTimeoutDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.evidence["max_silence_ms"], 12000)

    def test_silent_on_short_silence(self):
        state = _state()
        _vad(state, "silence", duration_ms=2000)
        self.assertIsNone(VoiceSilenceTimeoutDetector().on_run_completion(state))

    def test_ignores_non_silence_vad(self):
        state = _state()
        _vad(state, "speech_start", duration_ms=12000)
        self.assertIsNone(VoiceSilenceTimeoutDetector().on_run_completion(state))


# ── 3. Turn-taking collision ──────────────────────────────────────────────────


class TestTurnTakingCollision(unittest.TestCase):
    def test_fires_on_repeated_both_speaking(self):
        state = _state()
        _turn(state, "both_speaking")
        _turn(state, "both_speaking")
        sig = VoiceTurnTakingCollisionDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.evidence["collision_count"], 2)

    def test_silent_on_single_collision(self):
        state = _state()
        _turn(state, "both_speaking")
        _turn(state, "agent_speaking")
        self.assertIsNone(VoiceTurnTakingCollisionDetector().on_run_completion(state))


# ── 4. Latency-induced hangup ─────────────────────────────────────────────────


class TestLatencyInducedHangup(unittest.TestCase):
    def test_fires_when_last_tts_is_slow(self):
        state = _state()
        _transcription(state, latency_ms=100)
        _tts(state, latency_ms=9000)  # last voice event, slow, nothing after
        sig = VoiceLatencyInducedHangupDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.evidence["final_tts_latency_ms"], 9000)

    def test_silent_when_user_replied_after_slow_tts(self):
        state = _state()
        _tts(state, latency_ms=9000)
        _transcription(state, text="still here")  # user took their turn back
        self.assertIsNone(VoiceLatencyInducedHangupDetector().on_run_completion(state))

    def test_silent_when_last_tts_fast(self):
        state = _state()
        _transcription(state)
        _tts(state, latency_ms=200)
        self.assertIsNone(VoiceLatencyInducedHangupDetector().on_run_completion(state))


# ── 5. Audio quality degradation ──────────────────────────────────────────────


class TestAudioQualityDegradation(unittest.TestCase):
    def test_fires_on_downward_trend(self):
        state = _state()
        for c in (0.95, 0.9, 0.5, 0.4):
            _transcription(state, confidence=c)
        sig = VoiceAudioQualityDegradationDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertGreaterEqual(sig.evidence["drop"], 0.2)

    def test_silent_on_stable_confidence(self):
        state = _state()
        for c in (0.9, 0.92, 0.89, 0.91):
            _transcription(state, confidence=c)
        self.assertIsNone(VoiceAudioQualityDegradationDetector().on_run_completion(state))

    def test_silent_below_min_samples(self):
        state = _state()
        _transcription(state, confidence=0.95)
        _transcription(state, confidence=0.3)
        self.assertIsNone(VoiceAudioQualityDegradationDetector().on_run_completion(state))


# ── 6. Speaker confusion ──────────────────────────────────────────────────────


class TestSpeakerConfusion(unittest.TestCase):
    def test_fires_on_transcription_during_agent_speaking(self):
        state = _state()
        _turn(state, "agent_speaking", from_agent=True)
        _transcription(state, text="agent hearing itself")
        sig = VoiceSpeakerConfusionDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.evidence["self_heard_count"], 1)

    def test_silent_when_transcription_during_user_speaking(self):
        state = _state()
        _turn(state, "user_speaking", to_user=True)
        _transcription(state, text="real user input")
        self.assertIsNone(VoiceSpeakerConfusionDetector().on_run_completion(state))

    def test_silent_with_no_turn_context(self):
        state = _state()
        _transcription(state)
        self.assertIsNone(VoiceSpeakerConfusionDetector().on_run_completion(state))


# ── 7. Barge-in failure ───────────────────────────────────────────────────────


class TestBargeInFailure(unittest.TestCase):
    def test_fires_when_agent_keeps_talking_after_barge_in(self):
        state = _state()
        _vad(state, "barge_in", duration_ms=300)
        _tts(state, text="agent ignored interruption", truncated=False)
        sig = VoiceBargeInFailureDetector().on_run_completion(state)
        self.assertIsNotNone(sig)

    def test_silent_when_tts_truncated_after_barge_in(self):
        state = _state()
        _vad(state, "barge_in", duration_ms=300)
        _tts(state, truncated=True)  # correctly yielded
        self.assertIsNone(VoiceBargeInFailureDetector().on_run_completion(state))

    def test_silent_with_no_barge_in(self):
        state = _state()
        _tts(state, truncated=False)
        self.assertIsNone(VoiceBargeInFailureDetector().on_run_completion(state))


# ── 8. TTS truncation ─────────────────────────────────────────────────────────


class TestTtsTruncation(unittest.TestCase):
    def test_fires_on_unexplained_truncation(self):
        state = _state()
        _tts(state, truncated=True)
        sig = VoiceTtsTruncationDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.evidence["truncation_count"], 1)

    def test_does_not_count_barge_in_caused_truncation(self):
        # A truncation immediately after a barge-in is legitimate — that's
        # BARGE_IN_FAILURE's (non-)domain, not an unexplained cutoff.
        state = _state()
        _vad(state, "barge_in", duration_ms=300)
        _tts(state, truncated=True)
        self.assertIsNone(VoiceTtsTruncationDetector().on_run_completion(state))

    def test_silent_when_no_truncation(self):
        state = _state()
        _tts(state, truncated=False)
        self.assertIsNone(VoiceTtsTruncationDetector().on_run_completion(state))


# ── 9. VAD false trigger ──────────────────────────────────────────────────────


class TestVadFalseTrigger(unittest.TestCase):
    def test_fires_on_repeated_short_triggers(self):
        state = _state()
        for _ in range(3):
            _vad(state, "speech_start", duration_ms=40)
        sig = VoiceVadFalseTriggerDetector().on_run_completion(state)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.evidence["false_trigger_count"], 3)

    def test_silent_below_threshold(self):
        state = _state()
        _vad(state, "speech_start", duration_ms=40)
        _vad(state, "speech_start", duration_ms=40)
        self.assertIsNone(VoiceVadFalseTriggerDetector().on_run_completion(state))

    def test_zero_duration_not_counted_as_false_trigger(self):
        # duration_ms 0 = unknown, not a zero-length blip.
        state = _state()
        for _ in range(5):
            _vad(state, "speech_start", duration_ms=0)
        self.assertIsNone(VoiceVadFalseTriggerDetector().on_run_completion(state))

    def test_long_speech_not_counted(self):
        state = _state()
        for _ in range(5):
            _vad(state, "speech_start", duration_ms=800)
        self.assertIsNone(VoiceVadFalseTriggerDetector().on_run_completion(state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
