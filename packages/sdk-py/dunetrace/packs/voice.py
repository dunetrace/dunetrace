"""
Voice-agent detector pack (Phase 1.2). Nine detectors for failure modes
specific to real-time voice agents (speech-to-text, TTS, VAD, turn-taking).

These are first-party PACK detectors, not the customer-plugin kind:
  - every class sets `pack = "voice"`, which keeps it OUT of
    CUSTOM_DETECTOR_REGISTRY (see BaseDetector.__init_subclass__) so it only
    runs for an org that has activated the voice pack — never unconditionally.
  - the pack registers itself via register_pack() at import time (bottom of
    this module); importing dunetrace.packs is what makes it known.

Every detector reads the four voice event types straight out of
state.events (the same place RunawayIterationDetector reads llm.responded
from) — no new RunState fields. build_run_state (server) and RunContext
(in-process) both already append these events to state.events, so nothing
else needs plumbing. Each detector short-circuits to None when its own
event subset is empty, so activating the pack on a non-voice agent produces
zero signals.

Signals use FailureType.CUSTOM + evidence["detector_name"] = self.name, the
same TEXT-failure-type convention custom Python-class detectors use — the
detector worker routes them through write_custom_signal() accordingly.
SHADOW_BY_DEFAULT is True: these ship unvalidated, so they start in shadow
until an operator promotes the pack's signals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dunetrace.detectors import BaseDetector
from dunetrace.models import (
    AgentEvent,
    EventType,
    FailureSignal,
    FailureType,
    RunState,
    Severity,
)
from dunetrace.packs.base import DetectorPack, register_pack


def _events(state: RunState, event_type: EventType) -> List[AgentEvent]:
    """Voice events of one type, in emission order."""
    return [e for e in state.events if e.event_type == event_type]


def _signal(
    detector: BaseDetector,
    state: RunState,
    *,
    confidence: float,
    evidence: Dict[str, Any],
    severity: Severity,
    step_index: Optional[int] = None,
) -> FailureSignal:
    """Build a pack signal. Always stamps evidence["detector_name"] — the
    worker's TEXT-failure-type routing keys on it (a closed FailureType enum
    can't carry the pack detector's own identity)."""
    ev = dict(evidence)
    ev["detector_name"] = detector.name
    return FailureSignal(
        failure_type=FailureType.CUSTOM,
        severity=severity,
        run_id=state.run_id,
        agent_id=state.agent_id,
        agent_version=state.agent_version,
        step_index=step_index if step_index is not None else state.current_step,
        confidence=confidence,
        evidence=ev,
    )


# ── 1. Transcription confidence drop ──────────────────────────────────────────


class VoiceTranscriptionConfidenceDropDetector(BaseDetector):
    """Speech-to-text returned low-confidence transcripts — the agent is
    acting on words it isn't sure it heard. Absolute floor, not a trend
    (that's AUDIO_QUALITY_DEGRADATION)."""

    name = "VOICE_TRANSCRIPTION_CONFIDENCE_DROP"
    pack = "voice"
    MIN_CONFIDENCE = 0.5
    MIN_COUNT = 2

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        low = [
            e
            for e in _events(state, EventType.TRANSCRIPTION_RECEIVED)
            if e.payload.get("confidence", 1.0) < self.MIN_CONFIDENCE
        ]
        if len(low) < self.MIN_COUNT:
            return None
        worst = min(low, key=lambda e: e.payload.get("confidence", 1.0))
        return _signal(
            self,
            state,
            confidence=0.8,
            severity=Severity.MEDIUM,
            step_index=low[0].step_index,
            evidence={
                "low_confidence_count": len(low),
                "min_confidence": self.MIN_CONFIDENCE,
                "worst_confidence": round(worst.payload.get("confidence", 1.0), 3),
                "worst_text": worst.payload.get("text", ""),
            },
        )


# ── 2. Silence timeout ────────────────────────────────────────────────────────


class VoiceSilenceTimeoutDetector(BaseDetector):
    """A stretch of dead air past the timeout — the user likely disengaged or
    the agent stalled waiting for input that never came."""

    name = "VOICE_SILENCE_TIMEOUT"
    pack = "voice"
    MAX_SILENCE_MS = 8000

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        offenders = [
            e
            for e in _events(state, EventType.VOICE_ACTIVITY_DETECTED)
            if e.payload.get("type") == "silence"
            and e.payload.get("duration_ms", 0) > self.MAX_SILENCE_MS
        ]
        if not offenders:
            return None
        longest = max(offenders, key=lambda e: e.payload.get("duration_ms", 0))
        return _signal(
            self,
            state,
            confidence=0.75,
            severity=Severity.MEDIUM,
            step_index=longest.step_index,
            evidence={
                "silence_count": len(offenders),
                "max_silence_ms": longest.payload.get("duration_ms", 0),
                "threshold_ms": self.MAX_SILENCE_MS,
            },
        )


# ── 3. Turn-taking collision ──────────────────────────────────────────────────


class VoiceTurnTakingCollisionDetector(BaseDetector):
    """Agent and user talking over each other — repeated `both_speaking`
    turn-taking states, a sign the agent's endpointing/barge-in handling is
    off."""

    name = "VOICE_TURN_TAKING_COLLISION"
    pack = "voice"
    MAX_COLLISIONS = 2

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        collisions = [
            e
            for e in _events(state, EventType.TURN_TAKING)
            if e.payload.get("action") == "both_speaking"
        ]
        if len(collisions) < self.MAX_COLLISIONS:
            return None
        return _signal(
            self,
            state,
            confidence=0.8,
            severity=Severity.MEDIUM,
            step_index=collisions[0].step_index,
            evidence={
                "collision_count": len(collisions),
                "threshold": self.MAX_COLLISIONS,
            },
        )


# ── 4. Latency-induced hangup ─────────────────────────────────────────────────


class VoiceLatencyInducedHangupDetector(BaseDetector):
    """The last thing that happened was a slow agent response with no user
    reply after it — the caller most likely hung up waiting for the agent."""

    name = "VOICE_LATENCY_INDUCED_HANGUP"
    pack = "voice"
    MAX_LATENCY_MS = 5000

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        # Ordered stream of the two turn-bearing voice events. If the LAST one
        # is a slow TTS, nothing (no transcription) followed it — the user
        # never took their turn back.
        stream = [
            e
            for e in state.events
            if e.event_type in (EventType.TRANSCRIPTION_RECEIVED, EventType.TTS_GENERATED)
        ]
        if not stream:
            return None
        last = stream[-1]
        if last.event_type != EventType.TTS_GENERATED:
            return None
        latency = last.payload.get("latency_ms", 0)
        if latency <= self.MAX_LATENCY_MS:
            return None
        return _signal(
            self,
            state,
            confidence=0.7,
            severity=Severity.HIGH,
            step_index=last.step_index,
            evidence={
                "final_tts_latency_ms": latency,
                "threshold_ms": self.MAX_LATENCY_MS,
                "final_tts_text": last.payload.get("text", ""),
            },
        )


# ── 5. Audio quality degradation ──────────────────────────────────────────────


class VoiceAudioQualityDegradationDetector(BaseDetector):
    """Transcription confidence trends downward over the call — audio getting
    progressively worse (connection degrading, moving away from mic), distinct
    from CONFIDENCE_DROP's absolute floor."""

    name = "VOICE_AUDIO_QUALITY_DEGRADATION"
    pack = "voice"
    MIN_DROP = 0.2
    MIN_SAMPLES = 4

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        confs = [
            e.payload.get("confidence", 1.0)
            for e in _events(state, EventType.TRANSCRIPTION_RECEIVED)
        ]
        if len(confs) < self.MIN_SAMPLES:
            return None
        mid = len(confs) // 2
        first_mean = sum(confs[:mid]) / mid
        second_mean = sum(confs[mid:]) / (len(confs) - mid)
        drop = first_mean - second_mean
        if drop < self.MIN_DROP:
            return None
        return _signal(
            self,
            state,
            confidence=0.75,
            severity=Severity.MEDIUM,
            evidence={
                "first_half_mean_confidence": round(first_mean, 3),
                "second_half_mean_confidence": round(second_mean, 3),
                "drop": round(drop, 3),
                "min_drop": self.MIN_DROP,
                "sample_count": len(confs),
            },
        )


# ── 6. Speaker confusion ──────────────────────────────────────────────────────


class VoiceSpeakerConfusionDetector(BaseDetector):
    """A transcript arrived while the agent itself was speaking — the STT is
    hearing the agent's own TTS (echo / no acoustic echo cancellation), so the
    agent ends up 'replying' to itself."""

    name = "VOICE_SPEAKER_CONFUSION"
    pack = "voice"
    MIN_COUNT = 1

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        current_action: Optional[str] = None
        self_heard: List[AgentEvent] = []
        for e in state.events:
            if e.event_type == EventType.TURN_TAKING:
                current_action = e.payload.get("action")
            elif e.event_type == EventType.TRANSCRIPTION_RECEIVED:
                if current_action == "agent_speaking":
                    self_heard.append(e)
        if len(self_heard) < self.MIN_COUNT:
            return None
        return _signal(
            self,
            state,
            confidence=0.8,
            severity=Severity.HIGH,
            step_index=self_heard[0].step_index,
            evidence={
                "self_heard_count": len(self_heard),
                "sample_text": self_heard[0].payload.get("text", ""),
            },
        )


# ── 7. Barge-in failure ───────────────────────────────────────────────────────


class VoiceBargeInFailureDetector(BaseDetector):
    """The user interrupted (barge-in) but the agent produced a full,
    non-truncated TTS response afterward anyway — it talked over the
    interruption instead of yielding."""

    name = "VOICE_BARGE_IN_FAILURE"
    pack = "voice"

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        pending_barge_in: Optional[AgentEvent] = None
        for e in state.events:
            if (
                e.event_type == EventType.VOICE_ACTIVITY_DETECTED
                and e.payload.get("type") == "barge_in"
            ):
                pending_barge_in = e
            elif e.event_type == EventType.TTS_GENERATED and pending_barge_in is not None:
                if not e.payload.get("truncated", False):
                    return _signal(
                        self,
                        state,
                        confidence=0.8,
                        severity=Severity.HIGH,
                        step_index=e.step_index,
                        evidence={
                            "barge_in_step": pending_barge_in.step_index,
                            "tts_step": e.step_index,
                            "tts_text": e.payload.get("text", ""),
                        },
                    )
                pending_barge_in = None
        return None


# ── 8. TTS truncation ─────────────────────────────────────────────────────────


class VoiceTtsTruncationDetector(BaseDetector):
    """Agent responses cut off mid-utterance WITHOUT a preceding barge-in to
    explain it — the response hit a length cap or the synth failed, not a
    user interruption (that case is legitimate and excluded here, so this
    detector doesn't double-count with BARGE_IN_FAILURE)."""

    name = "VOICE_TTS_TRUNCATION"
    pack = "voice"
    MAX_TRUNCATIONS = 1

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        unexplained: List[AgentEvent] = []
        barge_in_since_last_tts = False
        for e in state.events:
            if (
                e.event_type == EventType.VOICE_ACTIVITY_DETECTED
                and e.payload.get("type") == "barge_in"
            ):
                barge_in_since_last_tts = True
            elif e.event_type == EventType.TTS_GENERATED:
                if e.payload.get("truncated", False) and not barge_in_since_last_tts:
                    unexplained.append(e)
                barge_in_since_last_tts = False
        if len(unexplained) < self.MAX_TRUNCATIONS:
            return None
        return _signal(
            self,
            state,
            confidence=0.75,
            severity=Severity.MEDIUM,
            step_index=unexplained[0].step_index,
            evidence={
                "truncation_count": len(unexplained),
                "threshold": self.MAX_TRUNCATIONS,
                "sample_text": unexplained[0].payload.get("text", ""),
            },
        )


# ── 9. VAD false trigger ──────────────────────────────────────────────────────


class VoiceVadFalseTriggerDetector(BaseDetector):
    """Voice-activity detection fired on segments too short to be real speech
    (noise, clicks) — repeated very-short speech_start/barge_in events. Only
    events carrying a positive duration_ms are considered; a 0/absent duration
    means 'unknown', not 'zero-length', so it isn't counted."""

    name = "VOICE_VAD_FALSE_TRIGGER"
    pack = "voice"
    MAX_FALSE_TRIGGERS = 3
    MIN_DURATION_MS = 100

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        false_triggers = [
            e
            for e in _events(state, EventType.VOICE_ACTIVITY_DETECTED)
            if e.payload.get("type") in ("speech_start", "barge_in")
            and 0 < e.payload.get("duration_ms", 0) < self.MIN_DURATION_MS
        ]
        if len(false_triggers) < self.MAX_FALSE_TRIGGERS:
            return None
        return _signal(
            self,
            state,
            confidence=0.7,
            severity=Severity.LOW,
            step_index=false_triggers[0].step_index,
            evidence={
                "false_trigger_count": len(false_triggers),
                "threshold": self.MAX_FALSE_TRIGGERS,
                "min_duration_ms": self.MIN_DURATION_MS,
            },
        )


class VoicePack(DetectorPack):
    name = "voice"
    description = (
        "Failure detectors for real-time voice agents: transcription confidence, "
        "silence timeouts, turn-taking collisions, latency-induced hangups, audio "
        "quality degradation, speaker confusion, barge-in handling, TTS truncation, "
        "and VAD false triggers."
    )
    detectors = [
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


register_pack(VoicePack())
