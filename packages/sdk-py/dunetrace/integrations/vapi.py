"""
Dunetrace integration for Vapi (https://vapi.ai) voice agents.

Vapi is cloud-orchestrated: Vapi runs the speech-to-text, LLM, and text-to-speech
pipeline itself and posts "server messages" to your server URL over HTTP. So the
integration is a webhook translator, not an in-process observer. You feed each
Vapi server message to the adapter; it maps them onto the voice events the pack
detectors read (transcription_received / tts_generated / voice_activity_detected
/ turn_taking).

Runtime enforcement caveat: because Vapi owns the audio pipeline, the voice
runtime policy actions (stop_current_tts, slow_response_pace, ...) cannot be
enforced in-path for Vapi. Detection here is post-hoc, per call. Do not present
the sub-millisecond runtime firewall claim for Vapi deployments. See BACKLOG.md
V2.

Run model: a Vapi call arrives as many separate webhook requests, and dt.run()
is a context manager that can't stay open across stateless HTTP handlers. The
adapter buffers a call's messages keyed by call id and replays them through one
dt.run() when the call ends (status-update status=ended, or end-of-call-report).
One Vapi call becomes one Dunetrace run; the call id is threaded as
conversation_id so multi-run modeling (Phase 2) can group it.

Coverage: Vapi server messages carry turn structure, transcripts, interruptions,
and call lifecycle, but NOT per-transcript STT confidence or per-event latency.
So the confidence-based detectors (VOICE_TRANSCRIPTION_CONFIDENCE_DROP,
VOICE_AUDIO_QUALITY_DEGRADATION) and the latency detector
(VOICE_LATENCY_INDUCED_HANGUP) get no signal from Vapi. Turn-taking, speaker
confusion, barge-in, and TTS truncation (via user-interrupted) do fire. This is a
Vapi surface limit, not a detector bug; documented in docs/detector-packs/voice.md.

Usage (FastAPI shown; any web framework works)::

    from dunetrace import Dunetrace
    from dunetrace.integrations.vapi import DunetraceVapiAdapter

    dt = Dunetrace()
    adapter = DunetraceVapiAdapter(dt, agent_id="support-line", model="gpt-4o")

    @app.post("/vapi/webhook")
    async def vapi_webhook(payload: dict):
        adapter.handle_message(payload)
        return {}  # Vapi request-type messages expect their own response; this
                   # adapter only observes and never answers them.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.integrations.vapi")

# Vapi roles for the agent side vary by message ("assistant" on transcript,
# "bot" historically); treat both as the agent.
_AGENT_ROLES = frozenset({"assistant", "bot"})

# Messages that end a call and trigger replay.
_TERMINAL_TYPES = frozenset({"end-of-call-report"})


class DunetraceVapiAdapter:
    """Translate Vapi server messages into Dunetrace voice events.

    Thread-safe: a single adapter handles concurrent calls, each keyed by its
    Vapi call id. Buffering is per call and cleared when the call is finalized.
    """

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str,
        *,
        model: str = "unknown",
        system_prompt: str = "",
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._model = model
        self._system_prompt = system_prompt
        self._buffers: Dict[str, List[dict]] = {}
        self._lock = Lock()

    # ── Public entry point ────────────────────────────────────────────────────

    def handle_message(self, payload: dict) -> None:
        """Route one Vapi server message. ``payload`` is the raw webhook body,
        i.e. the ``{"message": {...}}`` envelope. Unknown or unhandled message
        types are ignored, so new Vapi message types never raise."""
        msg = (payload or {}).get("message")
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        call_id = self._call_id(msg)

        # A terminal status-update or an end-of-call-report closes the call.
        if mtype in _TERMINAL_TYPES or (mtype == "status-update" and msg.get("status") == "ended"):
            self._finalize(call_id, msg)
            return
        self._buffer(call_id, msg)

    # ── Buffering + finalize ──────────────────────────────────────────────────

    def _buffer(self, call_id: str, msg: dict) -> None:
        with self._lock:
            self._buffers.setdefault(call_id, []).append(msg)

    def _finalize(self, call_id: str, terminal_msg: dict) -> None:
        """Replay this call's buffered messages through one dt.run(). Called when
        the call ends. If no messages were buffered (e.g. only a terminal message
        was received), nothing is emitted."""
        with self._lock:
            msgs = self._buffers.pop(call_id, [])
        if not msgs:
            return
        logger.debug("vapi: finalizing call %s (%d messages)", call_id, len(msgs))
        ended_reason = terminal_msg.get("endedReason")
        recording_url = self._recording_url(terminal_msg)
        with self._client.run(
            self._agent_id,
            model=self._model,
            system_prompt=self._system_prompt,
            conversation_id=call_id,
        ) as run:
            for m in msgs:
                self._apply(run, m)
            # Record how the call ended so call-metrics can tell a natural end from
            # a drop. The raw Vapi reason is stored as-is; the natural-vs-dropped
            # mapping lives server-side in the call-metrics query, not here.
            if ended_reason:
                run.external_signal("call_ended", source="vapi", reason=ended_reason)
            # The end-of-call-report carries the recording URL (Phase 2.3). Vapi
            # hosts the audio; Dunetrace only links to it, never fetches it.
            if recording_url:
                run.recording_metadata(recording_url, storage_provider="vapi")
            run.final_answer()

    @staticmethod
    def _recording_url(terminal_msg: dict) -> Optional[str]:
        """Pull a recording URL out of a Vapi end-of-call-report. Vapi has used a
        few shapes for this; check the common ones and return the first present."""
        artifact = terminal_msg.get("artifact")
        if not isinstance(artifact, dict):
            return None
        rec = artifact.get("recording")
        if isinstance(rec, dict):
            url = rec.get("stereoUrl") or rec.get("url")
            if url:
                return str(url)
        url = artifact.get("recordingUrl") or artifact.get("stereoRecordingUrl")
        return str(url) if url else None

    # ── Message → voice event mapping ─────────────────────────────────────────

    def _apply(self, run, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "transcript":
            self._apply_transcript(run, msg)
        elif mtype == "speech-update":
            self._apply_speech_update(run, msg)
        elif mtype == "user-interrupted":
            # The caller barged in. Record the interruption; VOICE_BARGE_IN_FAILURE
            # / VOICE_TTS_TRUNCATION read barge_in against the surrounding TTS.
            run.voice_activity_detected("barge_in")

    def _apply_transcript(self, run, msg: dict) -> None:
        # Only final transcripts are turns; partials are interim STT and would
        # double-count. Vapi does not surface a confidence score on this message,
        # so confidence defaults to 1.0 (see module docstring coverage note).
        if msg.get("transcriptType") != "final":
            return
        text = msg.get("transcript", "") or ""
        role = msg.get("role")
        if role in _AGENT_ROLES:
            run.tts_generated(text=text)
        else:
            run.transcription_received(text=text)

    def _apply_speech_update(self, run, msg: dict) -> None:
        status = msg.get("status")
        role = msg.get("role")
        is_agent = role in _AGENT_ROLES
        if status == "started":
            run.turn_taking(
                "agent_speaking" if is_agent else "user_speaking",
                from_agent=is_agent,
                to_user=not is_agent,
            )
        elif status == "stopped" and not is_agent:
            run.voice_activity_detected("speech_end")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _call_id(msg: dict) -> str:
        call = msg.get("call")
        if isinstance(call, dict) and call.get("id"):
            return str(call["id"])
        return "unknown"
