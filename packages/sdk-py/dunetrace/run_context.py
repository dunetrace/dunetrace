"""
The object returned by `dt.run(...)`. Provides emit helpers like tool_called
and llm_called, and builds up a RunState for local detection.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from dunetrace.models import (
    AgentEvent,
    EventType,
    ExternalSignal,
    LlmCall,
    MemoryEvent,
    RetrievalResult,
    RunState,
    ToolCall,
)
import logging
from dunetrace.policies import (
    ApprovalDenied,
    EvaluationContext,
    PolicyEvaluationRecord,
    PolicyViolation,
    build_metrics,
    build_reason,
    compute_run_cost,
    eval_logger,
)

logger = logging.getLogger("dunetrace.run")

# Valid values for the voice hooks — validated at the call site so an invalid
# state string never reaches the events table (where a voice-pack detector
# would silently never match it).
_VAD_TYPES = frozenset({"speech_start", "speech_end", "silence", "barge_in"})
_TURN_TAKING_ACTIONS = frozenset({"agent_speaking", "user_speaking", "both_speaking", "neither"})
# Where content written to agent memory came from (Capability 1). Optional on
# memory_written, but strongly encouraged: MEMORY_POISONING weighs injection risk
# higher for retrieval/tool_output/external sources than for user-set preferences.
_MEMORY_SOURCES = frozenset(
    {"user_input", "retrieval", "tool_output", "llm_output", "agent_reasoning", "external"}
)


def _omit_llm_output_text() -> bool:
    """True when DUNETRACE_OMIT_LLM_OUTPUT_TEXT opts out of transmitting raw LLM
    output text (bandwidth-sensitive deployments). Read per-call so it can be
    toggled at runtime/in tests; the check is a cheap env lookup off the hot path."""
    return os.environ.get("DUNETRACE_OMIT_LLM_OUTPUT_TEXT", "").lower() in ("1", "true", "yes")


if TYPE_CHECKING:
    from dunetrace.client import Dunetrace


class RunContext:
    """Thin wrapper around a single agent run."""

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str,
        agent_version: str,
        available_tools: list,
        input_text: str,
        system_prompt: str = "",
        parent_run_id: Optional[str] = None,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        self._client = client
        self.run_id = run_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.step = 0
        self.exit_reason: Optional[str] = None
        self._parent_run_id = parent_run_id
        # Correlation key for external evaluation integrations (Langfuse/
        # LangSmith/Braintrust) — not folded into agent_version's hash like
        # system_prompt is, since it identifies this run to an external
        # system rather than describing the agent's own identity/config.
        self._trace_id = trace_id
        # Groups this run with others from the same end-user interaction
        # (Phase 3.1 conversation modeling) — same non-identity rationale as
        # trace_id, threaded onto every event the same way.
        self._conversation_id = conversation_id

        self.state = RunState(
            run_id=self.run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            available_tools=available_tools,
            input_text=input_text,
            system_prompt=system_prompt or None,
        )

        # Policy enforcement state
        self.model_override: str | None = None  # set by switch_model action
        self.prompt_additions: list = []  # appended by inject_prompt action
        self._inject_prompt_fired = False  # audit Finding 25: unread-advisory check
        # Voice-agent policy outputs (Phase 1.3) — same read contract as
        # model_override: a policy sets these, the voice agent's own loop reads
        # them between turns and acts. Dunetrace never touches the audio
        # pipeline itself.
        self.stop_tts: bool = False  # set by stop_current_tts action
        self.escalate_to_human: bool = False  # set by escalate_to_human action
        self.escalation_reason: str | None = None  # from escalate_to_human params.reason
        self.recovery_prompt: str | None = None  # set by inject_recovery_prompt (last-wins)
        self.response_pace: str | None = None  # set by slow_response_pace action (last-wins)
        self._triggered_policies: set = set()  # policy keys fired in this run

        # Detector result cache — signal-trigger policies reuse this within a step
        # so detectors don't run twice for (llm_called, llm_responded) at the same step.
        self._detector_cache_step: int = -1
        self._detector_cache_signals: list = []

        # Cached per engine generation: whether any active policy uses trigger="signal".
        # Invalidated when the engine reloads (remote refresh) or add_policy() is called
        # so long-running agents pick up newly added signal-trigger policies.
        self._needs_signal: bool | None = None
        self._needs_signal_generation: int = -1

        # Running totals kept in sync with state — makes build_metrics O(1) for these fields.
        self._error_count: int = 0
        self._cost_usd: float = 0.0

        # Snapshotted once at run start; run.duration_ms in an EvaluationContext is
        # computed from this at context-build time (never read on the eval hot path,
        # keeping evaluation itself clock-free / deterministic).
        self._started_monotonic: float = time.monotonic()

    # ── LLM hooks ─────────────────────────────────────────────────────────────

    def llm_called(self, model: str, prompt_tokens: int = 0) -> None:
        self.state.llm_calls.append(
            LlmCall(
                model=model,
                prompt_tokens=prompt_tokens,
                finish_reason=None,
                latency_ms=None,
                step_index=self.step,
                timestamp=time.time(),
            )
        )
        self._emit(
            EventType.LLM_CALLED,
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
            },
        )

    def llm_responded(
        self,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        latency_ms: int = 0,
        finish_reason: str = "stop",
        output: str = "",
        output_length: int = 0,
        prompt_tokens: int = 0,
        error: Optional[str] = None,
    ) -> None:
        # Back-fill the most recent LlmCall with response data.
        # prompt_tokens is optional — pass it when the count is only known after
        # the call (e.g. taken from the API response), overriding the estimate
        # given to llm_called().
        if self.state.llm_calls:
            lc = self.state.llm_calls[-1]
            lc.finish_reason = finish_reason
            lc.latency_ms = latency_ms
            lc.output_length = output_length
            lc.completion_tokens = completion_tokens or None
            lc.reasoning_tokens = reasoning_tokens or None
            # Typed access to the output text. Stored on the struct even when
            # transmission is opted out below — it's already in-process here, so
            # local runtime detectors keep full fidelity at zero bandwidth cost.
            lc.output_text = output or None
            if prompt_tokens:
                lc.prompt_tokens = prompt_tokens
            self._cost_usd += compute_run_cost([lc])
        payload: dict = {
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "output_length": output_length,
        }
        # Transmit the output text by default; omit it (bandwidth) when
        # DUNETRACE_OMIT_LLM_OUTPUT_TEXT is set. output_length is always sent, so
        # size-based detectors work either way.
        if not _omit_llm_output_text():
            payload["output"] = output
        if prompt_tokens:
            payload["prompt_tokens"] = prompt_tokens
        if reasoning_tokens:
            payload["reasoning_tokens"] = reasoning_tokens
        if error:
            payload["error"] = error
        self._emit(EventType.LLM_RESPONDED, payload, advance=False)

    # ── Tool hooks ────────────────────────────────────────────────────────────

    def tool_called(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        _enforce_approval: bool = True,
    ) -> None:
        args_repr = str(args or {})
        self.state.tool_calls.append(
            ToolCall(
                tool_name=tool_name,
                args=args_repr,
                step_index=self.step,
                timestamp=time.time(),
            )
        )
        self._emit(
            EventType.TOOL_CALLED,
            {
                "tool_name": tool_name,
                "args": args_repr,
            },
        )
        # Human-in-the-loop gate (Capability 2, Phase 2.5): if a
        # require_approval policy matches this tool, block synchronously until a
        # human decides (raises ApprovalDenied on deny/timeout). Suppressed only
        # by the @dt.tool ASYNC wrapper, which instead awaits the async gate so
        # it doesn't block the event loop — see client.py. Sync callers (manual
        # dt.run() code, the sync @dt.tool wrapper) get the blocking gate here.
        if _enforce_approval:
            self._enforce_tool_approval_sync(tool_name, args)

    def tool_responded(
        self,
        tool_name: str,
        success: bool = True,
        output_length: Optional[int] = None,
        latency_ms: int = 0,
        error: Optional[str] = None,
        output: str = "",
    ) -> None:
        error_text = error if (not success and error) else None
        if output_length is None:
            computed_length = len(output) if output else 0
        else:
            computed_length = output_length
        # Back-fill success, error, and output on the most recent matching ToolCall.
        # _error_count increments only when a ToolCall is actually found and back-filled
        # so the running total stays in sync with the full-scan baseline in build_metrics.
        for tc in reversed(self.state.tool_calls):
            if tc.tool_name == tool_name and tc.success is None:
                tc.success = success
                tc.error = error_text
                tc.output_length = computed_length
                tc.output = output or None
                if not success:
                    self._error_count += 1
                break
        payload: dict = {
            "tool_name": tool_name,
            "success": success,
            "output_length": computed_length,
            "latency_ms": latency_ms,
            "output": output,
        }
        if error_text:
            payload["error"] = error_text
        self._emit(EventType.TOOL_RESPONDED, payload, advance=False)

    # ── Retrieval hooks (RAG) ─────────────────────────────────────────────────

    def retrieval_called(self, index_name: str, query: str = "") -> None:
        self._emit(
            EventType.RETRIEVAL_CALLED,
            {
                "index_name": index_name,
                "query": query,
            },
        )

    def retrieval_responded(
        self,
        index_name: str,
        result_count: int,
        top_score: Optional[float] = None,
        latency_ms: int = 0,
        content: str = "",
    ) -> None:
        self.state.retrievals.append(
            RetrievalResult(
                index_name=index_name,
                result_count=result_count,
                top_score=top_score,
                step_index=self.step,
                content=content or None,
            )
        )
        self._emit(
            EventType.RETRIEVAL_RESPONDED,
            {
                "index_name": index_name,
                "result_count": result_count,
                "top_score": top_score,
                "latency_ms": latency_ms,
                "content": content,
            },
            advance=False,
        )

    # ── Voice hooks (detector pack "voice") ───────────────────────────────────
    #
    # advance semantics mirror the non-voice hooks above: a turn-initiating
    # event advances the step counter (like tool_called / llm_called), while
    # completion and high-frequency annotation events do not (like
    # tool_responded / external_signal). This keeps a normal voice turn at
    # roughly one step — critical because the always-on built-in detectors
    # (RunawayIterationDetector's STEP_THRESHOLD, StepCountInflationDetector's
    # baseline) count steps for EVERY org, whether or not the voice pack is
    # active. A per-audio-frame VAD event advancing the counter would false-
    # fire those detectors on any real voice call.

    def transcription_received(
        self, text: str, confidence: float = 1.0, latency_ms: int = 0, audio_seconds: float = 0.0
    ) -> None:
        """Speech-to-text produced a transcript for one user turn. Turn-
        initiating — advances the step counter.

        audio_seconds: length of the transcribed audio. Optional, but pass it if
        you want per-minute STT cost attribution (Phase 2.2) — most STT providers
        bill per minute of audio, which the transcript text alone can't tell us.
        Omitted from the payload when 0 (backward compatible)."""
        payload: dict = {"text": text, "confidence": confidence, "latency_ms": latency_ms}
        if audio_seconds:
            payload["audio_seconds"] = audio_seconds
        self._emit(EventType.TRANSCRIPTION_RECEIVED, payload)

    def tts_generated(
        self, text: str, latency_ms: int = 0, truncated: bool = False, audio_seconds: float = 0.0
    ) -> None:
        """Text-to-speech rendered an agent response. Completes the turn —
        does not advance the step counter (like llm_responded).

        audio_seconds: length of the synthesized audio. Optional. TTS cost
        (Phase 2.2) is billed per character and derived from ``text``, so this is
        only needed if your TTS provider bills per audio-second instead. Omitted
        from the payload when 0 (backward compatible)."""
        payload: dict = {"text": text, "latency_ms": latency_ms, "truncated": truncated}
        if audio_seconds:
            payload["audio_seconds"] = audio_seconds
        self._emit(EventType.TTS_GENERATED, payload, advance=False)

    def voice_activity_detected(self, type: str, duration_ms: int = 0) -> None:
        """A VAD transition (speech_start / speech_end / silence / barge_in).
        High-frequency annotation — does not advance the step counter."""
        if type not in _VAD_TYPES:
            raise ValueError(
                f"voice_activity_detected: type must be one of {sorted(_VAD_TYPES)}, got {type!r}"
            )
        self._emit(
            EventType.VOICE_ACTIVITY_DETECTED,
            {
                "type": type,
                "duration_ms": duration_ms,
            },
            advance=False,
        )

    def turn_taking(self, action: str, from_agent: bool = False, to_user: bool = False) -> None:
        """A conversational-floor transition (agent_speaking / user_speaking /
        both_speaking / neither). State annotation — does not advance the step
        counter."""
        if action not in _TURN_TAKING_ACTIONS:
            raise ValueError(
                f"turn_taking: action must be one of {sorted(_TURN_TAKING_ACTIONS)}, got {action!r}"
            )
        self._emit(
            EventType.TURN_TAKING,
            {
                "action": action,
                "from_agent": from_agent,
                "to_user": to_user,
            },
            advance=False,
        )

    def recording_metadata(
        self,
        url: str,
        duration_seconds: float = 0.0,
        format: str = "",
        storage_provider: str = "",
        start_offset_seconds: float = 0.0,
    ) -> None:
        """Record where the call audio lives (Phase 2.3). Storage-agnostic: pass
        the URL and metadata; Dunetrace stores and links them but never fetches
        the audio, so a presigned or private URL is fine (mind its expiry).

        start_offset_seconds is when the recording began relative to the start of
        the call/run, used to deep-link a signal's moment into the audio.
        Annotation event — does not advance the step counter."""
        payload: dict = {"url": url}
        if duration_seconds:
            payload["duration_seconds"] = duration_seconds
        if format:
            payload["format"] = format
        if storage_provider:
            payload["storage_provider"] = storage_provider
        if start_offset_seconds:
            payload["start_offset_seconds"] = start_offset_seconds
        self._emit(EventType.RECORDING_AVAILABLE, payload, advance=False)

    # ── External signal hooks ─────────────────────────────────────────────────

    def external_signal(self, signal_name: str, source: str = "", **meta: Any) -> None:
        """
        Emit an infrastructure context event at the current agent step without advancing the step counter.

        Usage::

            run.external_signal("rate_limit", source="openai")
            run.external_signal("cache_miss", source="redis", key_prefix="emb:")
            run.external_signal("upstream_error", source="serp_api", http_status=503)

        SLOW_STEP and other detectors correlate these signals with failures so evidence reads
        "tool took 100s — coincided with rate_limit from openai" rather than just "tool took 100s".
        """
        ts = time.time()
        self.state.external_signals.append(
            ExternalSignal(
                signal_name=signal_name,
                step_index=self.step,
                timestamp=ts,
                source=source,
                meta=dict(meta),
            )
        )
        payload: dict = {"signal_name": signal_name}
        if source:
            payload["source"] = source
        if meta:
            payload["meta"] = dict(meta)
        # Emit directly — bypass _emit() so step counter does not advance.
        event = AgentEvent(
            event_type=EventType.EXTERNAL_SIGNAL,
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            step_index=self.step,
            timestamp=ts,
            payload=payload,
            parent_run_id=self._parent_run_id,
            trace_id=self._trace_id,
            conversation_id=self._conversation_id,
        )
        self.state.events.append(event)
        self._client._emit(event)

    # ── Agent memory channel (Capability 1) ───────────────────────────────────
    #
    # Instrumentation for what an agent persists to and reads from its own memory
    # (conversation buffers, scratchpads, long-term stores). None of these advance
    # the step counter — they annotate the run, like external_signal. The typed
    # RunState.memory_events view is built from these events (Phase 1.3); the
    # MEMORY_POISONING detector reads it (Phase 1.4).

    def memory_written(self, key: str, value: str, source: Optional[str] = None) -> None:
        """Record that ``value`` was written to agent memory under ``key``.

        ``source`` is optional but strongly encouraged — it names where the
        content originated so downstream detection can weigh injection risk:
        one of ``user_input``, ``retrieval``, ``tool_output``, ``llm_output``,
        ``agent_reasoning``, ``external``. Does not advance the step counter.
        """
        if source is not None and source not in _MEMORY_SOURCES:
            raise ValueError(
                f"memory_written: source must be one of {sorted(_MEMORY_SOURCES)} "
                f"or None, got {source!r}"
            )
        payload: Dict[str, Any] = {"key": key, "value": value}
        if source is not None:
            payload["source"] = source
        self.state.memory_events.append(
            MemoryEvent(
                op="written",
                key=key,
                value=value,
                source=source,
                step_index=self.step,
                timestamp=time.time(),
            )
        )
        self._emit(EventType.MEMORY_WRITTEN, payload, advance=False)

    def memory_read(self, key: str) -> None:
        """Record that agent memory was read at ``key``. Does not advance the step
        counter."""
        self.state.memory_events.append(
            MemoryEvent(op="read", key=key, step_index=self.step, timestamp=time.time())
        )
        self._emit(EventType.MEMORY_READ, {"key": key}, advance=False)

    def memory_cleared(self, key: Optional[str] = None) -> None:
        """Record that agent memory was cleared — a specific ``key``, or all
        memory when ``key`` is None. Does not advance the step counter."""
        self.state.memory_events.append(
            MemoryEvent(op="cleared", key=key, step_index=self.step, timestamp=time.time())
        )
        self._emit(EventType.MEMORY_CLEARED, {"key": key}, advance=False)

    # ── Human-in-the-loop approvals (Capability 2) ────────────────────────────
    #
    # A guarded tool call blocks here until a human decides (via Slack/dashboard
    # /API) or the timeout passes. Decision transport is polling the Customer
    # API — no inbound connection to the SDK. Timeout is fail-closed: if nobody
    # decides in time the tool is blocked, not allowed. request_approval() is
    # the sync path (blocks the calling thread); arequest_approval() is the
    # async path (yields the event loop between polls). Both share the same
    # quick synchronous HTTP calls — only the wait between polls differs.
    #
    # In Phase 2.2 these are called directly. Phase 2.5 wires them into the
    # require_approval policy action so a policy fires them from tool_called.

    def _begin_approval(self, tool_name: str, args: Any, timeout_s: int) -> int:
        row = self._client._create_approval_request(
            run_id=self.run_id,
            agent_id=self.agent_id,
            tool_name=tool_name,
            tool_args=str(args) if args is not None else None,
            timeout_seconds=timeout_s,
        )
        approval_id = int(row["id"])
        self._emit(
            EventType.APPROVAL_REQUESTED,
            {"approval_id": approval_id, "tool_name": tool_name},
            advance=False,
        )
        return approval_id

    def _finish_approval(self, tool_name: str, approval_id: int, status: str) -> None:
        """Emit the terminal event and either return (granted) or raise
        (denied/timeout, fail-closed)."""
        if status == "granted":
            self._emit(
                EventType.APPROVAL_GRANTED,
                {"approval_id": approval_id, "tool_name": tool_name},
                advance=False,
            )
            return
        event = EventType.APPROVAL_TIMEOUT if status == "timeout" else EventType.APPROVAL_DENIED
        self._emit(
            event,
            {"approval_id": approval_id, "tool_name": tool_name},
            advance=False,
        )
        raise ApprovalDenied(tool_name, status, approval_id)

    def _resolve_timeout(self, tool_name: str, approval_id: int) -> None:
        """The SDK's deadline passed while still pending. Mark the approval
        timed-out — but a human decision may have landed in the gap between our
        last poll and now, so if the timeout write is rejected (409) we re-read
        and honor the real decision instead of forcing a timeout over it."""
        updated = self._client._decide_approval(approval_id, "timeout")
        if updated is None:
            status = str(self._client._get_approval(approval_id)["status"])
            self._finish_approval(tool_name, approval_id, status)
        else:
            self._finish_approval(tool_name, approval_id, "timeout")

    def request_approval(
        self,
        tool_name: str,
        args: Any = None,
        *,
        timeout_s: int = 300,
        poll_interval_s: float = 2.0,
    ) -> None:
        """Block (synchronously) until a human approves `tool_name`, or raise
        ApprovalDenied on a deny/timeout. Safe for sync agents and manual
        dt.run() code; blocks the calling thread."""
        approval_id = self._begin_approval(tool_name, args, timeout_s)
        deadline = time.monotonic() + timeout_s
        while True:
            status = str(self._client._get_approval(approval_id)["status"])
            if status != "pending":
                self._finish_approval(tool_name, approval_id, status)
                return
            if time.monotonic() >= deadline:
                self._resolve_timeout(tool_name, approval_id)
                return
            time.sleep(poll_interval_s)

    async def arequest_approval(
        self,
        tool_name: str,
        args: Any = None,
        *,
        timeout_s: int = 300,
        poll_interval_s: float = 2.0,
    ) -> None:
        """Async counterpart to request_approval — yields the event loop
        between polls (asyncio.sleep) so an async agent doesn't block. Same
        fail-closed semantics."""
        approval_id = self._begin_approval(tool_name, args, timeout_s)
        deadline = time.monotonic() + timeout_s
        while True:
            status = str(self._client._get_approval(approval_id)["status"])
            if status != "pending":
                self._finish_approval(tool_name, approval_id, status)
                return
            if time.monotonic() >= deadline:
                self._resolve_timeout(tool_name, approval_id)
                return
            await asyncio.sleep(poll_interval_s)

    def _build_eval_context(
        self, *, args: Any = None, event: Optional[Dict[str, Any]] = None
    ) -> EvaluationContext:
        """Assemble the EvaluationContext for expression-condition evaluation.

        Populates the namespaces that have a real runtime source: ``run.*`` (run
        metadata), ``args.*`` (the current tool call's raw arguments, when
        provided — only available at the before_tool_call gate), and ``event.*``.
        ``agent.*``/``org.*`` are intentionally left empty (no SDK source yet —
        see BACKLOG.md); expressions referencing them evaluate as absent.
        """
        run_meta = {
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "agent_version": self.agent_version,
            "step_count": self.step,
            "tool_call_count": len(self.state.tool_calls),
            "error_count": self._error_count,
            "cost_usd": self._cost_usd,
            "event_count": len(self.state.events),
            "duration_ms": int((time.monotonic() - self._started_monotonic) * 1000),
        }
        # event.timestamp / event.hour are snapshotted here, once, from wall-clock
        # (UTC) — like run.duration_ms, they're captured at context-build time and
        # never re-read on the evaluation hot path, so evaluation stays clock-free
        # and deterministic for a given context. hour is UTC 0–23, for
        # business-hours-style policies.
        now = time.time()
        event_ns = dict(event or {})
        event_ns.setdefault("timestamp", now)
        event_ns.setdefault("hour", datetime.datetime.utcfromtimestamp(now).hour)
        return EvaluationContext(
            args=args if isinstance(args, dict) else {},
            run=run_meta,
            event=event_ns,
        )

    def _matched_approval_policy(self, tool_name: str, args: Any = None):
        """The require_approval policy gating this tool call, or None. ``args``
        (the raw call arguments) feed any ``condition.match`` expression so a
        policy can gate on argument values, not just the tool name."""
        engine = self._client._policy_engine  # type: ignore[attr-defined]
        if not len(engine):
            return None
        context = self._build_eval_context(
            args=args,
            event={"type": "before_tool_call", "tool_name": tool_name},
        )
        return engine.find_approval_policy(
            self.agent_id, tool_name, context, observer=self._policy_observer()
        )

    # ── Policy evaluation observability (Phase 5) ───────────────────────────────

    def _policy_observer(self):
        """Return the per-policy evaluation observer callback, or None when
        observability is inactive. Active iff structured DEBUG logging is enabled
        OR dashboard reporting is turned on. When None, the engine takes its
        zero-overhead short-circuit path."""
        reporting = getattr(self._client, "_policy_evaluation_reporting", False)
        if eval_logger.isEnabledFor(logging.DEBUG) or reporting:
            return self._observe_policy_eval
        return None

    def _observe_policy_eval(self, policy, trigger_matched: bool, trace: list, fired: bool) -> None:
        """Record one policy evaluation: structured DEBUG log and/or a shipped
        policy.evaluated event, both rate-limited per policy (100/min, sampled
        beyond). Never raises — observability must not break enforcement."""
        try:
            client = self._client
            log_on = eval_logger.isEnabledFor(logging.DEBUG)
            ship_on = getattr(client, "_policy_evaluation_reporting", False)
            if not (log_on or ship_on):
                return
            limiter = getattr(client, "_policy_eval_rate_limiter", None)
            admit, sampled = limiter.admit(policy.key) if limiter is not None else (True, False)
            if not admit:
                return
            conditions = [t.to_dict() for t in trace]
            trigger = policy.condition.get("trigger", "")
            reason = build_reason(trigger, trigger_matched, fired, conditions)
            record = PolicyEvaluationRecord(
                policy_name=policy.name,
                policy_id=policy.id,
                agent_id=self.agent_id,
                run_id=self.run_id,
                trigger=trigger,
                trigger_matched=trigger_matched,
                fired=fired,
                conditions=conditions,
                reason=reason,
                sampled=sampled,
                ts=time.time(),
            )
            if log_on:
                eval_logger.debug(
                    "policy %r: %s",
                    policy.name,
                    reason,
                    extra={"policy_evaluation": record.to_dict()},
                )
            if ship_on:
                self._ship_evaluation_record(record)
        except Exception as exc:  # observability is best-effort
            logger.debug("policy evaluation observability failed: %s", exc)

    def _ship_evaluation_record(self, record: PolicyEvaluationRecord) -> None:
        """Ship a policy.evaluated event through the normal transport. Uses the
        client's buffer directly (not self._emit) so it does NOT append to
        state.events or re-trigger policy evaluation — no recursion, no run-trace
        pollution. Ingest routes it into the policy_evaluations table."""
        event = AgentEvent(
            event_type=EventType.POLICY_EVALUATED,
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            step_index=self.step,
            payload=record.to_dict(),
            parent_run_id=self._parent_run_id,
            trace_id=self._trace_id,
            conversation_id=self._conversation_id,
        )
        self._client._emit(event)

    def _enforce_tool_approval_sync(self, tool_name: str, args: Any) -> None:
        policy = self._matched_approval_policy(tool_name, args)
        if policy is None:
            return
        params = policy.action.get("params") or {}
        self.request_approval(tool_name, args, timeout_s=params.get("timeout_s", 300))

    async def _enforce_tool_approval_async(self, tool_name: str, args: Any) -> None:
        policy = self._matched_approval_policy(tool_name, args)
        if policy is None:
            return
        params = policy.action.get("params") or {}
        await self.arequest_approval(tool_name, args, timeout_s=params.get("timeout_s", 300))

    def _warn_unread_advisory(self) -> None:
        """audit Finding 25: at run end, warn if an inject_prompt policy fired but
        the injected prompt was never consumed (a common bug — the agent must call
        run.pop_prompt_addition() and prepend it to its next LLM messages). Only the
        unconsumed-list case is reliably detectable without read-tracking; other
        advisory actions (switch_model/recovery_prompt/…) are covered in docs."""
        if self._inject_prompt_fired and self.prompt_additions:
            logger.warning(
                "A policy injected %d prompt(s) via inject_prompt, but they were never "
                "read (run.prompt_additions is still non-empty at run end). Call "
                "run.pop_prompt_addition() and prepend it to your next LLM call — otherwise "
                "the inject_prompt action had no effect. See docs/policies.md 'Enforced vs. "
                "advisory actions'.",
                len(self.prompt_additions),
            )

    def pop_prompt_addition(self) -> str | None:
        """Return and remove the oldest injected prompt addition, or None."""
        return self.prompt_additions.pop(0) if self.prompt_additions else None

    def final_answer(self) -> None:
        """Call when the agent produces its final answer."""
        self.exit_reason = "final_answer"
        self.state.exit_reason = "final_answer"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_policies(self, event_type: "Optional[EventType]" = None) -> None:
        """
        Evaluate registered policies against the current run metrics.

        Called after tool_called, llm_responded, and tool_responded. Raises
        PolicyViolation for 'stop' actions; sets model_override / prompt_additions
        for soft actions; emits POLICY_TRIGGERED for 'log' actions.

        ``event_type`` is the event that triggered this check; it populates
        ``event.type`` in the expression EvaluationContext. Raw tool ``args`` are
        not available in this path (they aren't retained on state) — an
        ``args.*`` expression is a before_tool_call-gate concern, not a metric one.
        """
        engine = self._client._policy_engine  # type: ignore[attr-defined]
        if not len(engine):
            return

        metrics = build_metrics(
            self.state,
            self.step,
            error_count=self._error_count,
            cost_usd=self._cost_usd,
        )

        # Augment metrics with signal detection if any policy uses trigger="signal".
        # We do this lazily to avoid running detectors when no signal policies exist.
        if self._needs_signal is None or self._needs_signal_generation != engine._generation:
            with engine._lock:
                self._needs_signal = any(
                    p.condition.get("trigger") == "signal"
                    for p in engine._policies
                    if p.enabled and p.agent_id in ("*", self.agent_id)
                )
                self._needs_signal_generation = engine._generation
        needs_signal = self._needs_signal
        if needs_signal:
            from dunetrace.detectors import run_detectors

            if self.step != self._detector_cache_step:
                self._detector_cache_signals = run_detectors(self.state, context="runtime")
                self._detector_cache_step = self.step
            sigs = self._detector_cache_signals
            metrics["signal"] = [s.failure_type.value for s in sigs] if sigs else []

        context = self._build_eval_context(
            event={"type": event_type.value} if event_type is not None else None
        )
        result = engine.evaluate(
            self.agent_id,
            metrics,
            self._triggered_policies,
            context,
            observer=self._policy_observer(),
        )
        if result is None:
            return

        policy, action = result
        action_type = action.get("type", "log")
        params = action.get("params") or {}

        # Emit a policy.triggered event for all action types
        self._emit(
            EventType.POLICY_TRIGGERED,
            {
                "policy_name": policy.name,
                "action_type": action_type,
                "trigger": policy.condition.get("trigger"),
                "value": metrics.get(policy.condition.get("trigger", "")),
            },
            advance=False,
        )

        if action_type == "stop":
            self._triggered_policies.add(policy.key)
            raise PolicyViolation(
                policy.name,
                action,
                params.get("message", f"Policy '{policy.name}' stopped the run"),
            )

        elif action_type == "switch_model":
            model = params.get("model")
            if model:
                self.model_override = model
                logger.info("Policy '%s': model_override → %s", policy.name, model)
            self._triggered_policies.add(policy.key)

        elif action_type == "inject_prompt":
            prompt = params.get("prompt", "")
            if prompt:
                self.prompt_additions.append(prompt)
                self._inject_prompt_fired = True  # audit Finding 25: for unread check
                logger.info("Policy '%s': injected prompt (%d chars)", policy.name, len(prompt))
            self._triggered_policies.add(policy.key)

        elif action_type == "stop_current_tts":
            self.stop_tts = True
            logger.info("Policy '%s': stop_current_tts", policy.name)
            self._triggered_policies.add(policy.key)

        elif action_type == "escalate_to_human":
            self.escalate_to_human = True
            reason = params.get("reason")
            if reason:
                self.escalation_reason = reason
            logger.info("Policy '%s': escalate_to_human (reason=%s)", policy.name, reason)
            self._triggered_policies.add(policy.key)

        elif action_type == "inject_recovery_prompt":
            prompt = params.get("prompt", "")
            if prompt:
                self.recovery_prompt = prompt  # last-wins, like model_override
                logger.info("Policy '%s': recovery_prompt set (%d chars)", policy.name, len(prompt))
            self._triggered_policies.add(policy.key)

        elif action_type == "slow_response_pace":
            # Advisory, same contract as the other voice actions: Dunetrace sets the
            # pacing intent, the agent's own loop reads run.response_pace between turns
            # and slows down (filler token while thinking, lower TTS rate). Last-wins.
            self.response_pace = params.get("pace", "slow")
            logger.info("Policy '%s': response_pace → %s", policy.name, self.response_pace)
            self._triggered_policies.add(policy.key)

        elif action_type == "log":
            # Log policies fire every time; don't add to triggered_already
            logger.info(
                "Policy '%s' logged: %s=%s",
                policy.name,
                policy.condition.get("trigger"),
                metrics.get(policy.condition.get("trigger", "")),
            )

    def _emit(self, event_type: EventType, payload: dict, *, advance: bool = True) -> None:
        if advance:
            self.step += 1
        event = AgentEvent(
            event_type=event_type,
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            step_index=self.step,
            payload=payload,
            parent_run_id=self._parent_run_id,
            trace_id=self._trace_id,
            conversation_id=self._conversation_id,
        )
        self.state.events.append(event)
        self._client._emit(event)

        # Evaluate policies after events that change tracked metrics.
        # POLICY_TRIGGERED is excluded to prevent recursive evaluation.
        if event_type in (
            EventType.TOOL_CALLED,
            EventType.LLM_RESPONDED,
            EventType.TOOL_RESPONDED,
        ):
            self._check_policies(event_type)
