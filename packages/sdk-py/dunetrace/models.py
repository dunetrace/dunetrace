"""
Core data models. No external dependencies.
Content fields carry raw text — tool args, tool output, LLM output, and errors
are transmitted as-is to the backend over TLS.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python 3.7
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


# ── Event Types ────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_ERRORED = "run.errored"
    LLM_CALLED = "llm.called"
    LLM_RESPONDED = "llm.responded"
    TOOL_CALLED = "tool.called"
    TOOL_RESPONDED = "tool.responded"
    RETRIEVAL_CALLED = "retrieval.called"
    RETRIEVAL_RESPONDED = "retrieval.responded"
    EXTERNAL_SIGNAL = "external.signal"
    POLICY_TRIGGERED = "policy.triggered"
    # Policy evaluation observability (rate-limited): one record per policy
    # evaluation, shipped via the normal transport but routed by ingest into the
    # policy_evaluations table rather than stored as a run event. Kept in sync
    # with dunetrace_schemas.enums.EventType (test_sdk_parity.py).
    POLICY_EVALUATED = "policy.evaluated"
    # Voice-agent events (detector pack "voice"). Additive: emitted only by the
    # optional voice-specific RunContext helpers; no built-in detector reads
    # them until the voice pack is active. Kept value-for-value in sync with
    # dunetrace_schemas.enums.EventType (enforced by test_sdk_parity.py).
    TRANSCRIPTION_RECEIVED = "transcription.received"
    TTS_GENERATED = "tts.generated"
    VOICE_ACTIVITY_DETECTED = "voice_activity.detected"
    TURN_TAKING = "turn_taking.changed"
    # Human-in-the-loop approval events (Capability 2). Emitted around a
    # require_approval policy gate. Kept value-for-value in sync with
    # dunetrace_schemas.enums.EventType (enforced by test_sdk_parity.py).
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_TIMEOUT = "approval.timeout"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FailureType(str, Enum):
    TOOL_LOOP = "TOOL_LOOP"
    TOOL_THRASHING = "TOOL_THRASHING"
    TOOL_AVOIDANCE = "TOOL_AVOIDANCE"
    GOAL_ABANDONMENT = "GOAL_ABANDONMENT"
    PROMPT_INJECTION_SIGNAL = "PROMPT_INJECTION_SIGNAL"
    RAG_EMPTY_RETRIEVAL = "RAG_EMPTY_RETRIEVAL"
    LLM_TRUNCATION_LOOP = "LLM_TRUNCATION_LOOP"
    CONTEXT_BLOAT = "CONTEXT_BLOAT"
    SLOW_STEP = "SLOW_STEP"
    RETRY_STORM = "RETRY_STORM"
    EMPTY_LLM_RESPONSE = "EMPTY_LLM_RESPONSE"
    STEP_COUNT_INFLATION = "STEP_COUNT_INFLATION"
    CASCADING_TOOL_FAILURE = "CASCADING_TOOL_FAILURE"
    FIRST_STEP_FAILURE = "FIRST_STEP_FAILURE"
    USER_DISSATISFACTION = "USER_DISSATISFACTION"
    INTENT_MISALIGNMENT = "INTENT_MISALIGNMENT"
    REASONING_STALL = "REASONING_STALL"
    CONFIDENT_HALLUCINATION = "CONFIDENT_HALLUCINATION_PROXY"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    COST_SPIKE = "COST_SPIKE"
    SESSION_LATENCY = "SESSION_LATENCY"
    PREMATURE_TERMINATION = "PREMATURE_TERMINATION"
    UNREAD_TOOL_ERROR = "UNREAD_TOOL_ERROR"
    TOOL_ARGUMENT_FABRICATION = "TOOL_ARGUMENT_FABRICATION"
    RETRIEVED_CONTENT_INJECTION = "RETRIEVED_CONTENT_INJECTION"
    HANDOFF_CONTEXT_LOSS = "HANDOFF_CONTEXT_LOSS"
    RUNAWAY_ITERATION = "RUNAWAY_ITERATION"
    CUSTOM = "CUSTOM"  # sentinel for user-defined custom detectors


# ── Agent Event ────────────────────────────────────────────────────────────────


@dataclass
class AgentEvent:
    """A single instrumentation event emitted by the SDK."""

    event_type: EventType
    run_id: str
    agent_id: str
    agent_version: str
    step_index: int
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    # audit Finding 14: a stable per-event id, generated at construction time so
    # it survives the durable retry queue unchanged — the ingest side dedups on
    # it (ON CONFLICT DO NOTHING), so an at-least-once retry never duplicates
    # events (which would otherwise inflate counts into phantom signals).
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "parent_run_id": self.parent_run_id,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "event_id": self.event_id,
        }


# ── Run-level structures (consumed by detectors) ──────────────────────────────


@dataclass
class ToolCall:
    tool_name: str
    args: str
    step_index: int
    timestamp: float
    success: Optional[bool] = None
    error: Optional[str] = None  # raw error message when success=False
    output: Optional[str] = (
        None  # raw tool response body, when the caller passes output= to tool_responded()
    )


@dataclass
class LlmCall:
    """Metadata from a single LLM call/response pair within a run."""

    model: str
    prompt_tokens: Optional[int]
    finish_reason: Optional[str]
    latency_ms: Optional[int]
    step_index: int
    timestamp: float
    output_length: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


@dataclass
class ExternalSignal:
    """
    Infrastructure context attached to an agent step (rate limits, cache misses, upstream outages).

    Emitted via ``run.external_signal("rate_limit", source="openai")``. Does not advance the step
    counter — it annotates the current step so detectors can correlate failures with infra events.
    """

    signal_name: str
    step_index: int
    timestamp: float
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    index_name: str
    result_count: int
    top_score: Optional[float]
    step_index: int
    content: Optional[str] = (
        None  # raw retrieved text, when the caller passes content= to retrieval_responded()
    )


@dataclass
class RunState:
    """Accumulated state for a single agent run. Detectors consume this, not raw events."""

    run_id: str
    agent_id: str
    agent_version: str
    available_tools: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    llm_calls: List[LlmCall] = field(default_factory=list)
    retrievals: List[RetrievalResult] = field(default_factory=list)
    events: List[AgentEvent] = field(default_factory=list)
    external_signals: List[ExternalSignal] = field(default_factory=list)
    step_durations_ms: Dict[int, int] = field(default_factory=dict)
    current_step: int = 0
    exit_reason: Optional[str] = None
    input_text: Optional[str] = None
    system_prompt: Optional[str] = None
    # Cross-run baselines populated by the server before detectors run.
    # None = insufficient history. Local self-hosted mode may leave these None.
    baseline_p75_steps: Optional[float] = None
    baseline_p75_latency_tool: Optional[float] = None  # ms, P75 tool step duration
    baseline_p75_latency_llm: Optional[float] = None  # ms, P75 LLM step duration
    baseline_p75_token_growth: Optional[float] = None  # ratio, P75 context growth factor
    baseline_p75_llm_tool_ratio: Optional[float] = None  # P75 LLM:tool call ratio
    baseline_p75_total_tokens: Optional[float] = (
        None  # P75 total tokens (prompt+completion) per run
    )
    baseline_p75_duration_s: Optional[float] = None  # P75 wall-clock run duration in seconds


# ── Risk Score ────────────────────────────────────────────────────────────────


@dataclass
class RiskScore:
    """
    Aggregate run-level risk assessment produced by RiskEngine.

    Captures the overall confidence that this run contains a behavioral failure,
    along with the per-feature breakdown that explains why. Used by the alert
    layer to gate, prioritize, and explain alerts — not a replacement for
    individual FailureSignals.

    :param confidence:     0.0–1.0. Probability the run contains a real failure.
    :param active_signals: Number of feature dimensions that scored > 0.6.
    :param scores:         Per-feature scores {loop, stagnation, token, retry, latency}.
    :param severity:       Set only when a hard rule fired ("CRITICAL" | "HIGH").
                           None for normal scored runs.
    """

    confidence: float
    active_signals: int
    scores: Dict[str, float]
    severity: Optional[str] = None


# ── Failure Signal ─────────────────────────────────────────────────────────────


@dataclass
class FailureSignal:
    """Output of a detector."""

    failure_type: FailureType
    severity: Severity
    run_id: str
    agent_id: str
    agent_version: str
    step_index: int
    confidence: float
    evidence: Dict[str, Any]
    detected_at: float = field(default_factory=time.time)
    co_signal_count: int = 0


# ── Exporter interface ────────────────────────────────────────────────────────


@runtime_checkable
class Exporter(Protocol):
    """
    Interface for custom event exporters.

    Implement ``handle`` to forward every ``AgentEvent`` to an external sink
    (Splunk, Datadog, a webhook, a custom file, etc.).

    Exporters are called synchronously in ``_emit`` on the agent's thread.
    Implementations should be fast and non-blocking; offload heavy I/O to a
    background thread internally if needed.

    Usage::

        class MyExporter:
            def handle(self, event: AgentEvent) -> None:
                requests.post("https://my-sink.example.com", json=event.to_dict())

        dt = Dunetrace(exporters=[MyExporter()])

    A plain callable also works via ``CallableExporter``::

        dt = Dunetrace(exporters=[CallableExporter(lambda e: print(e.to_dict()))])
    """

    def handle(self, event: "AgentEvent") -> None: ...


class CallableExporter:
    """Wraps a plain ``Callable[[AgentEvent], None]`` as an ``Exporter``."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def handle(self, event: "AgentEvent") -> None:
        self._fn(event)


# ── Helpers ───────────────────────────────────────────────────────────────────


def agent_version(system_prompt: str, model: str, tools: List[str]) -> str:
    """Same config always produces the same 8-char hash — any change produces a new version, preventing deploy-induced false positives."""
    fingerprint = f"{system_prompt}:{model}:{sorted(tools)}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:8]
