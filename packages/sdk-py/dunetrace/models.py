"""
Core data models. No external dependencies.
Content fields carry text, not hashes — tool args, tool output, LLM output and
errors reach the backend readable, over TLS. Before they are shipped the SDK
redacts credential-shaped keys and caps every content field (see
``dunetrace.redaction``: ``DEFAULT_DENYLIST``, ``max_field_chars`` — default
8192 characters, with ``<field>_truncated`` / ``<field>_original_length``
markers on the wire when a cap applies).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python 3.7
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


# ── Event Types ────────────────────────────────────────────────────────────────


# EventType, Severity and FailureType are generated — see
# packages/schemas-py/dunetrace_schemas/enum_source.py (the single source of
# truth, shared with dunetrace_schemas.enums) and scripts/gen_enums.py. They are
# re-exported here so `from dunetrace.models import EventType` keeps working.
from dunetrace._enums import EventType, FailureType, Severity


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
    output_length: Optional[int] = None
    output: Optional[str] = (
        None  # raw tool response body, when the caller passes output= to tool_responded()
    )
    # True length of `args` before any transport-side truncation. The OTLP
    # ingest path caps stored args at OTLP_MAX_ATTR_CHARS (8192), which is below
    # OVERSIZED_TOOL_ARGUMENTS' threshold — so `len(args)` alone could never fire
    # that detector on an OTel-ingested run. The SDK path sets it too whenever
    # its own cap (max_field_chars, default 8192) truncated `args`; None means
    # `len(args)` is already the true length.
    args_length: Optional[int] = None


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
    # Raw LLM output text. Nullable: None when the caller didn't pass output=, or
    # when transmission was opted out (DUNETRACE_OMIT_LLM_OUTPUT_TEXT=1) on the
    # server-reconstructed side. Detectors/evaluators that only need the size
    # keep reading output_length; those that need the text read this.
    output_text: Optional[str] = None
    # Which vendor served the call: "openai", "anthropic", "mistral". Set by the
    # auto-instrumentation patchers, which know the answer for free. Nullable:
    # manual run.llm_called() callers and events recorded before this field
    # existed leave it None, and cost lookup never depends on it (the price
    # tables key off the model name alone). Declared last so the positional
    # construction used across the detector tests keeps working.
    provider: Optional[str] = None
    # Per-run sequence number correlating this call with its response event.
    # Ordering alone cannot do it: a streamed call's llm.responded lands whenever
    # the caller drains the stream, so two overlapping streams emit
    # called(A), called(B), responded(A), responded(B) — and the server-side
    # builders' LIFO pop would hand B's response to A. None for manual callers
    # and for events recorded before this field existed; the builders fall back
    # to positional pairing then.
    call_id: Optional[int] = None
    # True when prompt_tokens is the SDK's own chars//4 estimate rather than a
    # figure the provider reported. The SDK has always known which it wrote
    # (auto.py's llm_called passes an estimate; _emit_*_response overrides with
    # usage) and threw the distinction away, so nothing downstream could tell an
    # approximate token count from an exact one.
    prompt_tokens_estimated: bool = False
    # Set when the response object could not be read. The value names the shape
    # that defeated extraction, e.g. "openai_response_shape:LegacyAPIResponse".
    # None means extraction succeeded — NOT that it was never attempted, which
    # is why the extractors return None on failure instead of a plausible
    # default: a fabricated ("", "stop") pair is indistinguishable from a real
    # empty response and fires EMPTY_LLM_RESPONSE on every run.
    instrumentation_degraded: Optional[str] = None


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
class MemoryEvent:
    """A single agent-memory operation (write/read/clear) within a run.

    Emitted via ``run.memory_written``/``memory_read``/``memory_cleared`` or
    auto-captured from framework memory (LangGraph store, CrewAI memory). Like
    ``ExternalSignal`` it annotates the current step rather than advancing it.
    Server-side, ``run_builder`` reconstructs these from ``memory.*`` events so
    the MEMORY_POISONING detector can inspect written content and its provenance.
    """

    op: str  # "written" | "read" | "cleared"
    key: Optional[str]  # None only for a clear-all (memory_cleared() with no key)
    step_index: int
    timestamp: float
    value: Optional[str] = None  # written content; None for read/clear
    source: Optional[str] = None  # provenance of a written value; None when unknown


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
    memory_events: List[MemoryEvent] = field(default_factory=list)
    step_durations_ms: Dict[int, int] = field(default_factory=dict)
    current_step: int = 0
    exit_reason: Optional[str] = None
    input_text: Optional[str] = None
    system_prompt: Optional[str] = None
    # Events the SDK's outbound buffer shed for this run under overload, read
    # from the run.completed / run.errored payload. > 0 means the run is
    # incomplete: the detector holds its signals in shadow with capped
    # confidence and severity rather than trusting a verdict on partial data.
    dropped_events: int = 0
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

    Captures the overall confidence that this run contains a structural failure,
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
