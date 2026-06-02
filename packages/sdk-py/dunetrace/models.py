"""
Core data models. No external dependencies.
Content fields store SHA-256 hashes — raw text never leaves your process.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


# ── Agent Event ────────────────────────────────────────────────────────────────


@dataclass
class AgentEvent:
    """A single instrumentation event emitted by the SDK. All content fields are hashed — no raw prompts or outputs."""

    event_type: EventType
    run_id: str
    agent_id: str
    agent_version: str
    step_index: int
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None

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
        }


# ── Run-level structures (consumed by detectors) ──────────────────────────────


@dataclass
class ToolCall:
    tool_name: str
    args_hash: str
    step_index: int
    timestamp: float
    success: Optional[bool] = None
    error_hash: Optional[str] = None  # hash_content(error_message) when success=False


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
    input_text_hash: Optional[str] = None
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def hash_content(text: str) -> str:
    """SHA-256, truncated to 16 chars. Used for all content fields."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def agent_version(system_prompt: str, model: str, tools: List[str]) -> str:
    """Same config always produces the same 8-char hash — any change produces a new version, preventing deploy-induced false positives."""
    fingerprint = f"{system_prompt}:{model}:{sorted(tools)}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:8]
