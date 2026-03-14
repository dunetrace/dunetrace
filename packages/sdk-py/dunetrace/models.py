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
    RUN_STARTED         = "run.started"
    RUN_COMPLETED       = "run.completed"
    RUN_ERRORED         = "run.errored"
    LLM_CALLED          = "llm.called"
    LLM_RESPONDED       = "llm.responded"
    TOOL_CALLED         = "tool.called"
    TOOL_RESPONDED      = "tool.responded"
    RETRIEVAL_CALLED    = "retrieval.called"
    RETRIEVAL_RESPONDED = "retrieval.responded"
    EXTERNAL_SIGNAL     = "external.signal"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class FailureType(str, Enum):
    TOOL_LOOP               = "TOOL_LOOP"
    TOOL_THRASHING          = "TOOL_THRASHING"
    TOOL_AVOIDANCE          = "TOOL_AVOIDANCE"
    GOAL_ABANDONMENT        = "GOAL_ABANDONMENT"
    PROMPT_INJECTION_SIGNAL = "PROMPT_INJECTION_SIGNAL"
    RAG_EMPTY_RETRIEVAL     = "RAG_EMPTY_RETRIEVAL"
    LLM_TRUNCATION_LOOP     = "LLM_TRUNCATION_LOOP"
    CONTEXT_BLOAT           = "CONTEXT_BLOAT"
    SLOW_STEP               = "SLOW_STEP"
    RETRY_STORM             = "RETRY_STORM"
    EMPTY_LLM_RESPONSE      = "EMPTY_LLM_RESPONSE"
    STEP_COUNT_INFLATION    = "STEP_COUNT_INFLATION"
    CASCADING_TOOL_FAILURE  = "CASCADING_TOOL_FAILURE"
    FIRST_STEP_FAILURE      = "FIRST_STEP_FAILURE"
    USER_DISSATISFACTION    = "USER_DISSATISFACTION"
    INTENT_MISALIGNMENT     = "INTENT_MISALIGNMENT"
    REASONING_STALL         = "REASONING_STALL"
    CONFIDENT_HALLUCINATION = "CONFIDENT_HALLUCINATION_PROXY"
    POLICY_VIOLATION        = "POLICY_VIOLATION"


# ── Agent Event ────────────────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    """
    A single instrumentation event emitted by the SDK.
    All content is hashed i.e. no raw prompts or outputs are ever sent.
    """
    event_type:    EventType
    run_id:        str
    agent_id:      str
    agent_version: str
    step_index:    int
    timestamp:     float = field(default_factory=time.time)
    payload:       Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type":    self.event_type.value,
            "run_id":        self.run_id,
            "agent_id":      self.agent_id,
            "agent_version": self.agent_version,
            "step_index":    self.step_index,
            "timestamp":     self.timestamp,
            "payload":       self.payload,
            "parent_run_id": self.parent_run_id,
        }


# ── Run-level structures (consumed by detectors) ──────────────────────────────

@dataclass
class ToolCall:
    tool_name:  str
    args_hash:  str
    step_index: int
    timestamp:  float
    success:    Optional[bool] = None
    error_hash: Optional[str]  = None  # hash_content(error_message) when success=False


@dataclass
class LlmCall:
    """Metadata from a single LLM call/response pair within a run."""
    model:         str
    prompt_tokens: Optional[int]
    finish_reason: Optional[str]
    latency_ms:    Optional[int]
    step_index:    int
    timestamp:     float
    output_length: Optional[int] = None


@dataclass
class ExternalSignal:
    """
    Infrastructure context emitted alongside agent events.

    Emitted via ``run.external_signal("rate_limit", source="openai")``.
    Does not advance the step counter — it annotates the current agent step,
    not a new one. Detectors use this to correlate failures with known
    infrastructure events (rate limits, cache misses, upstream outages).
    """
    signal_name: str
    step_index:  int
    timestamp:   float
    source:      str = ""
    meta:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    index_name:   str
    result_count: int
    top_score:    Optional[float]
    step_index:   int


@dataclass
class RunState:
    """
    Accumulated state for a single agent run.
    Detectors operate on this i.e. never on raw events.
    """
    run_id:          str
    agent_id:        str
    agent_version:   str
    available_tools: List[str]             = field(default_factory=list)
    tool_calls:      List[ToolCall]        = field(default_factory=list)
    llm_calls:       List[LlmCall]         = field(default_factory=list)
    retrievals:      List[RetrievalResult] = field(default_factory=list)
    events:          List[AgentEvent]      = field(default_factory=list)
    external_signals:  List[ExternalSignal] = field(default_factory=list)
    step_durations_ms: Dict[int, int]      = field(default_factory=dict)
    current_step:    int                   = 0
    exit_reason:     Optional[str]         = None
    input_text_hash: Optional[str]         = None
    # Cross-run baseline populated by the server before detectors run.
    # None = insufficient history. Local self-hosted mode may leave this None.
    baseline_p75_steps: Optional[float]    = None


# ── Failure Signal ─────────────────────────────────────────────────────────────

@dataclass
class FailureSignal:
    """Output of a detector."""
    failure_type:  FailureType
    severity:      Severity
    run_id:        str
    agent_id:      str
    agent_version: str
    step_index:    int
    confidence:    float
    evidence:      Dict[str, Any]
    detected_at:   float = field(default_factory=time.time)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_content(text: str) -> str:
    """SHA-256, truncated to 16 chars. Used for all content fields."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def agent_version(system_prompt: str, model: str, tools: List[str]) -> str:
    """
    Deterministic version hash.
    Same config (prompt + model + tools) -> same version string.
    Any change -> new version, preventing deploy-induced false positives.
    """
    fingerprint = f"{system_prompt}:{model}:{sorted(tools)}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:8]
