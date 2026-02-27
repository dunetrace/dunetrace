"""
dunetrace/models.py
Core data models shared across SDK and services.
Zero external dependencies.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ── Event Types ────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    RUN_STARTED       = "run.started"
    RUN_COMPLETED     = "run.completed"
    RUN_ERRORED       = "run.errored"
    LLM_CALLED        = "llm.called"
    LLM_RESPONDED     = "llm.responded"
    TOOL_CALLED       = "tool.called"
    TOOL_RESPONDED    = "tool.responded"
    RETRIEVAL_CALLED  = "retrieval.called"    # RAG support
    RETRIEVAL_RESPONDED = "retrieval.responded"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class FailureType(str, Enum):
    TOOL_LOOP                  = "TOOL_LOOP"
    TOOL_THRASHING             = "TOOL_THRASHING"
    TOOL_AVOIDANCE             = "TOOL_AVOIDANCE"
    GOAL_ABANDONMENT           = "GOAL_ABANDONMENT"
    PROMPT_INJECTION_SIGNAL    = "PROMPT_INJECTION_SIGNAL"
    RAG_EMPTY_RETRIEVAL        = "RAG_EMPTY_RETRIEVAL"
    LLM_TRUNCATION_LOOP        = "LLM_TRUNCATION_LOOP"
    CONTEXT_BLOAT              = "CONTEXT_BLOAT"
    SLOW_STEP                  = "SLOW_STEP"
    USER_DISSATISFACTION       = "USER_DISSATISFACTION"
    INTENT_MISALIGNMENT        = "INTENT_MISALIGNMENT"
    REASONING_STALL            = "REASONING_STALL"
    CONFIDENT_HALLUCINATION    = "CONFIDENT_HALLUCINATION_PROXY"
    POLICY_VIOLATION           = "POLICY_VIOLATION"


# ── Agent Event ────────────────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    """
    A single instrumentation event emitted by the SDK.
    All content is hashed — no raw prompts or outputs are ever sent.
    """
    event_type:    EventType
    run_id:        str
    agent_id:      str
    agent_version: str
    step_index:    int
    timestamp:     float = field(default_factory=time.time)
    payload:       Dict[str, Any] = field(default_factory=dict)

    # Optional: enables multi-agent trace linking
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


# ── Run State ──────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_name:  str
    args_hash:  str
    step_index: int
    timestamp:  float


@dataclass
class LlmCall:
    """Metadata from a single LLM call/response pair within a run."""
    model:         str
    prompt_tokens: Optional[int]      # from llm.called payload
    finish_reason: Optional[str]      # from llm.responded payload ("stop" | "length" | "tool_calls")
    latency_ms:    Optional[int]      # from llm.responded payload
    step_index:    int
    timestamp:     float


@dataclass
class RetrievalResult:
    index_name:    str
    result_count:  int
    top_score:     Optional[float]  # relevance score 0.0–1.0
    step_index:    int


@dataclass
class RunState:
    """
    Accumulated state for a single agent run.
    Detectors operate on this — never on raw events.
    """
    run_id:          str
    agent_id:        str
    agent_version:   str
    available_tools: List[str]           = field(default_factory=list)
    tool_calls:      List[ToolCall]      = field(default_factory=list)
    llm_calls:       List[LlmCall]       = field(default_factory=list)
    retrievals:      List[RetrievalResult] = field(default_factory=list)
    events:          List[AgentEvent]    = field(default_factory=list)
    # Gap (ms) between event[i] and event[i+1], keyed by the earlier event's step_index.
    # Computed by run_builder after all events are processed.
    # Used by SlowStepDetector and the dashboard duration strip.
    step_durations_ms: Dict[int, int]    = field(default_factory=dict)
    current_step:    int                 = 0
    exit_reason:     Optional[str]       = None
    input_text_hash: Optional[str]       = None


# ── Failure Signal ─────────────────────────────────────────────────────────────

@dataclass
class FailureSignal:
    """
    Output of a detector. Consumed by the Explain layer.
    """
    failure_type: FailureType
    severity:     Severity
    run_id:       str
    agent_id:     str
    agent_version: str
    step_index:   int
    confidence:   float              # 0.0–1.0
    evidence:     Dict[str, Any]     # raw data that triggered the signal
    detected_at:  float = field(default_factory=time.time)


# ── Helpers ────────────────────────────────────────────────────────────────────

def hash_content(text: str) -> str:
    """SHA-256, truncated to 16 chars. Used for all content fields."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def agent_version(system_prompt: str, model: str, tools: List[str]) -> str:
    """
    Deterministic version hash.
    Same config (prompt + model + tools) → same version string.
    Any change → new version. Prevents deploy-induced false positives.
    """
    fingerprint = f"{system_prompt}:{model}:{sorted(tools)}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:8]


# ── Explanation (output of explain layer) ──────────────────────────────────────
# Lives here so all services (explainer, alerts, API) can import it
# without cross-service imports.

@dataclass
class CodeFix:
    """A concrete, copy-pasteable code suggestion."""
    description: str
    language:    str           # "python" | "yaml" | "text"
    code:        str


@dataclass
class Explanation:
    """Human-readable explanation of a FailureSignal."""
    failure_type:     str
    severity:         str
    run_id:           str
    agent_id:         str
    agent_version:    str
    confidence:       float
    title:            str
    what:             str
    why_it_matters:   str
    evidence_summary: str
    suggested_fixes:  List["CodeFix"]
    step_index:       int
    detected_at:      float
    evidence:         Dict[str, Any] = field(default_factory=dict)

    def confidence_pct(self) -> str:
        return f"{int(self.confidence * 100)}%"

    def as_slack_text(self) -> str:
        lines = [
            f":rotating_light: *{self.title}*  |  {self.severity}  |  {self.confidence_pct()} confidence",
            f"*Agent:* `{self.agent_id}` (v`{self.agent_version}`)  •  *Run:* `{self.run_id}`",
            "",
            f"*What happened:* {self.what}",
            f"*Why it matters:* {self.why_it_matters}",
            "",
            f"*Evidence:* {self.evidence_summary}",
        ]
        if self.suggested_fixes:
            lines.append("")
            lines.append(f"*Top fix:* {self.suggested_fixes[0].description}")
            if self.suggested_fixes[0].code:
                lines.append(f"```{self.suggested_fixes[0].code}```")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_type":     self.failure_type,
            "severity":         self.severity,
            "run_id":           self.run_id,
            "agent_id":         self.agent_id,
            "agent_version":    self.agent_version,
            "confidence":       self.confidence,
            "title":            self.title,
            "what":             self.what,
            "why_it_matters":   self.why_it_matters,
            "evidence_summary": self.evidence_summary,
            "suggested_fixes": [
                {"description": f.description, "language": f.language, "code": f.code}
                for f in self.suggested_fixes
            ],
            "step_index":  self.step_index,
            "detected_at": self.detected_at,
            "evidence":    self.evidence,
        }


# ── Explanation (produced by the explain layer, consumed by alerts) ─────────────

from typing import List as _List  # avoid shadowing above

@dataclass
class CodeFix:
    """A concrete, copy-pasteable code suggestion."""
    description: str
    language:    str       # "python" | "yaml" | "text"
    code:        str


@dataclass
class Explanation:
    """Human-readable explanation of a FailureSignal."""
    failure_type:     str
    severity:         str
    run_id:           str
    agent_id:         str
    agent_version:    str
    confidence:       float
    title:            str
    what:             str
    why_it_matters:   str
    evidence_summary: str
    suggested_fixes:  _List[CodeFix]
    step_index:       int
    detected_at:      float
    evidence:         Dict[str, Any] = field(default_factory=dict)

    def confidence_pct(self) -> str:
        return f"{int(self.confidence * 100)}%"

    def as_slack_text(self) -> str:
        lines = [
            f":rotating_light: *{self.title}*  |  {self.severity}  |  {self.confidence_pct()} confidence",
            f"*Agent:* `{self.agent_id}` (v`{self.agent_version}`)  •  *Run:* `{self.run_id}`",
            "",
            f"*What happened:* {self.what}",
            f"*Why it matters:* {self.why_it_matters}",
            "",
            f"*Evidence:* {self.evidence_summary}",
        ]
        if self.suggested_fixes:
            lines.append("")
            lines.append(f"*Top fix:* {self.suggested_fixes[0].description}")
            if self.suggested_fixes[0].code:
                lines.append(f"```{self.suggested_fixes[0].code}```")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_type":     self.failure_type,
            "severity":         self.severity,
            "run_id":           self.run_id,
            "agent_id":         self.agent_id,
            "agent_version":    self.agent_version,
            "confidence":       self.confidence,
            "title":            self.title,
            "what":             self.what,
            "why_it_matters":   self.why_it_matters,
            "evidence_summary": self.evidence_summary,
            "suggested_fixes":  [{"description": f.description, "language": f.language, "code": f.code} for f in self.suggested_fixes],
            "step_index":       self.step_index,
            "detected_at":      self.detected_at,
            "evidence":         self.evidence,
        }