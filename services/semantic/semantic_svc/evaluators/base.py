"""
Dunetrace-native evaluator interface. Every evaluator wraps a DeepEval metric
but exposes a uniform result shape — so worker.py (and later, the signal
writer) never needs to know which underlying metric produced it, or which
direction that metric's raw score runs in (DeepEval's own metrics disagree:
HallucinationMetric succeeds on a LOW score, GEval succeeds on a HIGH one).
This is the seam Phase 1.3's brief asked for: swap the wrapped metric later
without touching worker.py.

Phase 3.2 adds ConversationEvaluator alongside RunEvaluator (renamed from the
original single `Evaluator` — no external module imported that name, so this
is a free rename) for evaluators that need multiple runs' worth of history
(e.g. UserFrustrationEvaluator), not just one. Both share the same EvalResult
shape; only the input differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ToolCallData:
    name: str
    input_parameters: dict | None = None
    output: str | None = None


@dataclass(frozen=True)
class EvaluationInput:
    """The minimum a semantic evaluator needs, reconstructed from Dunetrace's
    own stored events — deliberately not detector_svc.run_builder.RunState,
    which carries structural-detection fields (baselines, step timing) no
    evaluator here reads, and omits the one field they all need: the LLM's
    actual output text (RunState.LlmCall keeps only output_length)."""

    input_text: str
    actual_output: str
    context: list[str] = field(default_factory=list)
    tools_called: list[ToolCallData] = field(default_factory=list)


@dataclass(frozen=True)
class EvalResult:
    evaluator: str
    fired: bool  # True = this evaluator detected a problem
    confidence: float  # 0.0-1.0, direction-normalized — NOT the raw metric score
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class RunEvaluator(Protocol):
    name: str

    def evaluate(self, run: EvaluationInput) -> EvalResult: ...


@dataclass(frozen=True)
class ConversationTurn:
    """One run's worth of content, reduced to what UserFrustrationEvaluator
    (and any future ConversationEvaluator) needs — the same input_text/
    actual_output pair RunEvaluators use, plus the run_id it came from
    (evidence needs to cite which runs were actually considered)."""

    run_id: str
    input_text: str
    actual_output: str


@dataclass(frozen=True)
class ConversationEvaluationInput:
    """Ordered oldest -> newest. Deliberately just the last N runs (see
    worker.py), not the whole conversation history — bounding both LLM
    context size and cost."""

    turns: list[ConversationTurn] = field(default_factory=list)


class ConversationEvaluator(Protocol):
    name: str

    def evaluate(self, conversation: ConversationEvaluationInput) -> EvalResult: ...
