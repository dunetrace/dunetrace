"""
Wraps DeepEval's ConversationalGEval — DeepEval's real multi-turn evaluation
metric (verified against the installed 4.0.9 package: deepeval.metrics.
ConversationalGEval, deepeval.test_case.{ConversationalTestCase, Turn,
MultiTurnParams} all import and construct as expected). Same "no dedicated
metric exists for this rubric" situation TaskCompletionEvaluator already
handles with plain GEval — this is its conversational counterpart.

Same polarity convention as TaskCompletionEvaluator, not HallucinationEvaluator:
criteria is phrased so a HIGH score is healthy (a satisfied, non-frustrated
user), matching how ConversationalGEval's own `success = score >= threshold`
already works — fired = not metric.success, confidence = 1 - score. Keeping
every evaluator in this package on the same polarity (higher raw score always
means healthier) is deliberate, so callers never need per-evaluator direction
logic.
"""

from __future__ import annotations

from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn

from explainer_svc.cost import estimate_cost
from semantic_svc.evaluators.base import ConversationEvaluationInput, EvalResult
from semantic_svc.evaluators.models import build_deepeval_model, resolve_model_name

_CRITERIA = (
    "Determine how satisfied and calm the user seems across this multi-turn "
    "conversation with an AI agent. Look for escalating frustration signals: "
    "repeated rephrasing of the same request, explicit complaints, all-caps "
    "or exclamation-heavy messages, sarcasm, or the user giving up mid-task. "
    "Score low if the user shows clear or escalating frustration by the most "
    "recent turn; score high if the user seems satisfied or the conversation "
    "is proceeding normally."
)


class UserFrustrationEvaluator:
    name = "USER_FRUSTRATION"

    def __init__(self, provider: str, model_name: str | None = None, threshold: float = 0.5):
        self._model_name = resolve_model_name(provider, model_name)
        self._model = build_deepeval_model(provider, self._model_name)
        self._threshold = threshold

    def evaluate(self, conversation: ConversationEvaluationInput) -> EvalResult:
        turns: list[Turn] = []
        for t in conversation.turns:
            turns.append(Turn(role="user", content=t.input_text))
            turns.append(Turn(role="assistant", content=t.actual_output))

        test_case = ConversationalTestCase(turns=turns)
        metric = ConversationalGEval(
            name="UserFrustration",
            criteria=_CRITERIA,
            evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
            model=self._model,
            threshold=self._threshold,
        )
        metric.measure(test_case)

        score = metric.score if metric.score is not None else 1.0
        prompt_tokens = metric.input_tokens or 0
        completion_tokens = metric.output_tokens or 0
        return EvalResult(
            evaluator=self.name,
            fired=not metric.success,
            confidence=round(1.0 - score, 4),
            reasoning=metric.reason or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimate_cost(self._model_name, prompt_tokens, completion_tokens),
        )
