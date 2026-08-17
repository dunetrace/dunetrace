"""
CONFUSION_LOOP — the user keeps re-asking the same underlying question because
the agent's answers aren't resolving their intent.

Conversation-level, so it wraps DeepEval's ConversationalGEval, exactly like
UserFrustrationEvaluator (its nearest sibling). The distinction from
USER_FRUSTRATION: frustration is about the user's emotional state (complaints,
sarcasm, all-caps); a confusion loop is about intent — the same request coming
back around, rephrased, regardless of tone. A calm user politely re-asking the
same thing three times is a confusion loop but not necessarily frustrated.

Same polarity convention as every evaluator in this package: criteria phrased
so a HIGH score is healthy (the conversation is progressing / the request got
resolved), matching ConversationalGEval's own `success = score >= threshold`.
fired = not metric.success, confidence = 1 - score.
"""

from __future__ import annotations

from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn

from explainer_svc.cost import estimate_cost
from semantic_svc.evaluators.base import ConversationEvaluationInput, EvalResult
from semantic_svc.evaluators.models import build_deepeval_model, resolve_model_name

_CRITERIA = (
    "Determine whether the user is repeatedly asking the same underlying "
    "question across this multi-turn conversation because the agent's answers "
    "are not resolving their intent. Confusion-loop signals: the user "
    "rephrasing the same request, saying things like 'no, I meant...', 'that's "
    "not what I asked', or re-asking something the agent already attempted to "
    "answer. Judge intent similarity, not exact wording. Score LOW when the "
    "user is clearly stuck re-asking the same unresolved thing; score HIGH when "
    "each user turn moves on to a new or progressively different question, or "
    "the agent resolved the request. A genuine follow-up that builds on a "
    "resolved answer is NOT a confusion loop."
)


class ConfusionLoopEvaluator:
    name = "CONFUSION_LOOP"

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
            name="ConfusionLoop",
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
