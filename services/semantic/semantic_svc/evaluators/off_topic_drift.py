"""
OFF_TOPIC_DRIFT — the response starts on the user's question and then wanders
off it (into unrelated features, marketing language, tangents, a different
subject).

Run-level, wraps plain GEval like TaskCompletion and TaskUnderstandingFailure.

Distinct from TASK_UNDERSTANDING_FAILURE: that fires when the agent aimed at the
WRONG task from the start; drift is when the agent aimed at the RIGHT task and
lost the thread partway through. A response that is off-topic from its first
word is a task-understanding failure; one that answers the question and then
trails into unrelated material is drift.

Same polarity as the package: HIGH score = healthy (stayed on topic), fired =
not metric.success, confidence = 1 - score.
"""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

from explainer_svc.cost import estimate_cost
from semantic_svc.evaluators.base import EvalResult, EvaluationInput
from semantic_svc.evaluators.models import build_deepeval_model, resolve_model_name

_CRITERIA = (
    "Determine whether the agent's response stays focused on the user's question "
    "and topic, or drifts away from it partway through. This is about a response "
    "that STARTS addressing the right thing and then wanders off (into unrelated "
    "features, marketing language, tangents, or a different subject) — not about "
    "misreading the task from the start. Score LOW when the response begins on "
    "the user's question but ends somewhere unrelated. Score HIGH when the "
    "response stays on the user's topic throughout, even if it covers several "
    "related aspects of a broad or multi-faceted question. Covering multiple "
    "RELATED facets of what was asked is not drift; wandering into UNRELATED "
    "material is."
)


class OffTopicDriftEvaluator:
    name = "OFF_TOPIC_DRIFT"

    def __init__(self, provider: str, model_name: str | None = None, threshold: float = 0.5):
        self._model_name = resolve_model_name(provider, model_name)
        self._model = build_deepeval_model(provider, self._model_name)
        self._threshold = threshold

    def evaluate(self, run: EvaluationInput) -> EvalResult:
        tools_called = [
            ToolCall(name=t.name, input_parameters=t.input_parameters, output=t.output)
            for t in run.tools_called
        ]
        test_case = LLMTestCase(
            input=run.input_text,
            actual_output=run.actual_output,
            tools_called=tools_called or None,
        )
        # Drift is judged from the question vs. the response; tools are optional
        # context. Only ask GEval to weigh TOOLS_CALLED when the run has tools
        # (passing that param with no tools raises).
        params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
        if tools_called:
            params.append(SingleTurnParams.TOOLS_CALLED)
        metric = GEval(
            name="OffTopicDrift",
            criteria=_CRITERIA,
            evaluation_params=params,
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
