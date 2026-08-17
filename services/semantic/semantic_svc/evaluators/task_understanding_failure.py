"""
TASK_UNDERSTANDING_FAILURE — the agent solved the wrong problem. Its response is
aimed at a different task than the user actually asked for.

Run-level, so it wraps plain GEval exactly like TaskCompletionEvaluator (same
LLMTestCase-native, no-deprecated-fallback reasoning that evaluator's docstring
lays out).

Three-way distinction the criteria is worded to hold:
  - HALLUCINATION is about factuality (did it state unsupported things).
  - TASK_COMPLETION is about thoroughness (did it fully do the RIGHT task, or
    only partially).
  - TASK_UNDERSTANDING_FAILURE is about WHICH task: the agent aimed at a
    different task than asked. A perfect, fully-complete answer to the wrong
    question fails this evaluator while passing the other two; a partial attempt
    at the RIGHT task passes this one and is TASK_COMPLETION's concern.

Same polarity as the rest of the package: HIGH score = healthy (right task),
fired = not metric.success, confidence = 1 - score.
"""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

from explainer_svc.cost import estimate_cost
from semantic_svc.evaluators.base import EvalResult, EvaluationInput
from semantic_svc.evaluators.models import build_deepeval_model, resolve_model_name

_CRITERIA = (
    "Determine whether the agent's response addresses the SAME task the user "
    "actually requested, or a DIFFERENT one. This is about which task the agent "
    "chose to do, not how completely it did it. Score LOW when the agent "
    "answered a different question or performed a different task than asked (for "
    "example: the user asked about pricing and the agent described features, or "
    "the user asked to schedule a meeting and the agent explained scheduling "
    "best practices). Score HIGH when the response is aimed at the task the user "
    "asked for, even if it is incomplete — a partial attempt at the RIGHT task "
    "is not a task-understanding failure. If the request was genuinely ambiguous "
    "and the agent picked a reasonable interpretation, score HIGH; handling "
    "ambiguity reasonably is not a failure."
)


class TaskUnderstandingFailureEvaluator:
    name = "TASK_UNDERSTANDING_FAILURE"

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
        # Task understanding is judged from the input vs. the response; tools are
        # useful context but not required. Only ask GEval to weigh TOOLS_CALLED
        # when the run actually has tools — passing that param with no tools raises.
        params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
        if tools_called:
            params.append(SingleTurnParams.TOOLS_CALLED)
        metric = GEval(
            name="TaskUnderstandingFailure",
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
