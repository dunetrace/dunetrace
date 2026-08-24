"""
Task completion, via DeepEval's GEval — not TaskCompletionMetric.

TaskCompletionMetric's documented path expects a live DeepEval trace
(@observe-decorated agent); its fallback for a plain LLMTestCase (input,
actual_output, tools_called, no _trace_dict) exists in the 4.0.9 source but is
marked "# TODO: Deprecate this soon" there. semantic_svc has no live agent to
instrument — it evaluates already-completed runs reconstructed from stored
events — so that fallback would be the only path available, indefinitely.
GEval with a task-completion rubric is fully LLMTestCase-native (no
trace-shaped fallback, no deprecation risk) and still 100% DeepEval, so it
satisfies "embed DeepEval as the evaluator engine" without depending on a path
DeepEval's own maintainers flagged for removal. Maintainer decision, not a
silent substitution — see BACKLOG.md.
"""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

from explainer_svc.cost import estimate_cost
from semantic_svc.evaluators.base import EvalResult, EvaluationInput
from semantic_svc.evaluators.models import build_deepeval_model, resolve_model_name

_CRITERIA = (
    "Determine whether the agent accomplished the task described in the "
    "input. Consider the tools it called and their outputs as evidence of "
    "what the agent actually did, not just what it claims in its final "
    "output. Score low if the output is a refusal, a partial attempt, or "
    "unrelated to what the input asked for."
)


class TaskCompletionEvaluator:
    name = "TASK_COMPLETION"

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
        # Completion is judged from the input vs. the response; tools are
        # corroborating evidence but not required — a purely conversational turn
        # completes its task with no tool calls at all. Only ask GEval to weigh
        # TOOLS_CALLED when the run actually has tools: DeepEval's param check
        # rejects a None value for any declared param, and tools_called is None
        # here whenever the run called nothing, so passing that param with no
        # tools raises MissingTestCaseParamsError before anything is scored.
        params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
        if tools_called:
            params.append(SingleTurnParams.TOOLS_CALLED)
        metric = GEval(
            name="TaskCompletion",
            criteria=_CRITERIA,
            evaluation_params=params,
            model=self._model,
            threshold=self._threshold,
        )
        metric.measure(test_case)

        # success = score >= threshold (HIGH score = task completed).
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
