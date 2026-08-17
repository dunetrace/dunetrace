"""
Wraps DeepEval's HallucinationMetric. Verified against DeepEval 4.0.9 source
(Phase 1.3 discovery notes): requires INPUT, ACTUAL_OUTPUT, CONTEXT on the
LLMTestCase; success = score <= threshold (LOW score = less hallucination). An
empty context list (a run with no retrieval events) is accepted, not rejected
— DeepEval's own param check only rejects None, and the metric correctly
finds nothing to contradict, scoring low. Routing only retrieval-bearing runs
here for a meaningful finding is the adaptive sampling engine's job (Phase
1.2's retrieval-rate tier), not this evaluator's.
"""

from __future__ import annotations

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from explainer_svc.cost import estimate_cost
from semantic_svc.evaluators.base import EvalResult, EvaluationInput
from semantic_svc.evaluators.models import build_deepeval_model, resolve_model_name


class HallucinationEvaluator:
    name = "HALLUCINATION"

    def __init__(self, provider: str, model_name: str | None = None, threshold: float = 0.5):
        self._model_name = resolve_model_name(provider, model_name)
        self._model = build_deepeval_model(provider, self._model_name)
        self._threshold = threshold

    def evaluate(self, run: EvaluationInput) -> EvalResult:
        test_case = LLMTestCase(
            input=run.input_text,
            actual_output=run.actual_output,
            context=run.context,
        )
        metric = HallucinationMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        metric.measure(test_case)

        prompt_tokens = metric.input_tokens or 0
        completion_tokens = metric.output_tokens or 0
        return EvalResult(
            evaluator=self.name,
            fired=not metric.success,
            confidence=round(metric.score, 4) if metric.score is not None else 0.0,
            reasoning=metric.reason or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            # Cost is computed from Dunetrace's own price table, not
            # metric.evaluation_cost — see evaluators/models.py's docstring
            # for why DeepEval's own bundled pricing can't be trusted here.
            cost_usd=estimate_cost(self._model_name, prompt_tokens, completion_tokens),
        )
