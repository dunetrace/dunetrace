from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from semantic_svc.evaluators.base import EvaluationInput
from semantic_svc.evaluators.hallucination import HallucinationEvaluator


def _run(context=None):
    return EvaluationInput(
        input_text="What is the capital of France?",
        actual_output="The capital of France is Berlin.",
        context=context or ["The capital of France is Paris."],
    )


class TestHallucinationEvaluator(unittest.TestCase):
    def _build_evaluator(self):
        with patch("semantic_svc.evaluators.hallucination.build_deepeval_model"):
            return HallucinationEvaluator(provider="openai", model_name="gpt-4o-mini")

    def test_fires_when_metric_reports_failure(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=False,
            score=0.9,
            reason="Contradicts context",
            input_tokens=100,
            output_tokens=50,
        )
        with patch(
            "semantic_svc.evaluators.hallucination.HallucinationMetric", return_value=fake_metric
        ):
            result = evaluator.evaluate(_run())

        self.assertTrue(result.fired)
        self.assertEqual(result.evaluator, "HALLUCINATION")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.reasoning, "Contradicts context")

    def test_does_not_fire_when_metric_reports_success(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=0.1, reason="Grounded", input_tokens=100, output_tokens=50
        )
        with patch(
            "semantic_svc.evaluators.hallucination.HallucinationMetric", return_value=fake_metric
        ):
            result = evaluator.evaluate(_run())

        self.assertFalse(result.fired)

    def test_cost_computed_from_dunetrace_price_table_not_deepeval(self):
        evaluator = self._build_evaluator()
        # If cost were pulled from metric.evaluation_cost, this would be 999.0 —
        # asserting it isn't proves estimate_cost() is the actual source.
        fake_metric = MagicMock(
            success=True,
            score=0.1,
            reason="ok",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            evaluation_cost=999.0,
        )
        with patch(
            "semantic_svc.evaluators.hallucination.HallucinationMetric", return_value=fake_metric
        ):
            result = evaluator.evaluate(_run())

        # gpt-4o-mini: (0.15 + 0.60) per 1M/1M tokens = 0.75
        self.assertAlmostEqual(result.cost_usd, 0.75)
        self.assertNotEqual(result.cost_usd, 999.0)

    def test_token_counts_pass_through(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=0.0, reason="", input_tokens=42, output_tokens=7
        )
        with patch(
            "semantic_svc.evaluators.hallucination.HallucinationMetric", return_value=fake_metric
        ):
            result = evaluator.evaluate(_run())

        self.assertEqual(result.prompt_tokens, 42)
        self.assertEqual(result.completion_tokens, 7)

    def test_none_token_counts_default_to_zero(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=0.0, reason="", input_tokens=None, output_tokens=None
        )
        with patch(
            "semantic_svc.evaluators.hallucination.HallucinationMetric", return_value=fake_metric
        ):
            result = evaluator.evaluate(_run())

        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)

    def test_empty_context_accepted_not_rejected(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=0.0, reason="", input_tokens=10, output_tokens=5
        )
        with patch(
            "semantic_svc.evaluators.hallucination.HallucinationMetric", return_value=fake_metric
        ):
            result = evaluator.evaluate(_run(context=[]))
        self.assertFalse(result.fired)


if __name__ == "__main__":
    unittest.main()
