"""
Unit tests for TaskUnderstandingFailureEvaluator — wrapper mechanics only
(polarity, confidence, cost, test-case construction), GEval mocked so they run
offline. The behavioral cases (wrong-task fires, right-but-partial doesn't,
ambiguity doesn't) are LLM judgments verified in the calibration harness
(scripts/calibrate_task_understanding_failure.py).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from semantic_svc.evaluators.base import EvaluationInput, ToolCallData
from semantic_svc.evaluators.task_understanding_failure import (
    TaskUnderstandingFailureEvaluator,
)


def _run(
    input_text="How much does the Pro plan cost?",
    actual_output="Pro has SSO and audit logs.",
    tools=None,
):
    return EvaluationInput(
        input_text=input_text, actual_output=actual_output, tools_called=tools or []
    )


class TestTaskUnderstandingFailureEvaluator(unittest.TestCase):
    def _build(self):
        with patch("semantic_svc.evaluators.task_understanding_failure.build_deepeval_model"):
            return TaskUnderstandingFailureEvaluator(provider="openai", model_name="gpt-4o-mini")

    def test_fires_on_wrong_task(self):
        # LOW score = agent addressed a different task => not success => fired.
        fake = MagicMock(
            success=False,
            score=0.1,
            reason="User asked about pricing; the agent described features instead.",
            input_tokens=200,
            output_tokens=80,
        )
        with patch("semantic_svc.evaluators.task_understanding_failure.GEval", return_value=fake):
            result = self._build().evaluate(_run())
        self.assertTrue(result.fired)
        self.assertEqual(result.evaluator, "TASK_UNDERSTANDING_FAILURE")
        self.assertAlmostEqual(result.confidence, 0.9)
        self.assertIn("pricing", result.reasoning)

    def test_does_not_fire_when_right_task_addressed(self):
        fake = MagicMock(
            success=True,
            score=0.95,
            reason="The response addresses the pricing question the user asked.",
            input_tokens=200,
            output_tokens=80,
        )
        with patch("semantic_svc.evaluators.task_understanding_failure.GEval", return_value=fake):
            result = self._build().evaluate(_run())
        self.assertFalse(result.fired)
        self.assertAlmostEqual(result.confidence, 0.05)

    def test_cost_from_dunetrace_price_table(self):
        fake = MagicMock(
            success=True,
            score=0.9,
            reason="ok",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            evaluation_cost=999.0,
        )
        with patch("semantic_svc.evaluators.task_understanding_failure.GEval", return_value=fake):
            result = self._build().evaluate(_run())
        self.assertAlmostEqual(result.cost_usd, 0.75)  # gpt-4o-mini per-1M rates
        self.assertNotEqual(result.cost_usd, 999.0)

    def test_tools_called_passed_to_test_case(self):
        fake = MagicMock(success=True, score=1.0, reason="", input_tokens=10, output_tokens=5)
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            from deepeval.test_case import LLMTestCase as Real

            return Real(**kwargs)

        run = _run(
            tools=[
                ToolCallData(
                    name="search_pricing", input_parameters={"plan": "pro"}, output="$40/mo"
                )
            ]
        )
        with (
            patch("semantic_svc.evaluators.task_understanding_failure.GEval", return_value=fake),
            patch(
                "semantic_svc.evaluators.task_understanding_failure.LLMTestCase",
                side_effect=capture,
            ),
        ):
            self._build().evaluate(run)

        self.assertEqual(captured["input"], "How much does the Pro plan cost?")
        self.assertEqual(len(captured["tools_called"]), 1)
        self.assertEqual(captured["tools_called"][0].name, "search_pricing")


if __name__ == "__main__":
    unittest.main()
