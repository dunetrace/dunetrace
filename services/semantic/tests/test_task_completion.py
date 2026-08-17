from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from semantic_svc.evaluators.base import EvaluationInput, ToolCallData
from semantic_svc.evaluators.task_completion import TaskCompletionEvaluator


def _run(tools_called=None):
    return EvaluationInput(
        input_text="Book a flight to Paris.",
        actual_output="I'm not able to help with that.",
        tools_called=tools_called or [],
    )


class TestTaskCompletionEvaluator(unittest.TestCase):
    def _build_evaluator(self):
        with patch("semantic_svc.evaluators.task_completion.build_deepeval_model"):
            return TaskCompletionEvaluator(provider="openai", model_name="gpt-4o-mini")

    def test_fires_when_task_not_completed(self):
        evaluator = self._build_evaluator()
        # success = score >= threshold for GEval; low score => not successful => fired.
        fake_metric = MagicMock(
            success=False,
            score=0.1,
            reason="Agent refused the task",
            input_tokens=200,
            output_tokens=80,
        )
        with patch("semantic_svc.evaluators.task_completion.GEval", return_value=fake_metric):
            result = evaluator.evaluate(_run())

        self.assertTrue(result.fired)
        self.assertEqual(result.evaluator, "TASK_COMPLETION")
        self.assertAlmostEqual(result.confidence, 0.9)
        self.assertEqual(result.reasoning, "Agent refused the task")

    def test_does_not_fire_when_task_completed(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=0.95, reason="Task completed", input_tokens=200, output_tokens=80
        )
        with patch("semantic_svc.evaluators.task_completion.GEval", return_value=fake_metric):
            result = evaluator.evaluate(_run())

        self.assertFalse(result.fired)
        self.assertAlmostEqual(result.confidence, 0.05)

    def test_cost_computed_from_dunetrace_price_table_not_deepeval(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True,
            score=0.9,
            reason="ok",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            evaluation_cost=999.0,
        )
        with patch("semantic_svc.evaluators.task_completion.GEval", return_value=fake_metric):
            result = evaluator.evaluate(_run())

        self.assertAlmostEqual(result.cost_usd, 0.75)  # gpt-4o-mini per-1M rates
        self.assertNotEqual(result.cost_usd, 999.0)

    def test_tools_called_converted_to_deepeval_toolcall(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=1.0, reason="", input_tokens=10, output_tokens=5
        )
        tools = [ToolCallData(name="book_flight", input_parameters={"dest": "Paris"}, output="ok")]

        captured = {}

        def fake_llm_test_case_capture(*args, **kwargs):
            captured.update(kwargs)
            from deepeval.test_case import LLMTestCase as RealLLMTestCase

            return RealLLMTestCase(**kwargs)

        with (
            patch("semantic_svc.evaluators.task_completion.GEval", return_value=fake_metric),
            patch(
                "semantic_svc.evaluators.task_completion.LLMTestCase",
                side_effect=fake_llm_test_case_capture,
            ),
        ):
            evaluator.evaluate(_run(tools_called=tools))

        self.assertEqual(len(captured["tools_called"]), 1)
        self.assertEqual(captured["tools_called"][0].name, "book_flight")

    def test_no_tools_called_passes_none_not_empty_list(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=1.0, reason="", input_tokens=10, output_tokens=5
        )
        captured = {}

        def fake_llm_test_case_capture(*args, **kwargs):
            captured.update(kwargs)
            from deepeval.test_case import LLMTestCase as RealLLMTestCase

            return RealLLMTestCase(**kwargs)

        with (
            patch("semantic_svc.evaluators.task_completion.GEval", return_value=fake_metric),
            patch(
                "semantic_svc.evaluators.task_completion.LLMTestCase",
                side_effect=fake_llm_test_case_capture,
            ),
        ):
            evaluator.evaluate(_run(tools_called=[]))

        self.assertIsNone(captured["tools_called"])


if __name__ == "__main__":
    unittest.main()
