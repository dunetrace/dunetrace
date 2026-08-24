from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from deepeval.models import DeepEvalBaseLLM

from semantic_svc.evaluators.base import EvaluationInput, ToolCallData
from semantic_svc.evaluators.task_completion import TaskCompletionEvaluator


def _run(tools_called=None):
    return EvaluationInput(
        input_text="Book a flight to Paris.",
        actual_output="I'm not able to help with that.",
        tools_called=tools_called or [],
    )


class _StubJudge(DeepEvalBaseLLM):
    """A DeepEval model that answers from canned data instead of an API.

    GEval calls the judge twice — once to generate its evaluation steps, once to
    score — and passes a pydantic schema each time. Filling the schema keeps the
    real GEval code path intact while staying entirely offline.
    """

    def load_model(self):
        return self

    def get_model_name(self):
        return "stub-judge"

    def generate(self, prompt, schema=None, *args, **kwargs):
        if schema is None:
            return '{"score": 8, "reason": "stub reason"}'
        if "steps" in schema.model_fields:
            return schema(steps=["did the agent do what was asked"])
        return schema(score=8, reason="stub reason")

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema, *args, **kwargs)


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

    # ── evaluation_params guard ───────────────────────────────────────────────
    #
    # Regression: evaluation_params listed TOOLS_CALLED unconditionally while
    # tools_called is `... or None`, so every tool-free run (a purely
    # conversational turn) raised MissingTestCaseParamsError out of DeepEval's
    # own param check before anything was scored. The worker swallows that, so
    # the run got no TASK_COMPLETION verdict, no log row, and no visible error.

    def test_no_tools_param_when_no_tools(self):
        # With no tools, TOOLS_CALLED must be absent from evaluation_params
        # (DeepEval rejects a declared param whose value is None).
        evaluator = self._build_evaluator()
        fake = MagicMock(success=True, score=1.0, reason="", input_tokens=10, output_tokens=5)
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            return fake

        with patch("semantic_svc.evaluators.task_completion.GEval", side_effect=capture):
            evaluator.evaluate(_run(tools_called=[]))

        from deepeval.test_case import SingleTurnParams

        self.assertNotIn(SingleTurnParams.TOOLS_CALLED, captured["evaluation_params"])
        self.assertIn(SingleTurnParams.INPUT, captured["evaluation_params"])
        self.assertIn(SingleTurnParams.ACTUAL_OUTPUT, captured["evaluation_params"])

    def test_tools_param_present_when_tools_given(self):
        evaluator = self._build_evaluator()
        fake = MagicMock(success=True, score=1.0, reason="", input_tokens=10, output_tokens=5)
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            return fake

        tools = [ToolCallData(name="book_flight", input_parameters={"dest": "Paris"}, output="ok")]
        with patch("semantic_svc.evaluators.task_completion.GEval", side_effect=capture):
            evaluator.evaluate(_run(tools_called=tools))

        from deepeval.test_case import SingleTurnParams

        self.assertIn(SingleTurnParams.TOOLS_CALLED, captured["evaluation_params"])


class TestTaskCompletionAgainstRealGEval(unittest.TestCase):
    """The mocked-GEval tests above cannot see this bug — a MagicMock metric
    accepts any evaluation_params. These drive the REAL GEval and the REAL
    LLMTestCase so DeepEval's own check_llm_test_case_params runs; only the
    judge model is stubbed, so there is no network call and no API key needed.
    """

    def _evaluator(self):
        with patch(
            "semantic_svc.evaluators.task_completion.build_deepeval_model",
            return_value=_StubJudge(),
        ):
            return TaskCompletionEvaluator(provider="openai", model_name="gpt-4o-mini")

    def test_tool_free_run_is_scored_rather_than_raising(self):
        # Pre-fix this raised MissingTestCaseParamsError:
        #   'tools_called' cannot be None for the 'TaskCompletion [GEval]' metric
        result = self._evaluator().evaluate(_run(tools_called=[]))

        self.assertEqual(result.evaluator, "TASK_COMPLETION")
        self.assertFalse(result.fired)  # stub judge scores 8/10 => success
        self.assertAlmostEqual(result.confidence, 0.2)
        self.assertEqual(result.reasoning, "stub reason")

    def test_run_with_tools_still_scored(self):
        tools = [ToolCallData(name="book_flight", input_parameters={"dest": "Paris"}, output="ok")]
        result = self._evaluator().evaluate(_run(tools_called=tools))

        self.assertEqual(result.evaluator, "TASK_COMPLETION")
        self.assertAlmostEqual(result.confidence, 0.2)


if __name__ == "__main__":
    unittest.main()
