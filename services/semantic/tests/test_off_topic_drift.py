"""
Unit tests for OffTopicDriftEvaluator — wrapper mechanics only (polarity,
confidence, cost, tools-param handling), GEval mocked so they run offline. The
behavioral cases (drift fires, on-topic / broad-multi-aspect don't) are LLM
judgments verified in the calibration harness (scripts/calibrate_off_topic_drift.py).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from semantic_svc.evaluators.base import EvaluationInput, ToolCallData
from semantic_svc.evaluators.off_topic_drift import OffTopicDriftEvaluator


def _run(
    input_text="How much does the Pro plan cost?",
    actual_output="It's $40/mo. By the way, our roadmap is exciting!",
    tools=None,
):
    return EvaluationInput(
        input_text=input_text, actual_output=actual_output, tools_called=tools or []
    )


class TestOffTopicDriftEvaluator(unittest.TestCase):
    def _build(self):
        with patch("semantic_svc.evaluators.off_topic_drift.build_deepeval_model"):
            return OffTopicDriftEvaluator(provider="openai", model_name="gpt-4o-mini")

    def test_fires_on_drift(self):
        fake = MagicMock(
            success=False,
            score=0.2,
            reason="The response answered the price then wandered into unrelated roadmap marketing.",
            input_tokens=200,
            output_tokens=80,
        )
        with patch("semantic_svc.evaluators.off_topic_drift.GEval", return_value=fake):
            result = self._build().evaluate(_run())
        self.assertTrue(result.fired)
        self.assertEqual(result.evaluator, "OFF_TOPIC_DRIFT")
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertIn("wandered", result.reasoning)

    def test_does_not_fire_when_on_topic(self):
        fake = MagicMock(
            success=True,
            score=0.95,
            reason="The response stayed on the pricing question throughout.",
            input_tokens=200,
            output_tokens=80,
        )
        with patch("semantic_svc.evaluators.off_topic_drift.GEval", return_value=fake):
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
        with patch("semantic_svc.evaluators.off_topic_drift.GEval", return_value=fake):
            result = self._build().evaluate(_run())
        self.assertAlmostEqual(result.cost_usd, 0.75)
        self.assertNotEqual(result.cost_usd, 999.0)

    def test_no_tools_param_when_no_tools(self):
        # With no tools, TOOLS_CALLED must be absent from evaluation_params
        # (GEval rejects it otherwise).
        fake = MagicMock(success=True, score=1.0, reason="", input_tokens=10, output_tokens=5)
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            return fake

        with (
            patch("semantic_svc.evaluators.off_topic_drift.GEval", side_effect=capture),
            patch("semantic_svc.evaluators.off_topic_drift.build_deepeval_model"),
        ):
            OffTopicDriftEvaluator(provider="openai", model_name="gpt-4o-mini").evaluate(_run())

        from deepeval.test_case import SingleTurnParams

        self.assertNotIn(SingleTurnParams.TOOLS_CALLED, captured["evaluation_params"])

    def test_tools_param_present_when_tools_given(self):
        fake = MagicMock(success=True, score=1.0, reason="", input_tokens=10, output_tokens=5)
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            return fake

        run = _run(tools=[ToolCallData(name="get_price", output="$40")])
        with (
            patch("semantic_svc.evaluators.off_topic_drift.GEval", side_effect=capture),
            patch("semantic_svc.evaluators.off_topic_drift.build_deepeval_model"),
        ):
            OffTopicDriftEvaluator(provider="openai", model_name="gpt-4o-mini").evaluate(run)

        from deepeval.test_case import SingleTurnParams

        self.assertIn(SingleTurnParams.TOOLS_CALLED, captured["evaluation_params"])


if __name__ == "__main__":
    unittest.main()
