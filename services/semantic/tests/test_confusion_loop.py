"""
Unit tests for ConfusionLoopEvaluator — wrapper mechanics only (polarity,
confidence direction, cost, turn construction), with ConversationalGEval mocked
so they run offline. The behavioral cases (rephrasing fires, progressive
questions don't) are LLM judgments, verified in the calibration harness
(scripts/calibrate_confusion_loop.py), not here.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from semantic_svc.evaluators.base import ConversationEvaluationInput, ConversationTurn
from semantic_svc.evaluators.confusion_loop import ConfusionLoopEvaluator


def _conversation(turns=None):
    return ConversationEvaluationInput(
        turns=turns
        or [
            ConversationTurn(
                run_id="run-1",
                input_text="How do I export my data?",
                actual_output="You can use Settings.",
            ),
            ConversationTurn(
                run_id="run-2",
                input_text="No, I mean export to CSV specifically.",
                actual_output="Settings has options.",
            ),
            ConversationTurn(
                run_id="run-3",
                input_text="I'm asking how to get a CSV file.",
                actual_output="Check Settings.",
            ),
        ]
    )


class TestConfusionLoopEvaluator(unittest.TestCase):
    def _build(self):
        with patch("semantic_svc.evaluators.confusion_loop.build_deepeval_model"):
            return ConfusionLoopEvaluator(provider="openai", model_name="gpt-4o-mini")

    def test_fires_on_low_score_confusion_loop(self):
        # LOW score = user stuck re-asking => not success => fired (same polarity
        # as UserFrustration/TaskCompletion).
        fake = MagicMock(
            success=False,
            score=0.15,
            reason="User re-asked the same CSV-export question three times unresolved.",
            input_tokens=420,
            output_tokens=110,
        )
        with patch("semantic_svc.evaluators.confusion_loop.ConversationalGEval", return_value=fake):
            result = self._build().evaluate(_conversation())
        self.assertTrue(result.fired)
        self.assertEqual(result.evaluator, "CONFUSION_LOOP")
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertIn("CSV", result.reasoning)

    def test_does_not_fire_when_conversation_progresses(self):
        fake = MagicMock(
            success=True,
            score=0.92,
            reason="Each turn advanced to a new question; the agent resolved each.",
            input_tokens=420,
            output_tokens=110,
        )
        with patch("semantic_svc.evaluators.confusion_loop.ConversationalGEval", return_value=fake):
            result = self._build().evaluate(_conversation())
        self.assertFalse(result.fired)
        self.assertAlmostEqual(result.confidence, 0.08)

    def test_cost_from_dunetrace_price_table(self):
        fake = MagicMock(
            success=True,
            score=0.9,
            reason="ok",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            evaluation_cost=999.0,
        )
        with patch("semantic_svc.evaluators.confusion_loop.ConversationalGEval", return_value=fake):
            result = self._build().evaluate(_conversation())
        self.assertAlmostEqual(result.cost_usd, 0.75)  # gpt-4o-mini per-1M rates
        self.assertNotEqual(result.cost_usd, 999.0)

    def test_turns_alternate_user_and_assistant(self):
        fake = MagicMock(success=True, score=1.0, reason="", input_tokens=10, output_tokens=5)
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            from deepeval.test_case import ConversationalTestCase as Real

            return Real(**kwargs)

        with (
            patch("semantic_svc.evaluators.confusion_loop.ConversationalGEval", return_value=fake),
            patch(
                "semantic_svc.evaluators.confusion_loop.ConversationalTestCase", side_effect=capture
            ),
        ):
            self._build().evaluate(_conversation())

        turns = captured["turns"]
        self.assertEqual(len(turns), 6)  # 3 runs * (user + assistant)
        self.assertEqual([t.role for t in turns[:2]], ["user", "assistant"])
        self.assertEqual(turns[0].content, "How do I export my data?")


if __name__ == "__main__":
    unittest.main()
