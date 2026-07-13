from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from semantic_svc.evaluators.base import ConversationEvaluationInput, ConversationTurn
from semantic_svc.evaluators.user_frustration import UserFrustrationEvaluator


def _conversation(turns=None):
    return ConversationEvaluationInput(
        turns=turns
        or [
            ConversationTurn(
                run_id="run-1",
                input_text="How do I reset my password?",
                actual_output="Click here.",
            ),
            ConversationTurn(
                run_id="run-2",
                input_text="That link is broken, I told you already!",
                actual_output="Sorry, let me try again.",
            ),
        ]
    )


class TestUserFrustrationEvaluator(unittest.TestCase):
    def _build_evaluator(self):
        with patch("semantic_svc.evaluators.user_frustration.build_deepeval_model"):
            return UserFrustrationEvaluator(provider="openai", model_name="gpt-4o-mini")

    def test_fires_when_user_shows_frustration(self):
        evaluator = self._build_evaluator()
        # success = score >= threshold; low score (unsatisfied user) => not
        # successful => fired, same polarity as TaskCompletionEvaluator.
        fake_metric = MagicMock(
            success=False,
            score=0.1,
            reason="User repeated their complaint and expressed annoyance.",
            input_tokens=300,
            output_tokens=90,
        )
        with patch(
            "semantic_svc.evaluators.user_frustration.ConversationalGEval",
            return_value=fake_metric,
        ):
            result = evaluator.evaluate(_conversation())

        self.assertTrue(result.fired)
        self.assertEqual(result.evaluator, "USER_FRUSTRATION")
        self.assertAlmostEqual(result.confidence, 0.9)
        self.assertEqual(result.reasoning, "User repeated their complaint and expressed annoyance.")

    def test_does_not_fire_when_user_satisfied(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True,
            score=0.95,
            reason="User seems satisfied.",
            input_tokens=300,
            output_tokens=90,
        )
        with patch(
            "semantic_svc.evaluators.user_frustration.ConversationalGEval",
            return_value=fake_metric,
        ):
            result = evaluator.evaluate(_conversation())

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
        with patch(
            "semantic_svc.evaluators.user_frustration.ConversationalGEval",
            return_value=fake_metric,
        ):
            result = evaluator.evaluate(_conversation())

        self.assertAlmostEqual(result.cost_usd, 0.75)  # gpt-4o-mini per-1M rates
        self.assertNotEqual(result.cost_usd, 999.0)

    def test_each_run_becomes_a_user_and_assistant_turn(self):
        evaluator = self._build_evaluator()
        fake_metric = MagicMock(
            success=True, score=1.0, reason="", input_tokens=10, output_tokens=5
        )

        captured = {}

        def fake_conversational_test_case_capture(*args, **kwargs):
            captured.update(kwargs)
            from deepeval.test_case import ConversationalTestCase as RealTestCase

            return RealTestCase(**kwargs)

        with (
            patch(
                "semantic_svc.evaluators.user_frustration.ConversationalGEval",
                return_value=fake_metric,
            ),
            patch(
                "semantic_svc.evaluators.user_frustration.ConversationalTestCase",
                side_effect=fake_conversational_test_case_capture,
            ),
        ):
            evaluator.evaluate(_conversation())

        turns = captured["turns"]
        self.assertEqual(len(turns), 4)  # 2 runs * (user + assistant)
        self.assertEqual(turns[0].role, "user")
        self.assertEqual(turns[0].content, "How do I reset my password?")
        self.assertEqual(turns[1].role, "assistant")
        self.assertEqual(turns[1].content, "Click here.")
        self.assertEqual(turns[2].role, "user")
        self.assertEqual(turns[2].content, "That link is broken, I told you already!")
        self.assertEqual(turns[3].role, "assistant")
        self.assertEqual(turns[3].content, "Sorry, let me try again.")


if __name__ == "__main__":
    unittest.main()
