"""
API-side validation for expression conditions on policies (Phase 3).

_validate() is a pure function (no DB), so these run without a database. Covers
the new trigger="expression" path, the condition.match parse-check, and that
legacy flat conditions still validate exactly as before.

Run from services/api/ with:
  PYTHONPATH=../../packages/sdk-py:../explainer:. \
    python -m unittest discover -s tests -p "test_policy_expression_validation.py"
"""

import unittest

from fastapi import HTTPException

from api_svc.routers.policies import ActionModel, ConditionModel, _validate


def _cond(**kw):
    return ConditionModel(**kw)


ACTION = ActionModel(type="stop")
APPROVE = ActionModel(type="require_approval")


class TestLegacyValidationUnchanged(unittest.TestCase):
    def test_legacy_flat_condition_ok(self):
        _validate(_cond(trigger="tool_call_count", operator="gt", value=5), ACTION)

    def test_legacy_missing_operator_rejected(self):
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="tool_call_count", value=5), ACTION)
        self.assertEqual(e.exception.status_code, 422)
        self.assertIn("operator is required", e.exception.detail)

    def test_legacy_invalid_operator_rejected(self):
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="cost_usd", operator="bogus", value=1), ACTION)
        self.assertIn("Invalid operator", e.exception.detail)

    def test_invalid_trigger_rejected(self):
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="nope", operator="gt", value=1), ACTION)
        self.assertIn("Invalid trigger", e.exception.detail)


class TestExpressionValidation(unittest.TestCase):
    def test_expression_trigger_with_match_ok(self):
        _validate(
            _cond(trigger="expression", match={"args.amount": {"gt": 10000}}),
            ACTION,
            name="p",
        )

    def test_expression_trigger_without_match_rejected(self):
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="expression"), ACTION)
        self.assertIn("requires a condition.match", e.exception.detail)

    def test_invalid_operator_in_match_rejected_with_suggestion(self):
        with self.assertRaises(HTTPException) as e:
            _validate(
                _cond(trigger="expression", match={"args.amount": {"greaterthan": 1}}),
                ACTION,
                name="refund",
            )
        self.assertEqual(e.exception.status_code, 422)
        self.assertIn("Invalid condition.match", e.exception.detail)
        self.assertIn("Did you mean 'gt'?", e.exception.detail)

    def test_unknown_field_prefix_in_match_rejected(self):
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="expression", match={"argz.x": {"eq": 1}}), ACTION)
        self.assertIn("Invalid condition.match", e.exception.detail)

    def test_deeply_nested_match_rejected(self):
        deep = {"or": [{"or": [{"or": [{"args.a": {"eq": 1}}]}]}]}
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="expression", match=deep), ACTION)
        self.assertIn("maximum depth", e.exception.detail)


class TestMixedValidation(unittest.TestCase):
    def test_before_tool_call_with_match_and_approval_ok(self):
        _validate(
            _cond(
                trigger="before_tool_call",
                operator="eq",
                value="refund_customer",
                match={"args.amount": {"gt": 10000}},
            ),
            APPROVE,
            name="high-value-refund",
        )

    def test_signal_trigger_with_match_ok(self):
        _validate(
            _cond(
                trigger="signal",
                operator="contains",
                value="TOOL_ARGUMENT_FABRICATION",
                match={"args.destructive": {"eq": True}},
            ),
            ACTION,
        )

    def test_before_tool_call_still_requires_approval_action(self):
        # Pre-existing pairing rule must survive the schema change.
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="before_tool_call", operator="eq", value="x"), ACTION)
        self.assertIn("require_approval", e.exception.detail)


if __name__ == "__main__":
    unittest.main()
