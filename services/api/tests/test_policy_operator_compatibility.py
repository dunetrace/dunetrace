"""
API-side rejection of operator/trigger combinations that can never fire.

Covers entry points 4 and 5 (POST /v1/policies and PUT /v1/policies/{id}), both
of which route through `_validate`. A policy pushed from the dashboard, the MCP
`create_policy` tool, or curl all land here — none of them may install a silent
no-op.

The rule itself lives in the SDK (`dunetrace.policies.validate_condition`) and
is imported, not restated: the SDK's engine is what actually evaluates these
policies, so a combination this service accepts and that engine cannot match is
a row that stores cleanly, shows as enabled, and prevents nothing.

_validate() is a pure function (no DB), so these run without a database.

Run from services/api/ with:
  PYTHONPATH=../../packages/sdk-py:../explainer:. \
    python -m pytest tests/test_policy_operator_compatibility.py -v
"""

import unittest

from fastapi import HTTPException

from api_svc.routers.policies import (
    ActionModel,
    ConditionModel,
    _VALID_OPERATORS,
    _VALID_TRIGGERS,
    _validate,
)


def _cond(**kw):
    return ConditionModel(**kw)


LOG = ActionModel(type="log")
STOP = ActionModel(type="stop")
APPROVE = ActionModel(type="require_approval")


class TestSignalTriggerOperatorRejected(unittest.TestCase):
    """`signal` is list-valued (run_context.py:1049), so `contains` is the only
    operator that means anything against it."""

    def _rejects(self, operator, value="TOOL_LOOP"):
        with self.assertRaises(HTTPException) as e:
            _validate(_cond(trigger="signal", operator=operator, value=value), LOG, "p")
        self.assertEqual(e.exception.status_code, 422)
        return e.exception.detail

    def test_eq_rejected_with_422_naming_contains(self):
        detail = self._rejects("eq")
        self.assertIn("'contains'", detail)
        self.assertIn("can never match", detail)
        self.assertIn('"operator": "contains"', detail)

    def test_neq_rejected_as_always_fires(self):
        detail = self._rejects("neq")
        self.assertIn("every run", detail)
        self.assertIn("'contains'", detail)

    def test_ordering_operators_rejected(self):
        for operator in ("gt", "gte", "lt", "lte"):
            with self.subTest(operator=operator):
                self.assertIn("'contains'", self._rejects(operator))

    def test_contains_accepted(self):
        _validate(_cond(trigger="signal", operator="contains", value="TOOL_LOOP"), LOG, "p")

    def test_stop_action_on_signal_contains_accepted(self):
        _validate(_cond(trigger="signal", operator="contains", value="TOOL_LOOP"), STOP, "p")


class TestAmbiguousCellsStillAccepted(unittest.TestCase):
    """Scoped to a string value on purpose — a list value makes `eq` genuinely
    live, and rejecting it would be guessing at intent."""

    def test_signal_eq_with_list_value_accepted(self):
        _validate(_cond(trigger="signal", operator="eq", value=["TOOL_LOOP"]), LOG, "p")

    def test_scalar_trigger_with_contains_accepted(self):
        _validate(_cond(trigger="finish_reason", operator="contains", value="length"), LOG, "p")

    def test_string_trigger_with_ordering_operator_accepted(self):
        _validate(_cond(trigger="finish_reason", operator="gt", value="a"), LOG, "p")


class TestAllowlistsComeFromTheSDK(unittest.TestCase):
    """Not restated here — a divergence would be a policy this service stores
    and the engine cannot evaluate."""

    def test_triggers_are_the_sdk_set(self):
        from dunetrace.policies import VALID_TRIGGERS

        self.assertIs(_VALID_TRIGGERS, VALID_TRIGGERS)

    def test_operators_are_the_sdk_set(self):
        from dunetrace.policies import VALID_OPERATORS

        self.assertIs(_VALID_OPERATORS, VALID_OPERATORS)

    def test_allowlist_contents_unchanged_from_the_hand_written_sets(self):
        """The literals this file replaced, asserted so the import cannot
        silently widen or narrow what the API accepts."""
        self.assertEqual(
            set(_VALID_TRIGGERS),
            {
                "tool_call_count",
                "step_count",
                "cost_usd",
                "error_count",
                "finish_reason",
                "llm_latency_ms",
                "signal",
                "before_tool_call",
                "expression",
            },
        )
        self.assertEqual(set(_VALID_OPERATORS), {"gt", "gte", "lt", "lte", "eq", "neq", "contains"})


class TestKnownGoodCorpusStillValidates(unittest.TestCase):
    """Every shipped example policy plus everything build_suggested_policy()
    emits. None may start failing."""

    def test_example_yaml_policies(self):
        # examples/policies/high-value-refund-approval.yaml
        _validate(
            _cond(
                trigger="before_tool_call",
                operator="eq",
                value="refund_customer",
                match={"args.amount": {"gt": 10000}},
            ),
            APPROVE,
            "high-value-refund",
        )
        # examples/policies/business-hours-only-actions.yaml
        _validate(
            _cond(
                trigger="before_tool_call",
                operator="eq",
                value="send_customer_message",
                match={"or": [{"event.hour": {"lt": 9}}, {"event.hour": {"gte": 17}}]},
            ),
            APPROVE,
            "off-hours",
        )
        # examples/policies/trial-tier-strict-limits.yaml — pure expression
        _validate(
            _cond(
                trigger="expression",
                match={
                    "run.tool_call_count": {"gt": 8},
                    "or": [
                        {"agent.tier": {"eq": "trial"}},
                        {"org.plan": {"in": ["free", "starter"]}},
                    ],
                },
            ),
            STOP,
            "trial-tier",
        )

    def test_build_suggested_policy_shapes(self):
        """The four _NATIVE_POLICY_BUILDERS in fix_classification.py — the
        dashboard POSTs these verbatim after a confirmation dialog."""
        from api_svc.fix_classification import build_suggested_policy

        signals = [
            {"failure_type": "TOOL_LOOP", "agent_id": "a", "evidence": {"repeat_count": 12}},
            {"failure_type": "RETRY_STORM", "agent_id": "a", "evidence": {"retry_count": 5}},
            {
                "failure_type": "CASCADING_TOOL_FAILURE",
                "agent_id": "a",
                "evidence": {"consecutive_failures": 3},
            },
            {
                "failure_type": "STEP_COUNT_INFLATION",
                "agent_id": "a",
                "evidence": {"current_steps": 40},
            },
        ]
        built = 0
        for signal in signals:
            suggested = build_suggested_policy(signal)
            if suggested is None:
                continue
            built += 1
            with self.subTest(failure_type=signal["failure_type"]):
                _validate(
                    ConditionModel(**suggested["condition"]),
                    ActionModel(**suggested["action"]),
                    suggested["name"],
                )
        self.assertGreater(built, 0, "no suggested policies were built — fixture drifted")

    def test_dashboard_enable_from_incident_shape(self):
        """enablePolicyFromIncident() in mission-control.html."""
        _validate(_cond(trigger="signal", operator="contains", value="TOOL_LOOP"), LOG, "watch")


if __name__ == "__main__":
    unittest.main()
