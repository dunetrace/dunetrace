"""
Integration tests for expression conditions in the policy engine (Phase 3).

Covers three things the brief calls out:
  1. Backward compat — existing flat-condition policies (copied customer-shaped
     from test_policies.py / test_approval_policy_gate.py) behave EXACTLY as
     before, whether or not a context is passed.
  2. New expression conditions evaluate correctly via PolicyEngine.evaluate and
     find_approval_policy.
  3. Mixed old+new (legacy trigger AND condition.match) require BOTH to match.

Plus: the headline end-to-end case — require_approval on refund_customer only
when args.amount > 10000 — driven through a real dt.run() tool call.

Run: python -m unittest tests.test_policies_expression_integration -v
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock

from dunetrace.client import DunetraceClient
from dunetrace.policies import (
    EvaluationContext,
    Policy,
    PolicyEngine,
)


def _make_client() -> DunetraceClient:
    c = DunetraceClient(api_key="dt_test", api_url="http://localhost:8002", debug=False)
    c._ship = lambda batch: None
    return c


def _ctx(**ns) -> EvaluationContext:
    return EvaluationContext(
        args=ns.get("args", {}),
        run=ns.get("run", {}),
        agent=ns.get("agent", {}),
        org=ns.get("org", {}),
        event=ns.get("event", {}),
    )


# ── 1. Backward compatibility ─────────────────────────────────────────────────


class TestBackwardCompatibility(unittest.TestCase):
    """Legacy flat policies must be untouched by the expression machinery."""

    def test_flat_policy_still_fires_without_context(self):
        e = PolicyEngine()
        e.add(
            Policy(
                name="stop-at-5",
                condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
                action={"type": "stop"},
            )
        )
        # Old call shape (no context arg) still works.
        self.assertIsNotNone(e.evaluate("a", {"tool_call_count": 6}, set()))
        self.assertIsNone(e.evaluate("a", {"tool_call_count": 3}, set()))

    def test_flat_policy_unaffected_when_context_passed(self):
        e = PolicyEngine()
        e.add(
            Policy(
                name="stop-at-5",
                condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
                action={"type": "stop"},
            )
        )
        # A context is present but the policy has no match_expr — must ignore it.
        self.assertIsNotNone(e.evaluate("a", {"tool_call_count": 6}, set(), _ctx(args={"x": 1})))

    def test_flat_policy_has_no_match_expr(self):
        p = Policy(
            name="p",
            condition={"trigger": "cost_usd", "operator": "gte", "value": 1.0},
            action={"type": "log"},
        )
        self.assertIsNone(p.match_expr)

    def test_flat_approval_policy_still_gates_by_tool_name(self):
        e = PolicyEngine()
        e.add(
            Policy.from_dict(
                {
                    "name": "approve-wire",
                    "condition": {
                        "trigger": "before_tool_call",
                        "operator": "eq",
                        "value": "wire_money",
                    },
                    "action": {"type": "require_approval"},
                }
            )
        )
        # No context (old caller) and with context both still gate on the name.
        self.assertIsNotNone(e.find_approval_policy("a", "wire_money"))
        self.assertIsNone(e.find_approval_policy("a", "search"))
        self.assertIsNotNone(e.find_approval_policy("a", "wire_money", _ctx(args={"amt": 1})))


# ── 2. Pure expression policies ───────────────────────────────────────────────


class TestExpressionPolicies(unittest.TestCase):
    def _expr_policy(self, match, action_type="stop", name="expr"):
        return Policy(
            name=name,
            condition={"trigger": "expression", "match": match},
            action={"type": action_type},
        )

    def test_expression_policy_parsed_at_construction(self):
        p = self._expr_policy({"run.error_count": {"gte": 3}})
        self.assertIsNotNone(p.match_expr)

    def test_expression_fires_with_matching_context(self):
        e = PolicyEngine()
        e.add(self._expr_policy({"run.error_count": {"gte": 3}}))
        ctx = _ctx(run={"error_count": 4})
        self.assertIsNotNone(e.evaluate("a", {}, set(), ctx))

    def test_expression_does_not_fire_when_context_missing_field(self):
        e = PolicyEngine()
        e.add(self._expr_policy({"run.error_count": {"gte": 3}}))
        self.assertIsNone(e.evaluate("a", {}, set(), _ctx(run={"error_count": 1})))

    def test_expression_policy_never_fires_without_context(self):
        # An expression policy evaluated with no context cannot match.
        e = PolicyEngine()
        e.add(self._expr_policy({"run.error_count": {"gte": 3}}))
        self.assertIsNone(e.evaluate("a", {}, set()))

    def test_malformed_expression_policy_never_fires_defensively(self):
        # Construct a policy whose trigger says expression but match_expr is None
        # (bypassing normal construction). _legacy_matches must refuse it.
        p = self._expr_policy({"run.error_count": {"gte": 3}})
        object.__setattr__(p, "match_expr", None)
        self.assertFalse(p.matches({}, _ctx(run={"error_count": 99})))


# ── 3. Mixed legacy + expression (AND) ────────────────────────────────────────


class TestMixedConditions(unittest.TestCase):
    def _mixed(self):
        # signal fired AND args.destructive == true  (brief's coexistence example)
        return Policy(
            name="mixed",
            condition={
                "trigger": "signal",
                "operator": "contains",
                "value": "TOOL_ARGUMENT_FABRICATION",
                "match": {"args.destructive": {"eq": True}},
            },
            action={"type": "stop"},
        )

    def test_both_true_matches(self):
        p = self._mixed()
        metrics = {"signal": ["TOOL_ARGUMENT_FABRICATION"]}
        self.assertTrue(p.matches(metrics, _ctx(args={"destructive": True})))

    def test_legacy_true_expr_false_does_not_match(self):
        p = self._mixed()
        metrics = {"signal": ["TOOL_ARGUMENT_FABRICATION"]}
        self.assertFalse(p.matches(metrics, _ctx(args={"destructive": False})))

    def test_legacy_false_expr_true_does_not_match(self):
        p = self._mixed()
        metrics = {"signal": ["SOMETHING_ELSE"]}
        self.assertFalse(p.matches(metrics, _ctx(args={"destructive": True})))

    def test_mixed_expr_true_but_no_context_does_not_match(self):
        p = self._mixed()
        metrics = {"signal": ["TOOL_ARGUMENT_FABRICATION"]}
        self.assertFalse(p.matches(metrics))  # no context → expr can't match


# ── 4. Malformed policies skipped at load ─────────────────────────────────────


class TestLoadResilience(unittest.TestCase):
    def test_malformed_match_skipped_not_fatal(self):
        e = PolicyEngine()
        raw = [
            {
                "id": 1,
                "name": "good",
                "condition": {"trigger": "tool_call_count", "operator": "gt", "value": 0},
                "action": {"type": "log"},
            },
            {
                "id": 2,
                "name": "bad-op",
                "condition": {"trigger": "expression", "match": {"args.x": {"greaterthan": 1}}},
                "action": {"type": "stop"},
            },
        ]
        with self.assertLogs("dunetrace.policies", level="WARNING") as cm:
            e.load(raw)
        # Only the good policy loaded; the bad one was skipped with a warning.
        self.assertEqual(len(e), 1)
        self.assertTrue(any("invalid condition expression" in m for m in cm.output))


# ── 5. End-to-end: high-value refund approval ─────────────────────────────────


class TestHighValueRefundE2E(unittest.TestCase):
    """The headline use case, through a real dt.run() tool call."""

    def _policy(self):
        return dict(
            name="approve-high-value-refund",
            condition={
                "trigger": "before_tool_call",
                "operator": "eq",
                "value": "refund_customer",
                "match": {"args.amount": {"gt": 10000}},
            },
            action={"type": "require_approval", "params": {"timeout_s": 60}},
        )

    def test_high_amount_triggers_approval(self):
        c = _make_client()
        c.add_policy(**self._policy())
        with c.run("billing") as run:
            run.request_approval = MagicMock()
            run.tool_called("refund_customer", {"amount": 25000})
            run.request_approval.assert_called_once()
        c.shutdown(timeout=2)

    def test_low_amount_does_not_trigger_approval(self):
        c = _make_client()
        c.add_policy(**self._policy())
        with c.run("billing") as run:
            run.request_approval = MagicMock()
            run.tool_called("refund_customer", {"amount": 500})
            run.request_approval.assert_not_called()
        c.shutdown(timeout=2)

    def test_wrong_tool_never_gated_even_with_high_amount(self):
        c = _make_client()
        c.add_policy(**self._policy())
        with c.run("billing") as run:
            run.request_approval = MagicMock()
            run.tool_called("lookup_customer", {"amount": 999999})
            run.request_approval.assert_not_called()
        c.shutdown(timeout=2)

    def test_missing_amount_arg_does_not_gate(self):
        # args.amount absent → expression is False → no approval required.
        c = _make_client()
        c.add_policy(**self._policy())
        with c.run("billing") as run:
            run.request_approval = MagicMock()
            run.tool_called("refund_customer", {"reason": "duplicate"})
            run.request_approval.assert_not_called()
        c.shutdown(timeout=2)


class TestEventTimeContext(unittest.TestCase):
    """event.hour / event.timestamp are populated on the evaluation context so
    business-hours-style policies work (examples/policies/business-hours-only)."""

    def test_event_hour_and_timestamp_present(self):
        c = _make_client()
        with c.run("a") as run:
            ctx = run._build_eval_context(event={"type": "before_tool_call"})
        c.shutdown(timeout=2)
        self.assertIn("hour", ctx.event)
        self.assertIn("timestamp", ctx.event)
        self.assertIsInstance(ctx.event["hour"], int)
        self.assertTrue(0 <= ctx.event["hour"] <= 23)
        # caller-supplied fields are preserved
        self.assertEqual(ctx.event["type"], "before_tool_call")

    def test_business_hours_expression_evaluates(self):
        from dunetrace.policies import evaluate, parse_match_block

        c = _make_client()
        with c.run("a") as run:
            ctx = run._build_eval_context(event={})
        c.shutdown(timeout=2)
        expr = parse_match_block(
            {"or": [{"event.hour": {"lt": 9}}, {"event.hour": {"gte": 17}}]},
            policy_name="t",
        )
        hour = ctx.event["hour"]
        expected = hour < 9 or hour >= 17
        self.assertEqual(evaluate(expr, ctx), expected)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
