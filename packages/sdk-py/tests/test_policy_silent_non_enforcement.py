"""
Phase 1: a policy that registers and can never enforce must be refused.

A policy prevents nothing by being inert, and the dashboard cannot tell an
inert policy from a working one: both report enabled, both show zero firings,
and zero firings is the outcome you hope for. The user's trust is in the
absence of an event, which is exactly what a dead policy also produces. So the
only place to catch this is registration, while a human is reading the error.

Four ways a fully-parseable policy could never fire, all reproduced against the
real engine before the fix:

  1. ``agent.*`` / ``org.*`` — no runtime source. The sole EvaluationContext
     construction passes args/run/event only.
  2. ``args.*`` on any trigger but ``before_tool_call`` — args is supplied by
     find_approval_policy() and by nothing else.
  3. ``before_tool_call`` without ``require_approval`` — the gate filters on
     action type, and the trigger is not in build_metrics().
  4. ``require_approval`` without ``before_tool_call`` — same filter, other half.

3 and 4 were already rejected by the Customer API (routers/policies.py) and not
by the SDK, so the same policy was refused from the dashboard and accepted from
code.
"""

from __future__ import annotations

import unittest

from dunetrace.policies import (
    APPROVAL_ACTION,
    ARGS_TRIGGER,
    UNAVAILABLE_PREFIXES,
    Policy,
    PolicyConfigError,
    PolicyEngine,
)


def _policy(condition, action=None, name="p"):
    return Policy(name=name, condition=condition, action=action or {"type": "log"})


class TestUnavailablePrefixesRejected(unittest.TestCase):
    """1. agent.* and org.* parse, validate, and resolve to nothing."""

    def test_agent_prefix_rejected(self):
        with self.assertRaises(PolicyConfigError) as ctx:
            _policy(
                {
                    "trigger": "cost_usd",
                    "operator": "gt",
                    "value": 1,
                    "match": {"agent.tier": {"eq": "trial"}},
                }
            )
        msg = str(ctx.exception)
        self.assertIn("agent.*", msg)
        self.assertIn("not available at runtime", msg)

    def test_org_prefix_rejected(self):
        with self.assertRaises(PolicyConfigError) as ctx:
            _policy(
                {
                    "trigger": "cost_usd",
                    "operator": "gt",
                    "value": 1,
                    "match": {"org.plan": {"in": ["free"]}},
                }
            )
        self.assertIn("org.*", str(ctx.exception))

    def test_both_named_when_both_used(self):
        with self.assertRaises(PolicyConfigError) as ctx:
            _policy(
                {
                    "trigger": "expression",
                    "match": {
                        "or": [{"agent.tier": {"eq": "trial"}}, {"org.plan": {"in": ["free"]}}]
                    },
                }
            )
        msg = str(ctx.exception)
        self.assertIn("agent.*", msg)
        self.assertIn("org.*", msg)

    def test_rejected_however_deeply_nested(self):
        """field_paths() walks the whole tree, so depth does not hide it."""
        with self.assertRaises(PolicyConfigError):
            _policy(
                {
                    "trigger": "expression",
                    "match": {
                        "run.step_count": {"gt": 1},
                        "or": [
                            {"run.cost_usd": {"gt": 1}},
                            {
                                "and": [
                                    {"run.error_count": {"gte": 1}},
                                    {"agent.tier": {"eq": "trial"}},
                                ]
                            },
                        ],
                    },
                }
            )

    def test_error_names_what_is_available(self):
        """Naming the fault is not enough; the message must name the fix."""
        with self.assertRaises(PolicyConfigError) as ctx:
            _policy({"trigger": "expression", "match": {"agent.tier": {"eq": "x"}}})
        msg = str(ctx.exception)
        for available in ("args.*", "run.*", "event.*"):
            self.assertIn(available, msg)

    def test_the_constant_matches_what_is_rejected(self):
        self.assertEqual(UNAVAILABLE_PREFIXES, frozenset({"agent", "org"}))


class TestArgsOutsideTheGateRejected(unittest.TestCase):
    """2. args is only populated by find_approval_policy()."""

    def test_args_with_step_level_trigger_rejected(self):
        with self.assertRaises(PolicyConfigError) as ctx:
            _policy(
                {
                    "trigger": "cost_usd",
                    "operator": "gte",
                    "value": 0,
                    "match": {"args.amount": {"gt": 100}},
                },
                {"type": "stop"},
            )
        msg = str(ctx.exception)
        self.assertIn("args.*", msg)
        self.assertIn(ARGS_TRIGGER, msg)

    def test_args_with_expression_trigger_rejected(self):
        """A pure-expression policy never reaches the gate either: the gate
        filters trigger == before_tool_call before looking at anything else."""
        with self.assertRaises(PolicyConfigError):
            _policy({"trigger": "expression", "match": {"args.amount": {"gt": 1}}})

    def test_args_with_the_gate_trigger_is_accepted(self):
        """The whole point. This is the brief's worked example."""
        p = _policy(
            {
                "trigger": ARGS_TRIGGER,
                "operator": "eq",
                "value": "refund_customer",
                "match": {
                    "args.amount": {"gt": 10000},
                    "or": [{"run.error_count": {"gte": 2}}, {"run.cost_usd": {"gt": 5.0}}],
                },
            },
            {"type": APPROVAL_ACTION, "params": {"timeout_s": 300}},
        )
        self.assertIsNotNone(p.match_expr)


class TestTriggerActionPairingRejected(unittest.TestCase):
    """3 and 4. before_tool_call and require_approval are each other's only
    partner, and either half alone is inert."""

    def test_gate_trigger_without_approval_action_rejected(self):
        for action in ("stop", "log", "switch_model", "escalate_to_human"):
            with self.subTest(action=action):
                with self.assertRaises(PolicyConfigError) as ctx:
                    _policy(
                        {"trigger": ARGS_TRIGGER, "operator": "eq", "value": "wire_money"},
                        {"type": action},
                    )
                self.assertIn(APPROVAL_ACTION, str(ctx.exception))

    def test_approval_action_without_gate_trigger_rejected(self):
        for trigger in ("cost_usd", "tool_call_count", "signal", "expression"):
            with self.subTest(trigger=trigger):
                cond = (
                    {"trigger": "expression", "match": {"run.cost_usd": {"gt": 1}}}
                    if trigger == "expression"
                    else {"trigger": trigger, "operator": "gt", "value": 1}
                )
                with self.assertRaises(PolicyConfigError) as ctx:
                    _policy(cond, {"type": APPROVAL_ACTION})
                self.assertIn(ARGS_TRIGGER, str(ctx.exception))

    def test_the_valid_pairing_is_accepted(self):
        p = _policy(
            {"trigger": ARGS_TRIGGER, "operator": "eq", "value": "wire_money"},
            {"type": APPROVAL_ACTION, "params": {"timeout_s": 300}},
        )
        self.assertEqual(p.action["type"], APPROVAL_ACTION)


class TestExistingPoliciesUnaffected(unittest.TestCase):
    """The check must reject only what cannot fire. Over-rejection would break
    working deployments, which is a worse failure than the one being fixed."""

    VALID = [
        (
            "flat metric",
            {"trigger": "tool_call_count", "operator": "gt", "value": 5},
            {"type": "stop"},
        ),
        (
            "signal contains",
            {"trigger": "signal", "operator": "contains", "value": "TOOL_LOOP"},
            {"type": "log"},
        ),
        (
            "run.* expression",
            {
                "trigger": "cost_usd",
                "operator": "gt",
                "value": 1.0,
                "match": {"run.error_count": {"gte": 2}},
            },
            {"type": "switch_model", "params": {"model": "gpt-4o-mini"}},
        ),
        (
            "event.* expression",
            {
                "trigger": "step_count",
                "operator": "gt",
                "value": 3,
                "match": {"event.type": {"eq": "llm.called"}},
            },
            {"type": "inject_prompt", "params": {"text": "be brief"}},
        ),
        (
            "pure expression",
            {
                "trigger": "expression",
                "match": {
                    "run.step_count": {"gt": 20},
                    "or": [{"event.hour": {"gte": 22}}, {"event.hour": {"lt": 6}}],
                },
            },
            {"type": "escalate_to_human"},
        ),
        (
            "no match block",
            {"trigger": "finish_reason", "operator": "eq", "value": "length"},
            {"type": "inject_recovery_prompt"},
        ),
        (
            "approval gate",
            {"trigger": "before_tool_call", "operator": "eq", "value": "wire_money"},
            {"type": "require_approval"},
        ),
    ]

    def test_every_valid_shape_still_registers(self):
        for label, cond, action in self.VALID:
            with self.subTest(policy=label):
                _policy(cond, action, name=label)

    def test_shipped_examples_still_register(self):
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[3] / "examples" / "policies"
        files = sorted(root.glob("*.yaml"))
        self.assertTrue(files, f"no shipped examples found at {root}")
        for f in files:
            with self.subTest(example=f.name):
                d = yaml.safe_load(f.read_text())
                Policy(
                    name=d["name"],
                    condition=d["condition"],
                    action=d["action"],
                    agent_id=d.get("agent_id", "*"),
                    priority=d.get("priority", 100),
                )


class TestRemoteLoadSkipsRatherThanCrashes(unittest.TestCase):
    """A rejected policy must not take the rest of the bundle down with it.
    load() already handles PolicyConfigError this way; the new rejections have
    to travel the same route."""

    def test_one_dead_policy_does_not_block_the_others(self):
        e = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level="ERROR"):
            e.load(
                [
                    {
                        "id": 1,
                        "name": "dead",
                        "agent_id": "a",
                        "condition": {
                            "trigger": "expression",
                            "match": {"agent.tier": {"eq": "trial"}},
                        },
                        "action": {"type": "log"},
                    },
                    {
                        "id": 2,
                        "name": "live",
                        "agent_id": "a",
                        "condition": {"trigger": "tool_call_count", "operator": "gt", "value": 3},
                        "action": {"type": "log"},
                    },
                ],
                agent_id="a",
            )
        self.assertEqual([p.name for p in e._policies], ["live"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
