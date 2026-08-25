"""
Registration-time rejection of policy conditions that can never fire.

The bug: `metrics["signal"]` is a *list* of failure-type strings
(run_context.py:1049), but nothing stopped a policy from comparing it to a
string with `eq`. `Policy._legacy_matches` turns the mismatch into a plain
False, so the policy registered cleanly, reported as enabled, evaluated on
every step, and prevented nothing — silently, at every log level.

Four shapes of the same footgun are covered here, all confirmed by execution
against the real operator table before the fix:

    signal + eq  + str   -> never matches            (the reported bug)
    signal + neq + str   -> matches on EVERY run     (worse: always-fires)
    signal + gt/gte/lt/lte + str -> TypeError, swallowed -> never matches
    unknown trigger / unknown operator -> never matches

Deliberately NOT rejected (see the ambiguity note in
_reject_list_trigger_operator): `signal` with a *list* value, where `eq` is
genuinely live — `["TOOL_LOOP"] == ["TOOL_LOOP"]` is True. Fragile, but a real
comparison; rejecting it would be guessing at intent.

Run: python -m pytest tests/test_policy_condition_validation.py -v
"""

from __future__ import annotations

import logging
import unittest

from dunetrace import Dunetrace, Policy, PolicyConfigError
from dunetrace.policies import (
    LIST_VALUED_TRIGGERS,
    PolicyEngine,
    PolicyViolation,
    VALID_OPERATORS,
    VALID_TRIGGERS,
    validate_condition,
)


def _client():
    c = Dunetrace(api_key="k")
    c._ship = lambda batch: None
    return c


# ── Entry point 1: dt.add_policy() ───────────────────────────────────────────


class TestAddPolicyRejects(unittest.TestCase):
    """Raises into user code. Safe: no path loads policies during __init__ or
    dt.init(), so this cannot crash an agent at startup."""

    def _add(self, condition):
        dt = _client()
        try:
            with self.assertRaises(PolicyConfigError) as ctx:
                dt.add_policy(name="guard", condition=condition, action={"type": "stop"})
            return str(ctx.exception)
        finally:
            dt.shutdown(timeout=1)

    def test_signal_eq_rejected_and_names_contains(self):
        msg = self._add({"trigger": "signal", "operator": "eq", "value": "UNREAD_TOOL_ERROR"})
        self.assertIn("'contains'", msg)
        self.assertIn("can never match", msg)
        # The message must carry a copy-pasteable corrected condition.
        self.assertIn(
            '{"trigger": "signal", "operator": "contains", "value": "UNREAD_TOOL_ERROR"}', msg
        )

    def test_signal_neq_rejected_as_always_fires_not_as_never_matches(self):
        msg = self._add({"trigger": "signal", "operator": "neq", "value": "TOOL_LOOP"})
        self.assertIn("'contains'", msg)
        self.assertIn("every run", msg)
        self.assertNotIn("can never match", msg)

    def test_signal_ordering_operators_rejected(self):
        for op in ("gt", "gte", "lt", "lte"):
            with self.subTest(operator=op):
                msg = self._add({"trigger": "signal", "operator": op, "value": "TOOL_LOOP"})
                self.assertIn("'contains'", msg)

    def test_signal_with_operator_omitted_is_rejected(self):
        """An absent operator defaults to `gt` at evaluation, so it is just as
        dead as writing gt out — validate the effective operator, not the key."""
        msg = self._add({"trigger": "signal", "value": "TOOL_LOOP"})
        self.assertIn("'gt'", msg)
        self.assertIn("'contains'", msg)

    def test_unknown_trigger_rejected_with_suggestion(self):
        msg = self._add({"trigger": "toolcall_count", "operator": "gt", "value": 5})
        self.assertIn("unknown trigger", msg)
        self.assertIn("Did you mean 'tool_call_count'?", msg)

    def test_unknown_operator_rejected_with_suggestion(self):
        msg = self._add({"trigger": "tool_call_count", "operator": "greater", "value": 5})
        self.assertIn("unknown operator", msg)

    def test_symbolic_operator_rejected_and_mapped_to_its_name(self):
        """Custom detectors use >=/<=/==; policies do not. difflib cannot connect
        '>' to 'gt', so the hint table does."""
        for symbol, name in ((">", "gt"), (">=", "gte"), ("<", "lt"), ("==", "eq")):
            with self.subTest(symbol=symbol):
                msg = self._add({"trigger": "tool_call_count", "operator": symbol, "value": 5})
                self.assertIn(f"Did you mean {name!r}?", msg)


# ── Entry point 3: direct Policy(...) construction ───────────────────────────


class TestDirectConstructionRejects(unittest.TestCase):
    """Policy is in dunetrace.__all__, so this is a supported entry point and
    __post_init__ is the choke point that covers all three SDK paths."""

    def test_policy_constructor_rejects(self):
        with self.assertRaises(PolicyConfigError):
            Policy(
                name="guard",
                condition={"trigger": "signal", "operator": "eq", "value": "TOOL_LOOP"},
                action={"type": "stop"},
            )

    def test_from_dict_rejects(self):
        with self.assertRaises(PolicyConfigError):
            Policy.from_dict(
                {
                    "name": "guard",
                    "condition": {"trigger": "signal", "operator": "eq", "value": "TOOL_LOOP"},
                    "action": {"type": "stop"},
                }
            )


# ── Entry point 2: remote / dashboard-pushed policies ────────────────────────


class TestRemoteLoadRefusesInsteadOfRaising(unittest.TestCase):
    """PolicyEngine.load() runs on the daemon thread _fetch_policies() starts,
    under an `except Exception: logger.debug` in client.py. A raise there would
    be swallowed at DEBUG and would abort the rest of the batch — so this path
    refuses the one policy and logs at ERROR instead."""

    @staticmethod
    def _remote(policy_id, condition):
        return {
            "id": policy_id,
            "name": f"remote-{policy_id}",
            "agent_id": "*",
            "condition": condition,
            "action": {"type": "stop"},
            "enabled": True,
            "priority": 100,
        }

    def test_dead_remote_policy_is_not_installed(self):
        engine = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level=logging.ERROR) as logs:
            engine.load([self._remote(1, {"trigger": "signal", "operator": "eq", "value": "X"})])

        self.assertEqual(len(engine), 0, "a dead policy must not be installed")
        joined = "\n".join(logs.output)
        self.assertIn("NOT installed", joined)
        self.assertIn("'contains'", joined)

    def test_load_does_not_raise(self):
        """Raising here is swallowed at DEBUG by the caller — never do it."""
        engine = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level=logging.ERROR):
            engine.load([self._remote(1, {"trigger": "signal", "operator": "neq", "value": "X"})])

    def test_one_bad_policy_does_not_drop_the_good_ones(self):
        engine = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level=logging.ERROR):
            engine.load(
                [
                    self._remote(1, {"trigger": "signal", "operator": "eq", "value": "X"}),
                    self._remote(2, {"trigger": "signal", "operator": "contains", "value": "X"}),
                    self._remote(3, {"trigger": "tool_call_count", "operator": "gt", "value": 5}),
                ]
            )
        self.assertEqual(len(engine), 2)
        self.assertEqual(
            sorted(p.id for p in engine._policies), [2, 3], "only the dead one is dropped"
        )


# ── The correct form still works, end to end ─────────────────────────────────


class TestContainsStillFiresEndToEnd(unittest.TestCase):
    @staticmethod
    def _agent(dt, executed):
        """The failure pattern accumulates, THEN the agent reaches for the
        dangerous tool. A stop policy has to interrupt before that call — a
        guardrail that fires only after the damage is done is no guardrail."""

        def wire_money(amount):
            executed.append(amount)  # first statement — runs even if the rest raises
            return "sent"

        with dt.run("agent") as run:
            for i in range(6):
                run.tool_called("search", args={"q": f"q{i}"})
                # The failure the detector keys on: an error nothing reads.
                run.tool_responded("search", success=False, error="boom", output_length=0)
            run.tool_called("wire_money", args={"amount": 5000})
            wire_money(5000)

    def test_stop_policy_blocks_the_guarded_tool(self):
        """Asserted via a sentinel set as the FIRST statement of the tool body,
        not by absence of output: proving the tool never ran is the whole point
        of a stop policy, and 'no output' is also exactly what the silent no-op
        this fix removes would produce."""
        executed = []
        dt = _client()
        dt.add_policy(
            name="halt-on-unread-error",
            condition={"trigger": "signal", "operator": "contains", "value": "UNREAD_TOOL_ERROR"},
            action={"type": "stop"},
        )

        stopped = False
        try:
            self._agent(dt, executed)
        except PolicyViolation:
            stopped = True
        dt.shutdown(timeout=1)

        self.assertTrue(stopped, "the contains policy must actually fire")
        self.assertEqual(
            executed, [], "the guarded tool body must never have run — sentinel is empty"
        )

    def test_the_same_run_without_the_policy_reaches_the_tool(self):
        """Control: the identical run with no policy DOES reach the tool, so the
        empty sentinel above is the policy's doing and not a broken test."""
        executed = []
        dt = _client()
        self._agent(dt, executed)
        dt.shutdown(timeout=1)

        self.assertEqual(executed, [5000])

    def test_eq_would_not_have_blocked_it(self):
        """The bug, stated as a test: the same run under the eq form reaches the
        dangerous tool. It can no longer be registered, so this asserts against
        the raw engine — which is what policies already stored in the wild do."""
        executed = []
        dt = _client()
        # Bypass validation deliberately: this is the pre-fix state, and the
        # point is that it prevents nothing.
        bad = Policy.__new__(Policy)
        bad.name = "halt-on-unread-error"
        bad.condition = {"trigger": "signal", "operator": "eq", "value": "UNREAD_TOOL_ERROR"}
        bad.action = {"type": "stop"}
        bad.agent_id, bad.enabled, bad.priority, bad.id = "*", True, 100, None
        bad.match_expr = None
        dt._policy_engine.add(bad)

        self._agent(dt, executed)  # completes normally — no PolicyViolation
        dt.shutdown(timeout=1)

        self.assertEqual(executed, [5000], "eq never matched, so nothing was prevented")


# ── Ambiguous cells stay permissive ──────────────────────────────────────────


class TestAmbiguousCellsLeftPermissive(unittest.TestCase):
    """Cells that are surprising but genuinely live. Rejecting them would be
    guessing at intent rather than stating a fact."""

    def _accepts(self, condition):
        dt = _client()
        try:
            dt.add_policy(name="p", condition=condition, action={"type": "log"})
        finally:
            dt.shutdown(timeout=1)

    def test_signal_eq_with_a_list_value_is_allowed(self):
        # ["TOOL_LOOP"] == ["TOOL_LOOP"] is True — order-sensitive, but live.
        self._accepts({"trigger": "signal", "operator": "eq", "value": ["TOOL_LOOP"]})

    def test_signal_contains_with_a_non_string_value_is_allowed(self):
        # Matches test_policies.py::test_unresolvable_condition_runs_the_full_battery.
        self._accepts({"trigger": "signal", "operator": "contains", "value": {"any": ["A", "B"]}})

    def test_scalar_trigger_with_contains_is_allowed(self):
        # contains wraps a scalar as `b in [a]`, i.e. exactly eq. Redundant, live.
        self._accepts({"trigger": "finish_reason", "operator": "contains", "value": "length"})

    def test_string_trigger_with_ordering_operator_is_allowed(self):
        # Lexicographic string comparison. Almost certainly not intended, but it
        # can match, so it is not this check's business.
        self._accepts({"trigger": "finish_reason", "operator": "gt", "value": "a"})


# ── The known-good corpus must not start failing ─────────────────────────────


class TestKnownGoodCorpusStillRegisters(unittest.TestCase):
    """Every operator/trigger pair shipped in examples/policies/*.yaml, plus the
    conditions build_suggested_policy() emits and the ones the dashboard POSTs.
    None of these may start failing."""

    CORPUS = [
        # examples/policies/high-value-refund-approval.yaml
        {
            "trigger": "before_tool_call",
            "operator": "eq",
            "value": "refund_customer",
            "match": {"args.amount": {"gt": 10000}},
        },
        # examples/policies/business-hours-only-actions.yaml
        {
            "trigger": "before_tool_call",
            "operator": "eq",
            "value": "send_customer_message",
            "match": {"or": [{"event.hour": {"lt": 9}}, {"event.hour": {"gte": 17}}]},
        },
        # examples/policies/trial-tier-strict-limits.yaml — pure expression, no operator
        {
            "trigger": "expression",
            "match": {
                "run.tool_call_count": {"gt": 8},
                "or": [{"agent.tier": {"eq": "trial"}}, {"org.plan": {"in": ["free", "starter"]}}],
            },
        },
        # api_svc/fix_classification.py::build_suggested_policy — all four builders
        {"trigger": "tool_call_count", "operator": "gte", "value": 12},
        {"trigger": "error_count", "operator": "gte", "value": 3},
        {"trigger": "step_count", "operator": "gte", "value": 40},
        # dashboard enablePolicyFromIncident() / createSuggestedPolicy()
        {"trigger": "signal", "operator": "contains", "value": "TOOL_LOOP"},
        # docs/policies.md worked examples
        {"trigger": "cost_usd", "operator": "gt", "value": 0.50},
        {"trigger": "llm_latency_ms", "operator": "gt", "value": 10000},
        {"trigger": "finish_reason", "operator": "eq", "value": "length"},
    ]

    def test_every_shipped_condition_still_validates(self):
        for condition in self.CORPUS:
            with self.subTest(condition=condition):
                validate_condition(condition, policy_name="corpus")

    def test_every_shipped_condition_still_constructs_a_policy(self):
        for condition in self.CORPUS:
            with self.subTest(condition=condition):
                action = (
                    {"type": "require_approval"}
                    if condition["trigger"] == "before_tool_call"
                    else {"type": "log"}
                )
                Policy(name="corpus", condition=condition, action=action)


# ── Allowlist shape ──────────────────────────────────────────────────────────


class TestAllowlists(unittest.TestCase):
    def test_operators_are_derived_from_the_operator_table(self):
        from dunetrace.policies import _OPERATORS

        self.assertEqual(VALID_OPERATORS, frozenset(_OPERATORS))

    def test_every_build_metrics_key_is_a_valid_trigger(self):
        """A metric with no trigger is unreachable; a trigger with no metric can
        never fire. build_metrics + the two specials must equal the allowlist."""
        from dunetrace.models import RunState
        from dunetrace.policies import build_metrics

        produced = set(build_metrics(RunState(run_id="r", agent_id="a", agent_version="v"), 0))
        self.assertEqual(
            produced | {"signal", "before_tool_call", "expression"}, set(VALID_TRIGGERS)
        )

    def test_signal_is_the_only_list_valued_trigger(self):
        self.assertEqual(LIST_VALUED_TRIGGERS, frozenset({"signal"}))


if __name__ == "__main__":
    unittest.main()
