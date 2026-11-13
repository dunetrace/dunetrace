"""
Phase 2: dry run. A policy evaluates, records what it would have done, and
executes nothing.

The reason nobody arms a policy is that they cannot see what it will do first.
So the contract here is narrow and strict: identical evaluation, identical
verdict, zero effect on the run. Anything that makes a dry-run verdict differ
from what enforcement would really have done makes the preview a lie, which is
worse than having no preview.

Two design points the tests pin, because both are easy to get wrong:

  * A dry-run FIRING is never sampled and never gated behind the
    evaluation-reporting opt-in. "How many times would this have fired" is the
    whole question, and a sampled count answers it wrong.
  * A dry-run policy inherits enforcement's once-per-run dedupe, including the
    `log` exception. An enforcing stop fires once and ends the run; a dry-run
    stop that recorded on every later matching step would report six firings
    for something that would have happened once.
"""

from __future__ import annotations

import logging
import time
import unittest

from dunetrace import CallableExporter, Dunetrace, PolicyViolation
from dunetrace.policies import DEFAULT_MODE, DRY_RUN_MODE, Policy

_COND = {"trigger": "tool_call_count", "operator": "gt", "value": 2}
_STOP = {"type": "stop", "params": {"message": "too many tools"}}

#: Same budget the rest of the in-path work is held to (test_benchmark.py).
MAX_OVERHEAD_US = 500


def _client(mode, action=None, verdicts=None, condition=None):
    dt = Dunetrace(
        endpoint=None,
        exporters=(
            [
                CallableExporter(
                    lambda e: (
                        verdicts.append(e.payload)
                        if e.event_type.value == "policy.evaluated"
                        else None
                    )
                )
            ]
            if verdicts is not None
            else None
        ),
    )
    dt._policy_engine.add(
        Policy(
            name="cap-tools",
            agent_id="a",
            condition=dict(condition or _COND),
            action=dict(action or _STOP),
            mode=mode,
        )
    )
    return dt


def _drive(dt, calls=5):
    """Run an agent that trips the condition. Returns True if it was stopped."""
    try:
        with dt.run("a", model="gpt-4o") as run:
            for i in range(calls):
                run.tool_called("search", {"q": i})
                run.tool_responded("search", success=True)
            run.final_answer()
        return False
    except PolicyViolation:
        return True


class TestDryRunExecutesNothing(unittest.TestCase):
    """The headline guarantee."""

    def test_enforcing_stops_the_run(self):
        self.assertTrue(_drive(_client(DEFAULT_MODE)))

    def test_dry_run_does_not_stop_the_run(self):
        self.assertFalse(_drive(_client(DRY_RUN_MODE)))

    def test_dry_run_switch_model_does_not_change_the_model(self):
        dt = _client(DRY_RUN_MODE, action={"type": "switch_model", "params": {"model": "cheap"}})
        seen = {}
        with dt.run("a", model="gpt-4o") as run:
            for i in range(5):
                run.tool_called("s", {"q": i})
                run.tool_responded("s", success=True)
            seen["override"] = run.model_override
            run.final_answer()
        self.assertIsNone(seen["override"], "dry run mutated the run")

    def test_dry_run_inject_prompt_adds_nothing(self):
        dt = _client(DRY_RUN_MODE, action={"type": "inject_prompt", "params": {"prompt": "hi"}})
        seen = {}
        with dt.run("a") as run:
            for i in range(5):
                run.tool_called("s", {"q": i})
                run.tool_responded("s", success=True)
            seen["additions"] = list(run.prompt_additions)
            run.final_answer()
        self.assertEqual(seen["additions"], [])

    def test_dry_run_emits_no_policy_triggered_event(self):
        """policy.triggered is an enforcement record other systems act on. A
        dry-run policy must be invisible to everything but the verdict log."""
        seen = []
        dt = Dunetrace(
            endpoint=None, exporters=[CallableExporter(lambda e: seen.append(e.event_type.value))]
        )
        dt._policy_engine.add(
            Policy(
                name="p", agent_id="a", condition=dict(_COND), action=dict(_STOP), mode=DRY_RUN_MODE
            )
        )
        _drive(dt)
        self.assertNotIn("policy.triggered", seen)

    def test_enforcing_still_emits_policy_triggered(self):
        seen = []
        dt = Dunetrace(
            endpoint=None, exporters=[CallableExporter(lambda e: seen.append(e.event_type.value))]
        )
        dt._policy_engine.add(
            Policy(name="p", agent_id="a", condition=dict(_COND), action=dict(_STOP))
        )
        _drive(dt)
        self.assertIn("policy.triggered", seen)


class TestVerdictRecord(unittest.TestCase):
    """Enough context that someone reading it later knows exactly what would
    have happened and why."""

    def _verdict(self, action=None, condition=None):
        verdicts = []
        _drive(_client(DRY_RUN_MODE, action=action, verdicts=verdicts, condition=condition))
        fired = [v for v in verdicts if v.get("fired")]
        self.assertTrue(fired, "no dry-run verdict was recorded")
        return fired[0]

    def test_records_every_required_field(self):
        v = self._verdict()
        self.assertEqual(v["policy_name"], "cap-tools")
        self.assertEqual(v["run_id"], v["run_id"])  # present
        self.assertEqual(v["mode"], DRY_RUN_MODE)
        self.assertEqual(v["trigger"], "tool_call_count")
        self.assertIsInstance(v["step_index"], int)
        self.assertEqual(v["would_action_type"], "stop")
        self.assertEqual(v["would_action_params"], {"message": "too many tools"})
        self.assertTrue(v["matched_branch"])
        self.assertTrue(v["ts"])

    def test_records_the_action_params_verbatim(self):
        """A verdict has to be readable without going back to the policy."""
        v = self._verdict(action={"type": "switch_model", "params": {"model": "gpt-4o-mini"}})
        self.assertEqual(v["would_action_type"], "switch_model")
        self.assertEqual(v["would_action_params"], {"model": "gpt-4o-mini"})

    def test_matched_branch_names_the_expression_clause(self):
        v = self._verdict(
            condition={**_COND, "match": {"run.error_count": {"gte": 0}}},
        )
        self.assertIn("run.error_count", v["matched_branch"])

    def test_verdict_is_never_sampled(self):
        self.assertFalse(self._verdict()["sampled"])

    def test_verdict_is_recorded_without_the_reporting_opt_in(self):
        """The opt-in is off by default. If verdicts depended on it, a policy in
        dry run would look exactly like a policy doing nothing."""
        verdicts = []
        dt = _client(DRY_RUN_MODE, verdicts=verdicts)
        self.assertFalse(getattr(dt, "_policy_evaluation_reporting", False))
        _drive(dt)
        self.assertTrue([v for v in verdicts if v.get("fired")])


class TestFiringCountsMatchEnforcement(unittest.TestCase):
    """A verdict count that does not line up one-to-one with enforcement makes
    the dashboard's central number meaningless."""

    def _fired_count(self, action):
        verdicts = []
        _drive(_client(DRY_RUN_MODE, action=action, verdicts=verdicts))
        return len([v for v in verdicts if v.get("fired")])

    def test_stop_records_once_per_run(self):
        self.assertEqual(self._fired_count(_STOP), 1)

    def test_log_records_every_match(self):
        """log is exempt from dedupe when enforcing, so it is here too."""
        self.assertGreater(self._fired_count({"type": "log"}), 1)


class TestEnforcingPoliciesUnaffected(unittest.TestCase):
    def test_default_mode_is_enforcing(self):
        p = Policy(name="p", condition=dict(_COND), action=dict(_STOP))
        self.assertEqual(p.mode, DEFAULT_MODE)
        self.assertFalse(p.is_dry_run)

    def test_enforcing_evaluation_still_samples(self):
        """Only dry-run firings bypass the limiter. Ordinary observability is
        unchanged, or a hot loop floods the table."""
        dt = _client(DEFAULT_MODE)
        limiter = getattr(dt, "_policy_eval_rate_limiter", None)
        self.assertIsNotNone(limiter, "the limiter must still be installed")

    def test_no_observer_installed_when_nothing_is_in_dry_run(self):
        """The zero-overhead short-circuit has to survive this feature."""
        dt = _client(DEFAULT_MODE)
        logging.getLogger("dunetrace.policies.evaluation").setLevel(logging.WARNING)
        with dt.run("a") as run:
            self.assertIsNone(run._policy_observer())
            run.final_answer()

    def test_observer_installed_when_a_policy_is_in_dry_run(self):
        dt = _client(DRY_RUN_MODE)
        logging.getLogger("dunetrace.policies.evaluation").setLevel(logging.WARNING)
        with dt.run("a") as run:
            self.assertIsNotNone(run._policy_observer())
            run.final_answer()


class TestPromotionPreservesHistory(unittest.TestCase):
    """Promotion is a field change on the same policy id, so verdicts written
    before it stay attached and readable next to what came after."""

    def test_promotion_is_a_field_change_on_the_same_policy(self):
        dt = Dunetrace(endpoint=None)
        engine = dt._policy_engine
        remote = {
            "id": 42,
            "name": "cap-tools",
            "agent_id": "a",
            "condition": dict(_COND),
            "action": dict(_STOP),
            "mode": DRY_RUN_MODE,
        }
        engine.load([remote], secret="", agent_id="a")
        self.assertTrue(engine._policies[0].is_dry_run)
        self.assertEqual(engine._policies[0].id, 42)

        engine.load([{**remote, "mode": DEFAULT_MODE}], secret="", agent_id="a")
        promoted = engine._policies[0]
        self.assertFalse(promoted.is_dry_run)
        self.assertEqual(promoted.id, 42, "the id is what verdict history hangs off")
        self.assertEqual(promoted.name, "cap-tools")

    def test_promoted_policy_enforces(self):
        dt = _client(DRY_RUN_MODE)
        self.assertFalse(_drive(dt))
        dt._policy_engine._policies[0].mode = DEFAULT_MODE
        dt._policy_engine._refresh_dry_run_flag()
        self.assertTrue(_drive(dt), "promotion did not arm the policy")


class TestLatency(unittest.TestCase):
    def test_dry_run_stays_within_the_in_path_budget(self):
        """Dry run runs where enforcement runs, so it answers to the same
        budget. Measured per tool call, against the same MAX_OVERHEAD_US the
        rest of the in-path work uses."""
        dt = _client(DRY_RUN_MODE)
        iterations = 200
        t0 = time.perf_counter()
        with dt.run("a") as run:
            for i in range(iterations):
                run.tool_called("s", {"q": i})
                run.tool_responded("s", success=True)
            run.final_answer()
        us = (time.perf_counter() - t0) / iterations * 1_000_000
        self.assertLess(us, MAX_OVERHEAD_US, f"{us:.0f}us per call exceeds the in-path budget")


if __name__ == "__main__":
    unittest.main(verbosity=2)
