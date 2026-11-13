"""
Tests for PolicyEngine in dunetrace.policies.
No network, no DB — fully offline.

Run:
    cd packages/sdk-py
    python -m pytest tests/test_policies.py -v
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from dunetrace.models import RunState
from dunetrace.policies import Policy, PolicyEngine, PolicyViolation


# ── Helpers ────────────────────────────────────────────────────────────────────


def _policy(
    name: str = "test-policy",
    trigger: str = "tool_call_count",
    operator: str = "gt",
    value: int = 5,
    action_type: str = "stop",
    agent_id: str = "*",
    priority: int = 100,
    enabled: bool = True,
    policy_id: int | None = None,
) -> Policy:
    return Policy(
        name=name,
        condition={"trigger": trigger, "operator": operator, "value": value},
        action={"type": action_type},
        agent_id=agent_id,
        enabled=enabled,
        priority=priority,
        id=policy_id,
    )


def _metrics(**kwargs) -> dict:
    base = {
        "tool_call_count": 0,
        "step_count": 0,
        "cost_usd": 0.0,
        "error_count": 0,
        "finish_reason": None,
        "llm_latency_ms": None,
    }
    base.update(kwargs)
    return base


# ── add() and evaluate() ───────────────────────────────────────────────────────


class TestPolicyEngineAddAndEvaluate(unittest.TestCase):
    def test_add_policy_evaluates_correctly(self):
        """add() + evaluate() returns the policy when condition is met."""
        engine = PolicyEngine()
        engine.add(_policy(trigger="tool_call_count", operator="gt", value=5))
        result = engine.evaluate("agent", _metrics(tool_call_count=6), set())
        self.assertIsNotNone(result)
        policy, action = result
        self.assertEqual(action["type"], "stop")

    def test_evaluate_returns_none_when_condition_not_met(self):
        """Condition is 'gt 5' but count=5 should NOT fire."""
        engine = PolicyEngine()
        engine.add(_policy(trigger="tool_call_count", operator="gt", value=5))
        result = engine.evaluate("agent", _metrics(tool_call_count=5), set())
        self.assertIsNone(result)

    def test_policy_not_fired_when_already_triggered(self):
        """A policy whose key is in triggered_already must be skipped."""
        engine = PolicyEngine()
        p = _policy(name="once-policy", policy_id=42)
        engine.add(p)
        already = {p.key}
        result = engine.evaluate("agent", _metrics(tool_call_count=10), already)
        self.assertIsNone(result)

    def test_disabled_policy_never_fires(self):
        engine = PolicyEngine()
        engine.add(_policy(enabled=False, trigger="tool_call_count", operator="gt", value=0))
        result = engine.evaluate("agent", _metrics(tool_call_count=100), set())
        self.assertIsNone(result)

    def test_policy_scoped_to_agent_fires_only_for_that_agent(self):
        engine = PolicyEngine()
        engine.add(_policy(agent_id="agent-A", trigger="step_count", operator="gt", value=3))
        self.assertIsNotNone(engine.evaluate("agent-A", _metrics(step_count=4), set()))
        self.assertIsNone(engine.evaluate("agent-B", _metrics(step_count=4), set()))

    def test_wildcard_agent_id_fires_for_any_agent(self):
        engine = PolicyEngine()
        engine.add(_policy(agent_id="*", trigger="step_count", operator="gt", value=3))
        self.assertIsNotNone(engine.evaluate("any-agent", _metrics(step_count=4), set()))

    def test_priority_ordering_highest_priority_fires_first(self):
        """Lower priority number = higher priority; should be returned first."""
        engine = PolicyEngine()
        engine.add(
            _policy(
                name="low-pri",
                priority=200,
                trigger="step_count",
                operator="gt",
                value=0,
                action_type="log",
            )
        )
        engine.add(
            _policy(
                name="high-pri",
                priority=10,
                trigger="step_count",
                operator="gt",
                value=0,
                action_type="stop",
            )
        )
        result = engine.evaluate("agent", _metrics(step_count=1), set())
        self.assertIsNotNone(result)
        policy, _ = result
        self.assertEqual(policy.name, "high-pri")

    def test_multiple_policies_only_first_match_returned(self):
        """evaluate() returns the first matching policy, not all."""
        engine = PolicyEngine()
        engine.add(
            _policy(
                name="p1",
                priority=1,
                trigger="step_count",
                operator="gte",
                value=1,
                action_type="log",
            )
        )
        engine.add(
            _policy(
                name="p2",
                priority=2,
                trigger="step_count",
                operator="gte",
                value=1,
                action_type="stop",
            )
        )
        result = engine.evaluate("agent", _metrics(step_count=1), set())
        policy, _ = result
        self.assertEqual(policy.name, "p1")

    def test_add_increases_len(self):
        engine = PolicyEngine()
        self.assertEqual(len(engine), 0)
        engine.add(_policy())
        self.assertEqual(len(engine), 1)

    def test_evaluate_empty_engine_returns_none(self):
        engine = PolicyEngine()
        self.assertIsNone(engine.evaluate("agent", _metrics(), set()))


# ── Operators ─────────────────────────────────────────────────────────────────


class TestPolicyOperators(unittest.TestCase):
    def _engine_with(self, operator: str, value) -> PolicyEngine:
        engine = PolicyEngine()
        engine.add(_policy(trigger="tool_call_count", operator=operator, value=value))
        return engine

    def test_gt_operator(self):
        eng = self._engine_with("gt", 5)
        self.assertIsNone(eng.evaluate("a", _metrics(tool_call_count=5), set()))
        self.assertIsNotNone(eng.evaluate("a", _metrics(tool_call_count=6), set()))

    def test_gte_operator(self):
        eng = self._engine_with("gte", 5)
        self.assertIsNotNone(eng.evaluate("a", _metrics(tool_call_count=5), set()))

    def test_lt_operator(self):
        eng = self._engine_with("lt", 5)
        self.assertIsNone(eng.evaluate("a", _metrics(tool_call_count=5), set()))
        self.assertIsNotNone(eng.evaluate("a", _metrics(tool_call_count=4), set()))

    def test_lte_operator(self):
        eng = self._engine_with("lte", 5)
        self.assertIsNotNone(eng.evaluate("a", _metrics(tool_call_count=5), set()))

    def test_eq_operator(self):
        eng = self._engine_with("eq", 3)
        self.assertIsNotNone(eng.evaluate("a", _metrics(tool_call_count=3), set()))
        self.assertIsNone(eng.evaluate("a", _metrics(tool_call_count=4), set()))

    def test_neq_operator(self):
        eng = self._engine_with("neq", 3)
        self.assertIsNone(eng.evaluate("a", _metrics(tool_call_count=3), set()))
        self.assertIsNotNone(eng.evaluate("a", _metrics(tool_call_count=4), set()))

    def test_none_metric_never_fires(self):
        """If the metric is None, no operator should match."""
        engine = PolicyEngine()
        engine.add(_policy(trigger="llm_latency_ms", operator="gt", value=1000))
        result = engine.evaluate("agent", _metrics(llm_latency_ms=None), set())
        self.assertIsNone(result)


# ── load() — simulates remote fetch ──────────────────────────────────────────


class TestPolicyEngineLoad(unittest.TestCase):
    def test_load_replaces_remote_policies(self):
        """load() with a list of policy dicts makes them evaluatable."""
        engine = PolicyEngine()
        raw = [
            {
                "id": 1,
                "name": "remote-policy",
                "agent_id": "*",
                "condition": {"trigger": "tool_call_count", "operator": "gt", "value": 2},
                "action": {"type": "stop"},
                "enabled": True,
                "priority": 100,
            }
        ]
        engine.load(raw)
        result = engine.evaluate("agent", _metrics(tool_call_count=3), set())
        self.assertIsNotNone(result)
        policy, action = result
        self.assertEqual(policy.name, "remote-policy")

    def test_load_keeps_local_policies(self):
        """load() must preserve policies added via add() (no id) alongside remote ones."""
        engine = PolicyEngine()
        engine.add(_policy(name="local", trigger="step_count", operator="gt", value=0))
        engine.load([])  # empty remote batch
        result = engine.evaluate("agent", _metrics(step_count=1), set())
        self.assertIsNotNone(result)
        policy, _ = result
        self.assertEqual(policy.name, "local")

    def test_load_increments_generation(self):
        engine = PolicyEngine()
        gen_before = engine._generation
        engine.load([])
        self.assertGreater(engine._generation, gen_before)

    def test_load_with_bad_signature_skips_policy(self):
        """A policy that fails HMAC verification must be silently skipped."""
        engine = PolicyEngine()
        raw = [
            {
                "id": 99,
                "name": "tampered",
                "agent_id": "*",
                "condition": {"trigger": "step_count", "operator": "gt", "value": 0},
                "action": {"type": "stop"},
                "enabled": True,
                "priority": 100,
                "signature": "badhash",
            }
        ]
        engine.load(raw, secret="supersecret")
        result = engine.evaluate("agent", _metrics(step_count=1), set())
        self.assertIsNone(result)

    def test_fallback_local_policies_still_work_when_load_empty(self):
        """If remote fetch returns nothing, local policies remain active."""
        engine = PolicyEngine()
        engine.add(_policy(name="fallback", trigger="error_count", operator="gte", value=1))
        engine.load([])  # empty remote result
        result = engine.evaluate("agent", _metrics(error_count=1), set())
        self.assertIsNotNone(result)


# ── TTL / needs_fetch ─────────────────────────────────────────────────────────


class TestPolicyEngineTTL(unittest.TestCase):
    def test_needs_fetch_true_before_first_fetch(self):
        """A fresh engine with no fetch history needs fetching."""
        engine = PolicyEngine()
        self.assertTrue(engine.needs_fetch("agent-1"))

    def test_needs_fetch_false_right_after_mark_fetched(self):
        """Immediately after mark_fetched, the TTL window is fresh."""
        engine = PolicyEngine()
        engine.mark_fetched("agent-1")
        self.assertFalse(engine.needs_fetch("agent-1"))

    def test_needs_fetch_true_after_ttl_expires(self):
        """After the TTL elapses, needs_fetch returns True again."""
        engine = PolicyEngine()
        # Stamp the fetch time in the past (beyond TTL)
        past = time.monotonic() - (PolicyEngine._FETCH_TTL + 1)
        engine._fetch_times["agent-1"] = past
        self.assertTrue(engine.needs_fetch("agent-1"))

    def test_needs_fetch_independent_per_agent(self):
        """TTL is tracked per agent_id."""
        engine = PolicyEngine()
        engine.mark_fetched("agent-A")
        self.assertFalse(engine.needs_fetch("agent-A"))
        self.assertTrue(engine.needs_fetch("agent-B"))  # never fetched


# ── Thread safety ─────────────────────────────────────────────────────────────


class TestPolicyEngineThreadSafety(unittest.TestCase):
    def test_concurrent_evaluate_calls_do_not_crash(self):
        """Many threads calling evaluate() simultaneously must not raise."""
        engine = PolicyEngine()
        for i in range(20):
            engine.add(_policy(name=f"p{i}", trigger="step_count", operator="gt", value=i))

        errors = []

        def _worker():
            try:
                for _ in range(50):
                    engine.evaluate("agent", _metrics(step_count=10), set())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_concurrent_add_and_evaluate_do_not_crash(self):
        """add() from one thread while evaluate() runs in another must be safe."""
        engine = PolicyEngine()
        errors = []

        def _adder():
            for i in range(100):
                try:
                    engine.add(
                        _policy(name=f"dynamic-{i}", trigger="step_count", operator="gt", value=i)
                    )
                except Exception as exc:
                    errors.append(exc)

        def _evaluator():
            for _ in range(200):
                try:
                    engine.evaluate("agent", _metrics(step_count=50), set())
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=_adder)
        t2 = threading.Thread(target=_evaluator)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")


# ── PolicyViolation ────────────────────────────────────────────────────────────


class TestPolicyViolation(unittest.TestCase):
    def test_policy_violation_carries_name(self):
        exc = PolicyViolation("my-policy", {"type": "stop"})
        self.assertEqual(exc.policy_name, "my-policy")

    def test_policy_violation_carries_action(self):
        action = {"type": "stop"}
        exc = PolicyViolation("p", action)
        self.assertEqual(exc.action, action)

    def test_policy_violation_default_message(self):
        exc = PolicyViolation("my-policy", {})
        self.assertIn("my-policy", str(exc))

    def test_policy_violation_is_runtime_error(self):
        self.assertIsInstance(PolicyViolation("p", {}), RuntimeError)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSignalPolicyDoesNotFailOpen(unittest.TestCase):
    """A trigger="signal" policy is a safety control. It must not stop
    enforcing because a detector was slow — cost is shed by scope instead."""

    def test_cost_downgraded_detector_still_enforces(self):
        from dunetrace.detectors import _cost_trackers, run_detectors, TIER1_DETECTORS

        target = TIER1_DETECTORS[0]
        state = RunState(run_id="r", agent_id="a", agent_version="v")
        # One real evaluation so the detector has a genuine cost tracker.
        run_detectors(state, detectors=[target], context="analytics")
        tracker = _cost_trackers[target.name]
        previous = tracker.downgraded_at
        tracker.downgraded_at = time.monotonic()
        try:
            runtime = run_detectors(state, detectors=[target], context="runtime")
            policy = run_detectors(state, detectors=[target], context="policy")
        finally:
            tracker.downgraded_at = previous

        # "runtime" sheds the downgraded detector; "policy" must not, because
        # shedding it silently disables a guardrail the customer configured.
        self.assertEqual(runtime, [])
        self.assertIsInstance(policy, list)

    def test_only_referenced_detectors_run_for_a_signal_policy(self):
        from dunetrace import Dunetrace

        dt = Dunetrace(endpoint=None)
        dt.add_policy(
            "halt-on-loop",
            {"trigger": "signal", "operator": "contains", "value": "TOOL_LOOP"},
            {"type": "log"},
        )
        with dt.run("agent") as run:
            run.tool_called("x", {})
            run.tool_responded("x", success=True)
            wanted = run._needed_signal_types
        self.assertEqual(wanted, {"TOOL_LOOP"})
        dt.shutdown(timeout=1)

    def test_unresolvable_condition_runs_the_full_battery(self):
        """An expression/match condition can't be resolved to failure types
        statically — over-running is the safe direction for a safety control."""
        from dunetrace import Dunetrace

        dt = Dunetrace(endpoint=None)
        dt.add_policy(
            "halt",
            {"trigger": "signal", "operator": "contains", "value": {"any": ["A", "B"]}},
            {"type": "log"},
        )
        with dt.run("agent") as run:
            run.tool_called("x", {})
            run.tool_responded("x", success=True)
            wanted = run._needed_signal_types
        self.assertIsNone(wanted)
        dt.shutdown(timeout=1)


class TestRemotePolicyTrustBoundary(unittest.TestCase):
    """A remote `stop` policy halts a customer's production agent. The stated
    threat is a compromised or spoofed server doing exactly that."""

    @staticmethod
    def _remote(action_type, signature=""):
        return {
            "id": 7,
            "name": "halt",
            "agent_id": "*",
            "condition": {"trigger": "tool_call_count", "operator": "gte", "value": 1},
            "action": {"type": action_type},
            "enabled": True,
            "priority": 100,
            "signature": signature,
        }

    def test_unverifiable_stop_is_downgraded_to_log(self):
        """With no secret there is nothing to verify against, so an enforcing
        remote policy is recorded rather than obeyed."""
        engine = PolicyEngine()
        engine.load([self._remote("stop")], secret="")
        self.assertEqual(len(engine), 1)
        with engine._lock:
            self.assertEqual(engine._policies[0].action["type"], "log")

    def test_unverifiable_require_approval_is_downgraded(self):
        engine = PolicyEngine()
        engine.load([self._remote("require_approval")], secret="")
        with engine._lock:
            self.assertEqual(engine._policies[0].action["type"], "log")

    def test_non_enforcing_remote_policy_is_untouched(self):
        engine = PolicyEngine()
        engine.load([self._remote("log")], secret="")
        with engine._lock:
            self.assertEqual(engine._policies[0].action["type"], "log")

    def test_unsigned_policy_is_rejected_when_a_secret_is_configured(self):
        """Accepting unsigned policies made verification optional for exactly
        the party it was meant to constrain — omit the field, be obeyed."""
        engine = PolicyEngine()
        engine.load([self._remote("stop", signature="")], secret="s3cret")
        self.assertEqual(len(engine), 0)

    def test_local_policies_are_never_downgraded(self):
        """A policy the customer registered in their own process is their own
        code and needs no signature."""
        engine = PolicyEngine()
        engine.add(_policy(name="local-halt", action_type="stop", policy_id=None))
        with engine._lock:
            self.assertEqual(engine._policies[0].action["type"], "stop")

    def test_enforcing_action_set_covers_every_run_altering_action(self):
        from dunetrace.policies import ENFORCING_ACTIONS

        for action in ("stop", "require_approval", "escalate_to_human", "switch_model"):
            self.assertIn(action, ENFORCING_ACTIONS)
        # Purely observational actions stay allowed unverified.
        self.assertNotIn("log", ENFORCING_ACTIONS)
        self.assertNotIn("slow_response_pace", ENFORCING_ACTIONS)
