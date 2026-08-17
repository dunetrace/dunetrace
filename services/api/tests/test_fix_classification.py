"""
Tests for fix_classification.py — the dunetrace_native vs customer_code split
and the deterministic suggested-policy builder.

No DB, no network, no LLM calls.
"""

from __future__ import annotations

import unittest

from api_svc.fix_classification import build_suggested_policy, classify_fix


def _signal(failure_type: str, evidence: dict, agent_id: str = "agent-1") -> dict:
    return {"failure_type": failure_type, "evidence": evidence, "agent_id": agent_id}


class TestClassifyFix(unittest.TestCase):
    def test_tool_loop_is_dunetrace_native(self):
        signal = _signal("TOOL_LOOP", {"count": 5})
        self.assertEqual(classify_fix(signal), "dunetrace_native")

    def test_retry_storm_is_dunetrace_native(self):
        signal = _signal("RETRY_STORM", {"consecutive_fails": 3})
        self.assertEqual(classify_fix(signal), "dunetrace_native")

    def test_cascading_tool_failure_is_dunetrace_native(self):
        signal = _signal("CASCADING_TOOL_FAILURE", {"consecutive_failures": 4})
        self.assertEqual(classify_fix(signal), "dunetrace_native")

    def test_step_count_inflation_is_dunetrace_native(self):
        signal = _signal("STEP_COUNT_INFLATION", {"current_steps": 12})
        self.assertEqual(classify_fix(signal), "dunetrace_native")

    def test_tool_thrashing_is_customer_code(self):
        """Alternating A/B pattern — no count-threshold trigger fits."""
        signal = _signal("TOOL_THRASHING", {"tool_a": "x", "tool_b": "y", "count": 6})
        self.assertEqual(classify_fix(signal), "customer_code")

    def test_cost_spike_is_customer_code(self):
        """Evidence is in tokens; no token-count PolicyCondition trigger exists."""
        signal = _signal("COST_SPIKE", {"total_tokens": 80000, "threshold": 50000})
        self.assertEqual(classify_fix(signal), "customer_code")

    def test_prompt_injection_is_customer_code(self):
        signal = _signal("PROMPT_INJECTION_SIGNAL", {"matched_pattern_count": 2})
        self.assertEqual(classify_fix(signal), "customer_code")

    def test_rag_empty_retrieval_is_customer_code(self):
        signal = _signal("RAG_EMPTY_RETRIEVAL", {"result_count": 0})
        self.assertEqual(classify_fix(signal), "customer_code")

    def test_native_type_with_missing_evidence_degrades_to_customer_code(self):
        """A detector in the native set whose evidence is missing/malformed
        must not offer a broken apply button — degrade gracefully."""
        signal = _signal("TOOL_LOOP", {})  # no "count" key at all
        self.assertEqual(classify_fix(signal), "customer_code")

    def test_native_type_with_zero_evidence_degrades_to_customer_code(self):
        signal = _signal("STEP_COUNT_INFLATION", {"current_steps": 0})
        self.assertEqual(classify_fix(signal), "customer_code")

    def test_unknown_failure_type_is_customer_code(self):
        signal = _signal("SOME_FUTURE_DETECTOR", {})
        self.assertEqual(classify_fix(signal), "customer_code")


class TestBuildSuggestedPolicy(unittest.TestCase):
    def test_tool_loop_policy_shape(self):
        signal = _signal("TOOL_LOOP", {"count": 5}, agent_id="my-agent")
        policy = build_suggested_policy(signal)
        self.assertIsNotNone(policy)
        self.assertEqual(policy["agent_id"], "my-agent")
        self.assertEqual(
            policy["condition"], {"trigger": "tool_call_count", "operator": "gte", "value": 5}
        )
        self.assertEqual(policy["action"], {"type": "stop"})
        self.assertEqual(policy["priority"], 100)
        self.assertTrue(policy["enabled"])

    def test_retry_storm_policy_uses_error_count(self):
        signal = _signal("RETRY_STORM", {"consecutive_fails": 3})
        policy = build_suggested_policy(signal)
        self.assertEqual(
            policy["condition"], {"trigger": "error_count", "operator": "gte", "value": 3}
        )

    def test_cascading_tool_failure_policy_uses_error_count(self):
        signal = _signal("CASCADING_TOOL_FAILURE", {"consecutive_failures": 4})
        policy = build_suggested_policy(signal)
        self.assertEqual(
            policy["condition"], {"trigger": "error_count", "operator": "gte", "value": 4}
        )

    def test_step_count_inflation_policy_uses_step_count(self):
        signal = _signal("STEP_COUNT_INFLATION", {"current_steps": 15})
        policy = build_suggested_policy(signal)
        self.assertEqual(
            policy["condition"], {"trigger": "step_count", "operator": "gte", "value": 15}
        )

    def test_policy_shape_matches_policycreate_schema(self):
        """The dashboard submits this dict verbatim to POST /v1/policies —
        every key PolicyCreate requires must be present."""
        signal = _signal("TOOL_LOOP", {"count": 5})
        policy = build_suggested_policy(signal)
        self.assertEqual(
            set(policy.keys()), {"name", "agent_id", "condition", "action", "priority", "enabled"}
        )

    def test_non_native_type_returns_none(self):
        signal = _signal("TOOL_AVOIDANCE", {})
        self.assertIsNone(build_suggested_policy(signal))

    def test_missing_evidence_returns_none(self):
        signal = _signal("TOOL_LOOP", {})
        self.assertIsNone(build_suggested_policy(signal))

    def test_non_numeric_evidence_returns_none(self):
        signal = _signal("TOOL_LOOP", {"count": "not-a-number"})
        self.assertIsNone(build_suggested_policy(signal))


if __name__ == "__main__":
    unittest.main(verbosity=2)
