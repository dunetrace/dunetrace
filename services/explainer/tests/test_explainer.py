"""
tests/test_explainer.py

Tests for the explain layer.
Zero external dependencies. No DB, no LLM, no network.

Run:
    cd services/explainer
    python -m unittest tests.test_explainer -v
"""
from __future__ import annotations

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/sdk-py")
))

from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.explainer import explain
from explainer_svc.models import Explanation, CodeFix


# ── Factories ──────────────────────────────────────────────────────────────────

def make_signal(
    failure_type: FailureType,
    severity: Severity = Severity.HIGH,
    confidence: float = 0.90,
    step_index: int = 5,
    evidence: dict = None,
) -> FailureSignal:
    return FailureSignal(
        failure_type=failure_type,
        severity=severity,
        run_id="run-test-abc",
        agent_id="agent-test",
        agent_version="abc12345",
        step_index=step_index,
        confidence=confidence,
        evidence=evidence or {},
        detected_at=time.time(),
    )


# ── Contract tests — every explanation must satisfy these ──────────────────────

class TestExplanationContract(unittest.TestCase):
    """
    Every Explanation produced by explain() must satisfy
    the same structural contract regardless of failure type.
    """

    ALL_TYPES = [
        (FailureType.TOOL_LOOP,               {"tool": "web_search", "count": 5, "window": 5}),
        (FailureType.TOOL_THRASHING,          {"tool_a": "search", "tool_b": "browse", "oscillation_count": 3}),
        (FailureType.TOOL_AVOIDANCE,          {"available_tools": ["web_search", "calculator"], "tool_calls_made": 0}),
        (FailureType.GOAL_ABANDONMENT,        {"stall_steps": 4, "last_tool_used": "database_lookup"}),
        (FailureType.PROMPT_INJECTION_SIGNAL, {"matched_patterns": ["ignore_instructions", "you_are_now"], "pattern_count": 2}),
        (FailureType.RAG_EMPTY_RETRIEVAL,     {"index_name": "product-docs", "result_count": 0, "top_score": None, "bad_retrievals": 1}),
    ]

    def _check_contract(self, exp: Explanation, failure_type: FailureType):
        self.assertIsInstance(exp, Explanation)

        # Identity fields are passed through unchanged
        self.assertEqual(exp.failure_type, failure_type.value)
        self.assertEqual(exp.run_id,        "run-test-abc")
        self.assertEqual(exp.agent_id,      "agent-test")
        self.assertEqual(exp.agent_version, "abc12345")
        self.assertAlmostEqual(exp.confidence, 0.90)

        # Text fields must be non-empty strings
        self.assertIsInstance(exp.title, str)
        self.assertGreater(len(exp.title), 10,
                           f"{failure_type}: title too short: {exp.title!r}")

        self.assertIsInstance(exp.what, str)
        self.assertGreater(len(exp.what), 20,
                           f"{failure_type}: 'what' too short")

        self.assertIsInstance(exp.why_it_matters, str)
        self.assertGreater(len(exp.why_it_matters), 20,
                           f"{failure_type}: 'why_it_matters' too short")

        self.assertIsInstance(exp.evidence_summary, str)
        self.assertGreater(len(exp.evidence_summary), 5,
                           f"{failure_type}: evidence_summary too short")

        # Must have at least one fix
        self.assertIsInstance(exp.suggested_fixes, list)
        self.assertGreater(len(exp.suggested_fixes), 0,
                           f"{failure_type}: no suggested fixes")

        for fix in exp.suggested_fixes:
            self.assertIsInstance(fix, CodeFix)
            self.assertIsInstance(fix.description, str)
            self.assertGreater(len(fix.description), 5)
            self.assertIn(fix.language, ("python", "yaml", "text"))
            self.assertIsInstance(fix.code, str)
            self.assertGreater(len(fix.code), 5)

    def test_all_types_satisfy_contract(self):
        for failure_type, evidence in self.ALL_TYPES:
            with self.subTest(failure_type=failure_type):
                signal = make_signal(failure_type, evidence=evidence)
                exp = explain(signal)
                self._check_contract(exp, failure_type)

    def test_confidence_pct_format(self):
        signal = make_signal(FailureType.TOOL_LOOP,
                             evidence={"tool": "search", "count": 5, "window": 5})
        exp = explain(signal)
        self.assertEqual(exp.confidence_pct(), "90%")

    def test_as_dict_is_json_serialisable(self):
        import json
        signal = make_signal(FailureType.TOOL_LOOP,
                             evidence={"tool": "search", "count": 5, "window": 5})
        d = explain(signal).as_dict()
        # Should not raise
        serialised = json.dumps(d)
        self.assertIsInstance(serialised, str)

    def test_as_dict_shape(self):
        signal = make_signal(FailureType.TOOL_LOOP,
                             evidence={"tool": "search", "count": 5, "window": 5})
        d = explain(signal).as_dict()
        required_keys = {
            "failure_type", "severity", "run_id", "agent_id",
            "agent_version", "confidence", "title", "what",
            "why_it_matters", "evidence_summary", "suggested_fixes",
            "step_index", "detected_at", "evidence",
        }
        self.assertTrue(required_keys.issubset(set(d.keys())))

    def test_slack_text_contains_title(self):
        signal = make_signal(FailureType.TOOL_LOOP,
                             evidence={"tool": "search", "count": 5, "window": 5})
        exp = explain(signal)
        slack = exp.as_slack_text()
        self.assertIn(exp.title, slack)

    def test_slack_text_contains_run_id(self):
        signal = make_signal(FailureType.TOOL_LOOP,
                             evidence={"tool": "search", "count": 5, "window": 5})
        exp = explain(signal)
        slack = exp.as_slack_text()
        self.assertIn("run-test-abc", slack)


# ── TOOL_LOOP specific ─────────────────────────────────────────────────────────

class TestToolLoopExplanation(unittest.TestCase):

    def setUp(self):
        self.signal = make_signal(
            FailureType.TOOL_LOOP,
            confidence=0.95,
            evidence={"tool": "web_search", "count": 5, "window": 5},
        )
        self.exp = explain(self.signal)

    def test_title_contains_tool_name(self):
        self.assertIn("web_search", self.exp.title)

    def test_title_contains_count(self):
        self.assertIn("5", self.exp.title)

    def test_what_mentions_tool(self):
        self.assertIn("web_search", self.exp.what)

    def test_evidence_summary_mentions_confidence(self):
        self.assertIn("95%", self.exp.evidence_summary)

    def test_has_code_fix(self):
        python_fixes = [f for f in self.exp.suggested_fixes if f.language == "python"]
        self.assertGreater(len(python_fixes), 0)

    def test_severity_passthrough(self):
        self.assertEqual(self.exp.severity, "HIGH")


# ── TOOL_THRASHING specific ────────────────────────────────────────────────────

class TestToolThrashingExplanation(unittest.TestCase):

    def setUp(self):
        self.signal = make_signal(
            FailureType.TOOL_THRASHING,
            confidence=0.90,
            evidence={"tool_a": "search", "tool_b": "browse", "oscillation_count": 3},
        )
        self.exp = explain(self.signal)

    def test_title_contains_both_tools(self):
        self.assertIn("search", self.exp.title)
        self.assertIn("browse", self.exp.title)

    def test_what_mentions_oscillation_count(self):
        self.assertIn("3", self.exp.what)

    def test_has_detection_code(self):
        python_fixes = [f for f in self.exp.suggested_fixes if f.language == "python"]
        self.assertGreater(len(python_fixes), 0)
        combined_code = " ".join(f.code for f in python_fixes)
        self.assertIn("deque", combined_code)


# ── TOOL_AVOIDANCE specific ────────────────────────────────────────────────────

class TestToolAvoidanceExplanation(unittest.TestCase):

    def setUp(self):
        self.signal = make_signal(
            FailureType.TOOL_AVOIDANCE,
            severity=Severity.MEDIUM,
            confidence=0.75,
            evidence={"available_tools": ["web_search", "calculator"], "tool_calls_made": 0},
        )
        self.exp = explain(self.signal)

    def test_title_mentions_no_tools(self):
        self.assertIn("without", self.exp.title.lower())

    def test_what_lists_tools(self):
        self.assertIn("web_search", self.exp.what)
        self.assertIn("calculator", self.exp.what)

    def test_has_tool_choice_fix(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("tool_choice", all_code)

    def test_evidence_summary_shows_zero_calls(self):
        self.assertIn("0", self.exp.evidence_summary)


# ── GOAL_ABANDONMENT specific ──────────────────────────────────────────────────

class TestGoalAbandonmentExplanation(unittest.TestCase):

    def setUp(self):
        self.signal = make_signal(
            FailureType.GOAL_ABANDONMENT,
            confidence=0.70,
            evidence={"stall_steps": 4, "last_tool_used": "database_lookup"},
        )
        self.exp = explain(self.signal)

    def test_title_contains_stall_steps(self):
        self.assertIn("4", self.exp.title)

    def test_title_contains_last_tool(self):
        self.assertIn("database_lookup", self.exp.title)

    def test_what_explains_stall(self):
        self.assertIn("4", self.exp.what)
        self.assertIn("database_lookup", self.exp.what)

    def test_fix_addresses_tool_results(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("tool_name", all_code)


# ── PROMPT_INJECTION specific ──────────────────────────────────────────────────

class TestPromptInjectionExplanation(unittest.TestCase):

    def setUp(self):
        self.signal = make_signal(
            FailureType.PROMPT_INJECTION_SIGNAL,
            severity=Severity.CRITICAL,
            confidence=0.85,
            evidence={
                "matched_patterns": ["ignore_instructions", "you_are_now"],
                "pattern_count": 2,
            },
        )
        self.exp = explain(self.signal)

    def test_severity_is_critical(self):
        self.assertEqual(self.exp.severity, "CRITICAL")

    def test_title_shows_pattern_count(self):
        self.assertIn("2", self.exp.title)

    def test_what_names_patterns(self):
        self.assertIn("ignore_instructions", self.exp.what)

    def test_explains_security_risk(self):
        self.assertIn("security", self.exp.why_it_matters.lower())

    def test_has_logging_fix(self):
        all_text = " ".join(f.description + " " + f.code
                            for f in self.exp.suggested_fixes)
        self.assertIn("log", all_text.lower())

    def test_single_pattern_grammar(self):
        """Title should say 'pattern' not 'patterns' for count=1."""
        signal = make_signal(
            FailureType.PROMPT_INJECTION_SIGNAL,
            evidence={"matched_patterns": ["ignore_instructions"], "pattern_count": 1},
        )
        exp = explain(signal)
        # Should not say "1 patterns"
        self.assertNotIn("1 patterns", exp.title)


# ── RAG_EMPTY_RETRIEVAL specific ───────────────────────────────────────────────

class TestRagEmptyRetrievalExplanation(unittest.TestCase):

    def setUp(self):
        self.signal = make_signal(
            FailureType.RAG_EMPTY_RETRIEVAL,
            confidence=0.88,
            evidence={
                "index_name": "product-docs",
                "result_count": 0,
                "top_score": None,
                "bad_retrievals": 1,
            },
        )
        self.exp = explain(self.signal)

    def test_title_contains_index_name(self):
        self.assertIn("product-docs", self.exp.title)

    def test_what_mentions_zero_results(self):
        self.assertIn("0", self.exp.what)

    def test_what_explains_fallback_to_memory(self):
        self.assertIn("training", self.exp.what.lower())

    def test_fix_includes_score_threshold(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("score", all_code.lower())

    def test_low_score_variant(self):
        """When result_count > 0 but score is low, explanation should reflect that."""
        signal = make_signal(
            FailureType.RAG_EMPTY_RETRIEVAL,
            evidence={
                "index_name": "docs",
                "result_count": 2,
                "top_score": 0.18,
                "bad_retrievals": 1,
            },
        )
        exp = explain(signal)
        self.assertIn("0.18", exp.evidence_summary)

    def test_multiple_bad_retrievals_reflected(self):
        signal = make_signal(
            FailureType.RAG_EMPTY_RETRIEVAL,
            evidence={
                "index_name": "docs",
                "result_count": 0,
                "top_score": None,
                "bad_retrievals": 3,
            },
        )
        exp = explain(signal)
        self.assertIn("3", exp.what)


# ── Fallback ───────────────────────────────────────────────────────────────────

class TestFallback(unittest.TestCase):

    def test_unknown_failure_type_returns_explanation(self):
        """explain() must never raise — even for future failure types."""
        signal = make_signal(FailureType.POLICY_VIOLATION, evidence={})
        exp = explain(signal)
        self.assertIsInstance(exp, Explanation)
        self.assertIsInstance(exp.title, str)
        self.assertGreater(len(exp.title), 0)
        self.assertGreater(len(exp.suggested_fixes), 0)

    def test_broken_evidence_does_not_crash(self):
        """Template should handle missing evidence keys gracefully."""
        signal = make_signal(FailureType.TOOL_LOOP, evidence={})  # missing all keys
        exp = explain(signal)
        self.assertIsInstance(exp, Explanation)

    def test_missing_pattern_fields_do_not_crash(self):
        signal = make_signal(FailureType.PROMPT_INJECTION_SIGNAL, evidence={})
        exp = explain(signal)
        self.assertIsInstance(exp, Explanation)

    def test_explain_never_raises(self):
        """explain() must swallow all template errors."""
        for failure_type in FailureType:
            with self.subTest(failure_type=failure_type):
                signal = make_signal(failure_type, evidence={})
                try:
                    exp = explain(signal)
                    self.assertIsInstance(exp, Explanation)
                except Exception as e:
                    self.fail(f"explain() raised for {failure_type}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
