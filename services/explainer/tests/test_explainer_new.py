"""
Tests for:
  - Updated explainer templates (new evidence fields, fixed keys)
  - rate_context parameter in explain()
  - Templates not covered before: LlmTruncationLoop, ContextBloat, RetrySotorm, ReasoningStall,
    EmptyLlmResponse, StepCountInflation, CascadingToolFailure, FirstStepFailure, SlowStep

No DB, no LLM, no network.
Run:
    cd services/explainer
    python -m pytest tests/test_explainer_new.py -v
"""

from __future__ import annotations

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../packages/sdk-py")
    ),
)

from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.explainer import explain
from explainer_svc.models import Explanation


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
        run_id="run-test",
        agent_id="agent-test",
        agent_version="abc12345",
        step_index=step_index,
        confidence=confidence,
        evidence=evidence or {},
        detected_at=time.time(),
    )


# ── TOOL_THRASHING with new evidence key ──────────────────────────────────────


class TestToolThrashingCountKey(unittest.TestCase):
    """Template reads ev.get('count', ev.get('oscillation_count', '?'))
    — test both code paths."""

    def test_count_key_used_in_what(self):
        """New detectors emit 'count' — template should use it."""
        signal = make_signal(
            FailureType.TOOL_THRASHING,
            evidence={"tool_a": "search", "tool_b": "browse", "count": 6},
        )
        exp = explain(signal)
        self.assertIn("6", exp.what)

    def test_oscillation_count_key_backward_compatible(self):
        """Old stored signals used 'oscillation_count' — still works."""
        signal = make_signal(
            FailureType.TOOL_THRASHING,
            evidence={"tool_a": "alpha", "tool_b": "beta", "oscillation_count": 4},
        )
        exp = explain(signal)
        self.assertIn("4", exp.what)

    def test_fallback_when_neither_key_present(self):
        """Missing both keys → '?' inserted, no crash."""
        signal = make_signal(
            FailureType.TOOL_THRASHING,
            evidence={"tool_a": "a", "tool_b": "b"},
        )
        exp = explain(signal)
        self.assertIn("?", exp.what)

    def test_count_takes_precedence_over_oscillation_count(self):
        """If both keys are present, 'count' wins."""
        signal = make_signal(
            FailureType.TOOL_THRASHING,
            evidence={
                "tool_a": "s",
                "tool_b": "b",
                "count": 8,
                "oscillation_count": 3,  # should be ignored
            },
        )
        exp = explain(signal)
        self.assertIn("8", exp.what)
        # The old value (3) should not appear in what if 8 is used
        # (they could both appear if the strings match other numeric mentions)
        # Just verify 8 is there
        self.assertIn("8", exp.what)


# ── GOAL_ABANDONMENT — new evidence fields ────────────────────────────────────


class TestGoalAbandonmentNewEvidence(unittest.TestCase):
    def _make_exp(self, evidence: dict) -> Explanation:
        signal = make_signal(
            FailureType.GOAL_ABANDONMENT,
            evidence=evidence,
        )
        return explain(signal)

    def test_stall_event_sequence_in_evidence_summary(self):
        exp = self._make_exp(
            {
                "stall_steps": 4,
                "last_tool_used": "db_query",
                "steps_since_last_tool": 5,
                "stall_event_sequence": [
                    "llm.called",
                    "llm.called",
                    "llm.called",
                    "llm.called",
                ],
                "last_tool_step": 3,
            }
        )
        # The sequence should appear in the evidence_summary
        self.assertIn("llm", exp.evidence_summary)

    def test_steps_since_last_tool_in_evidence_summary(self):
        exp = self._make_exp(
            {
                "stall_steps": 4,
                "last_tool_used": "search",
                "steps_since_last_tool": 7,
                "stall_event_sequence": [],
                "last_tool_step": 2,
            }
        )
        self.assertIn("7", exp.evidence_summary)

    def test_no_event_sequence_section_when_empty(self):
        """Empty stall_event_sequence → evidence_summary ends with a period, not an arrow."""
        exp = self._make_exp(
            {
                "stall_steps": 4,
                "last_tool_used": "search",
                "steps_since_last_tool": 4,
                "stall_event_sequence": [],
                "last_tool_step": 2,
            }
        )
        self.assertNotIn("→", exp.evidence_summary)

    def test_missing_stall_event_sequence_no_crash(self):
        """Older signals without stall_event_sequence key still work."""
        exp = self._make_exp({"stall_steps": 4, "last_tool_used": "tool"})
        self.assertIsInstance(exp, Explanation)

    def test_last_tool_in_title(self):
        exp = self._make_exp(
            {
                "stall_steps": 3,
                "last_tool_used": "vector_search",
                "steps_since_last_tool": 3,
                "stall_event_sequence": [],
            }
        )
        self.assertIn("vector_search", exp.title)

    def test_confidence_in_evidence_summary(self):
        signal = make_signal(
            FailureType.GOAL_ABANDONMENT,
            confidence=0.70,
            evidence={"stall_steps": 4, "last_tool_used": "x"},
        )
        exp = explain(signal)
        self.assertIn("70%", exp.evidence_summary)


# ── CONTEXT_BLOAT — token_growth_sequence → growth curve ─────────────────────


class TestContextBloatGrowthCurve(unittest.TestCase):
    def _make_exp(self, evidence: dict) -> Explanation:
        return explain(make_signal(FailureType.CONTEXT_BLOAT, evidence=evidence))

    def test_growth_curve_built_from_sequence(self):
        exp = self._make_exp(
            {
                "first_tokens": 500,
                "last_tokens": 6000,
                "growth_factor": 12.0,
                "llm_call_count": 3,
                "first_call_step": 1,
                "last_call_step": 3,
                "token_growth_sequence": [
                    {"step": 1, "tokens": 500},
                    {"step": 2, "tokens": 2000},
                    {"step": 3, "tokens": 6000},
                ],
            }
        )
        # Evidence summary should show the step-by-step curve
        self.assertIn("500", exp.evidence_summary)
        self.assertIn("2000", exp.evidence_summary)
        self.assertIn("6000", exp.evidence_summary)
        self.assertIn("→", exp.evidence_summary)

    def test_fallback_to_first_last_when_no_sequence(self):
        exp = self._make_exp(
            {
                "first_tokens": 300,
                "last_tokens": 4500,
                "growth_factor": 15.0,
                "llm_call_count": 3,
                "first_call_step": 1,
                "last_call_step": 3,
                # No token_growth_sequence key
            }
        )
        # Falls back to "300→4500" style
        self.assertIn("300", exp.evidence_summary)
        self.assertIn("4500", exp.evidence_summary)
        self.assertIn("→", exp.evidence_summary)

    def test_growth_factor_in_title(self):
        exp = self._make_exp(
            {
                "first_tokens": 200,
                "last_tokens": 800,
                "growth_factor": 4.0,
                "llm_call_count": 3,
                "first_call_step": 1,
                "last_call_step": 3,
            }
        )
        self.assertIn("4.0", exp.title)

    def test_what_mentions_step_range(self):
        exp = self._make_exp(
            {
                "first_tokens": 500,
                "last_tokens": 5000,
                "growth_factor": 10.0,
                "llm_call_count": 4,
                "first_call_step": 1,
                "last_call_step": 4,
            }
        )
        self.assertIn("step 1", exp.what)
        self.assertIn("step 4", exp.what)


# ── LLM_TRUNCATION_LOOP — enriched evidence in summary ───────────────────────


class TestLlmTruncationLoopEnriched(unittest.TestCase):
    def _make_exp(self, evidence: dict) -> Explanation:
        return explain(make_signal(FailureType.LLM_TRUNCATION_LOOP, evidence=evidence))

    def test_token_counts_in_evidence_summary(self):
        exp = self._make_exp(
            {
                "truncation_count": 2,
                "total_llm_calls": 4,
                "first_truncation_step": 2,
                "last_truncation_step": 4,
                "token_counts_at_truncation": [8000, 12000],
                "models": ["gpt-4o"],
            }
        )
        self.assertIn("8000", exp.evidence_summary)
        self.assertIn("12000", exp.evidence_summary)

    def test_model_name_in_evidence_summary(self):
        exp = self._make_exp(
            {
                "truncation_count": 2,
                "total_llm_calls": 3,
                "first_truncation_step": 1,
                "last_truncation_step": 3,
                "token_counts_at_truncation": [],
                "models": ["claude-3-5-sonnet"],
            }
        )
        self.assertIn("claude-3-5-sonnet", exp.evidence_summary)

    def test_multiple_models_in_evidence_summary(self):
        exp = self._make_exp(
            {
                "truncation_count": 2,
                "total_llm_calls": 3,
                "first_truncation_step": 1,
                "last_truncation_step": 3,
                "token_counts_at_truncation": [],
                "models": ["gpt-4o", "gpt-4o-mini"],
            }
        )
        self.assertIn("gpt-4o", exp.evidence_summary)
        self.assertIn("gpt-4o-mini", exp.evidence_summary)

    def test_no_token_note_when_list_empty(self):
        exp = self._make_exp(
            {
                "truncation_count": 2,
                "total_llm_calls": 3,
                "first_truncation_step": 1,
                "last_truncation_step": 3,
                "token_counts_at_truncation": [],
                "models": [],
            }
        )
        self.assertNotIn("Prompt tokens", exp.evidence_summary)

    def test_no_model_note_when_empty(self):
        exp = self._make_exp(
            {
                "truncation_count": 2,
                "total_llm_calls": 3,
                "first_truncation_step": 1,
                "last_truncation_step": 3,
                "token_counts_at_truncation": [],
                "models": [],
            }
        )
        self.assertNotIn("Model", exp.evidence_summary)

    def test_singular_model_label(self):
        """One model → 'Model:', not 'Models:'."""
        exp = self._make_exp(
            {
                "truncation_count": 2,
                "total_llm_calls": 3,
                "first_truncation_step": 1,
                "last_truncation_step": 3,
                "token_counts_at_truncation": [],
                "models": ["gpt-4o"],
            }
        )
        self.assertIn("Model:", exp.evidence_summary)
        self.assertNotIn("Models:", exp.evidence_summary)

    def test_plural_model_label(self):
        """Two models → 'Models:'."""
        exp = self._make_exp(
            {
                "truncation_count": 2,
                "total_llm_calls": 3,
                "first_truncation_step": 1,
                "last_truncation_step": 3,
                "token_counts_at_truncation": [],
                "models": ["gpt-4o", "gpt-4o-mini"],
            }
        )
        self.assertIn("Models:", exp.evidence_summary)


# ── explain() rate_context parameter ─────────────────────────────────────────


class TestExplainRateContext(unittest.TestCase):
    def test_rate_context_attached_to_explanation(self):
        signal = make_signal(
            FailureType.TOOL_LOOP,
            evidence={"tool": "web_search", "count": 5, "window": 5},
        )
        rc = {"total_runs": 10, "affected_runs": 3, "rate": 0.3, "is_systemic": False}
        exp = explain(signal, rate_context=rc)
        self.assertEqual(exp.rate_context, rc)

    def test_rate_context_defaults_to_empty_dict(self):
        signal = make_signal(
            FailureType.TOOL_LOOP,
            evidence={"tool": "web_search", "count": 5, "window": 5},
        )
        exp = explain(signal)
        self.assertEqual(exp.rate_context, {})

    def test_rate_context_none_becomes_empty_dict(self):
        signal = make_signal(
            FailureType.TOOL_LOOP,
            evidence={"tool": "web_search", "count": 5, "window": 5},
        )
        exp = explain(signal, rate_context=None)
        self.assertEqual(exp.rate_context, {})

    def test_rate_context_preserved_for_fallback(self):
        """rate_context must be attached even when the fallback path is used."""
        signal = make_signal(FailureType.POLICY_VIOLATION, evidence={})
        rc = {"total_runs": 5, "affected_runs": 5, "rate": 1.0, "is_systemic": True}
        exp = explain(signal, rate_context=rc)
        self.assertEqual(exp.rate_context, rc)

    def test_rate_context_preserves_all_keys(self):
        signal = make_signal(
            FailureType.TOOL_LOOP, evidence={"tool": "x", "count": 5, "window": 5}
        )
        rc = {
            "total_runs": 20,
            "affected_runs": 4,
            "rate": 0.2,
            "is_systemic": False,
            "extra_key": "extra_value",
        }
        exp = explain(signal, rate_context=rc)
        self.assertEqual(exp.rate_context["extra_key"], "extra_value")


# ── New template coverage: RETRY_STORM ───────────────────────────────────────


class TestRetryStormExplanation(unittest.TestCase):
    def setUp(self):
        self.signal = make_signal(
            FailureType.RETRY_STORM,
            severity=Severity.HIGH,
            confidence=0.92,
            evidence={
                "tool": "payment_api",
                "consecutive_fails": 4,
                "threshold": 3,
                "first_fail_step": 2,
                "step_indices": [2, 3, 4, 5],
                "args_identical": False,
                "error_hashes": ["e1", "e1", "e1", "e1"],
                "failure_reason_hash": "e1",
                "reason_identical": True,
            },
        )
        self.exp = explain(self.signal)

    def test_title_contains_tool_name(self):
        self.assertIn("payment_api", self.exp.title)

    def test_title_contains_fail_count(self):
        self.assertIn("4", self.exp.title)

    def test_what_mentions_consecutive_fails(self):
        self.assertIn("4", self.exp.what)
        self.assertIn("payment_api", self.exp.what)

    def test_evidence_summary_contains_step_range(self):
        self.assertIn("2", self.exp.evidence_summary)  # first_fail_step

    def test_has_retry_fix(self):
        """Should suggest exponential backoff or similar."""
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("retry", all_code.lower())

    def test_severity_passthrough(self):
        self.assertEqual(self.exp.severity, "HIGH")


# ── New template coverage: REASONING_STALL ───────────────────────────────────


class TestReasoningStallExplanation(unittest.TestCase):
    def setUp(self):
        self.signal = make_signal(
            FailureType.REASONING_STALL,
            severity=Severity.MEDIUM,
            confidence=0.70,
            evidence={
                "llm_calls": 12,
                "tool_calls": 2,
                "ratio": 6.0,
                "threshold": 4.0,
                "event_sequence": ["llm", "llm", "tool", "llm", "llm", "llm"],
            },
        )
        self.exp = explain(self.signal)

    def test_title_contains_counts(self):
        self.assertIn("12", self.exp.title)
        self.assertIn("2", self.exp.title)

    def test_title_contains_ratio(self):
        self.assertIn("6.0", self.exp.title)

    def test_what_mentions_threshold(self):
        self.assertIn("4.0", self.exp.what)

    def test_evidence_summary_contains_ratio(self):
        self.assertIn("6.0", self.exp.evidence_summary)

    def test_has_reasoning_cap_fix(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("consecutive_llm", all_code)


# ── New template coverage: EMPTY_LLM_RESPONSE ────────────────────────────────


class TestEmptyLlmResponseExplanation(unittest.TestCase):
    def setUp(self):
        self.signal = make_signal(
            FailureType.EMPTY_LLM_RESPONSE,
            severity=Severity.HIGH,
            confidence=0.95,
            evidence={"occurrences": 2, "first_step": 3, "finish_reason": "stop"},
        )
        self.exp = explain(self.signal)

    def test_title_contains_step(self):
        self.assertIn("3", self.exp.title)

    def test_what_mentions_occurrences(self):
        self.assertIn("2", self.exp.what)

    def test_evidence_summary_mentions_step(self):
        self.assertIn("3", self.exp.evidence_summary)

    def test_has_guard_fix(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("empty", all_code.lower())

    def test_single_occurrence_grammar(self):
        signal = make_signal(
            FailureType.EMPTY_LLM_RESPONSE,
            evidence={"occurrences": 1, "first_step": 1, "finish_reason": "stop"},
        )
        exp = explain(signal)
        self.assertIn("once", exp.evidence_summary)


# ── New template coverage: STEP_COUNT_INFLATION ───────────────────────────────


class TestStepCountInflationExplanation(unittest.TestCase):
    def setUp(self):
        self.signal = make_signal(
            FailureType.STEP_COUNT_INFLATION,
            severity=Severity.MEDIUM,
            confidence=0.82,
            evidence={
                "current_steps": 30,
                "baseline_p75": 10.0,
                "inflation_ratio": 3.0,
                "threshold_factor": 2.0,
            },
        )
        self.exp = explain(self.signal)

    def test_title_contains_current_steps(self):
        self.assertIn("30", self.exp.title)

    def test_title_contains_baseline(self):
        self.assertIn("10.0", self.exp.title)

    def test_what_mentions_ratio(self):
        self.assertIn("3.0", self.exp.what)

    def test_evidence_summary_mentions_factor(self):
        self.assertIn("2.0", self.exp.evidence_summary)

    def test_has_step_limit_fix(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("MAX_STEPS", all_code)


# ── New template coverage: CASCADING_TOOL_FAILURE ────────────────────────────


class TestCascadingToolFailureExplanation(unittest.TestCase):
    def setUp(self):
        self.signal = make_signal(
            FailureType.CASCADING_TOOL_FAILURE,
            severity=Severity.HIGH,
            confidence=0.88,
            evidence={
                "consecutive_failures": 4,
                "distinct_tools": ["database", "search_api", "cache"],
                "threshold": 3,
                "first_fail_step": 2,
            },
        )
        self.exp = explain(self.signal)

    def test_title_contains_fail_count(self):
        self.assertIn("4", self.exp.title)

    def test_title_contains_tool_names(self):
        # At least one tool name should appear in the title
        title_lower = self.exp.title.lower()
        self.assertTrue(
            "database" in title_lower
            or "search_api" in title_lower
            or "cache" in title_lower
        )

    def test_what_mentions_distinct_tools(self):
        self.assertIn("database", self.exp.what)

    def test_evidence_summary_mentions_first_fail_step(self):
        self.assertIn("2", self.exp.evidence_summary)

    def test_has_health_check_fix(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("dependencies", all_code.lower())


# ── New template coverage: FIRST_STEP_FAILURE ────────────────────────────────


class TestFirstStepFailureExplanation(unittest.TestCase):
    def _make_exp(self, trigger: str, tool: str | None = None) -> Explanation:
        evidence = {"trigger": trigger, "failed_step": 1, "max_step": 2}
        if tool:
            evidence["tool"] = tool
        return explain(
            make_signal(
                FailureType.FIRST_STEP_FAILURE,
                severity=Severity.MEDIUM,
                evidence=evidence,
            )
        )

    def test_run_error_trigger(self):
        exp = self._make_exp("run_errored")
        self.assertIn("exception", exp.what.lower())

    def test_empty_llm_trigger(self):
        exp = self._make_exp("empty_llm_response")
        self.assertIn("empty", exp.what.lower())

    def test_tool_failure_trigger_with_tool_name(self):
        exp = self._make_exp("tool_failure", tool="init_db")
        self.assertIn("init_db", exp.what)

    def test_step_in_evidence_summary(self):
        exp = self._make_exp("run_errored")
        self.assertIn("1", exp.evidence_summary)

    def test_tool_in_evidence_summary_when_present(self):
        exp = self._make_exp("tool_failure", tool="connect_db")
        self.assertIn("connect_db", exp.evidence_summary)

    def test_no_tool_in_evidence_summary_when_absent(self):
        exp = self._make_exp("run_errored")
        # Should not blow up on missing 'tool' key
        self.assertIsInstance(exp, Explanation)


# ── New template coverage: SLOW_STEP ─────────────────────────────────────────


class TestSlowStepExplanation(unittest.TestCase):
    def setUp(self):
        self.signal = make_signal(
            FailureType.SLOW_STEP,
            severity=Severity.HIGH,
            confidence=0.92,
            evidence={
                "step_index": 3,
                "duration_ms": 75_000,
                "threshold_ms": 15_000,
                "event_type": "tool.called",
                "step_label": "tool execution",
                "ratio": 5.0,
                "all_slow_steps": {3: 75_000},
            },
        )
        self.exp = explain(self.signal)

    def test_title_contains_duration(self):
        self.assertIn("75.0", self.exp.title)

    def test_title_contains_threshold(self):
        self.assertIn("15.0", self.exp.title)

    def test_what_mentions_step_index(self):
        self.assertIn("3", self.exp.what)

    def test_evidence_summary_shows_ms(self):
        self.assertIn("75000", self.exp.evidence_summary)

    def test_has_timeout_fix(self):
        all_code = " ".join(f.code for f in self.exp.suggested_fixes)
        self.assertIn("timeout", all_code.lower())

    def test_coincident_signal_in_what_when_present(self):
        signal = make_signal(
            FailureType.SLOW_STEP,
            evidence={
                "step_index": 2,
                "duration_ms": 30_000,
                "threshold_ms": 15_000,
                "event_type": "tool.called",
                "step_label": "tool execution",
                "ratio": 2.0,
                "all_slow_steps": {2: 30_000},
                "coincident_signals": [{"signal_name": "db_high_latency"}],
            },
        )
        exp = explain(signal)
        self.assertIn("db_high_latency", exp.what)

    def test_no_coincident_note_when_absent(self):
        self.assertNotIn("coincident", self.exp.what.lower())


# ── Edge cases across all templates ──────────────────────────────────────────


class TestTemplateEdgeCases(unittest.TestCase):
    """Templates must handle missing/None evidence values without crashing."""

    PARTIALLY_EMPTY = [
        (FailureType.RETRY_STORM, {"tool": "api"}),
        (FailureType.CONTEXT_BLOAT, {"growth_factor": 5.0}),
        (FailureType.LLM_TRUNCATION_LOOP, {"truncation_count": 3}),
        (FailureType.EMPTY_LLM_RESPONSE, {}),
        (FailureType.STEP_COUNT_INFLATION, {"current_steps": 20}),
        (FailureType.CASCADING_TOOL_FAILURE, {"consecutive_failures": 4}),
        (FailureType.FIRST_STEP_FAILURE, {}),
        (FailureType.SLOW_STEP, {"duration_ms": 60_000, "threshold_ms": 15_000}),
        (FailureType.REASONING_STALL, {"llm_calls": 10, "tool_calls": 1}),
    ]

    def test_no_crash_with_partial_evidence(self):
        for failure_type, evidence in self.PARTIALLY_EMPTY:
            with self.subTest(failure_type=failure_type):
                signal = make_signal(failure_type, evidence=evidence)
                try:
                    exp = explain(signal)
                    self.assertIsInstance(exp, Explanation)
                    self.assertGreater(len(exp.title), 0)
                except Exception as e:
                    self.fail(
                        f"explain() raised for {failure_type} with partial evidence: {e}"
                    )

    def test_all_templates_return_json_serialisable_explanation(self):
        import json

        for failure_type, evidence in self.PARTIALLY_EMPTY:
            with self.subTest(failure_type=failure_type):
                signal = make_signal(failure_type, evidence=evidence)
                exp = explain(signal)
                try:
                    json.dumps(exp.as_dict())
                except TypeError as e:
                    self.fail(
                        f"as_dict() not JSON-serialisable for {failure_type}: {e}"
                    )

    def test_tool_loop_with_all_none_success_calls(self):
        """ToolLoop where no call has a success field set."""
        signal = make_signal(
            FailureType.TOOL_LOOP,
            evidence={
                "tool": "x",
                "count": 5,
                "window": 5,
                "first_step": 0,
                "last_step": 4,
                "step_indices": [0, 1, 2, 3, 4],
                "args_hashes": ["a"] * 5,
                "args_identical": True,
                "args_similar": True,
                "success_rate": None,  # edge case
            },
        )
        exp = explain(signal)
        self.assertIsInstance(exp, Explanation)

    def test_goal_abandonment_no_stall_event_sequence_key(self):
        """Old signals without stall_event_sequence don't crash."""
        signal = make_signal(
            FailureType.GOAL_ABANDONMENT,
            evidence={
                "stall_steps": 4,
                "last_tool_used": "db",
                # no stall_event_sequence, no steps_since_last_tool
            },
        )
        exp = explain(signal)
        self.assertIsInstance(exp, Explanation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
