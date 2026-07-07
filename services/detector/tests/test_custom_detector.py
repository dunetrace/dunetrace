"""
Tests for detector_svc.custom_detector: numeric-metric conditions (existing
behavior, regression-covered here for the first time since evaluate_custom_detector
was modified to also dispatch content conditions), content-condition operators,
ReDoS/timeout protection, and the per-detector evaluation budget stopgap.

Run:
    PYTHONPATH=packages/sdk-py:services/detector pytest services/detector/tests/test_custom_detector.py -v
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from dunetrace.models import AgentEvent, EventType, RunState, ToolCall

from detector_svc.custom_detector import evaluate_custom_detector


def make_state(**kwargs) -> RunState:
    defaults = dict(run_id="run-1", agent_id="agent-1", agent_version="v1")
    defaults.update(kwargs)
    return RunState(**defaults)


def _numeric_config(metric: str, operator: str, threshold: float) -> dict:
    return {
        "detector_name": "CUSTOM_TEST",
        "conditions": [{"metric": metric, "operator": operator, "threshold": threshold}],
        "severity": "HIGH",
    }


def _content_config(field: str, operator: str, value: object, case_sensitive: bool = True) -> dict:
    return {
        "detector_name": "CUSTOM_CONTENT_TEST",
        "conditions": [
            {
                "field": field,
                "operator": operator,
                "value": value,
                "case_sensitive": case_sensitive,
            }
        ],
        "severity": "HIGH",
    }


class TestNumericMetricConditionsStillWork(unittest.TestCase):
    """Regression coverage for the pre-existing metric path — no test file
    existed for this before content conditions were added."""

    def test_fires_when_condition_met(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="a", step_index=i, timestamp=0.0) for i in range(5)
        ]
        result = evaluate_custom_detector(_numeric_config("tool_call_count", ">=", 3), state)
        self.assertIsNotNone(result)
        self.assertEqual(result["failure_type"], "CUSTOM_TEST")

    def test_does_not_fire_when_condition_not_met(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="a", step_index=0, timestamp=0.0)]
        result = evaluate_custom_detector(_numeric_config("tool_call_count", ">=", 3), state)
        self.assertIsNone(result)

    def test_unknown_operator_returns_none_and_warns(self):
        state = make_state()
        config = _numeric_config("tool_call_count", "~=", 1)
        with self.assertLogs("dunetrace.detector.custom", level="WARNING"):
            result = evaluate_custom_detector(config, state)
        self.assertIsNone(result)

    def test_multiple_conditions_are_anded(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="a", step_index=i, timestamp=0.0) for i in range(5)
        ]
        config = {
            "detector_name": "CUSTOM_AND_TEST",
            "conditions": [
                {"metric": "tool_call_count", "operator": ">=", "threshold": 3},
                {"metric": "tool_call_count", "operator": "<=", "threshold": 1},  # fails
            ],
            "severity": "HIGH",
        }
        self.assertIsNone(evaluate_custom_detector(config, state))


class TestContentConditions(unittest.TestCase):
    def test_contains_fires_on_matching_tool_args(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(
                tool_name="x", args="{'query': 'delete all rows'}", step_index=0, timestamp=0.0
            )
        ]
        result = evaluate_custom_detector(
            _content_config("tool_args", "contains", "delete all"), state
        )
        self.assertIsNotNone(result)

    def test_contains_does_not_fire_when_absent(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="{'query': 'safe read'}", step_index=0, timestamp=0.0)
        ]
        result = evaluate_custom_detector(
            _content_config("tool_args", "contains", "delete all"), state
        )
        self.assertIsNone(result)

    def test_contains_is_case_sensitive_by_default(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="ERROR occurred", step_index=0, timestamp=0.0)
        ]
        self.assertIsNone(
            evaluate_custom_detector(_content_config("tool_args", "contains", "error"), state)
        )

    def test_contains_case_insensitive_variant(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="ERROR occurred", step_index=0, timestamp=0.0)
        ]
        result = evaluate_custom_detector(
            _content_config("tool_args", "contains", "error", case_sensitive=False), state
        )
        self.assertIsNotNone(result)

    def test_starts_with(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="ERROR: disk full", step_index=0, timestamp=0.0)
        ]
        self.assertIsNotNone(
            evaluate_custom_detector(_content_config("tool_args", "starts_with", "ERROR"), state)
        )
        self.assertIsNone(
            evaluate_custom_detector(_content_config("tool_args", "starts_with", "disk"), state)
        )

    def test_ends_with(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="disk full", step_index=0, timestamp=0.0)]
        self.assertIsNotNone(
            evaluate_custom_detector(_content_config("tool_args", "ends_with", "full"), state)
        )

    def test_equals(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="exact", step_index=0, timestamp=0.0)]
        self.assertIsNotNone(
            evaluate_custom_detector(_content_config("tool_args", "equals", "exact"), state)
        )
        self.assertIsNone(
            evaluate_custom_detector(_content_config("tool_args", "equals", "exac"), state)
        )

    def test_length_gt(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="x" * 100, step_index=0, timestamp=0.0)]
        self.assertIsNotNone(
            evaluate_custom_detector(_content_config("tool_args", "length_gt", 50), state)
        )
        self.assertIsNone(
            evaluate_custom_detector(_content_config("tool_args", "length_gt", 500), state)
        )

    def test_length_lt(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="short", step_index=0, timestamp=0.0)]
        self.assertIsNotNone(
            evaluate_custom_detector(_content_config("tool_args", "length_lt", 50), state)
        )

    def test_regex_matches(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="user@example.com", step_index=0, timestamp=0.0)
        ]
        result = evaluate_custom_detector(
            _content_config("tool_args", "regex_matches", r"[\w.]+@[\w.]+"), state
        )
        self.assertIsNotNone(result)

    def test_tool_error_field(self):
        state = make_state()
        tc = ToolCall(tool_name="x", args="a", step_index=0, timestamp=0.0)
        tc.success = False
        tc.error = "connection refused"
        state.tool_calls = [tc]
        self.assertIsNotNone(
            evaluate_custom_detector(_content_config("tool_error", "contains", "refused"), state)
        )

    def test_llm_output_field_reads_from_events_not_llm_calls(self):
        # LlmCall itself has no raw output text (only output_length) — the text
        # only exists in the raw llm.responded event payload.
        state = make_state()
        state.events = [
            AgentEvent(
                event_type=EventType.LLM_RESPONDED,
                run_id="run-1",
                agent_id="agent-1",
                agent_version="v1",
                step_index=1,
                payload={"output": "I cannot help with that request"},
            )
        ]
        result = evaluate_custom_detector(
            _content_config("llm_output", "contains", "cannot help"), state
        )
        self.assertIsNotNone(result)

    def test_input_text_field(self):
        state = make_state(input_text="please delete the production database")
        result = evaluate_custom_detector(
            _content_config("input_text", "contains", "delete the production"), state
        )
        self.assertIsNotNone(result)

    def test_no_occurrences_does_not_fire(self):
        state = make_state()  # no tool calls at all
        result = evaluate_custom_detector(
            _content_config("tool_args", "contains", "anything"), state
        )
        self.assertIsNone(result)

    def test_unknown_field_returns_false_and_warns(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="a", step_index=0, timestamp=0.0)]
        with self.assertLogs("dunetrace.detector.custom", level="WARNING"):
            result = evaluate_custom_detector(
                _content_config("nonexistent_field", "contains", "a"), state
            )
        self.assertIsNone(result)

    def test_unknown_content_operator_returns_false_and_warns(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="a", step_index=0, timestamp=0.0)]
        with self.assertLogs("dunetrace.detector.custom", level="WARNING"):
            result = evaluate_custom_detector(
                _content_config("tool_args", "matches_vibe", "a"), state
            )
        self.assertIsNone(result)

    def test_mixed_metric_and_content_conditions_are_anded(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="ERROR: timeout", step_index=i, timestamp=0.0)
            for i in range(3)
        ]
        config = {
            "detector_name": "CUSTOM_MIXED",
            "conditions": [
                {"metric": "tool_call_count", "operator": ">=", "threshold": 3},
                {
                    "field": "tool_args",
                    "operator": "contains",
                    "value": "ERROR",
                    "case_sensitive": True,
                },
            ],
            "severity": "HIGH",
        }
        self.assertIsNotNone(evaluate_custom_detector(config, state))


class TestRegexTimeoutProtection(unittest.TestCase):
    def test_regex_timeout_is_treated_as_non_matching_not_an_exception(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="a" * 40, step_index=0, timestamp=0.0)]
        config = _content_config("tool_args", "regex_matches", r"(a+)+$")
        # An absurdly small timeout forces the TimeoutError path deterministically,
        # rather than depending on real catastrophic-backtracking timing (flaky).
        with self.assertLogs("dunetrace.detector.custom", level="WARNING") as cm:
            result = evaluate_custom_detector(config, state, regex_timeout_ms=1e-6)
        self.assertIsNone(result)  # times out -> non-matching, never raises
        self.assertTrue(any("timed out" in msg for msg in cm.output))

    def test_invalid_regex_pattern_is_treated_as_non_matching(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="anything", step_index=0, timestamp=0.0)]
        config = _content_config("tool_args", "regex_matches", "[unclosed")
        with self.assertLogs("dunetrace.detector.custom", level="WARNING") as cm:
            result = evaluate_custom_detector(config, state)
        self.assertIsNone(result)
        self.assertTrue(any("invalid pattern" in msg for msg in cm.output))


class TestEvaluationBudget(unittest.TestCase):
    def test_slow_evaluation_aborts_and_returns_none(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="a", step_index=0, timestamp=0.0)]
        config = {
            "detector_name": "CUSTOM_SLOW",
            "conditions": [
                {"metric": "tool_call_count", "operator": ">=", "threshold": 0},
                {"metric": "tool_call_count", "operator": ">=", "threshold": 0},
            ],
            "severity": "HIGH",
        }
        # Absurdly tight budget guarantees the second condition's pre-check trips it.
        with self.assertLogs("dunetrace.detector.custom", level="WARNING") as cm:
            result = evaluate_custom_detector(config, state, evaluation_budget_ms=-1)
        self.assertIsNone(result)
        self.assertTrue(any("exceeded its evaluation budget" in msg for msg in cm.output))

    def test_budget_warning_is_rate_limited(self):
        state = make_state()
        state.tool_calls = [ToolCall(tool_name="x", args="a", step_index=0, timestamp=0.0)]
        config = {
            "detector_name": "CUSTOM_RATE_LIMIT_TEST",
            "conditions": [
                {"metric": "tool_call_count", "operator": ">=", "threshold": 0},
                {"metric": "tool_call_count", "operator": ">=", "threshold": 0},
            ],
            "severity": "HIGH",
        }
        with self.assertLogs("dunetrace.detector.custom", level="WARNING"):
            evaluate_custom_detector(config, state, evaluation_budget_ms=-1)

        with self.assertNoLogs("dunetrace.detector.custom", level="WARNING"):
            evaluate_custom_detector(config, state, evaluation_budget_ms=-1)

    def test_default_budget_does_not_abort_a_fast_detector(self):
        state = make_state()
        state.tool_calls = [
            ToolCall(tool_name="x", args="a", step_index=i, timestamp=0.0) for i in range(5)
        ]
        result = evaluate_custom_detector(_numeric_config("tool_call_count", ">=", 3), state)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
