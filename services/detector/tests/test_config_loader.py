"""
Tests for detectors.yml parsing, in particular the new `severity` override —
tunable for every detector regardless of _PARAM_MAP, since SEVERITY is a
cross-cutting BaseDetector attribute rather than a detector-specific tunable.

Run:
    cd services/detector
    PYTHONPATH=../../packages/sdk-py:. python -m pytest tests/test_config_loader.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/detector"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dunetrace.models import Severity
from detector_svc.config_loader import load_detector_kwargs, load_custom_detector_budget


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestExistingParamMapBehavior(unittest.TestCase):
    """Regression coverage — the pre-existing per-detector tunables must keep working."""

    def test_missing_file_returns_empty_dict(self):
        result = load_detector_kwargs("/nonexistent/path/detectors.yml")
        self.assertEqual(result, {})

    def test_threshold_and_window_parsed(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
    window: 8
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(result["default"]["tool_loop"], {"THRESHOLD": 5, "WINDOW": 8})
        finally:
            os.unlink(path)

    def test_unknown_param_key_ignored(self):
        path = _write_yaml("""
default:
  tool_loop:
    not_a_real_param: 999
""")
        try:
            result = load_detector_kwargs(path)
            self.assertNotIn("tool_loop", result.get("default", {}))
        finally:
            os.unlink(path)


class TestSeverityOverride(unittest.TestCase):
    def test_severity_parsed_for_a_detector_with_other_tunables(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
    severity: CRITICAL
""")
        try:
            result = load_detector_kwargs(path)
            kwargs = result["default"]["tool_loop"]
            self.assertEqual(kwargs["THRESHOLD"], 5)
            self.assertEqual(kwargs["SEVERITY"], Severity.CRITICAL)
        finally:
            os.unlink(path)

    def test_severity_parsed_for_a_detector_with_no_other_tunables(self):
        """empty_llm_response has no entry in _PARAM_MAP at all — severity must still work."""
        path = _write_yaml("""
default:
  empty_llm_response:
    severity: LOW
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(result["default"]["empty_llm_response"], {"SEVERITY": Severity.LOW})
        finally:
            os.unlink(path)

    def test_severity_is_case_insensitive(self):
        path = _write_yaml("""
default:
  tool_loop:
    severity: medium
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(result["default"]["tool_loop"]["SEVERITY"], Severity.MEDIUM)
        finally:
            os.unlink(path)

    def test_invalid_severity_is_ignored_not_raised(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
    severity: SUPER_BAD
""")
        try:
            result = load_detector_kwargs(path)
            kwargs = result["default"]["tool_loop"]
            self.assertEqual(kwargs["THRESHOLD"], 5)
            self.assertNotIn("SEVERITY", kwargs)
        finally:
            os.unlink(path)

    def test_no_severity_key_means_no_severity_override(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
""")
        try:
            result = load_detector_kwargs(path)
            self.assertNotIn("SEVERITY", result["default"]["tool_loop"])
        finally:
            os.unlink(path)

    def test_severity_per_category_override(self):
        path = _write_yaml("""
default:
  tool_loop:
    severity: HIGH
web-research:
  tool_loop:
    severity: LOW
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(result["default"]["tool_loop"]["SEVERITY"], Severity.HIGH)
            self.assertEqual(result["web-research"]["tool_loop"]["SEVERITY"], Severity.LOW)
        finally:
            os.unlink(path)


class TestMaxCostNsOverride(unittest.TestCase):
    """max_cost_ns is accepted for every detector regardless of _PARAM_MAP, same
    cross-cutting treatment as severity — see A2 in BACKLOG.md's Done section."""

    def test_max_cost_ns_parsed_for_a_detector_with_other_tunables(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
    max_cost_ns: 500000
""")
        try:
            result = load_detector_kwargs(path)
            kwargs = result["default"]["tool_loop"]
            self.assertEqual(kwargs["THRESHOLD"], 5)
            self.assertEqual(kwargs["MAX_COST_NS"], 500000)
        finally:
            os.unlink(path)

    def test_max_cost_ns_parsed_for_a_detector_with_no_other_tunables(self):
        path = _write_yaml("""
default:
  empty_llm_response:
    max_cost_ns: 250000
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(result["default"]["empty_llm_response"], {"MAX_COST_NS": 250000})
        finally:
            os.unlink(path)

    def test_non_numeric_max_cost_ns_is_ignored_not_raised(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
    max_cost_ns: "fast please"
""")
        try:
            result = load_detector_kwargs(path)
            kwargs = result["default"]["tool_loop"]
            self.assertEqual(kwargs["THRESHOLD"], 5)
            self.assertNotIn("MAX_COST_NS", kwargs)
        finally:
            os.unlink(path)

    def test_negative_max_cost_ns_is_ignored_not_raised(self):
        path = _write_yaml("""
default:
  tool_loop:
    max_cost_ns: -1
""")
        try:
            result = load_detector_kwargs(path)
            self.assertNotIn("MAX_COST_NS", result["default"].get("tool_loop", {}))
        finally:
            os.unlink(path)

    def test_no_max_cost_ns_key_means_no_override(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
""")
        try:
            result = load_detector_kwargs(path)
            self.assertNotIn("MAX_COST_NS", result["default"]["tool_loop"])
        finally:
            os.unlink(path)


class TestCustomDetectorBudget(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        result = load_custom_detector_budget("/nonexistent/path/detectors.yml")
        self.assertEqual(result, {"evaluation_budget_ms": 10.0, "regex_timeout_ms": 5.0})

    def test_missing_section_returns_defaults(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
""")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result, {"evaluation_budget_ms": 10.0, "regex_timeout_ms": 5.0})
        finally:
            os.unlink(path)

    def test_custom_values_parsed(self):
        path = _write_yaml("""
custom_detectors:
  evaluation_budget_ms: 25
  regex_timeout_ms: 8
""")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result, {"evaluation_budget_ms": 25.0, "regex_timeout_ms": 8.0})
        finally:
            os.unlink(path)

    def test_partial_override_keeps_other_default(self):
        path = _write_yaml("""
custom_detectors:
  evaluation_budget_ms: 25
""")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result["evaluation_budget_ms"], 25.0)
            self.assertEqual(result["regex_timeout_ms"], 5.0)
        finally:
            os.unlink(path)

    def test_non_numeric_value_falls_back_to_default(self):
        path = _write_yaml("""
custom_detectors:
  evaluation_budget_ms: "not a number"
""")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result["evaluation_budget_ms"], 10.0)
        finally:
            os.unlink(path)

    def test_negative_value_falls_back_to_default(self):
        path = _write_yaml("""
custom_detectors:
  evaluation_budget_ms: -5
""")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result["evaluation_budget_ms"], 10.0)
        finally:
            os.unlink(path)

    def test_regex_timeout_capped_to_evaluation_budget_if_larger(self):
        path = _write_yaml("""
custom_detectors:
  evaluation_budget_ms: 5
  regex_timeout_ms: 20
""")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result["evaluation_budget_ms"], 5.0)
            self.assertEqual(result["regex_timeout_ms"], 5.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
