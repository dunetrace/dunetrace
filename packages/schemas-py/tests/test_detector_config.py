"""
Tests for the shared detectors.yml parser (dunetrace_schemas.detector_config),
moved here from services/detector so the ingest service can serve the same
merged kwargs at GET /v1/detector-config. The detector's config_loader is a
thin adapter over this module and keeps its own (unchanged) test file.

Run:
    PYTHONPATH=packages/schemas-py python -m pytest packages/schemas-py/tests/test_detector_config.py -v
"""

from __future__ import annotations

import builtins
import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

from dunetrace_schemas.enums import Severity
from dunetrace_schemas.detector_config import (
    BUILTIN_DETECTOR_KEYS,
    DEFAULT_CONFIG_PATH,
    _PARAM_MAP,
    effective_detector_kwargs,
    json_safe_kwargs,
    load_custom_detector_budget,
    load_detector_kwargs,
    resolve_config_path,
)

_LOGGER = "dunetrace.detector_config"


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestParamMapBehavior(unittest.TestCase):
    """Regression coverage — the per-detector tunables must keep working."""

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

    def test_scattershot_thresholds_parsed(self):
        path = _write_yaml("""
default:
  scattershot_tool_use:
    min_distinct_tools: 7
    min_total_calls: 10
    min_repeat_ratio: 2.0
    scan_limit: 100
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(
                result["default"]["scattershot_tool_use"],
                {
                    "MIN_DISTINCT_TOOLS": 7,
                    "MIN_TOTAL_CALLS": 10,
                    "MIN_REPEAT_RATIO": 2.0,
                    "SCAN_LIMIT": 100,
                },
            )
        finally:
            os.unlink(path)

    def test_unknown_param_key_ignored_and_warned(self):
        path = _write_yaml("""
default:
  tool_loop:
    not_a_real_param: 999
""")
        try:
            with self.assertLogs(_LOGGER, level="WARNING") as logs:
                result = load_detector_kwargs(path)
            self.assertNotIn("tool_loop", result.get("default", {}))
            self.assertTrue(any("not_a_real_param" in line for line in logs.output))
        finally:
            os.unlink(path)

    def test_reserved_alert_keys_are_not_flagged(self):
        """alert_policy / destinations belong to alerts_svc — same file, other consumer."""
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 3
    alert_policy: {mode: consecutive, n: 2}
    destinations: [slack]
""")
        try:
            with mock.patch.object(logging.getLogger(_LOGGER), "warning") as warn:
                result = load_detector_kwargs(path)
            warn.assert_not_called()
            self.assertEqual(result["default"]["tool_loop"], {"THRESHOLD": 3})
        finally:
            os.unlink(path)

    def test_custom_detectors_section_is_not_a_category(self):
        path = _write_yaml("""
custom_detectors:
  evaluation_budget_ms: 25
default:
  tool_loop:
    threshold: 3
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(list(result), ["default"])
        finally:
            os.unlink(path)

    def test_unparseable_file_returns_empty_dict_with_warning(self):
        path = _write_yaml("default: [unclosed")
        try:
            with self.assertLogs(_LOGGER, level="WARNING"):
                self.assertEqual(load_detector_kwargs(path), {})
        finally:
            os.unlink(path)

    def test_non_mapping_document_returns_empty_dict(self):
        path = _write_yaml("- just\n- a list\n")
        try:
            with self.assertLogs(_LOGGER, level="WARNING"):
                self.assertEqual(load_detector_kwargs(path), {})
        finally:
            os.unlink(path)


class TestTypoWarning(unittest.TestCase):
    """A misspelled section name is a WARNING, not a silent no-op — the
    lookup is `.get(key, {})`, so without this the operator's tuning never
    applies and nothing says so."""

    def test_unknown_section_is_warned_and_dropped_when_known_set_given(self):
        path = _write_yaml("""
default:
  tool_lop:
    threshold: 5
  tool_loop:
    threshold: 7
""")
        try:
            with self.assertLogs(_LOGGER, level="WARNING") as logs:
                result = load_detector_kwargs(path, known_detectors={"tool_loop"})
            self.assertEqual(result["default"], {"tool_loop": {"THRESHOLD": 7}})
            self.assertTrue(any("tool_lop is not a known detector" in line for line in logs.output))
        finally:
            os.unlink(path)

    def test_unknown_section_is_silently_kept_without_known_set(self):
        """Callers without the class list opt out of the check entirely."""
        path = _write_yaml("""
default:
  tool_lop:
    severity: HIGH
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(result["default"]["tool_lop"], {"SEVERITY": Severity.HIGH})
        finally:
            os.unlink(path)

    def test_builtin_keys_cover_every_param_map_entry(self):
        self.assertTrue(set(_PARAM_MAP) <= BUILTIN_DETECTOR_KEYS)
        self.assertEqual(len(BUILTIN_DETECTOR_KEYS), 34)


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
            with self.assertLogs(_LOGGER, level="WARNING"):
                result = load_detector_kwargs(path)
            kwargs = result["default"]["tool_loop"]
            self.assertEqual(kwargs["THRESHOLD"], 5)
            self.assertNotIn("SEVERITY", kwargs)
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
    def test_max_cost_ns_parsed(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
    max_cost_ns: 500000
  empty_llm_response:
    max_cost_ns: 250000
""")
        try:
            result = load_detector_kwargs(path)
            self.assertEqual(
                result["default"]["tool_loop"], {"THRESHOLD": 5, "MAX_COST_NS": 500000}
            )
            self.assertEqual(result["default"]["empty_llm_response"], {"MAX_COST_NS": 250000})
        finally:
            os.unlink(path)

    def test_invalid_max_cost_ns_is_ignored_not_raised(self):
        path = _write_yaml("""
default:
  tool_loop:
    threshold: 5
    max_cost_ns: "fast please"
  tool_thrashing:
    max_cost_ns: -1
""")
        try:
            with self.assertLogs(_LOGGER, level="WARNING"):
                result = load_detector_kwargs(path)
            self.assertEqual(result["default"]["tool_loop"], {"THRESHOLD": 5})
            self.assertNotIn("tool_thrashing", result["default"])
        finally:
            os.unlink(path)


class TestCustomDetectorBudget(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        result = load_custom_detector_budget("/nonexistent/path/detectors.yml")
        self.assertEqual(result, {"evaluation_budget_ms": 10.0, "regex_timeout_ms": 5.0})

    def test_missing_section_returns_defaults(self):
        path = _write_yaml("default:\n  tool_loop:\n    threshold: 5\n")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result, {"evaluation_budget_ms": 10.0, "regex_timeout_ms": 5.0})
        finally:
            os.unlink(path)

    def test_custom_values_parsed_and_partial_override_keeps_other_default(self):
        path = _write_yaml("custom_detectors:\n  evaluation_budget_ms: 25\n")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result, {"evaluation_budget_ms": 25.0, "regex_timeout_ms": 5.0})
        finally:
            os.unlink(path)

    def test_invalid_values_fall_back_to_default(self):
        path = _write_yaml(
            'custom_detectors:\n  evaluation_budget_ms: "not a number"\n  regex_timeout_ms: -5\n'
        )
        try:
            with self.assertLogs(_LOGGER, level="WARNING"):
                result = load_custom_detector_budget(path)
            self.assertEqual(result, {"evaluation_budget_ms": 10.0, "regex_timeout_ms": 5.0})
        finally:
            os.unlink(path)

    def test_regex_timeout_capped_to_evaluation_budget_if_larger(self):
        path = _write_yaml("custom_detectors:\n  evaluation_budget_ms: 5\n  regex_timeout_ms: 20\n")
        try:
            result = load_custom_detector_budget(path)
            self.assertEqual(result, {"evaluation_budget_ms": 5.0, "regex_timeout_ms": 5.0})
        finally:
            os.unlink(path)


class TestWithoutPyYAML(unittest.TestCase):
    """Ingest must import and serve without PyYAML installed: empty config,
    one WARNING, no exception."""

    def _without_yaml(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "yaml" or name.startswith("yaml."):
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        return mock.patch.dict(sys.modules, {"yaml": None}), mock.patch.object(
            builtins, "__import__", fake_import
        )

    def test_load_detector_kwargs_is_empty_with_warning(self):
        path = _write_yaml("default:\n  tool_loop:\n    threshold: 5\n")
        try:
            p1, p2 = self._without_yaml()
            with p1, p2, self.assertLogs(_LOGGER, level="WARNING") as logs:
                self.assertEqual(load_detector_kwargs(path), {})
            self.assertEqual(len(logs.output), 1)
            self.assertIn("PyYAML not installed", logs.output[0])
        finally:
            os.unlink(path)

    def test_budget_falls_back_to_defaults(self):
        path = _write_yaml("custom_detectors:\n  evaluation_budget_ms: 25\n")
        try:
            p1, p2 = self._without_yaml()
            with p1, p2, self.assertLogs(_LOGGER, level="WARNING"):
                result = load_custom_detector_budget(path)
            self.assertEqual(result, {"evaluation_budget_ms": 10.0, "regex_timeout_ms": 5.0})
        finally:
            os.unlink(path)


class TestConfigPath(unittest.TestCase):
    def test_explicit_path_wins(self):
        with mock.patch.dict(os.environ, {"DETECTOR_CONFIG": "/from/env.yml"}):
            self.assertEqual(resolve_config_path("/explicit.yml"), "/explicit.yml")

    def test_env_then_default(self):
        with mock.patch.dict(os.environ, {"DETECTOR_CONFIG": "/from/env.yml"}):
            self.assertEqual(resolve_config_path(), "/from/env.yml")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DETECTOR_CONFIG", None)
            self.assertEqual(resolve_config_path(), DEFAULT_CONFIG_PATH)
        self.assertEqual(DEFAULT_CONFIG_PATH, "/app/detectors.yml")


class TestEffectiveKwargs(unittest.TestCase):
    """The merge GET /v1/detector-config serves must be the one
    detector_svc.detectors.get_detectors performs: category over default,
    key by key, default alone for an unknown category."""

    CONFIG = {
        "default": {
            "tool_loop": {"THRESHOLD": 2, "WINDOW": 5},
            "retry_storm": {"THRESHOLD": 3},
        },
        "web-research": {
            "tool_loop": {"THRESHOLD": 5},
            "reasoning_stall": {"INFLATION_FACTOR": 3.0, "SEVERITY": Severity.LOW},
        },
    }

    def test_category_overrides_default_key_by_key(self):
        merged = effective_detector_kwargs(self.CONFIG, "web-research")
        self.assertEqual(
            merged,
            {
                "tool_loop": {"THRESHOLD": 5, "WINDOW": 5},
                "retry_storm": {"THRESHOLD": 3},
                "reasoning_stall": {"INFLATION_FACTOR": 3.0, "SEVERITY": Severity.LOW},
            },
        )

    def test_unknown_category_gets_default(self):
        self.assertEqual(effective_detector_kwargs(self.CONFIG, "nope"), self.CONFIG["default"])
        self.assertEqual(effective_detector_kwargs(self.CONFIG, "default"), self.CONFIG["default"])

    def test_result_is_a_copy(self):
        merged = effective_detector_kwargs(self.CONFIG, "nope")
        merged["tool_loop"]["THRESHOLD"] = 99
        self.assertEqual(self.CONFIG["default"]["tool_loop"]["THRESHOLD"], 2)

    def test_empty_config_gives_empty_map(self):
        self.assertEqual(effective_detector_kwargs({}, "anything"), {})

    def test_json_safe_kwargs_flattens_enums(self):
        merged = effective_detector_kwargs(self.CONFIG, "web-research")
        safe = json_safe_kwargs(merged)
        self.assertEqual(safe["reasoning_stall"]["SEVERITY"], "LOW")
        self.assertIs(type(safe["reasoning_stall"]["SEVERITY"]), str)
        self.assertEqual(safe["tool_loop"], {"THRESHOLD": 5, "WINDOW": 5})


if __name__ == "__main__":
    unittest.main(verbosity=2)
