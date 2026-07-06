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
from detector_svc.config_loader import load_detector_kwargs


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
