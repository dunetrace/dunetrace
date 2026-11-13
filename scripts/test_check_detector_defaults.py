#!/usr/bin/env python3
"""
Tests for check_detector_defaults.py itself. A drift check that cannot detect
drift is worse than no check — it reads as a guarantee — so each of the three
failure modes gets a test that fails without the check, plus a test that the
marker escape hatch actually suppresses, plus a regression guard that the
repo's own detectors.yml stays clean.

Run: python scripts/test_check_detector_defaults.py
  or python -m unittest scripts.test_check_detector_defaults
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_detector_defaults as cdd  # noqa: E402

# tool_loop's real shipped values, which match ToolLoopDetector's class
# defaults — the baseline every fixture below perturbs one field of.
CLEAN = """\
default:

  tool_loop:
    threshold: 3
    window: 5
    alert_policy:
      mode: consecutive
      threshold: 2
      window_runs: 10
"""


def run_check(text: str) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "detectors.yml"
        path.write_text(text)
        return cdd.check(path)


class TestCleanConfig(unittest.TestCase):
    def test_shipped_values_matching_class_defaults_pass(self):
        self.assertEqual(run_check(CLEAN), [])

    def test_repo_detectors_yml_has_no_drift(self):
        """The real file. This is the check the CI job actually runs; keeping
        it here means `make test`-style local runs catch drift too."""
        self.assertEqual(cdd.check(cdd.REPO_ROOT / "detectors.yml"), [])


class TestUnmappedKey(unittest.TestCase):
    """Case 1: a key the parser silently drops with a startup warning."""

    def test_key_with_no_param_map_entry_is_flagged(self):
        text = CLEAN.replace("    window: 5\n", "    window: 5\n    no_such_knob: 9\n")
        violations = run_check(text)
        self.assertTrue(any("no_such_knob" in v for v in violations), violations)
        self.assertTrue(any("reaches no constructor kwarg" in v for v in violations))

    def test_flagged_with_the_line_number(self):
        text = CLEAN.replace("    window: 5\n", "    window: 5\n    no_such_knob: 9\n")
        (v,) = [x for x in run_check(text) if "no_such_knob" in x]
        self.assertIn("detectors.yml:6", v)

    def test_unmapped_key_in_an_agent_category_is_also_flagged(self):
        """Category sections are exempt from the value check, not from this
        one — a typo'd knob does nothing wherever it is written."""
        text = CLEAN + "\nweb-research:\n  tool_loop:\n    no_such_knob: 9\n"
        self.assertTrue(any("web-research" in v and "no_such_knob" in v for v in run_check(text)))

    def test_alert_policy_and_destinations_are_not_flagged(self):
        """Both are read by alerts_svc, not by this parser."""
        text = CLEAN.replace("    window: 5\n", "    window: 5\n    destinations: [slack]\n")
        self.assertEqual(run_check(text), [])


class TestDanglingParamMapEntry(unittest.TestCase):
    """Case 2: _PARAM_MAP aims at an attribute the class does not have. This
    is the one that produced no warning at all — the YAML key parsed fine."""

    def test_missing_attribute_is_flagged(self):
        with patch.dict(cdd._PARAM_MAP["tool_loop"], {"threshold": "GONE_AWAY"}):
            violations = run_check(CLEAN)
        self.assertTrue(any("GONE_AWAY" in v and "does not exist" in v for v in violations))

    def test_param_map_entry_for_an_unknown_detector_is_flagged(self):
        with patch.dict(cdd._PARAM_MAP, {"not_a_detector": {"x": "X"}}):
            violations = run_check(CLEAN)
        self.assertTrue(any("not_a_detector" in v for v in violations))


class TestDefaultValueDrift(unittest.TestCase):
    """Case 3: the shipped baseline stops being the class default."""

    def test_value_differing_from_class_default_is_flagged(self):
        violations = run_check(CLEAN.replace("threshold: 3", "threshold: 99"))
        self.assertTrue(any("defaults to 3" in v for v in violations), violations)

    def test_marker_suppresses_the_drift(self):
        text = CLEAN.replace(
            "    threshold: 3",
            "    # default-drift-ok: deliberately louder than the class default\n    threshold: 99",
        )
        self.assertEqual(run_check(text), [])

    def test_marker_does_not_suppress_an_unmapped_key(self):
        """The escape hatch is for value drift only — a key that reaches
        nothing is always a bug, however it is annotated."""
        text = CLEAN.replace(
            "    window: 5\n",
            "    window: 5\n    # default-drift-ok: nope\n    no_such_knob: 9\n",
        )
        self.assertTrue(any("no_such_knob" in v for v in run_check(text)))

    def test_agent_category_may_differ_from_the_class_default(self):
        """An override is supposed to differ. That is what it is for."""
        text = CLEAN + "\nweb-research:\n  tool_loop:\n    threshold: 5\n"
        self.assertEqual(run_check(text), [])

    def test_int_float_equality_is_not_drift(self):
        """YAML 3 and a class default of 3.0 are the same threshold."""
        text = "default:\n  context_bloat:\n    growth_factor: 3\n"
        self.assertEqual(run_check(text), [])


class TestUnknownSection(unittest.TestCase):
    def test_unknown_detector_section_is_flagged(self):
        text = CLEAN + "\n  tool_lop:\n    threshold: 3\n"
        self.assertTrue(any("tool_lop" in v for v in run_check(text)))


class TestScanPositions(unittest.TestCase):
    def test_records_line_numbers_and_markers(self):
        text = (
            "default:\n"
            "  tool_loop:\n"
            "    threshold: 3\n"
            "    # default-drift-ok: because\n"
            "    window: 5\n"
        )
        pos = cdd.scan_positions(text)
        self.assertEqual(pos[("default", "tool_loop", "threshold")], (3, False))
        self.assertEqual(pos[("default", "tool_loop", "window")], (5, True))

    def test_a_blank_line_breaks_the_marker(self):
        """The marker has to be immediately above the key, so a comment left
        floating further up cannot silently cover a later edit."""
        text = "default:\n  tool_loop:\n    # default-drift-ok: x\n\n    threshold: 3\n"
        self.assertEqual(cdd.scan_positions(text)[("default", "tool_loop", "threshold")][1], False)

    def test_alert_policy_body_is_not_mistaken_for_a_tunable(self):
        pos = cdd.scan_positions(CLEAN)
        self.assertIn(("default", "tool_loop", "threshold"), pos)
        # alert_policy's own `threshold: 2` sits at a deeper indent and must
        # not overwrite the detector's.
        self.assertEqual(pos[("default", "tool_loop", "threshold")][0], 4)


if __name__ == "__main__":
    unittest.main()
