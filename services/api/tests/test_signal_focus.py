"""
Phase 3: signal → step context.

A fired signal says something went wrong. The drill-down says what the step
actually contained. The only non-obvious part is *which* steps to show, and
that is per-detector: the evidence key holding the step span differs by failure
type, and ten detectors span a window while the other twenty-four collapse to
the signal's own step.

That mapping already exists once, in explain_common._STEP_RANGE_FIELDS. These
tests pin that the API computes `focus` from it, so the dashboard never grows a
second copy. A second copy would not fail loudly — it would focus the wrong
step, which is indistinguishable from the detector having pointed there.
"""

from __future__ import annotations

import unittest

from api_svc.db.queries import _focus_range
from api_svc.explain_common import _STEP_RANGE_FIELDS


class TestFocusRangeUsesTheDetectorsOwnEvidence(unittest.TestCase):
    def test_tool_loop_spans_its_whole_window(self):
        """The case the mapping exists for. TOOL_LOOP's step_index is the LAST
        call; focusing only that hides the repetition that defines it."""
        focus = _focus_range({"first_step": 2, "last_step": 10}, "TOOL_LOOP", 10)
        self.assertEqual(focus, {"first_step": 2, "last_step": 10})

    def test_context_bloat_uses_its_own_key_names(self):
        focus = _focus_range({"first_call_step": 1, "last_call_step": 9}, "CONTEXT_BLOAT", 9)
        self.assertEqual(focus, {"first_step": 1, "last_step": 9})

    def test_retry_storm_collapses_to_the_first_failure(self):
        focus = _focus_range({"first_fail_step": 4}, "RETRY_STORM", 8)
        self.assertEqual(focus, {"first_step": 4, "last_step": 4})

    def test_unmapped_detector_falls_back_to_the_signal_step(self):
        """Twenty-four of thirty-four detectors have no range."""
        focus = _focus_range({"key": "value"}, "MEMORY_POISONING", 3)
        self.assertEqual(focus, {"first_step": 3, "last_step": 3})

    def test_every_mapped_detector_resolves(self):
        """A detector listed in the mapping but whose keys do not resolve would
        silently focus the fallback. Walk the whole table."""
        for failure_type, (first_key, last_key) in _STEP_RANGE_FIELDS.items():
            with self.subTest(failure_type=failure_type):
                evidence = {first_key: 2, last_key: 7}
                focus = _focus_range(evidence, failure_type, 99)
                # Five entries name the same key twice (EMPTY_LLM_RESPONSE is
                # first_step/first_step) because the detector points at one
                # step, not a span. Both ends then collapse to that key's value
                # — which for this fixture is whichever write landed last.
                expected = {"first_step": evidence[first_key], "last_step": evidence[last_key]}
                self.assertEqual(
                    focus,
                    expected,
                    f"{failure_type} did not read its own evidence keys",
                )
                self.assertNotIn(
                    99, focus.values(), f"{failure_type} fell back instead of reading evidence"
                )


class TestFocusRangeNeverBreaksRunDetail(unittest.TestCase):
    """Run detail is the most-loaded view in the product. A malformed evidence
    dict must degrade to the signal's own step, never raise."""

    def test_non_numeric_evidence_falls_back(self):
        self.assertEqual(
            _focus_range({"first_step": "two", "last_step": 7}, "TOOL_LOOP", 5),
            {"first_step": 5, "last_step": 5},
        )

    def test_missing_evidence_falls_back(self):
        self.assertEqual(_focus_range({}, "TOOL_LOOP", 6), {"first_step": 6, "last_step": 6})

    def test_none_evidence_falls_back(self):
        self.assertEqual(_focus_range(None, "TOOL_LOOP", 6), {"first_step": 6, "last_step": 6})

    def test_non_integer_step_index_falls_back_to_zero(self):
        self.assertEqual(
            _focus_range({}, "MEMORY_POISONING", None), {"first_step": 0, "last_step": 0}
        )

    def test_unknown_failure_type_falls_back(self):
        """A custom detector's failure type is free TEXT and is not in the map."""
        self.assertEqual(
            _focus_range({"first_step": 1}, "MY_CUSTOM_DETECTOR", 4),
            {"first_step": 4, "last_step": 4},
        )


class TestRunDetailShape(unittest.TestCase):
    """The drill-down reads one response. Both fields have to survive the
    typed response model, which silently drops anything undeclared — that is
    exactly how the first attempt at this lost them."""

    def test_run_signal_model_declares_focus(self):
        from api_svc.schemas import RunSignal

        fields = getattr(RunSignal, "model_fields", None) or getattr(
            RunSignal, "__dataclass_fields__", {}
        )
        self.assertIn("focus", fields)

    def test_run_detail_model_declares_dry_run_verdicts(self):
        from api_svc.schemas import RunDetail

        fields = getattr(RunDetail, "model_fields", None) or getattr(
            RunDetail, "__dataclass_fields__", {}
        )
        self.assertIn("dry_run_verdicts", fields)

    def test_router_passes_both_through(self):
        """Declared on the model but not mapped in the router is the same
        silent drop, one layer up."""
        import inspect

        from api_svc.routers import runs

        src = inspect.getsource(runs.get_run)
        self.assertIn("dry_run_verdicts=", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
