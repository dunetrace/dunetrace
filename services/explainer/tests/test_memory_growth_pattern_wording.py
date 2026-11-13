"""MEMORY_GROWTH_PATTERN must stay readable when the ratio is undefined.

Zero reads is a real fire condition (the comparison never divides), and the
evidence stores ratio as JSON null. The template has to say that in words.
Coverage also calls every template with {} and with explicit nulls, so those
shapes must not raise and must not print a Python None.
"""

from __future__ import annotations

import unittest

from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.templates import TEMPLATES

TEMPLATE = TEMPLATES[FailureType.MEMORY_GROWTH_PATTERN]


def _signal(evidence: dict) -> FailureSignal:
    return FailureSignal(
        failure_type=FailureType.MEMORY_GROWTH_PATTERN,
        severity=Severity.MEDIUM,
        run_id="run-1",
        agent_id="agent-1",
        agent_version="v1",
        step_index=4,
        confidence=0.8,
        evidence=evidence,
        detected_at=0.0,
    )


def _blob(evidence: dict) -> str:
    exp = TEMPLATE(_signal(evidence))
    return f"{exp.title} {exp.what} {exp.why_it_matters} {exp.evidence_summary}"


class TestMemoryGrowthPatternWording(unittest.TestCase):
    def test_ratio_and_keys_are_explained(self):
        text = _blob(
            {
                "write_count": 3,
                "read_count": 1,
                "ratio": 3.0,
                "write_keys": ["user_prefs", "notes"],
                "read_keys": ["notes"],
                "write_read_ratio": 3,
            }
        )
        self.assertIn("3 writes", text)
        self.assertIn("3×", text)
        self.assertIn("user_prefs", text)
        self.assertIn("later run", text, "the within-run scope is stated")
        self.assertNotIn("None", text)

    def test_zero_reads_does_not_render_a_null_ratio(self):
        text = _blob({"write_count": 3, "read_count": 0, "ratio": None, "write_keys": []})
        self.assertIn("3 writes", text)
        self.assertIn("no reads", text)
        self.assertNotIn("None", text)
        self.assertNotIn("null", text.lower())

    def test_suggested_cap_is_per_run_not_per_process(self):
        exp = TEMPLATE(_signal({"write_count": 3, "read_count": 0}))
        code = " ".join(f.code for f in exp.suggested_fixes)
        self.assertNotIn("global writes", code)

    def test_empty_and_null_evidence_stay_readable(self):
        for evidence in ({}, {key: None for key in ("write_count", "read_count", "ratio")}):
            with self.subTest(evidence=evidence):
                text = _blob(evidence)
                self.assertTrue(text.strip())
                self.assertNotIn("None", text)
