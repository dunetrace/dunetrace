"""
Tests for coerce_failure_type — reading open-set failure types off the wire.

`failure_signals.failure_type` is TEXT, and several first-class writers store
values that aren't `FailureType` members by design: JSON-config custom detectors
(`CUSTOM_*`), detector packs (`VOICE_*`), semantic evaluators, and the operational
markers `SEMANTIC_QUOTA_EXCEEDED` / `EXTERNAL_INTEGRATION_DOWN`.

Read paths that called `FailureType(value)` directly raised on all of them. The
signal wasn't dropped — it degraded to a bare type name with an empty explanation,
plus one ERROR log line per signal per request. This helper is what lets those rows
reach `explain()`, whose `_fallback` already knows how to render them.

Run:
    cd services/explainer
    python -m unittest tests.test_coerce_failure_type -v
"""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.explainer import coerce_failure_type, explain


class TestCoerceFailureType(unittest.TestCase):
    def test_known_type_passes_through(self):
        ft, ev = coerce_failure_type("TOOL_LOOP", {"tool": "search"})
        self.assertEqual(ft, FailureType.TOOL_LOOP)
        self.assertEqual(ev, {"tool": "search"})

    def test_known_type_does_not_add_raw_failure_type(self):
        _, ev = coerce_failure_type("TOOL_LOOP", {})
        self.assertNotIn("raw_failure_type", ev)

    def test_unknown_type_maps_to_custom_sentinel(self):
        ft, ev = coerce_failure_type("CUSTOM_EXCESSIVE_TOOL_CALLS", {})
        self.assertEqual(ft, FailureType.CUSTOM)
        self.assertEqual(ev["raw_failure_type"], "CUSTOM_EXCESSIVE_TOOL_CALLS")

    def test_none_evidence_is_tolerated(self):
        ft, ev = coerce_failure_type("VOICE_SILENCE_TIMEOUT", None)
        self.assertEqual(ft, FailureType.CUSTOM)
        self.assertEqual(ev["raw_failure_type"], "VOICE_SILENCE_TIMEOUT")

    def test_caller_evidence_is_not_mutated(self):
        original = {"month": "2026-07"}
        _, ev = coerce_failure_type("SEMANTIC_QUOTA_EXCEEDED", original)
        self.assertNotIn("raw_failure_type", original)
        self.assertIn("raw_failure_type", ev)

    def test_existing_raw_failure_type_is_preserved(self):
        """setdefault, not assignment — a writer that already recorded the raw
        name keeps its value."""
        _, ev = coerce_failure_type("CUSTOM_X", {"raw_failure_type": "ORIGINAL"})
        self.assertEqual(ev["raw_failure_type"], "ORIGINAL")

    def test_every_real_world_unknown_type_is_handled(self):
        """The exact set found in a live database — 2,056 rows across these types
        all failed to explain before this helper existed."""
        for raw in (
            "SEMANTIC_QUOTA_EXCEEDED",
            "EXTERNAL_INTEGRATION_DOWN",
            "CUSTOM_CONSECUTIVE_IDENTICAL_TOOL_CALLS",
            "CUSTOM_EXCESSIVE_TOOL_CALLS",
            "CUSTOM_TOOL_OSCILLATION",
            "TASK_COMPLETION",
            "VOICE_SPEAKER_CONFUSION",
            "VOICE_SILENCE_TIMEOUT",
            "VOICE_TRANSCRIPTION_CONFIDENCE_DROP",
        ):
            with self.subTest(raw=raw):
                ft, ev = coerce_failure_type(raw, {})
                self.assertEqual(ft, FailureType.CUSTOM)
                self.assertEqual(ev["raw_failure_type"], raw)


class TestCoercedSignalExplainsProperly(unittest.TestCase):
    """The point of the coercion is that explain() then produces real content
    rather than the empty strings the read path used to fall back to."""

    def _explain(self, raw):
        ft, ev = coerce_failure_type(raw, {"scope": "org_quota"})
        return explain(
            FailureSignal(
                failure_type=ft,
                severity=Severity.LOW,
                run_id="r1",
                agent_id="a1",
                agent_version="v1",
                step_index=0,
                confidence=1.0,
                evidence=ev,
                detected_at=0.0,
            )
        )

    def test_title_uses_the_raw_name_not_the_sentinel(self):
        exp = self._explain("SEMANTIC_QUOTA_EXCEEDED")
        self.assertIn("Semantic Quota Exceeded", exp.title)
        self.assertNotIn("CUSTOM", exp.title.upper())

    def test_explanation_body_is_populated(self):
        exp = self._explain("VOICE_SILENCE_TIMEOUT")
        self.assertTrue(exp.what.strip(), "what must not be empty")
        self.assertTrue(exp.why_it_matters.strip(), "why_it_matters must not be empty")

    def test_known_type_still_uses_its_real_template(self):
        ft, ev = coerce_failure_type(
            "TOOL_LOOP",
            {
                "tool": "search",
                "count": 6,
                "window": 5,
                "first_step": 2,
                "last_step": 7,
                "args_identical": True,
            },
        )
        exp = explain(
            FailureSignal(
                failure_type=ft,
                severity=Severity.HIGH,
                run_id="r1",
                agent_id="a1",
                agent_version="v1",
                step_index=3,
                confidence=0.9,
                evidence=ev,
                detected_at=0.0,
            )
        )
        self.assertIn("search", exp.what)


if __name__ == "__main__":
    unittest.main()
