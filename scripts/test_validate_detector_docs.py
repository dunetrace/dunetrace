"""
Tests for the detector-docs drift check (scripts/validate_detector_docs.py).

Run: python -m unittest scripts.test_validate_detector_docs   (from repo root)
  or: cd scripts && python -m unittest test_validate_detector_docs
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validate_detector_docs as v  # noqa: E402


class TestValidateLogic(unittest.TestCase):
    def test_fake_detector_in_docs_fails(self):
        # The exact drift class this check exists to stop: a doc claims a detector
        # that doesn't exist in code.
        unknown, _ = v.validate(
            {"TOOL_LOOP", "SILENT_TRUNCATION"},
            {"TOOL_LOOP": ["README.md"], "MEMORY_POISONING": ["README.md"]},
            allowlist=set(),
        )
        self.assertIn("MEMORY_POISONING", unknown)
        self.assertNotIn("TOOL_LOOP", unknown)

    def test_all_known_names_pass(self):
        unknown, _ = v.validate(
            {"TOOL_LOOP", "HALLUCINATION"},
            {"TOOL_LOOP": ["a.md"], "HALLUCINATION": ["b.md"]},
            allowlist=set(),
        )
        self.assertEqual(unknown, {})

    def test_allowlisted_non_detector_passes(self):
        unknown, _ = v.validate(
            {"TOOL_LOOP"},
            {"SLACK_WEBHOOK_URL": ["docs/alerts.md"]},
            allowlist={"SLACK_WEBHOOK_URL"},
        )
        self.assertEqual(unknown, {})

    def test_in_code_but_undocumented_warns_only(self):
        unknown, undoc = v.validate(
            {"TOOL_LOOP", "RARE_DETECTOR"},
            {"TOOL_LOOP": ["a.md"]},
            allowlist=set(),
        )
        self.assertEqual(unknown, {})  # not a failure
        self.assertIn("RARE_DETECTOR", undoc)


class TestScanText(unittest.TestCase):
    def test_extracts_backticked_multi_and_single_word(self):
        text = "We ship `TOOL_LOOP` and `HALLUCINATION`, configured via `SLACK_WEBHOOK_URL`."
        toks = v.scan_text(text)
        self.assertEqual(toks, {"TOOL_LOOP", "HALLUCINATION", "SLACK_WEBHOOK_URL"})

    def test_ignores_unbackticked_and_lowercase(self):
        text = "TOOL_LOOP without backticks, and `lower_case`, should not match."
        self.assertEqual(v.scan_text(text), set())

    def test_fake_readme_fails_end_to_end(self):
        fake_readme = "Detectors include `TOOL_LOOP` and `MEMORY_POISONING`."
        mentions = {t: ["FAKE_README"] for t in v.scan_text(fake_readme)}
        unknown, _ = v.validate({"TOOL_LOOP"}, mentions, allowlist=set())
        self.assertIn("MEMORY_POISONING", unknown)

    def test_correct_readme_passes_end_to_end(self):
        fake_readme = "Detectors include `TOOL_LOOP` and `SILENT_TRUNCATION`."
        mentions = {t: ["FAKE_README"] for t in v.scan_text(fake_readme)}
        unknown, _ = v.validate({"TOOL_LOOP", "SILENT_TRUNCATION"}, mentions, allowlist=set())
        self.assertEqual(unknown, {})


class TestAgainstRealCodeAndDocs(unittest.TestCase):
    def test_new_detectors_and_evaluators_are_in_code_names(self):
        names = v.code_names()
        for n in (
            "SILENT_TRUNCATION",
            "MODEL_FALLBACK_DRIFT",  # Phase 1, 4
            "CONFUSION_LOOP",
            "TASK_UNDERSTANDING_FAILURE",
            "SYCOPHANCY_SIGNAL",
            "OFF_TOPIC_DRIFT",  # Phase 5-8
            "MEMORY_POISONING",  # Capability 1
            "DELEGATION_LOOP",  # Capability 2
            "AGENT_HANDOFF_FAILURE",  # PR #52
            "TOOL_LOOP",
            "HALLUCINATION",
            "VOICE_SILENCE_TIMEOUT",  # pre-existing
        ):
            self.assertIn(n, names)

    def test_real_docs_have_no_drift(self):
        # Catches any current drift between code and the detector-claim docs.
        names = v.code_names()
        mentions = v.scan_docs()
        unknown, _ = v.validate(names, mentions)
        self.assertEqual(unknown, {}, f"doc drift detected: {unknown}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
