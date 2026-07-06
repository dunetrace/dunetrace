"""
Tests for FailureSignalSchema — the validated wire-format failure signal.

Run:
    cd packages/schemas-py
    python -m pytest tests/test_signals.py -v
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from dunetrace_schemas import FailureSignalSchema


class TestFailureSignalSchema(unittest.TestCase):
    def _base(self, **overrides):
        fields = {
            "failure_type": "TOOL_LOOP",
            "severity": "HIGH",
            "run_id": "run-1",
            "agent_id": "agent-1",
            "agent_version": "v1",
            "step_index": 3,
            "confidence": 0.9,
            "evidence": {"tool": "search", "count": 5},
        }
        fields.update(overrides)
        return fields

    def test_valid_signal_constructs(self):
        sig = FailureSignalSchema(**self._base())
        self.assertEqual(sig.failure_type, "TOOL_LOOP")
        self.assertEqual(sig.severity, "HIGH")

    def test_custom_detector_failure_type_accepted(self):
        """Custom detectors store arbitrary CUSTOM_* names as raw TEXT, not the built-in enum."""
        sig = FailureSignalSchema(**self._base(failure_type="CUSTOM_HIGH_LATENCY"))
        self.assertEqual(sig.failure_type, "CUSTOM_HIGH_LATENCY")

    def test_invalid_severity_rejected(self):
        with self.assertRaises(ValidationError):
            FailureSignalSchema(**self._base(severity="EXTREME"))

    def test_confidence_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            FailureSignalSchema(**self._base(confidence=1.5))
        with self.assertRaises(ValidationError):
            FailureSignalSchema(**self._base(confidence=-0.1))

    def test_shadow_defaults_true(self):
        sig = FailureSignalSchema(**self._base())
        self.assertTrue(sig.shadow)

    def test_alerted_defaults_false(self):
        sig = FailureSignalSchema(**self._base())
        self.assertFalse(sig.alerted)

    def test_co_signal_count_defaults_zero(self):
        sig = FailureSignalSchema(**self._base())
        self.assertEqual(sig.co_signal_count, 0)

    def test_empty_failure_type_rejected(self):
        with self.assertRaises(ValidationError):
            FailureSignalSchema(**self._base(failure_type=""))

    def test_detected_at_defaults_when_omitted(self):
        sig = FailureSignalSchema(**self._base())
        self.assertIsInstance(sig.detected_at, float)
        self.assertGreater(sig.detected_at, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
