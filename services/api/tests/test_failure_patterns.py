"""
Regression tests for the failure-pattern deep-dive endpoint validation.

The allow-list of valid failure types was hardcoded and went stale: real, live
failure types added to the FailureType enum later (e.g. TOOL_ARGUMENT_FABRICATION,
the top pattern on the overview) were rejected with 422, so clicking them in the
dashboard broke. These tests assert the allow-list stays in sync with the enum.

Run: make test-api  (or python -m pytest services/api/tests/test_failure_patterns.py)
"""

from __future__ import annotations

import unittest

from api_svc.routers.failure_patterns import _VALID_FAILURE_TYPES
from dunetrace.models import FailureType


class TestFailureTypeAllowList(unittest.TestCase):
    def test_allow_list_covers_every_enum_failure_type(self):
        enum_values = {t.value for t in FailureType}
        missing = enum_values - _VALID_FAILURE_TYPES
        self.assertEqual(missing, set(), f"allow-list missing enum types: {sorted(missing)}")

    def test_previously_rejected_types_are_now_allowed(self):
        # The exact types that were live in the DB but rejected by the stale list.
        for ft in (
            "TOOL_ARGUMENT_FABRICATION",
            "COST_SPIKE",
            "SESSION_LATENCY",
            "UNREAD_TOOL_ERROR",
            "RUNAWAY_ITERATION",
            "PREMATURE_TERMINATION",
            "RETRIEVED_CONTENT_INJECTION",
        ):
            self.assertIn(ft, _VALID_FAILURE_TYPES, ft)


if __name__ == "__main__":
    unittest.main(verbosity=2)
