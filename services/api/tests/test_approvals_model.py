"""
Tests for the approval state model (Capability 2, Phase 2.1). Pure logic, no
DB — the whole point of api_svc/approvals.py is that the legal-transition rule
lives in one testable place.

Run: PYTHONPATH=../../packages/sdk-py:../explainer:. python -m unittest tests.test_approvals_model -v
"""

from __future__ import annotations

import unittest

from api_svc.approvals import (
    ApprovalStatus,
    DECISION_STATUSES,
    TERMINAL_STATUSES,
    coerce_status,
    is_terminal,
    is_valid_transition,
)


class TestApprovalStatus(unittest.TestCase):
    def test_four_statuses(self):
        self.assertEqual(
            {s.value for s in ApprovalStatus},
            {"pending", "granted", "denied", "timeout"},
        )

    def test_decision_statuses_exclude_pending(self):
        self.assertNotIn(ApprovalStatus.PENDING, DECISION_STATUSES)
        self.assertEqual(len(DECISION_STATUSES), 3)


class TestTerminal(unittest.TestCase):
    def test_pending_is_not_terminal(self):
        self.assertFalse(is_terminal(ApprovalStatus.PENDING))

    def test_decisions_are_terminal(self):
        for s in (ApprovalStatus.GRANTED, ApprovalStatus.DENIED, ApprovalStatus.TIMEOUT):
            self.assertTrue(is_terminal(s), s)

    def test_terminal_set_matches_decisions(self):
        self.assertEqual(TERMINAL_STATUSES, set(DECISION_STATUSES))


class TestTransitions(unittest.TestCase):
    def test_pending_to_each_decision_is_legal(self):
        for target in DECISION_STATUSES:
            self.assertTrue(is_valid_transition(ApprovalStatus.PENDING, target), target)

    def test_pending_to_pending_is_illegal(self):
        self.assertFalse(is_valid_transition(ApprovalStatus.PENDING, ApprovalStatus.PENDING))

    def test_terminal_has_no_outgoing_transitions(self):
        for current in TERMINAL_STATUSES:
            for target in ApprovalStatus:
                self.assertFalse(
                    is_valid_transition(current, target),
                    f"{current} -> {target} should be illegal",
                )

    def test_late_decision_after_timeout_is_rejected(self):
        # The concrete scenario the guard protects: a human clicks Approve in
        # Slack after the SDK already gave up (timeout). It must not flip a
        # recorded timeout to granted.
        self.assertFalse(is_valid_transition(ApprovalStatus.TIMEOUT, ApprovalStatus.GRANTED))


class TestCoerce(unittest.TestCase):
    def test_valid_string_coerces(self):
        self.assertIs(coerce_status("granted"), ApprovalStatus.GRANTED)

    def test_unknown_string_raises(self):
        with self.assertRaises(ValueError):
            coerce_status("maybe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
