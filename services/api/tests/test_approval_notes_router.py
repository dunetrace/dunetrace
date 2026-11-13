"""
Decision notes on the approvals endpoint (api_svc/routers/approvals.py).

A note rides the terminal decision write and only that write. That single design
choice is what makes it immutable and what makes the race behaviour fall out for
free: a decision that loses the `status = 'pending'` guard writes nothing, so its
note is discarded with it. There is deliberately no note-update endpoint.

Calls route functions directly with mocked DB queries — the established pattern
in this suite. No network, no DB.

Run: PYTHONPATH=../../packages/sdk-py:../explainer:. python -m pytest tests/test_approval_notes_router.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import pydantic
from fastapi import HTTPException

from api_svc.routers.approvals import MAX_NOTE_CHARS, ApprovalDecision, post_approval_decision

NOTE = "wrong Chen — it's Sarah, CUST_8834"


def _row(status="denied", note=NOTE, **extra):
    row = {
        "id": 7,
        "org_id": "org",
        "run_id": "r1",
        "agent_id": "a1",
        "tool_name": "delete_customer",
        "status": status,
        "note": note,
        "decided_by": "alice@corp.com",
        "decision_channel": "dashboard",
    }
    row.update(extra)
    return row


class TestNoteRidesTheDecision(unittest.IsolatedAsyncioTestCase):
    async def test_deny_with_note_reaches_the_db_layer(self):
        body = ApprovalDecision(
            decision="denied", decided_by="alice@corp.com", decision_channel="dashboard", note=NOTE
        )
        with patch(
            "api_svc.routers.approvals.set_approval_decision", AsyncMock(return_value=_row())
        ) as mock_set:
            result = await post_approval_decision(7, body, org_id="org")
        self.assertEqual(mock_set.call_args.kwargs["note"], NOTE)
        self.assertEqual(result["note"], NOTE)

    async def test_note_is_allowed_on_a_grant(self):
        """ "approved, but watch X" belongs in the audit trail exactly as much as
        a refusal does."""
        body = ApprovalDecision(decision="granted", note="approved — use CUST_8834")
        with patch(
            "api_svc.routers.approvals.set_approval_decision",
            AsyncMock(return_value=_row(status="granted", note="approved — use CUST_8834")),
        ) as mock_set:
            result = await post_approval_decision(7, body, org_id="org")
        self.assertEqual(mock_set.call_args.kwargs["note"], "approved — use CUST_8834")
        self.assertEqual(result["status"], "granted")

    async def test_note_is_allowed_on_a_timeout(self):
        """Not a path anything drives today — the SDK's own timeout write sends
        no note — but the endpoint takes a note on ANY terminal decision and
        should not special-case one of the three."""
        body = ApprovalDecision(decision="timeout", note="expired during the incident call")
        with patch(
            "api_svc.routers.approvals.set_approval_decision",
            AsyncMock(return_value=_row(status="timeout")),
        ) as mock_set:
            await post_approval_decision(7, body, org_id="org")
        self.assertEqual(mock_set.call_args.kwargs["note"], "expired during the incident call")

    async def test_no_note_passes_none_not_empty_string(self):
        body = ApprovalDecision(decision="denied")
        with patch(
            "api_svc.routers.approvals.set_approval_decision",
            AsyncMock(return_value=_row(note=None)),
        ) as mock_set:
            await post_approval_decision(7, body, org_id="org")
        self.assertIsNone(mock_set.call_args.kwargs["note"])

    async def test_whitespace_only_note_is_no_note(self):
        """ "" and "   " and absent all mean the same thing downstream; storing
        the difference would only give the trust surfaces a distinction they'd
        have to strip anyway."""
        for blank in ("", "   ", "\n\t "):
            with self.subTest(blank=repr(blank)):
                body = ApprovalDecision(decision="denied", note=blank)
                with patch(
                    "api_svc.routers.approvals.set_approval_decision",
                    AsyncMock(return_value=_row(note=None)),
                ) as mock_set:
                    await post_approval_decision(7, body, org_id="org")
                self.assertIsNone(mock_set.call_args.kwargs["note"])


class TestNoteLengthIsRejectedNotTruncated(unittest.IsolatedAsyncioTestCase):
    """Silently storing half of what a human wrote — and shipping that half back
    into the agent's next planning step — is worse than making them shorten it.

    Validation runs before the handler body, so an oversized note cannot
    half-apply: the decision is not written at all."""

    async def test_at_the_limit_is_accepted(self):
        body = ApprovalDecision(decision="denied", note="x" * MAX_NOTE_CHARS)
        with patch(
            "api_svc.routers.approvals.set_approval_decision",
            AsyncMock(return_value=_row(note="x" * MAX_NOTE_CHARS)),
        ) as mock_set:
            await post_approval_decision(7, body, org_id="org")
        self.assertEqual(len(mock_set.call_args.kwargs["note"]), MAX_NOTE_CHARS)

    async def test_over_the_limit_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            ApprovalDecision(decision="denied", note="x" * (MAX_NOTE_CHARS + 1))

    async def test_rejection_is_atomic_the_decision_is_not_applied(self):
        """The whole point of rejecting at the model boundary: the DB layer is
        never reached, so there is no window where the decision landed and the
        note didn't."""
        mock_set = AsyncMock(return_value=_row())
        with patch("api_svc.routers.approvals.set_approval_decision", mock_set):
            with self.assertRaises(pydantic.ValidationError):
                body = ApprovalDecision(decision="denied", note="x" * (MAX_NOTE_CHARS + 1))
                await post_approval_decision(7, body, org_id="org")
        mock_set.assert_not_called()


class TestLosingDecisionDiscardsItsNote(unittest.IsolatedAsyncioTestCase):
    """The pre-existing 409 path, unchanged. The `status = 'pending'` guard
    rejects the whole UPDATE, so the note never lands — no separate cleanup, and
    no way for a late note to attach itself to someone else's decision."""

    async def test_409_when_already_decided_and_nothing_is_written(self):
        mock_set = AsyncMock(return_value=None)  # guard blocked the write
        with (
            patch("api_svc.routers.approvals.set_approval_decision", mock_set),
            patch(
                "api_svc.routers.approvals.get_approval",
                AsyncMock(return_value=_row(status="granted", note="the winner's note")),
            ),
        ):
            body = ApprovalDecision(decision="denied", note="the loser's note")
            with self.assertRaises(HTTPException) as ctx:
                await post_approval_decision(7, body, org_id="org")

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("granted", ctx.exception.detail)
        # The losing note was offered to the DB layer and refused there — it is
        # not written, and the winner's note is untouched.
        self.assertEqual(mock_set.call_args.kwargs["note"], "the loser's note")

    async def test_404_when_the_approval_does_not_exist(self):
        with (
            patch("api_svc.routers.approvals.set_approval_decision", AsyncMock(return_value=None)),
            patch("api_svc.routers.approvals.get_approval", AsyncMock(return_value=None)),
        ):
            body = ApprovalDecision(decision="denied", note=NOTE)
            with self.assertRaises(HTTPException) as ctx:
                await post_approval_decision(7, body, org_id="org")
        self.assertEqual(ctx.exception.status_code, 404)


class TestBackwardCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_a_body_without_a_note_still_validates(self):
        """Every existing caller — the Slack handler, the SDK's timeout write,
        any operator script — sends no note field at all."""
        body = ApprovalDecision(decision="granted", decided_by="u", decision_channel="slack")
        self.assertIsNone(body.note)

    async def test_set_approval_decision_note_is_keyword_with_a_default(self):
        """routers/slack.py calls this function positionally-by-keyword without
        `note`. If the parameter were required, the Slack path would break — and
        the Slack surface is explicitly out of scope for this change."""
        import inspect

        from api_svc.db.queries import set_approval_decision

        param = inspect.signature(set_approval_decision).parameters["note"]
        self.assertIs(param.default, None)
        self.assertEqual(param.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)


if __name__ == "__main__":
    unittest.main()
