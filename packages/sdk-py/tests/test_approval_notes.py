"""
Human decision notes on the approval flow.

A human deciding an approval can attach a note. It reaches the agent process on
the ApprovalDenied exception, lands in the run's event stream, and — the part
that stops this feature from being self-defeating — counts as human-authored
input to the detector trust surfaces, so a corrected retry is not flagged by our
own detectors as ungrounded or unwarranted.

Naming: the exception's terminal status is `.status` ("denied" | "timeout") and
the human's text is `.note`. There is deliberately no `.reason` and no alias —
`reason` was the wrong word once grants started carrying notes too.

Run: python -m pytest tests/test_approval_notes.py -v
"""

from __future__ import annotations

import asyncio
import unittest
import unittest.mock
from unittest.mock import MagicMock

from dunetrace.client import DunetraceClient
from dunetrace.models import EventType
from dunetrace.policies import ApprovalDenied

NOTE = "wrong Chen — it's Sarah, CUST_8834"


def _make_client() -> DunetraceClient:
    c = DunetraceClient(api_key="dt_test", api_url="http://localhost:8002", debug=False)
    c._ship = lambda batch: None
    return c


def _run(client):
    cm = client.run("agent")
    return cm.__enter__(), cm


def _event(run, event_type):
    for e in run.state.events:
        if e.event_type == event_type:
            return e
    return None


def _decided(status, **extra):
    """An approval row as the Customer API returns it after a decision."""
    row = {"id": 7, "status": status, "note": None, "decided_by": None}
    row.update(extra)
    return row


class TestDenyWithNote(unittest.TestCase):
    def test_note_and_decided_by_reach_the_exception(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(
            return_value=_decided("denied", note=NOTE, decided_by="alice@corp.com")
        )
        run, cm = _run(c)
        with self.assertRaises(ApprovalDenied) as ctx:
            run.request_approval("delete_customer")
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        self.assertEqual(ctx.exception.note, NOTE)
        self.assertEqual(ctx.exception.decided_by, "alice@corp.com")
        self.assertEqual(ctx.exception.status, "denied")
        self.assertEqual(ctx.exception.tool_name, "delete_customer")

    def test_note_appears_in_the_denied_event_payload(self):
        """The exception is ephemeral — caught, logged, maybe swallowed. The
        event stream is the audit trail, and it is what the server-side trust
        surfaces read."""
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(
            return_value=_decided("denied", note=NOTE, decided_by="alice@corp.com")
        )
        run, cm = _run(c)
        with self.assertRaises(ApprovalDenied):
            run.request_approval("delete_customer")
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        payload = _event(run, EventType.APPROVAL_DENIED).payload
        self.assertEqual(payload["note"], NOTE)
        self.assertEqual(payload["decided_by"], "alice@corp.com")

    def test_note_is_in_the_exception_message(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(return_value=_decided("denied", note=NOTE))
        run, cm = _run(c)
        with self.assertRaises(ApprovalDenied) as ctx:
            run.request_approval("delete_customer")
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)
        self.assertIn(NOTE, str(ctx.exception))

    def test_reason_attribute_is_gone(self):
        """One name per concept. `.reason` used to hold the status; it does not
        exist now, and no alias was shipped in its place."""
        exc = ApprovalDenied("t", "denied", 1, note="n", decided_by="d")
        self.assertFalse(hasattr(exc, "reason"))
        self.assertEqual(exc.status, "denied")


class TestGrantWithNote(unittest.TestCase):
    def test_grant_note_rides_the_event_and_nothing_raises(self):
        """ "approved, but watch X" belongs in the audit trail exactly as much as
        a refusal does."""
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(
            return_value=_decided("granted", note="approved — use CUST_8834", decided_by="bob")
        )
        run, cm = _run(c)
        run.request_approval("delete_customer")  # must not raise
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        payload = _event(run, EventType.APPROVAL_GRANTED).payload
        self.assertEqual(payload["note"], "approved — use CUST_8834")
        self.assertEqual(payload["decided_by"], "bob")


class TestTimeout(unittest.TestCase):
    def test_timeout_note_is_none(self):
        """Nobody decided, so nobody wrote anything — by construction."""
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(return_value={"status": "pending"})
        c._decide_approval = MagicMock(return_value=_decided("timeout"))
        run, cm = _run(c)
        with self.assertRaises(ApprovalDenied) as ctx:
            run.request_approval("delete_customer", timeout_s=0)
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        self.assertEqual(ctx.exception.status, "timeout")
        self.assertIsNone(ctx.exception.note)
        self.assertIsNone(ctx.exception.decided_by)
        self.assertNotIn("note", _event(run, EventType.APPROVAL_TIMEOUT).payload)


class TestLateDecisionRace(unittest.TestCase):
    """The pre-existing 409 path: the SDK's timeout write loses to a human who
    decided in the gap. Behaviour is unchanged; the note now rides along."""

    def test_winning_human_note_is_honored(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(
            side_effect=[
                {"status": "pending"},
                _decided("denied", note=NOTE, decided_by="alice"),
            ]
        )
        c._decide_approval = MagicMock(return_value=None)  # 409 — human won
        run, cm = _run(c)
        with self.assertRaises(ApprovalDenied) as ctx:
            run.request_approval("delete_customer", timeout_s=0)
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        self.assertEqual(ctx.exception.status, "denied")
        self.assertEqual(ctx.exception.note, NOTE)
        self.assertIn(EventType.APPROVAL_DENIED, [e.event_type for e in run.state.events])

    def test_grant_wins_the_race_and_its_note_survives(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(
            side_effect=[
                {"status": "pending"},
                _decided("granted", note="fine, but only this once", decided_by="alice"),
            ]
        )
        c._decide_approval = MagicMock(return_value=None)
        run, cm = _run(c)
        run.request_approval("delete_customer", timeout_s=0)  # must not raise
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        self.assertEqual(
            _event(run, EventType.APPROVAL_GRANTED).payload["note"], "fine, but only this once"
        )


class TestMalformedPayloadDoesNotCrash(unittest.TestCase):
    """The approval row is untyped JSON from the API. A detector — and the
    raising path itself — must never turn a weird row into an AttributeError at
    the catch site."""

    def test_row_shapes_that_must_degrade_to_no_note(self):
        for row in (
            {"status": "denied"},  # key absent
            {"status": "denied", "note": None},  # explicit null
            {"status": "denied", "note": ""},  # empty
            {"status": "denied", "note": "   "},  # whitespace only
            {"status": "denied", "note": 42},  # wrong type
            {"status": "denied", "note": {"a": 1}},  # wrong type
            {"status": "denied", "decided_by": 7},  # wrong type
        ):
            with self.subTest(row=row):
                c = _make_client()
                c._create_approval_request = MagicMock(return_value={"id": 7})
                c._get_approval = MagicMock(return_value=dict(row))
                run, cm = _run(c)
                with self.assertRaises(ApprovalDenied) as ctx:
                    run.request_approval("delete_customer")
                cm.__exit__(None, None, None)
                c.shutdown(timeout=2)
                self.assertIsNone(ctx.exception.note)
                self.assertNotIn("note", _event(run, EventType.APPROVAL_DENIED).payload)


class TestAsyncPath(unittest.TestCase):
    def test_arequest_approval_carries_the_note(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 7})
        c._get_approval = MagicMock(return_value=_decided("denied", note=NOTE, decided_by="alice"))
        run, cm = _run(c)

        async def go():
            with self.assertRaises(ApprovalDenied) as ctx:
                await run.arequest_approval("delete_customer")
            return ctx.exception

        exc = asyncio.run(go())
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)
        self.assertEqual(exc.note, NOTE)
        self.assertEqual(exc.decided_by, "alice")


if __name__ == "__main__":
    unittest.main()
