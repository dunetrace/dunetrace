"""
Tests for the SDK approval flow (Capability 2, Phase 2.2): request_approval
(sync) and arequest_approval (async). The Customer-API HTTP helpers on the
client are the seam — patched here so no network and no real waiting. The
blocking poll is exercised by controlling what _get_approval returns on
successive calls.

Run: python -m unittest tests.test_approval_flow -v
"""

from __future__ import annotations

import asyncio
import unittest
import unittest.mock
from unittest.mock import MagicMock

from dunetrace.client import DunetraceClient
from dunetrace.models import EventType
from dunetrace.policies import ApprovalDenied


def _make_client() -> DunetraceClient:
    c = DunetraceClient(api_key="dt_test", api_url="http://localhost:8002", debug=False)
    c._ship = lambda batch: None
    return c


def _run(client):
    cm = client.run("agent")
    return cm.__enter__(), cm


def _event_types(run):
    return [e.event_type for e in run.state.events]


class TestSyncApprovalGranted(unittest.TestCase):
    def test_returns_and_emits_requested_then_granted(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 11})
        # pending on the first poll, granted on the second
        c._get_approval = MagicMock(side_effect=[{"status": "pending"}, {"status": "granted"}])
        run, cm = _run(c)
        with unittest.mock.patch("time.sleep"):
            run.request_approval("wire_money", {"amt": 500}, timeout_s=300)
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        types = _event_types(run)
        self.assertIn(EventType.APPROVAL_REQUESTED, types)
        self.assertIn(EventType.APPROVAL_GRANTED, types)
        self.assertNotIn(EventType.APPROVAL_DENIED, types)

    def test_create_request_carries_run_and_tool(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 1})
        c._get_approval = MagicMock(return_value={"status": "granted"})
        run, cm = _run(c)
        run.request_approval("delete_db", {"table": "users"}, timeout_s=120)
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        kwargs = c._create_approval_request.call_args.kwargs
        self.assertEqual(kwargs["tool_name"], "delete_db")
        self.assertEqual(kwargs["run_id"], run.run_id)
        self.assertEqual(kwargs["timeout_seconds"], 120)


class TestSyncApprovalDenied(unittest.TestCase):
    def test_raises_and_emits_denied(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 22})
        c._get_approval = MagicMock(return_value={"status": "denied"})
        run, cm = _run(c)
        with self.assertRaises(ApprovalDenied) as ctx:
            run.request_approval("wire_money")
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        self.assertEqual(ctx.exception.reason, "denied")
        self.assertEqual(ctx.exception.tool_name, "wire_money")
        self.assertIn(EventType.APPROVAL_DENIED, _event_types(run))


class TestSyncApprovalTimeout(unittest.TestCase):
    def test_fail_closed_on_timeout(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 33})
        c._get_approval = MagicMock(return_value={"status": "pending"})
        # SDK marks its own timeout; the write succeeds (no human raced it).
        c._decide_approval = MagicMock(return_value={"id": 33, "status": "timeout"})
        run, cm = _run(c)
        with self.assertRaises(ApprovalDenied) as ctx:
            run.request_approval("wire_money", timeout_s=0)  # deadline already passed
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        self.assertEqual(ctx.exception.reason, "timeout")
        c._decide_approval.assert_called_once_with(33, "timeout")
        self.assertIn(EventType.APPROVAL_TIMEOUT, _event_types(run))

    def test_human_wins_race_against_timeout(self):
        """Deadline passes, SDK tries to mark timeout, but a human granted in
        the gap — the timeout write 409s (returns None), SDK re-reads and
        honors the grant instead of forcing a timeout."""
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 44})
        c._get_approval = MagicMock(
            side_effect=[
                {"status": "pending"},  # last poll before deadline
                {"status": "granted"},  # re-read after the 409
            ]
        )
        c._decide_approval = MagicMock(return_value=None)  # 409 — already decided
        run, cm = _run(c)
        run.request_approval("wire_money", timeout_s=0)  # returns, no raise
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)

        self.assertIn(EventType.APPROVAL_GRANTED, _event_types(run))
        self.assertNotIn(EventType.APPROVAL_TIMEOUT, _event_types(run))


class TestAsyncApproval(unittest.TestCase):
    def test_async_granted(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 55})
        c._get_approval = MagicMock(side_effect=[{"status": "pending"}, {"status": "granted"}])
        run, cm = _run(c)

        async def go():
            with unittest.mock.patch("asyncio.sleep", new=_noop_async):
                await run.arequest_approval("wire_money", timeout_s=300)

        asyncio.run(go())
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)
        self.assertIn(EventType.APPROVAL_GRANTED, _event_types(run))

    def test_async_denied_raises(self):
        c = _make_client()
        c._create_approval_request = MagicMock(return_value={"id": 66})
        c._get_approval = MagicMock(return_value={"status": "denied"})
        run, cm = _run(c)

        async def go():
            await run.arequest_approval("wire_money", timeout_s=300)

        with self.assertRaises(ApprovalDenied):
            asyncio.run(go())
        cm.__exit__(None, None, None)
        c.shutdown(timeout=2)
        self.assertIn(EventType.APPROVAL_DENIED, _event_types(run))


async def _noop_async(*args, **kwargs):
    return None


if __name__ == "__main__":
    unittest.main(verbosity=2)
