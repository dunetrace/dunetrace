"""
Endpoint-level tests for the approvals API (api_svc/routers/approvals.py).
Calls route functions directly with mocked DB queries — the established
pattern in this suite. No network, no DB.

Run: PYTHONPATH=../../packages/sdk-py:../explainer:. python -m unittest tests.test_approvals_router -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.approvals import (
    ApprovalCreate,
    ApprovalDecision,
    get_one_approval,
    list_org_approvals,
    post_approval_decision,
    post_create_approval,
)


class TestListApprovals(unittest.IsolatedAsyncioTestCase):
    async def test_lists_org_approvals(self):
        with patch(
            "api_svc.routers.approvals.list_approvals",
            AsyncMock(return_value=[{"id": 1, "status": "pending"}]),
        ) as mock_list:
            result = await list_org_approvals(status="pending", org_id="org")
        self.assertEqual(len(result), 1)
        # org_id from require_org threads through, status filter passed
        self.assertEqual(mock_list.call_args.args[0], "org")
        self.assertEqual(mock_list.call_args.args[1], "pending")

    async def test_no_filter_lists_all(self):
        with patch(
            "api_svc.routers.approvals.list_approvals", AsyncMock(return_value=[])
        ) as mock_list:
            await list_org_approvals(status=None, org_id="org")
        self.assertIsNone(mock_list.call_args.args[1])

    async def test_invalid_status_is_422(self):
        with self.assertRaises(HTTPException) as ctx:
            await list_org_approvals(status="bogus", org_id="org")
        self.assertEqual(ctx.exception.status_code, 422)


class TestCreateApproval(unittest.IsolatedAsyncioTestCase):
    async def test_creates_and_returns_serialized_row(self):
        body = ApprovalCreate(
            run_id="r1",
            agent_id="a1",
            tool_name="wire_money",
            tool_args='{"amt": 5}',
            timeout_seconds=120,
        )
        with patch(
            "api_svc.routers.approvals.create_approval",
            AsyncMock(return_value={"id": 9, "org_id": "org", "status": "pending"}),
        ) as mock_create:
            result = await post_create_approval(body, org_id="org")

        self.assertEqual(result["id"], 9)
        # org comes from require_org, and expires_at was computed and passed.
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["org_id"], "org")
        self.assertIsNotNone(kwargs["expires_at"])

    async def test_timeout_is_clamped(self):
        body = ApprovalCreate(
            run_id="r1", agent_id="a1", tool_name="t", timeout_seconds=999999
        )  # over the max
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return {"id": 1, "status": "pending"}

        with patch("api_svc.routers.approvals.create_approval", _capture):
            await post_create_approval(body, org_id="org")
        # expires_at was computed from a clamped timeout (<= 3600s from now),
        # so it's well under the requested ~11.5 days.
        from datetime import datetime, timezone, timedelta

        self.assertLess(captured["expires_at"], datetime.now(timezone.utc) + timedelta(hours=2))

    async def test_pool_down_returns_503(self):
        body = ApprovalCreate(run_id="r1", agent_id="a1", tool_name="t")
        with patch("api_svc.routers.approvals.create_approval", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as ctx:
                await post_create_approval(body, org_id="org")
        self.assertEqual(ctx.exception.status_code, 503)


class TestGetApproval(unittest.IsolatedAsyncioTestCase):
    async def test_returns_row(self):
        with patch(
            "api_svc.routers.approvals.get_approval",
            AsyncMock(return_value={"id": 3, "status": "pending"}),
        ):
            result = await get_one_approval(3, org_id="org")
        self.assertEqual(result["status"], "pending")

    async def test_missing_is_404(self):
        with patch("api_svc.routers.approvals.get_approval", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as ctx:
                await get_one_approval(999, org_id="org")
        self.assertEqual(ctx.exception.status_code, 404)


class TestDecision(unittest.IsolatedAsyncioTestCase):
    async def test_records_granted(self):
        body = ApprovalDecision(
            decision="granted", decided_by="alice", decision_channel="dashboard"
        )
        with patch(
            "api_svc.routers.approvals.set_approval_decision",
            AsyncMock(return_value={"id": 5, "status": "granted"}),
        ) as mock_set:
            result = await post_approval_decision(5, body, org_id="org")
        self.assertEqual(result["status"], "granted")
        self.assertEqual(mock_set.call_args.kwargs["new_status"], "granted")

    async def test_invalid_decision_is_422(self):
        body = ApprovalDecision(decision="maybe")
        with self.assertRaises(HTTPException) as ctx:
            await post_approval_decision(5, body, org_id="org")
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_pending_is_not_a_decision_422(self):
        body = ApprovalDecision(decision="pending")
        with self.assertRaises(HTTPException) as ctx:
            await post_approval_decision(5, body, org_id="org")
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_already_decided_is_409(self):
        body = ApprovalDecision(decision="granted")
        with patch("api_svc.routers.approvals.set_approval_decision", AsyncMock(return_value=None)):
            with patch(
                "api_svc.routers.approvals.get_approval",
                AsyncMock(return_value={"id": 5, "status": "denied"}),
            ):
                with self.assertRaises(HTTPException) as ctx:
                    await post_approval_decision(5, body, org_id="org")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("denied", ctx.exception.detail)

    async def test_missing_on_decide_is_404(self):
        body = ApprovalDecision(decision="granted")
        with patch("api_svc.routers.approvals.set_approval_decision", AsyncMock(return_value=None)):
            with patch("api_svc.routers.approvals.get_approval", AsyncMock(return_value=None)):
                with self.assertRaises(HTTPException) as ctx:
                    await post_approval_decision(5, body, org_id="org")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
