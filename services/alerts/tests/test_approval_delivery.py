"""
Tests for approval delivery (Capability 2, Phase 2.3): the approval formatter
and the alerts worker's deliver_pending_approvals() pass. No network, no DB —
the DB helpers, destination resolver, and senders are patched in the worker
namespace, the same convention test_delivery.py uses.

Run: PYTHONPATH=../../packages/sdk-py:../explainer:. python -m unittest tests.test_approval_delivery -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alerts_svc.formatters.approval import format_slack_approval, format_webhook_approval
import alerts_svc.worker as worker_module


_APPROVAL = {
    "id": 42,
    "org_id": "org-1",
    "run_id": "run-1",
    "agent_id": "payments-agent",
    "tool_name": "wire_money",
    "tool_args": '{"amount": 5000}',
}


class TestApprovalFormatter(unittest.TestCase):
    def test_slack_has_approve_and_deny_buttons(self):
        msg = format_slack_approval(_APPROVAL)
        actions = [b for b in msg["blocks"] if b["type"] == "actions"][0]
        ids = {e["action_id"] for e in actions["elements"]}
        self.assertEqual(ids, {"approve_request", "deny_request"})

    def test_slack_button_value_carries_approval_and_org(self):
        msg = format_slack_approval(_APPROVAL)
        actions = [b for b in msg["blocks"] if b["type"] == "actions"][0]
        val = json.loads(actions["elements"][0]["value"])
        self.assertEqual(val["approval_id"], 42)
        self.assertEqual(val["org_id"], "org-1")

    def test_slack_summary_mentions_tool_and_agent(self):
        msg = format_slack_approval(_APPROVAL)
        text = msg["blocks"][0]["text"]["text"]
        self.assertIn("wire_money", text)
        self.assertIn("payments-agent", text)

    def test_notification_text_names_agent_and_tool(self):
        """Top-level `text` is the desktop notification preview — it must say
        what is being approved, in plain text (no mrkdwn markers)."""
        text = format_slack_approval(_APPROVAL)["text"]
        self.assertIn("wire_money", text)
        self.assertIn("payments-agent", text)
        for ch in "`*~":
            self.assertNotIn(ch, text)

    def test_notification_text_strips_mrkdwn_from_identifiers(self):
        """A notification renders `text` verbatim, so a backtick or asterisk
        in an agent id or tool name would leak the marker through. The
        underscore in the tool name must survive — stripping it would
        corrupt the identifier."""
        approval = dict(_APPROVAL, agent_id="pay*ments", tool_name="wire_`money`")
        text = format_slack_approval(approval)["text"]
        for ch in "`*~":
            self.assertNotIn(ch, text)
        self.assertIn("payments", text)
        self.assertIn("wire_money", text)

    def test_webhook_payload_shape(self):
        p = format_webhook_approval(_APPROVAL)
        self.assertEqual(p["event"], "approval_request")
        self.assertEqual(p["approval_id"], 42)
        self.assertEqual(p["tool_name"], "wire_money")


class TestDeliverPendingApprovals(unittest.IsolatedAsyncioTestCase):
    def _patches(self, *, slack_enabled=False, webhook_enabled=False, slack_dest=None):
        settings = MagicMock()
        settings.slack_enabled = slack_enabled
        settings.webhook_enabled = webhook_enabled
        settings.WEBHOOK_SECRET = "secret"
        return {
            "settings": patch("alerts_svc.worker.settings", settings),
            "resolve": patch(
                "alerts_svc.worker._resolve_slack_destination",
                AsyncMock(return_value=slack_dest),
            ),
            "mark": patch("alerts_svc.worker.mark_approval_delivered", AsyncMock()),
            "send_slack": patch("alerts_svc.worker.send_slack", MagicMock()),
            "send_webhook": patch("alerts_svc.worker.send_webhook", MagicMock()),
        }

    async def test_no_approvals_returns_zero(self):
        with patch("alerts_svc.worker.fetch_undelivered_approvals", AsyncMock(return_value=[])):
            self.assertEqual(await worker_module.deliver_pending_approvals(), 0)

    async def test_delivers_over_per_org_slack_and_marks_delivered(self):
        p = self._patches(slack_dest="https://hooks.slack.com/org1")
        with (
            patch(
                "alerts_svc.worker.fetch_undelivered_approvals",
                AsyncMock(return_value=[dict(_APPROVAL)]),
            ),
            p["settings"],
            p["resolve"],
            p["mark"] as mock_mark,
            p["send_slack"] as mock_slack,
            p["send_webhook"] as mock_wh,
        ):
            handled = await worker_module.deliver_pending_approvals()

        self.assertEqual(handled, 1)
        mock_slack.assert_called_once()
        self.assertEqual(mock_slack.call_args.kwargs["webhook_url"], "https://hooks.slack.com/org1")
        mock_wh.assert_not_called()
        mock_mark.assert_awaited_once_with(42)

    async def test_delivers_over_webhook_when_enabled(self):
        p = self._patches(webhook_enabled=True)
        with (
            patch(
                "alerts_svc.worker.fetch_undelivered_approvals",
                AsyncMock(return_value=[dict(_APPROVAL)]),
            ),
            p["settings"],
            p["resolve"],
            p["mark"] as mock_mark,
            p["send_slack"] as mock_slack,
            p["send_webhook"] as mock_wh,
        ):
            await worker_module.deliver_pending_approvals()

        mock_wh.assert_called_once()
        mock_slack.assert_not_called()
        mock_mark.assert_awaited_once_with(42)

    async def test_no_channel_still_marks_delivered_to_avoid_hot_loop(self):
        p = self._patches()  # slack + webhook both disabled, no per-org slack
        with (
            patch(
                "alerts_svc.worker.fetch_undelivered_approvals",
                AsyncMock(return_value=[dict(_APPROVAL)]),
            ),
            p["settings"],
            p["resolve"],
            p["mark"] as mock_mark,
            p["send_slack"] as mock_slack,
            p["send_webhook"] as mock_wh,
        ):
            handled = await worker_module.deliver_pending_approvals()

        self.assertEqual(handled, 1)
        mock_slack.assert_not_called()
        mock_wh.assert_not_called()
        mock_mark.assert_awaited_once_with(42)  # marked so it isn't re-fetched forever


if __name__ == "__main__":
    unittest.main(verbosity=2)
