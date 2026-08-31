"""
Tests for the weekly digest: should_send_digest(), format_digest_slack(),
send_weekly_digest() with all DB/HTTP mocked.

Run:
    cd services/alerts
    python -m pytest tests/test_digest.py -v
"""

from __future__ import annotations

import json
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alerts_svc.digest import (
    should_send_digest,
    format_digest_slack,
    send_weekly_digest,
)
from alerts_svc.sender import SendResult

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _sample_data(
    total_runs=100,
    total_agents=3,
    total_signals=20,
    top_failures=None,
    top_agents=None,
    systemic=None,
    issues_opened=3,
    issues_resolved=1,
):
    return {
        "total_runs": total_runs,
        "total_agents": total_agents,
        "total_signals": total_signals,
        "top_failures": (
            top_failures
            if top_failures is not None
            else [
                {
                    "failure_type": "TOOL_LOOP",
                    "affected_runs": 10,
                    "total_runs": 100,
                    "rate": 0.10,
                },
                {
                    "failure_type": "RETRY_STORM",
                    "affected_runs": 5,
                    "total_runs": 100,
                    "rate": 0.05,
                },
            ]
        ),
        "top_agents": (
            top_agents
            if top_agents is not None
            else [
                {
                    "agent_id": "research-agent",
                    "signal_count": 12,
                    "run_count": 50,
                    "dominant_failure": "TOOL_LOOP",
                },
                {
                    "agent_id": "code-agent",
                    "signal_count": 8,
                    "run_count": 50,
                    "dominant_failure": "RETRY_STORM",
                },
            ]
        ),
        "systemic": (
            systemic
            if systemic is not None
            else [
                {
                    "agent_id": "research-agent",
                    "failure_type": "TOOL_LOOP",
                    "total_runs": 50,
                    "affected_runs": 10,
                    "rate": 0.20,
                },
            ]
        ),
        "issues_opened": issues_opened,
        "issues_resolved": issues_resolved,
    }


# ── should_send_digest ─────────────────────────────────────────────────────────


class TestShouldSendDigest(unittest.TestCase):
    def _frozen(self, weekday: int, hour: int):
        """Return a UTC datetime with the given weekday (0=Mon) and hour."""
        # Find next date with desired weekday
        base = datetime.datetime(2026, 4, 13, hour, 0, 0, tzinfo=datetime.timezone.utc)  # Monday
        delta = (weekday - base.weekday()) % 7
        return base + datetime.timedelta(days=delta)

    def test_returns_true_on_correct_day_and_hour(self):
        from alerts_svc import digest as digest_mod
        from alerts_svc import config as cfg_mod

        now = self._frozen(cfg_mod.settings.DIGEST_DAY, cfg_mod.settings.DIGEST_HOUR)
        with patch("alerts_svc.digest.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = now
            mock_dt.timezone.utc = datetime.timezone.utc
            mock_dt.timedelta = datetime.timedelta
            result = should_send_digest()
        self.assertTrue(result)

    def test_returns_false_on_wrong_day(self):
        from alerts_svc import config as cfg_mod

        wrong_day = (cfg_mod.settings.DIGEST_DAY + 1) % 7
        now = self._frozen(wrong_day, cfg_mod.settings.DIGEST_HOUR)
        with patch("alerts_svc.digest.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = now
            mock_dt.timezone.utc = datetime.timezone.utc
            mock_dt.timedelta = datetime.timedelta
            result = should_send_digest()
        self.assertFalse(result)

    def test_returns_false_on_wrong_hour(self):
        from alerts_svc import config as cfg_mod

        wrong_hour = (cfg_mod.settings.DIGEST_HOUR + 1) % 24
        now = self._frozen(cfg_mod.settings.DIGEST_DAY, wrong_hour)
        with patch("alerts_svc.digest.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = now
            mock_dt.timezone.utc = datetime.timezone.utc
            mock_dt.timedelta = datetime.timedelta
            result = should_send_digest()
        self.assertFalse(result)


# ── format_digest_slack ────────────────────────────────────────────────────────


class TestFormatDigestSlack(unittest.TestCase):
    def _blocks(self, data=None, org_id="org-1"):
        payload = format_digest_slack(data or _sample_data(), org_id)
        return payload["attachments"][0]["blocks"]

    def test_returns_valid_slack_payload(self):
        payload = format_digest_slack(_sample_data(), "org-1")
        self.assertIn("attachments", payload)
        self.assertIsInstance(json.dumps(payload), str)

    def test_has_top_level_notification_text(self):
        """Without a top-level `text`, Slack desktop/mobile notifications
        show "no preview available"."""
        payload = format_digest_slack(_sample_data(), "org-1")
        self.assertTrue(payload.get("text"))
        self.assertIn("Weekly agent health digest", payload["text"])
        self.assertEqual(payload["attachments"][0]["fallback"], payload["text"])

    def test_notification_text_has_no_mrkdwn_markers(self):
        text = format_digest_slack(_sample_data(), "org-1")["text"]
        for ch in "`*~":
            self.assertNotIn(ch, text)

    def test_header_block_contains_digest_label(self):
        blocks = self._blocks()
        header = blocks[0]
        self.assertEqual(header["type"], "header")
        self.assertIn("Digest", header["text"]["text"])

    def test_header_block_contains_org_id(self):
        blocks = self._blocks(org_id="org-acme")
        header = blocks[0]
        self.assertIn("org-acme", header["text"]["text"])

    def test_context_block_shows_totals(self):
        blocks = self._blocks(_sample_data(total_runs=50, total_agents=2, total_signals=10))
        context = blocks[1]
        text = context["elements"][0]["text"]
        self.assertIn("50", text)
        self.assertIn("2", text)
        self.assertIn("10", text)

    def test_top_failure_types_appear_in_blocks(self):
        blocks = self._blocks()
        all_text = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        self.assertIn("TOOL_LOOP", all_text)
        self.assertIn("RETRY_STORM", all_text)

    def test_systemic_pattern_in_blocks(self):
        blocks = self._blocks()
        all_text = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        self.assertIn("Systemic", all_text)
        self.assertIn("research-agent", all_text)

    def test_issue_counts_in_blocks(self):
        blocks = self._blocks(_sample_data(issues_opened=4, issues_resolved=2))
        all_text = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        self.assertIn("4", all_text)
        self.assertIn("2", all_text)

    def test_no_failures_shows_clean_message(self):
        blocks = self._blocks(_sample_data(top_failures=[], top_agents=[], systemic=[]))
        all_text = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        self.assertIn("No failures", all_text)

    def test_dashboard_button_present(self):
        blocks = self._blocks()
        action_blocks = [b for b in blocks if b["type"] == "actions"]
        self.assertGreater(len(action_blocks), 0)
        button = action_blocks[0]["elements"][0]
        self.assertEqual(button["type"], "button")

    def test_pct_computed_correctly(self):
        data = _sample_data(
            top_failures=[
                {
                    "failure_type": "TOOL_LOOP",
                    "affected_runs": 1,
                    "total_runs": 3,
                    "rate": 1 / 3,
                }
            ]
        )
        blocks = self._blocks(data)
        all_text = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        self.assertIn("33%", all_text)

    def test_color_is_blue(self):
        payload = format_digest_slack(_sample_data(), "org-1")
        self.assertEqual(payload["attachments"][0]["color"], "#4a9ede")


# ── send_weekly_digest (fan-out over active orgs) ─────────────────────────────


class TestSendWeeklyDigest(unittest.IsolatedAsyncioTestCase):
    async def test_skips_when_digest_not_enabled(self):
        with (
            patch("alerts_svc.digest.settings") as mock_settings,
            patch("alerts_svc.digest.fetch_active_org_ids", AsyncMock()) as mock_active,
        ):
            mock_settings.digest_enabled = False
            result = await send_weekly_digest()
        self.assertEqual(result, 0)
        mock_active.assert_not_called()

    async def test_skips_when_not_correct_time(self):
        with (
            patch("alerts_svc.digest.settings") as mock_settings,
            patch("alerts_svc.digest.should_send_digest", return_value=False),
            patch("alerts_svc.digest.fetch_active_org_ids", AsyncMock()) as mock_active,
        ):
            mock_settings.digest_enabled = True
            result = await send_weekly_digest()
        self.assertEqual(result, 0)
        mock_active.assert_not_called()

    async def test_no_active_orgs_sends_nothing(self):
        with (
            patch("alerts_svc.digest.settings") as mock_settings,
            patch("alerts_svc.digest.should_send_digest", return_value=True),
            patch("alerts_svc.digest.fetch_active_org_ids", AsyncMock(return_value=[])),
        ):
            mock_settings.digest_enabled = True
            result = await send_weekly_digest()
        self.assertEqual(result, 0)

    async def test_sends_one_digest_per_active_org(self):
        """Two active orgs → two independent sends, each scoped to its own org_id."""
        fetched_for = []

        async def fake_fetch_data(org_id):
            fetched_for.append(org_id)
            return _sample_data()

        with (
            patch("alerts_svc.digest.settings") as mock_settings,
            patch("alerts_svc.digest.should_send_digest", return_value=True),
            patch(
                "alerts_svc.digest.fetch_active_org_ids",
                AsyncMock(return_value=["org-a", "org-b"]),
            ),
            patch(
                "alerts_svc.digest.was_digest_sent_recently",
                AsyncMock(return_value=False),
            ),
            patch("alerts_svc.digest.fetch_weekly_digest_data", side_effect=fake_fetch_data),
            patch(
                "alerts_svc.digest.send_slack",
                return_value=SendResult(True, "slack", 1, 200),
            ),
            patch("alerts_svc.digest.log_digest_sent", AsyncMock()) as mock_log,
        ):
            mock_settings.digest_enabled = True
            mock_settings.DASHBOARD_URL = "https://app.dunetrace.io"
            result = await send_weekly_digest()

        self.assertEqual(result, 2)
        self.assertEqual(set(fetched_for), {"org-a", "org-b"})
        self.assertEqual(mock_log.call_count, 2)

    async def test_one_org_failing_does_not_block_the_other(self):
        """If one org's digest fails, the other org still gets sent."""

        async def fake_fetch_data(org_id):
            if org_id == "org-bad":
                raise Exception("DB down")
            return _sample_data()

        with (
            patch("alerts_svc.digest.settings") as mock_settings,
            patch("alerts_svc.digest.should_send_digest", return_value=True),
            patch(
                "alerts_svc.digest.fetch_active_org_ids",
                AsyncMock(return_value=["org-bad", "org-good"]),
            ),
            patch(
                "alerts_svc.digest.was_digest_sent_recently",
                AsyncMock(return_value=False),
            ),
            patch("alerts_svc.digest.fetch_weekly_digest_data", side_effect=fake_fetch_data),
            patch(
                "alerts_svc.digest.send_slack",
                return_value=SendResult(True, "slack", 1, 200),
            ),
            patch("alerts_svc.digest.log_digest_sent", AsyncMock()) as mock_log,
        ):
            mock_settings.digest_enabled = True
            mock_settings.DASHBOARD_URL = "https://app.dunetrace.io"
            result = await send_weekly_digest()

        self.assertEqual(result, 1)
        mock_log.assert_called_once_with("org-good")


# ── _send_org_digest (single-org behavior) ────────────────────────────────────


class TestSendOrgDigest(unittest.IsolatedAsyncioTestCase):
    async def test_skips_when_already_sent_recently(self):
        from alerts_svc.digest import _send_org_digest

        with (
            patch(
                "alerts_svc.digest.was_digest_sent_recently",
                AsyncMock(return_value=True),
            ),
        ):
            result = await _send_org_digest("org-1")
        self.assertFalse(result)

    async def test_skips_and_logs_when_no_runs(self):
        from alerts_svc.digest import _send_org_digest

        with (
            patch(
                "alerts_svc.digest.was_digest_sent_recently",
                AsyncMock(return_value=False),
            ),
            patch(
                "alerts_svc.digest.fetch_weekly_digest_data",
                AsyncMock(return_value={"total_runs": 0}),
            ),
            patch("alerts_svc.digest.log_digest_sent", AsyncMock()) as mock_log,
        ):
            result = await _send_org_digest("org-1")
        self.assertFalse(result)
        mock_log.assert_called_once_with("org-1")  # still log so we don't retry until next week

    async def test_sends_and_logs_on_success(self):
        from alerts_svc.digest import _send_org_digest

        with (
            patch(
                "alerts_svc.digest.was_digest_sent_recently",
                AsyncMock(return_value=False),
            ),
            patch(
                "alerts_svc.digest.fetch_weekly_digest_data",
                AsyncMock(return_value=_sample_data()),
            ),
            patch(
                "alerts_svc.digest.send_slack",
                return_value=SendResult(True, "slack", 1, 200),
            ),
            patch("alerts_svc.digest.log_digest_sent", AsyncMock()) as mock_log,
        ):
            result = await _send_org_digest("org-1")
        self.assertTrue(result)
        mock_log.assert_called_once_with("org-1")

    async def test_returns_false_on_slack_failure(self):
        from alerts_svc.digest import _send_org_digest

        with (
            patch(
                "alerts_svc.digest.was_digest_sent_recently",
                AsyncMock(return_value=False),
            ),
            patch(
                "alerts_svc.digest.fetch_weekly_digest_data",
                AsyncMock(return_value=_sample_data()),
            ),
            patch(
                "alerts_svc.digest.send_slack",
                return_value=SendResult(False, "slack", 0, 500, error="timeout"),
            ),
            patch("alerts_svc.digest.log_digest_sent", AsyncMock()) as mock_log,
        ):
            result = await _send_org_digest("org-1")
        self.assertFalse(result)
        mock_log.assert_not_called()  # don't log — allow retry next cycle

    async def test_returns_false_on_data_fetch_error(self):
        from alerts_svc.digest import _send_org_digest

        with (
            patch(
                "alerts_svc.digest.was_digest_sent_recently",
                AsyncMock(return_value=False),
            ),
            patch(
                "alerts_svc.digest.fetch_weekly_digest_data",
                AsyncMock(side_effect=Exception("DB down")),
            ),
        ):
            result = await _send_org_digest("org-1")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
