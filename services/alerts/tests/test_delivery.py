"""
Tests for alert delivery: deliver(), send_with_retry, Slack/webhook send paths.
No DB, no real HTTP calls.

Run:
    cd services/alerts
    PYTHONPATH=packages/sdk-py:services/explainer:services/alerts \
        python -m pytest tests/test_delivery.py -v
"""

from __future__ import annotations

import sys
import os
import time
import unittest
from unittest.mock import patch, MagicMock, call

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explainer_svc.models import Explanation, CodeFix
from alerts_svc.sender import SendResult, send_with_retry
import alerts_svc.worker as worker_module


# ── Factories ──────────────────────────────────────────────────────────────────


def make_explanation(
    failure_type: str = "TOOL_LOOP",
    severity: str = "HIGH",
    confidence: float = 0.9,
) -> Explanation:
    return Explanation(
        failure_type=failure_type,
        severity=severity,
        run_id="run-del-001",
        agent_id="test-agent",
        agent_version="abc00001",
        confidence=confidence,
        step_index=3,
        detected_at=time.time(),
        evidence={"tool": "search", "count": 5},
        title="Tool loop detected: `search` called 5×",
        what="The agent called search repeatedly without progress.",
        why_it_matters="Loops waste tokens.",
        evidence_summary="search called 5 times. Confidence: 90%.",
        suggested_fixes=[
            CodeFix(
                description="Limit tool calls",
                language="python",
                code="if count > 3: break",
            )
        ],
    )


# ── deliver() — Slack ──────────────────────────────────────────────────────────


class TestDeliverSlack(unittest.TestCase):
    """deliver() sends to Slack when slack_enabled and severity meets threshold."""

    def setUp(self):
        self.exp = make_explanation()

    def test_successful_slack_delivery_returns_success(self):
        """When Slack returns 200, deliver() includes a success SendResult."""
        with (
            patch("alerts_svc.worker.settings") as ms,
            patch(
                "alerts_svc.worker.send_slack",
                return_value=SendResult(True, "slack", 1, 200),
            ) as mock_slack,
        ):
            ms.slack_enabled = True
            ms.webhook_enabled = False
            ms.SLACK_MIN_SEVERITY = "LOW"
            ms.ALERT_DEDUP_WINDOW = 0
            results = worker_module.deliver(self.exp, signal_id=1)

        self.assertIn("slack", results)
        self.assertTrue(results["slack"].success)
        mock_slack.assert_called_once()

    def test_slack_payload_contains_explanation(self):
        """The payload passed to send_slack must include the formatted explanation."""
        captured_payload = []
        with (
            patch("alerts_svc.worker.settings") as ms,
            patch(
                "alerts_svc.worker.send_slack",
                side_effect=lambda p: (
                    captured_payload.append(p) or SendResult(True, "slack", 1, 200)
                ),
            ),
            patch(
                "alerts_svc.worker.format_slack",
                return_value={"attachments": [{"text": "alert"}]},
            ) as mock_fmt,
        ):
            ms.slack_enabled = True
            ms.webhook_enabled = False
            ms.SLACK_MIN_SEVERITY = "LOW"
            ms.ALERT_DEDUP_WINDOW = 0
            worker_module.deliver(self.exp, signal_id=5)

        mock_fmt.assert_called_once()
        self.assertEqual(len(captured_payload), 1)

    def test_slack_skipped_when_severity_below_threshold(self):
        """LOW explanation should not be sent when SLACK_MIN_SEVERITY=HIGH."""
        low_exp = make_explanation(severity="LOW")
        with (
            patch("alerts_svc.worker.settings") as ms,
            patch("alerts_svc.worker.send_slack") as mock_slack,
        ):
            ms.slack_enabled = True
            ms.webhook_enabled = False
            ms.SLACK_MIN_SEVERITY = "HIGH"
            results = worker_module.deliver(low_exp)

        mock_slack.assert_not_called()
        self.assertNotIn("slack", results)

    def test_deliver_returns_empty_dict_when_no_destinations(self):
        """No Slack, no webhook → deliver() returns {}."""
        with patch("alerts_svc.worker.settings") as ms:
            ms.slack_enabled = False
            ms.webhook_enabled = False
            results = worker_module.deliver(self.exp)
        self.assertEqual(results, {})


# ── deliver() — Webhook ────────────────────────────────────────────────────────


class TestDeliverWebhook(unittest.TestCase):
    """deliver() sends to the generic webhook when webhook_enabled."""

    def setUp(self):
        self.exp = make_explanation()

    def test_successful_webhook_delivery_returns_success(self):
        """When the webhook returns 200, deliver() includes a success SendResult."""
        with (
            patch("alerts_svc.worker.settings") as ms,
            patch(
                "alerts_svc.worker.send_webhook",
                return_value=SendResult(True, "webhook", 1, 200),
            ) as mock_wh,
            patch("alerts_svc.worker.build_signed_request", return_value=(b"{}", {})),
        ):
            ms.slack_enabled = False
            ms.webhook_enabled = True
            ms.WEBHOOK_SECRET = ""
            results = worker_module.deliver(self.exp, signal_id=2)

        self.assertIn("webhook", results)
        self.assertTrue(results["webhook"].success)
        mock_wh.assert_called_once()

    def test_webhook_receives_signed_body_and_headers(self):
        """build_signed_request output is forwarded to send_webhook as-is."""
        captured = []
        body = b'{"event": "failure_signal"}'
        headers = {"Content-Type": "application/json", "X-Dunetrace-Signature": "abc"}

        with (
            patch("alerts_svc.worker.settings") as ms,
            patch("alerts_svc.worker.build_signed_request", return_value=(body, headers)),
            patch(
                "alerts_svc.worker.send_webhook",
                side_effect=lambda b, h: (
                    captured.append((b, h)) or SendResult(True, "webhook", 1, 200)
                ),
            ),
        ):
            ms.slack_enabled = False
            ms.webhook_enabled = True
            ms.WEBHOOK_SECRET = "s3cr3t"
            worker_module.deliver(self.exp)

        self.assertEqual(len(captured), 1)
        sent_body, sent_headers = captured[0]
        self.assertEqual(sent_body, body)
        self.assertIn("X-Dunetrace-Signature", sent_headers)

    def test_both_slack_and_webhook_sent_when_both_enabled(self):
        """When both destinations are enabled, both get a delivery attempt."""
        with (
            patch("alerts_svc.worker.settings") as ms,
            patch(
                "alerts_svc.worker.send_slack",
                return_value=SendResult(True, "slack", 1, 200),
            ) as mock_slack,
            patch(
                "alerts_svc.worker.send_webhook",
                return_value=SendResult(True, "webhook", 1, 200),
            ) as mock_wh,
            patch("alerts_svc.worker.build_signed_request", return_value=(b"{}", {})),
        ):
            ms.slack_enabled = True
            ms.webhook_enabled = True
            ms.SLACK_MIN_SEVERITY = "LOW"
            ms.WEBHOOK_SECRET = ""
            ms.ALERT_DEDUP_WINDOW = 0
            results = worker_module.deliver(self.exp)

        mock_slack.assert_called_once()
        mock_wh.assert_called_once()
        self.assertIn("slack", results)
        self.assertIn("webhook", results)


# ── send_with_retry — retry logic ─────────────────────────────────────────────


class TestSendWithRetryRetries(unittest.TestCase):
    """Retry-on-transient-failure and exponential-backoff behaviour."""

    def test_retry_on_transient_500(self):
        """First attempt returns 500, second returns 200 — function succeeds."""
        responses = [(500, "err"), (200, "ok")]
        call_count = [0]

        def mock_post(url, body, headers):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with (
            patch("alerts_svc.sender._post", side_effect=mock_post),
            patch("alerts_svc.sender.time.sleep"),
        ):
            result = send_with_retry(
                "http://test", b"body", {}, "slack", max_retries=2, retry_backoff=0.01
            )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)

    def test_exponential_backoff_sleep_calls(self):
        """Backoff delays double on each retry: backoff, 2*backoff, 4*backoff, ..."""
        sleep_calls = []
        responses = [(500, "err"), (500, "err"), (200, "ok")]
        call_count = [0]

        def mock_post(url, body, headers):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with (
            patch("alerts_svc.sender._post", side_effect=mock_post),
            patch("alerts_svc.sender.time.sleep", side_effect=lambda d: sleep_calls.append(d)),
        ):
            send_with_retry("http://test", b"body", {}, "slack", max_retries=3, retry_backoff=1.0)

        # Two retries → two sleep calls; each delay doubles
        self.assertEqual(len(sleep_calls), 2)
        self.assertLess(sleep_calls[0], sleep_calls[1])

    def test_permanent_failure_after_max_retries(self):
        """Exhausting all retries returns a failed SendResult with error string."""
        with (
            patch("alerts_svc.sender._post", return_value=(503, "unavailable")),
            patch("alerts_svc.sender.time.sleep"),
        ):
            result = send_with_retry(
                "http://test", b"body", {}, "slack", max_retries=2, retry_backoff=0.01
            )

        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 3)  # 1 initial + 2 retries
        self.assertIsNotNone(result.error)
        self.assertIn("503", result.error)

    def test_signal_not_marked_alerted_when_all_destinations_fail(self):
        """A failed delivery should return None from _deliver_one (not mark alerted)."""
        # This is a unit test on the retry result shape; the poll_once integration
        # is covered in test_alerts.py::TestWorkerPollOnce::test_failed_delivery_not_marked_alerted
        with (
            patch("alerts_svc.sender._post", return_value=(500, "err")),
            patch("alerts_svc.sender.time.sleep"),
        ):
            result = send_with_retry(
                "http://test", b"body", {}, "dest", max_retries=1, retry_backoff=0
            )

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_network_error_treated_as_transient(self):
        """URLError is retried; after max_retries it returns a failure."""
        import urllib.error

        with (
            patch(
                "alerts_svc.sender._post",
                side_effect=urllib.error.URLError("network unreachable"),
            ),
            patch("alerts_svc.sender.time.sleep"),
        ):
            result = send_with_retry(
                "http://test", b"body", {}, "slack", max_retries=1, retry_backoff=0.01
            )

        self.assertFalse(result.success)
        self.assertIn("network unreachable", result.error)

    def test_success_on_first_attempt_no_sleep(self):
        """When the first attempt succeeds, time.sleep should never be called."""
        with (
            patch("alerts_svc.sender._post", return_value=(200, "ok")),
            patch("alerts_svc.sender.time.sleep") as mock_sleep,
        ):
            result = send_with_retry(
                "http://test", b"body", {}, "slack", max_retries=3, retry_backoff=1.0
            )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)
        mock_sleep.assert_not_called()

    def test_correct_attempt_count_on_success_after_retry(self):
        """Attempt counter must reflect total attempts including retries."""
        responses = [(500, "err"), (500, "err"), (200, "ok")]
        idx = [0]

        def mock_post(*_):
            r = responses[idx[0]]
            idx[0] += 1
            return r

        with (
            patch("alerts_svc.sender._post", side_effect=mock_post),
            patch("alerts_svc.sender.time.sleep"),
        ):
            result = send_with_retry(
                "http://test", b"body", {}, "dest", max_retries=5, retry_backoff=0.01
            )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 3)

    def test_success_result_has_no_error(self):
        """A successful SendResult must have error=None."""
        with patch("alerts_svc.sender._post", return_value=(200, "ok")):
            result = send_with_retry(
                "http://test", b"body", {}, "slack", max_retries=0, retry_backoff=0
            )
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_failure_result_carries_status_code(self):
        """A failed SendResult should include the HTTP status code."""
        with (
            patch("alerts_svc.sender._post", return_value=(502, "bad gateway")),
            patch("alerts_svc.sender.time.sleep"),
        ):
            result = send_with_retry(
                "http://test", b"body", {}, "dest", max_retries=0, retry_backoff=0
            )
        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 502)


if __name__ == "__main__":
    unittest.main(verbosity=2)
