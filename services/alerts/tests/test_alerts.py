"""
tests/test_alerts.py

Alert service tests — formatters, sender retry, worker pipeline.
Zero external deps. No DB, no real HTTP calls.

Run:
    cd services/alerts
    python -m unittest tests.test_alerts -v
"""
from __future__ import annotations

import json
import sys
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dunetrace.models import FailureSignal, FailureType, Severity, Explanation, CodeFix
from alerts_svc.explainer import explain

from alerts_svc.formatters.slack   import format_slack, format_slack_simple
from alerts_svc.formatters.webhook import format_webhook, sign_payload, build_signed_request
from alerts_svc.sender  import SendResult, send_with_retry
from alerts_svc.config  import SEVERITY_ORDER
import alerts_svc.worker as worker_module
# ── Factories ──────────────────────────────────────────────────────────────────

def make_signal(
    failure_type = FailureType.TOOL_LOOP,
    severity     = Severity.HIGH,
    confidence   = 0.95,
    evidence     = None,
) -> FailureSignal:
    return FailureSignal(
        failure_type  = failure_type,
        severity      = severity,
        run_id        = "run-abc123",
        agent_id      = "agent-test",
        agent_version = "abc12345",
        step_index    = 5,
        confidence    = confidence,
        evidence      = evidence or {"tool": "web_search", "count": 5, "window": 5},
        detected_at   = time.time(),
    )


def make_explanation(
    failure_type = "TOOL_LOOP",
    severity     = "HIGH",
    confidence   = 0.95,
) -> Explanation:
    return Explanation(
        failure_type    = failure_type,
        severity        = severity,
        run_id          = "run-abc123",
        agent_id        = "agent-test",
        agent_version   = "abc12345",
        confidence      = confidence,
        step_index      = 5,
        detected_at     = time.time(),
        evidence        = {"tool": "web_search", "count": 5, "window": 5},
        title           = "Tool loop detected: `web_search` called 5× in 5 steps",
        what            = "The agent called web_search 5 times without progress.",
        why_it_matters  = "Loops burn tokens and delay the user.",
        evidence_summary= "web_search called 5 times. Confidence: 95%.",
        suggested_fixes = [
            CodeFix(
                description="Add a per-tool call limit",
                language="python",
                code="MAX_CALLS = 3\nif count > MAX_CALLS:\n    raise RuntimeError('too many calls')",
            ),
            CodeFix(
                description="Add a step limit",
                language="python",
                code="if step > 15: return early_exit()",
            ),
        ],
    )


# ── Slack formatter ────────────────────────────────────────────────────────────

class TestSlackFormatter(unittest.TestCase):

    def setUp(self):
        self.exp = make_explanation()

    def test_returns_dict(self):
        payload = format_slack(self.exp)
        self.assertIsInstance(payload, dict)

    def test_has_attachments(self):
        payload = format_slack(self.exp)
        self.assertIn("attachments", payload)
        self.assertEqual(len(payload["attachments"]), 1)

    def test_attachment_has_color(self):
        attachment = format_slack(self.exp)["attachments"][0]
        self.assertIn("color", attachment)
        self.assertRegex(attachment["color"], r"^#[0-9A-F]{6}$")

    def test_critical_is_red(self):
        exp = make_explanation(severity="CRITICAL")
        color = format_slack(exp)["attachments"][0]["color"]
        self.assertEqual(color, "#FF0000")

    def test_high_is_orange(self):
        color = format_slack(make_explanation(severity="HIGH"))["attachments"][0]["color"]
        self.assertEqual(color, "#FF6B00")

    def test_medium_is_amber(self):
        color = format_slack(make_explanation(severity="MEDIUM"))["attachments"][0]["color"]
        self.assertEqual(color, "#FFB800")

    def test_low_is_green(self):
        color = format_slack(make_explanation(severity="LOW"))["attachments"][0]["color"]
        self.assertEqual(color, "#36A64F")

    def test_blocks_present(self):
        blocks = format_slack(self.exp)["attachments"][0]["blocks"]
        self.assertIsInstance(blocks, list)
        self.assertGreater(len(blocks), 3)

    def test_title_in_header_block(self):
        blocks = format_slack(self.exp)["attachments"][0]["blocks"]
        header = blocks[0]
        self.assertEqual(header["type"], "header")
        self.assertIn(self.exp.title, header["text"]["text"])

    def test_run_id_in_context(self):
        blocks  = format_slack(self.exp)["attachments"][0]["blocks"]
        context = blocks[1]
        self.assertEqual(context["type"], "context")
        text = context["elements"][0]["text"]
        self.assertIn("run-abc123", text)

    def test_what_in_blocks(self):
        payload = json.dumps(format_slack(self.exp))
        self.assertIn("What happened", payload)

    def test_why_in_blocks(self):
        payload = json.dumps(format_slack(self.exp))
        self.assertIn("Why it matters", payload)

    def test_evidence_in_blocks(self):
        payload = json.dumps(format_slack(self.exp))
        self.assertIn("Evidence", payload)

    def test_top_fix_included(self):
        payload = json.dumps(format_slack(self.exp))
        self.assertIn("Add a per-tool call limit", payload)

    def test_view_run_button_present(self):
        blocks  = format_slack(self.exp)["attachments"][0]["blocks"]
        actions = [b for b in blocks if b["type"] == "actions"]
        self.assertGreater(len(actions), 0)
        buttons = [e for e in actions[0]["elements"] if e["type"] == "button"]
        self.assertGreater(len(buttons), 0)

    def test_is_json_serialisable(self):
        payload = format_slack(self.exp)
        serialised = json.dumps(payload)
        self.assertIsInstance(serialised, str)

    def test_simple_format_has_color(self):
        payload = format_slack_simple(self.exp)
        self.assertIn("color", payload["attachments"][0])

    def test_simple_format_has_text(self):
        payload = format_slack_simple(self.exp)
        self.assertIn("text", payload["attachments"][0])
        self.assertIn(self.exp.title, payload["attachments"][0]["text"])


# ── Webhook formatter ──────────────────────────────────────────────────────────

class TestWebhookFormatter(unittest.TestCase):

    def setUp(self):
        self.exp = make_explanation()

    def test_returns_dict(self):
        self.assertIsInstance(format_webhook(self.exp), dict)

    def test_schema_version_present(self):
        self.assertEqual(format_webhook(self.exp)["schema_version"], "1.0")

    def test_event_field(self):
        self.assertEqual(format_webhook(self.exp)["event"], "failure_signal")

    def test_identity_fields_present(self):
        p = format_webhook(self.exp)
        self.assertEqual(p["failure_type"],  "TOOL_LOOP")
        self.assertEqual(p["severity"],      "HIGH")
        self.assertEqual(p["run_id"],        "run-abc123")
        self.assertEqual(p["agent_id"],      "agent-test")
        self.assertEqual(p["agent_version"], "abc12345")

    def test_explanation_fields_present(self):
        p = format_webhook(self.exp)
        self.assertIn("title",            p)
        self.assertIn("what",             p)
        self.assertIn("why_it_matters",   p)
        self.assertIn("evidence_summary", p)

    def test_suggested_fixes_is_list(self):
        p = format_webhook(self.exp)
        self.assertIsInstance(p["suggested_fixes"], list)
        self.assertGreater(len(p["suggested_fixes"]), 0)

    def test_each_fix_has_required_fields(self):
        for fix in format_webhook(self.exp)["suggested_fixes"]:
            self.assertIn("description", fix)
            self.assertIn("language",    fix)
            self.assertIn("code",        fix)

    def test_evidence_passthrough(self):
        evidence = {"tool": "web_search", "count": 5, "window": 5}
        exp = make_explanation()
        exp.evidence = evidence
        self.assertEqual(format_webhook(exp)["evidence"], evidence)

    def test_is_json_serialisable(self):
        serialised = json.dumps(format_webhook(self.exp))
        self.assertIsInstance(serialised, str)

    def test_sign_payload_produces_hex_string(self):
        body = b'{"test": "data"}'
        sig  = sign_payload(body, secret="mysecret")
        self.assertRegex(sig, r"^[0-9a-f]{64}$")

    def test_sign_payload_is_deterministic(self):
        body = b'{"test": "data"}'
        sig1 = sign_payload(body, "secret")
        sig2 = sign_payload(body, "secret")
        self.assertEqual(sig1, sig2)

    def test_sign_payload_different_secret_different_sig(self):
        body = b'test'
        self.assertNotEqual(sign_payload(body, "secret1"), sign_payload(body, "secret2"))

    def test_sign_payload_empty_secret_returns_empty_string(self):
        self.assertEqual(sign_payload(b"body", ""), "")

    def test_build_signed_request_returns_bytes_and_headers(self):
        body, headers = build_signed_request(self.exp, secret="s3cr3t")
        self.assertIsInstance(body, bytes)
        self.assertIsInstance(headers, dict)

    def test_build_signed_request_content_type(self):
        _, headers = build_signed_request(self.exp)
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_build_signed_request_signature_header_present_when_secret_set(self):
        _, headers = build_signed_request(self.exp, secret="s3cr3t")
        self.assertIn("X-Dunetrace-Signature", headers)

    def test_build_signed_request_no_signature_header_without_secret(self):
        _, headers = build_signed_request(self.exp, secret="")
        self.assertNotIn("X-Dunetrace-Signature", headers)

    def test_signature_matches_body(self):
        body, headers = build_signed_request(self.exp, secret="abc")
        expected = sign_payload(body, "abc")
        self.assertEqual(headers["X-Dunetrace-Signature"], expected)


# ── Sender retry logic ─────────────────────────────────────────────────────────

class TestSenderRetry(unittest.TestCase):

    def test_success_on_first_attempt(self):
        with patch("alerts_svc.sender._post", return_value=(200, "ok")):
            result = send_with_retry("http://test", b"body", {}, "test",
                                     max_retries=2, retry_backoff=0)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.status_code, 200)

    def test_retries_on_non_2xx(self):
        responses = [(500, "err"), (500, "err"), (200, "ok")]
        call_count = [0]
        def mock_post(url, body, headers):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with patch("alerts_svc.sender._post", side_effect=mock_post), \
             patch("alerts_svc.sender.time.sleep"):
            result = send_with_retry("http://test", b"body", {}, "test",
                                     max_retries=3, retry_backoff=0.01)

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 3)

    def test_fails_after_max_retries(self):
        with patch("alerts_svc.sender._post", return_value=(500, "err")), \
             patch("alerts_svc.sender.time.sleep"):
            result = send_with_retry("http://test", b"body", {}, "test",
                                     max_retries=2, retry_backoff=0.01)

        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 3)  # 1 initial + 2 retries

    def test_handles_url_error(self):
        import urllib.error
        with patch("alerts_svc.sender._post",
                   side_effect=urllib.error.URLError("connection refused")), \
             patch("alerts_svc.sender.time.sleep"):
            result = send_with_retry("http://test", b"body", {}, "test",
                                     max_retries=1, retry_backoff=0.01)

        self.assertFalse(result.success)
        self.assertIn("connection refused", result.error)

    def test_success_result_has_no_error(self):
        with patch("alerts_svc.sender._post", return_value=(200, "ok")):
            result = send_with_retry("http://test", b"body", {}, "test",
                                     max_retries=1, retry_backoff=0)
        self.assertIsNone(result.error)

    def test_failure_result_has_error_string(self):
        with patch("alerts_svc.sender._post", return_value=(503, "unavailable")), \
             patch("alerts_svc.sender.time.sleep"):
            result = send_with_retry("http://test", b"body", {}, "test",
                                     max_retries=0, retry_backoff=0)
        self.assertIsNotNone(result.error)
        self.assertIn("503", result.error)


# ── Severity threshold ─────────────────────────────────────────────────────────

class TestSeverityThreshold(unittest.TestCase):

    def test_severity_order_values(self):
        self.assertLess(SEVERITY_ORDER["LOW"],      SEVERITY_ORDER["MEDIUM"])
        self.assertLess(SEVERITY_ORDER["MEDIUM"],   SEVERITY_ORDER["HIGH"])
        self.assertLess(SEVERITY_ORDER["HIGH"],     SEVERITY_ORDER["CRITICAL"])

    def test_high_meets_high_threshold(self):
        from alerts_svc.config import settings as s, SEVERITY_ORDER
        original = s.SLACK_MIN_SEVERITY
        s.SLACK_MIN_SEVERITY = "HIGH"
        meets = SEVERITY_ORDER.get("HIGH", 0) >= SEVERITY_ORDER.get("HIGH", 2)
        self.assertTrue(meets)
        s.SLACK_MIN_SEVERITY = original

    def test_low_does_not_meet_high_threshold(self):
        meets = SEVERITY_ORDER.get("LOW", 0) >= SEVERITY_ORDER.get("HIGH", 2)
        self.assertFalse(meets)

    def test_critical_meets_all_thresholds(self):
        for threshold in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            meets = SEVERITY_ORDER.get("CRITICAL", 0) >= SEVERITY_ORDER.get(threshold, 0)
            self.assertTrue(meets, f"CRITICAL should meet {threshold} threshold")


# ── Worker pipeline ────────────────────────────────────────────────────────────

class TestWorkerRowToSignal(unittest.TestCase):

    def test_reconstructs_signal_from_row(self):
        row = {
            "id":            1,
            "failure_type":  "TOOL_LOOP",
            "severity":      "HIGH",
            "run_id":        "run-1",
            "agent_id":      "agent-1",
            "agent_version": "v1",
            "step_index":    5,
            "confidence":    0.95,
            "evidence":      {"tool": "web_search", "count": 5, "window": 5},
            "detected_at":   time.time(),
        }
        signal = worker_module._row_to_signal(row)
        self.assertEqual(signal.failure_type, FailureType.TOOL_LOOP)
        self.assertEqual(signal.severity,     Severity.HIGH)
        self.assertEqual(signal.run_id,       "run-1")
        self.assertAlmostEqual(signal.confidence, 0.95)

    def test_handles_datetime_detected_at(self):
        """asyncpg returns datetime objects for TIMESTAMPTZ columns."""
        from datetime import datetime, timezone
        row = {
            "id": 1, "failure_type": "TOOL_LOOP", "severity": "HIGH",
            "run_id": "r", "agent_id": "a", "agent_version": "v",
            "step_index": 1, "confidence": 0.9, "evidence": {},
            "detected_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        signal = worker_module._row_to_signal(row)
        self.assertIsInstance(signal.detected_at, float)

    def test_handles_none_detected_at(self):
        row = {
            "id": 1, "failure_type": "TOOL_LOOP", "severity": "HIGH",
            "run_id": "r", "agent_id": "a", "agent_version": "v",
            "step_index": 1, "confidence": 0.9, "evidence": {},
            "detected_at": None,
        }
        signal = worker_module._row_to_signal(row)
        self.assertIsInstance(signal.detected_at, float)


class TestWorkerDeliver(unittest.TestCase):

    def setUp(self):
        self.exp = make_explanation()

    def test_no_destinations_returns_empty_dict(self):
        with patch("alerts_svc.worker.settings") as mock_settings:
            mock_settings.slack_enabled   = False
            mock_settings.webhook_enabled = False
            mock_settings.SLACK_MIN_SEVERITY = "HIGH"
            results = worker_module.deliver(self.exp)
        self.assertEqual(results, {})

    def test_slack_called_when_enabled_and_severity_meets_threshold(self):
        with patch("alerts_svc.worker.settings") as mock_s, \
             patch("alerts_svc.worker.send_slack",
                   return_value=SendResult(True, "slack", 1, 200)) as mock_slack, \
             patch("alerts_svc.worker.send_webhook",
                   return_value=SendResult(True, "webhook", 1, 200)):
            mock_s.slack_enabled      = True
            mock_s.webhook_enabled    = False
            mock_s.SLACK_MIN_SEVERITY = "HIGH"
            results = worker_module.deliver(self.exp)

        mock_slack.assert_called_once()
        self.assertIn("slack", results)

    def test_slack_skipped_when_severity_too_low(self):
        low_exp = make_explanation(severity="LOW")
        with patch("alerts_svc.worker.settings") as mock_s, \
             patch("alerts_svc.worker.send_slack") as mock_slack:
            mock_s.slack_enabled      = True
            mock_s.webhook_enabled    = False
            mock_s.SLACK_MIN_SEVERITY = "HIGH"
            worker_module.deliver(low_exp)

        mock_slack.assert_not_called()

    def test_webhook_called_when_enabled(self):
        with patch("alerts_svc.worker.settings") as mock_s, \
             patch("alerts_svc.worker.send_webhook",
                   return_value=SendResult(True, "webhook", 1, 200)) as mock_wh, \
             patch("alerts_svc.worker.build_signed_request", return_value=(b"{}", {})):
            mock_s.slack_enabled   = False
            mock_s.webhook_enabled = True
            mock_s.WEBHOOK_SECRET  = ""
            results = worker_module.deliver(self.exp)

        mock_wh.assert_called_once()
        self.assertIn("webhook", results)


class TestWorkerPollOnce(unittest.IsolatedAsyncioTestCase):

    async def test_poll_once_empty_returns_zeros(self):
        with patch("alerts_svc.worker.fetch_unalerted_signals", AsyncMock(return_value=[])):
            found, delivered = await worker_module.poll_once()
        self.assertEqual(found, 0)
        self.assertEqual(delivered, 0)

    async def test_poll_once_processes_signals(self):
        rows = [{
            "id":            42,
            "failure_type":  "TOOL_LOOP",
            "severity":      "HIGH",
            "run_id":        "run-1",
            "agent_id":      "agent-1",
            "agent_version": "v1",
            "step_index":    5,
            "confidence":    0.95,
            "evidence":      {"tool": "web_search", "count": 5, "window": 5},
            "detected_at":   time.time(),
        }]

        with patch("alerts_svc.worker.fetch_unalerted_signals",  AsyncMock(return_value=rows)), \
             patch("alerts_svc.worker.mark_alerted_batch",       AsyncMock()) as mock_mark, \
             patch("alerts_svc.worker.deliver",
                   return_value={"slack": SendResult(True, "slack", 1, 200)}):
            found, delivered = await worker_module.poll_once()

        self.assertEqual(found,     1)
        self.assertEqual(delivered, 1)
        mock_mark.assert_called_once_with([42])

    async def test_failed_delivery_not_marked_alerted(self):
        rows = [{
            "id":            99,
            "failure_type":  "TOOL_LOOP",
            "severity":      "HIGH",
            "run_id":        "r",
            "agent_id":      "a",
            "agent_version": "v",
            "step_index":    1,
            "confidence":    0.9,
            "evidence":      {"tool": "x", "count": 3, "window": 3},
            "detected_at":   time.time(),
        }]

        with patch("alerts_svc.worker.fetch_unalerted_signals",  AsyncMock(return_value=rows)), \
             patch("alerts_svc.worker.mark_alerted_batch",       AsyncMock()) as mock_mark, \
             patch("alerts_svc.worker.deliver",
                   return_value={"slack": SendResult(False, "slack", 3, 503, "err")}):
            found, delivered = await worker_module.poll_once()

        self.assertEqual(found,     1)
        self.assertEqual(delivered, 0)
        mock_mark.assert_not_called()

    async def test_no_destinations_marks_signal_done(self):
        """If no destinations configured, signal is still marked alerted."""
        rows = [{
            "id":            7,
            "failure_type":  "TOOL_LOOP",
            "severity":      "HIGH",
            "run_id":        "r",
            "agent_id":      "a",
            "agent_version": "v",
            "step_index":    1,
            "confidence":    0.9,
            "evidence":      {"tool": "x", "count": 3, "window": 3},
            "detected_at":   time.time(),
        }]

        with patch("alerts_svc.worker.fetch_unalerted_signals", AsyncMock(return_value=rows)), \
             patch("alerts_svc.worker.mark_alerted_batch",      AsyncMock()) as mock_mark, \
             patch("alerts_svc.worker.deliver",                 return_value={}):
            found, delivered = await worker_module.poll_once()

        self.assertEqual(delivered, 1)
        mock_mark.assert_called_once_with([7])


if __name__ == "__main__":
    unittest.main(verbosity=2)
