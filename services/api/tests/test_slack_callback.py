"""
Endpoint-level tests for the Slack interactive callback handler
(api_svc/routers/slack.py), focused on Phase 4.1's new "snooze" action
(the pre-existing mark_resolved/false_positive branches had no test
coverage before this — not backfilled here, out of scope for this change).

Calls the route function directly with a minimal fake Request (this
codebase's established pattern), mocked DB calls. No network, no DB.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from api_svc.config import settings
from api_svc.routers.slack import slack_callback


class _FakeRequest:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


def _slack_form_body(action_id: str, value: dict, user_name: str = "alice") -> bytes:
    payload = {
        "actions": [{"action_id": action_id, "value": json.dumps(value)}],
        "user": {"name": user_name},
    }
    return ("payload=" + json.dumps(payload)).encode()


def _btn_value(**overrides):
    fields = {
        "signal_id": 42,
        "agent_id": "support-bot",
        "failure_type": "TOOL_LOOP",
        "org_id": "org-1",
    }
    fields.update(overrides)
    return fields


class TestSlackCallbackSnooze(unittest.IsolatedAsyncioTestCase):
    async def test_snooze_calls_snooze_pattern_with_correct_args(self):
        request = _FakeRequest(_slack_form_body("snooze", _btn_value()))
        with (
            patch("api_svc.routers.slack.settings") as mock_settings,
            patch(
                "api_svc.routers.slack.snooze_pattern",
                AsyncMock(return_value={"snoozed_until": "2026-07-13T00:00:00Z"}),
            ) as mock_snooze,
        ):
            mock_settings.SLACK_SIGNING_SECRET = ""
            response = await slack_callback(request)

        mock_snooze.assert_called_once_with("org-1", "support-bot", "TOOL_LOOP", hours=24)
        self.assertEqual(response.status_code, 200)

    async def test_snooze_response_mentions_failure_type_and_agent(self):
        request = _FakeRequest(
            _slack_form_body(
                "snooze", _btn_value(failure_type="RETRY_STORM", agent_id="billing-bot")
            )
        )
        with (
            patch("api_svc.routers.slack.settings") as mock_settings,
            patch(
                "api_svc.routers.slack.snooze_pattern",
                AsyncMock(return_value={"snoozed_until": None}),
            ),
        ):
            mock_settings.SLACK_SIGNING_SECRET = ""
            response = await slack_callback(request)

        body = json.loads(response.body)
        self.assertIn("RETRY_STORM", body["text"])
        self.assertIn("billing-bot", body["text"])
        self.assertIn("24h", body["text"])

    async def test_missing_required_fields_returns_400(self):
        request = _FakeRequest(
            _slack_form_body(
                "snooze", {"signal_id": 1, "agent_id": "", "failure_type": "", "org_id": ""}
            )
        )
        with patch("api_svc.routers.slack.settings") as mock_settings:
            mock_settings.SLACK_SIGNING_SECRET = ""
            response = await slack_callback(request)

        self.assertEqual(response.status_code, 400)

    async def test_no_actions_in_payload_returns_200_without_calling_db(self):
        payload = {"actions": [], "user": {"name": "alice"}}
        request = _FakeRequest(("payload=" + json.dumps(payload)).encode())
        with (
            patch("api_svc.routers.slack.settings") as mock_settings,
            patch("api_svc.routers.slack.snooze_pattern", AsyncMock()) as mock_snooze,
        ):
            mock_settings.SLACK_SIGNING_SECRET = ""
            response = await slack_callback(request)

        self.assertEqual(response.status_code, 200)
        mock_snooze.assert_not_called()

    async def test_unknown_action_id_returns_200_without_calling_snooze(self):
        request = _FakeRequest(_slack_form_body("some_future_action", _btn_value()))
        with (
            patch("api_svc.routers.slack.settings") as mock_settings,
            patch("api_svc.routers.slack.snooze_pattern", AsyncMock()) as mock_snooze,
        ):
            mock_settings.SLACK_SIGNING_SECRET = ""
            response = await slack_callback(request)

        self.assertEqual(response.status_code, 200)
        mock_snooze.assert_not_called()

    async def test_invalid_signature_returns_403(self):
        request = _FakeRequest(
            _slack_form_body("snooze", _btn_value()),
            headers={"X-Slack-Request-Timestamp": "123", "X-Slack-Signature": "v0=bogus"},
        )
        with patch("api_svc.routers.slack.settings") as mock_settings:
            mock_settings.SLACK_SIGNING_SECRET = "shh"
            response = await slack_callback(request)

        self.assertEqual(response.status_code, 403)


_TEST_SIGNING_SECRET = "test-signing-secret"


def _signed_request(action_id: str, value: dict, user_name: str = "alice") -> _FakeRequest:
    """A callback request signed the way Slack signs one.

    These tests used to send an UNSIGNED request and rely on the callback
    accepting it. It no longer does outside AUTH_MODE=dev: the route carries no
    API key and takes its target org straight from the payload, so the
    signature is the only credential on it. Signing here means they exercise
    the path that actually ships, and they now cover verification too rather
    than bypassing it.
    """
    body = _slack_form_body(action_id, value, user_name)
    timestamp = str(int(time.time()))
    digest = hmac.new(
        _TEST_SIGNING_SECRET.encode(),
        f"v0:{timestamp}:{body.decode()}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return _FakeRequest(
        body,
        headers={
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": f"v0={digest}",
        },
    )


class TestSlackApprovalActions(unittest.IsolatedAsyncioTestCase):
    """Capability 2, Phase 2.3: Approve/Deny buttons record an approval
    decision via set_approval_decision."""

    def setUp(self):
        # Pin the production configuration: a signing secret is set and every
        # request below is signed. AUTH_MODE is pinned too so the suite does
        # not inherit it from a local .env — api_svc.config loads one at import,
        # CI has none, and that difference is exactly how these tests passed
        # locally while failing in CI with 403.
        for p in (
            patch.object(settings, "SLACK_SIGNING_SECRET", _TEST_SIGNING_SECRET),
            patch.object(settings, "AUTH_MODE", "prod"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _approval_value(self, **overrides):
        v = {"approval_id": 7, "org_id": "org-1", "tool_name": "wire_money"}
        v.update(overrides)
        return v

    async def test_approve_records_granted(self):
        request = _signed_request("approve_request", self._approval_value())
        with patch(
            "api_svc.routers.slack.set_approval_decision",
            AsyncMock(return_value={"id": 7, "status": "granted"}),
        ) as mock_set:
            response = await slack_callback(request)

        self.assertEqual(response.status_code, 200)
        args, kwargs = mock_set.call_args
        self.assertEqual(args[0], "org-1")
        self.assertEqual(args[1], 7)
        self.assertEqual(args[2], "granted")
        self.assertEqual(kwargs["decided_by"], "alice")
        self.assertEqual(kwargs["decision_channel"], "slack")
        self.assertIn("approved", response.body.decode())

    async def test_deny_records_denied(self):
        request = _signed_request("deny_request", self._approval_value())
        with patch(
            "api_svc.routers.slack.set_approval_decision",
            AsyncMock(return_value={"id": 7, "status": "denied"}),
        ) as mock_set:
            response = await slack_callback(request)

        self.assertEqual(mock_set.call_args.args[2], "denied")
        self.assertIn("denied", response.body.decode())

    async def test_missing_fields_returns_400(self):
        request = _signed_request("approve_request", {"tool_name": "x"})  # no id/org
        with patch("api_svc.routers.slack.set_approval_decision", AsyncMock()) as mock_set:
            response = await slack_callback(request)
        self.assertEqual(response.status_code, 400)
        mock_set.assert_not_called()

    async def test_already_decided_reports_current_state(self):
        request = _signed_request("approve_request", self._approval_value())
        with (
            patch("api_svc.routers.slack.set_approval_decision", AsyncMock(return_value=None)),
            patch(
                "api_svc.routers.slack.get_approval",
                AsyncMock(return_value={"id": 7, "status": "timeout"}),
            ),
        ):
            response = await slack_callback(request)
        self.assertEqual(response.status_code, 200)
        self.assertIn("timeout", response.body.decode())


if __name__ == "__main__":
    unittest.main()


class TestSignatureIsRequiredOutsideDev(unittest.IsolatedAsyncioTestCase):
    """The route is mounted with no API-key dependency and writes approval
    decisions against an org_id taken from the caller's own payload, so the
    Slack signature is the only credential. A missing secret must fail closed.

    It used to fall through to "skip verification" whenever the secret was
    empty — which is the default, and the normal state for an install that
    delivers alerts through an incoming webhook (those need no signing
    secret). That left cross-tenant approval writes open to an anonymous POST
    in AUTH_MODE=prod.
    """

    @staticmethod
    def _request():
        return _FakeRequest(
            _slack_form_body("approve_request", {"approval_id": 7, "org_id": "victim-org"})
        )

    async def test_no_secret_in_prod_is_refused_and_writes_nothing(self):
        decide = AsyncMock()
        with (
            patch.object(settings, "SLACK_SIGNING_SECRET", ""),
            patch.object(settings, "AUTH_MODE", "prod"),
            patch("api_svc.routers.slack.set_approval_decision", decide),
        ):
            resp = await slack_callback(self._request())
        self.assertEqual(resp.status_code, 403)
        decide.assert_not_awaited()

    async def test_no_secret_in_dev_still_works(self):
        """Local development keeps working without a Slack app configured."""
        decide = AsyncMock(return_value={"status": "granted"})
        with (
            patch.object(settings, "SLACK_SIGNING_SECRET", ""),
            patch.object(settings, "AUTH_MODE", "dev"),
            patch("api_svc.routers.slack.set_approval_decision", decide),
        ):
            resp = await slack_callback(self._request())
        self.assertEqual(resp.status_code, 200)
        decide.assert_awaited()

    async def test_bad_signature_is_refused_when_a_secret_is_set(self):
        decide = AsyncMock()
        req = _FakeRequest(
            _slack_form_body("approve_request", {"approval_id": 7, "org_id": "victim-org"}),
            headers={"X-Slack-Request-Timestamp": "1", "X-Slack-Signature": "v0=nope"},
        )
        with (
            patch.object(settings, "SLACK_SIGNING_SECRET", "s3cret"),
            patch.object(settings, "AUTH_MODE", "prod"),
            patch("api_svc.routers.slack.set_approval_decision", decide),
        ):
            resp = await slack_callback(req)
        self.assertEqual(resp.status_code, 403)
        decide.assert_not_awaited()
