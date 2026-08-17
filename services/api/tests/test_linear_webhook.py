"""
Endpoint-level tests for the Linear webhook receiver (Phase 4.1's
bi-directional sync — api_svc/routers/linear_webhook.py). Calls the route
function directly with a minimal fake Request, mocked DB/Linear calls. No
network, no DB.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from api_svc.routers.linear_webhook import linear_webhook


class _FakeRequest:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


def _signed_request(payload: dict, secret: str) -> _FakeRequest:
    body = json.dumps(payload).encode()
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return _FakeRequest(body, headers={"Linear-Signature": sig})


def _issue_update_payload(issue_id="issue-1", new_state_id="state-done", old_state_id="state-todo"):
    return {
        "action": "update",
        "type": "Issue",
        "data": {"id": issue_id, "stateId": new_state_id},
        "updatedFrom": {"stateId": old_state_id},
        "webhookTimestamp": int(time.time() * 1000),
    }


_ENCRYPTED = "encrypted-blob"
_CREDS = {"api_key": "lin_key", "webhook_secret": "whsec_test"}


class TestLinearWebhookNoIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_no_integration_configured_returns_404(self):
        request = _FakeRequest(b"{}")
        with patch(
            "api_svc.routers.linear_webhook.get_org_linear_webhook_secret",
            AsyncMock(return_value=None),
        ):
            response = await linear_webhook("org-1", request)
        self.assertEqual(response.status_code, 404)


class TestLinearWebhookSignatureVerification(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_signature_returns_403(self):
        request = _FakeRequest(
            json.dumps(_issue_update_payload()).encode(),
            headers={"Linear-Signature": "wrong-signature"},
        )
        with (
            patch(
                "api_svc.routers.linear_webhook.get_org_linear_webhook_secret",
                AsyncMock(return_value=_ENCRYPTED),
            ),
            patch(
                "api_svc.routers.linear_webhook.decrypt_credentials_for_webhook_verification",
                return_value=_CREDS,
            ),
        ):
            response = await linear_webhook("org-1", request)
        self.assertEqual(response.status_code, 403)

    async def test_valid_signature_accepted(self):
        request = _signed_request(_issue_update_payload(), _CREDS["webhook_secret"])
        with (
            patch(
                "api_svc.routers.linear_webhook.get_org_linear_webhook_secret",
                AsyncMock(return_value=_ENCRYPTED),
            ),
            patch(
                "api_svc.routers.linear_webhook.decrypt_credentials_for_webhook_verification",
                return_value=_CREDS,
            ),
            patch(
                "api_svc.routers.linear_webhook.get_signal_id_for_linear_issue",
                AsyncMock(return_value=None),
            ),
        ):
            response = await linear_webhook("org-1", request)
        self.assertEqual(response.status_code, 200)


class TestLinearWebhookStateChangeSync(unittest.IsolatedAsyncioTestCase):
    def _patches(self, mapping, state_type, fetch_state_side_effect=None):
        p = [
            patch(
                "api_svc.routers.linear_webhook.get_org_linear_webhook_secret",
                AsyncMock(return_value=_ENCRYPTED),
            ),
            patch(
                "api_svc.routers.linear_webhook.decrypt_credentials_for_webhook_verification",
                return_value=_CREDS,
            ),
            patch(
                "api_svc.routers.linear_webhook.get_signal_id_for_linear_issue",
                AsyncMock(return_value=mapping),
            ),
        ]
        if fetch_state_side_effect is not None:
            p.append(
                patch(
                    "api_svc.routers.linear_webhook.fetch_workflow_state_type",
                    AsyncMock(side_effect=fetch_state_side_effect),
                )
            )
        else:
            p.append(
                patch(
                    "api_svc.routers.linear_webhook.fetch_workflow_state_type",
                    AsyncMock(return_value=state_type),
                )
            )
        return p

    async def test_completed_state_marks_signal_resolved(self):
        request = _signed_request(
            _issue_update_payload(issue_id="issue-42"), _CREDS["webhook_secret"]
        )
        patches = self._patches({"org_id": "org-1", "signal_id": 99}, "completed")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "api_svc.routers.linear_webhook.mark_signal_resolved", AsyncMock(return_value=True)
            ) as mark_mock,
        ):
            response = await linear_webhook("org-1", request)

        self.assertEqual(response.status_code, 200)
        mark_mock.assert_called_once_with("org-1", 99)

    async def test_canceled_state_marks_signal_resolved(self):
        request = _signed_request(_issue_update_payload(), _CREDS["webhook_secret"])
        patches = self._patches({"org_id": "org-1", "signal_id": 5}, "canceled")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "api_svc.routers.linear_webhook.mark_signal_resolved", AsyncMock(return_value=True)
            ) as mark_mock,
        ):
            await linear_webhook("org-1", request)

        mark_mock.assert_called_once_with("org-1", 5)

    async def test_non_resolved_state_does_not_mark_resolved(self):
        """Issue moved to e.g. 'started', not 'completed'/'canceled' —
        must not sync as resolved."""
        request = _signed_request(_issue_update_payload(), _CREDS["webhook_secret"])
        patches = self._patches({"org_id": "org-1", "signal_id": 5}, "started")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("api_svc.routers.linear_webhook.mark_signal_resolved", AsyncMock()) as mark_mock,
        ):
            response = await linear_webhook("org-1", request)

        self.assertEqual(response.status_code, 200)
        mark_mock.assert_not_called()

    async def test_unknown_issue_not_created_by_dunetrace_ignored(self):
        request = _signed_request(_issue_update_payload(), _CREDS["webhook_secret"])
        patches = self._patches(None, "completed")
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("api_svc.routers.linear_webhook.mark_signal_resolved", AsyncMock()) as mark_mock,
        ):
            response = await linear_webhook("org-1", request)

        self.assertEqual(response.status_code, 200)
        mark_mock.assert_not_called()

    async def test_non_state_update_ignored(self):
        """An issue edit that isn't a state change (updatedFrom has no
        stateId key) must not trigger the workflow-state lookup at all."""
        payload = _issue_update_payload()
        del payload["updatedFrom"]["stateId"]
        request = _signed_request(payload, _CREDS["webhook_secret"])
        with (
            patch(
                "api_svc.routers.linear_webhook.get_org_linear_webhook_secret",
                AsyncMock(return_value=_ENCRYPTED),
            ),
            patch(
                "api_svc.routers.linear_webhook.decrypt_credentials_for_webhook_verification",
                return_value=_CREDS,
            ),
            patch(
                "api_svc.routers.linear_webhook.fetch_workflow_state_type", AsyncMock()
            ) as fetch_mock,
        ):
            response = await linear_webhook("org-1", request)

        self.assertEqual(response.status_code, 200)
        fetch_mock.assert_not_called()

    async def test_non_issue_type_ignored(self):
        payload = {"action": "update", "type": "Comment", "data": {}, "updatedFrom": {}}
        request = _signed_request(payload, _CREDS["webhook_secret"])
        with (
            patch(
                "api_svc.routers.linear_webhook.get_org_linear_webhook_secret",
                AsyncMock(return_value=_ENCRYPTED),
            ),
            patch(
                "api_svc.routers.linear_webhook.decrypt_credentials_for_webhook_verification",
                return_value=_CREDS,
            ),
            patch(
                "api_svc.routers.linear_webhook.get_signal_id_for_linear_issue", AsyncMock()
            ) as lookup_mock,
        ):
            response = await linear_webhook("org-1", request)

        self.assertEqual(response.status_code, 200)
        lookup_mock.assert_not_called()

    async def test_stale_timestamp_rejected(self):
        payload = _issue_update_payload()
        payload["webhookTimestamp"] = int((time.time() - 3600) * 1000)  # 1h old
        request = _signed_request(payload, _CREDS["webhook_secret"])
        with (
            patch(
                "api_svc.routers.linear_webhook.get_org_linear_webhook_secret",
                AsyncMock(return_value=_ENCRYPTED),
            ),
            patch(
                "api_svc.routers.linear_webhook.decrypt_credentials_for_webhook_verification",
                return_value=_CREDS,
            ),
        ):
            response = await linear_webhook("org-1", request)

        self.assertEqual(response.status_code, 400)

    async def test_workflow_state_fetch_failure_acks_without_crashing(self):
        """A transient Linear API failure fetching the state type must not
        propagate as a 5xx — Linear would just retry the same webhook
        forever; ack it and let the next event (if any) resolve it."""
        request = _signed_request(_issue_update_payload(), _CREDS["webhook_secret"])
        patches = self._patches(
            {"org_id": "org-1", "signal_id": 5},
            None,
            fetch_state_side_effect=RuntimeError("Linear API down"),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            response = await linear_webhook("org-1", request)

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
