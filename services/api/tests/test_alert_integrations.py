"""
Endpoint-level tests for Phase 4.1's per-org Slack/Linear alert-destination
config (api_svc/routers/alert_integrations.py). Calls route functions
directly (this codebase's established pattern), mocked DB/Linear calls. No
network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.linear_client import LinearApiError
from api_svc.routers.alert_integrations import (
    LinearIntegrationRequest,
    LinearPreviewTeamsRequest,
    SlackIntegrationRequest,
    get_linear_integration,
    get_slack_integration,
    preview_linear_teams,
    remove_linear_integration,
    remove_slack_integration,
    set_linear_integration,
    set_slack_integration,
)


class TestSetSlackIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_encrypts_before_storing(self):
        body = SlackIntegrationRequest(webhook_url="https://hooks.slack.com/x", channel="#alerts")
        with (
            patch(
                "api_svc.routers.alert_integrations.encrypt_credentials",
                return_value="encrypted-token",
            ) as encrypt_mock,
            patch(
                "api_svc.routers.alert_integrations.upsert_org_alert_integration", AsyncMock()
            ) as upsert_mock,
            patch(
                "api_svc.routers.alert_integrations.get_org_alert_integration_status",
                AsyncMock(return_value={"config": {"channel": "#alerts"}, "enabled": True}),
            ),
        ):
            result = await set_slack_integration(body, org_id="org-1")

        encrypt_mock.assert_called_once_with({"webhook_url": "https://hooks.slack.com/x"})
        upsert_mock.assert_awaited_once_with(
            "org-1", "slack", "encrypted-token", {"channel": "#alerts"}
        )
        self.assertTrue(result.configured)

    async def test_missing_master_key_returns_503(self):
        body = SlackIntegrationRequest(webhook_url="https://hooks.slack.com/x")
        with patch(
            "api_svc.routers.alert_integrations.encrypt_credentials",
            side_effect=ValueError("DUNETRACE_MASTER_KEY is not configured"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_slack_integration(body, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_response_never_contains_the_webhook_url(self):
        body = SlackIntegrationRequest(webhook_url="https://hooks.slack.com/x")
        with (
            patch("api_svc.routers.alert_integrations.encrypt_credentials", return_value="tok"),
            patch("api_svc.routers.alert_integrations.upsert_org_alert_integration", AsyncMock()),
            patch(
                "api_svc.routers.alert_integrations.get_org_alert_integration_status",
                AsyncMock(return_value={"config": {"channel": ""}, "enabled": True}),
            ),
        ):
            result = await set_slack_integration(body, org_id="org-1")

        self.assertNotIn("hooks.slack.com", str(result.model_dump()))


class TestGetSlackIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_returns_configured_false(self):
        with patch(
            "api_svc.routers.alert_integrations.get_org_alert_integration_status",
            AsyncMock(return_value=None),
        ):
            result = await get_slack_integration(org_id="org-1")
        self.assertFalse(result.configured)


class TestRemoveSlackIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.alert_integrations.delete_org_alert_integration",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_slack_integration(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_found_deletes_without_raising(self):
        with patch(
            "api_svc.routers.alert_integrations.delete_org_alert_integration",
            AsyncMock(return_value=True),
        ):
            await remove_slack_integration(org_id="org-1")  # must not raise


class TestSetLinearIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_encrypts_api_key_and_webhook_secret_together(self):
        body = LinearIntegrationRequest(
            api_key="lin_api_x", webhook_secret="whsec_y", team_id="team-1", project_id="proj-1"
        )
        with (
            patch(
                "api_svc.routers.alert_integrations.encrypt_credentials",
                return_value="encrypted-token",
            ) as encrypt_mock,
            patch(
                "api_svc.routers.alert_integrations.upsert_org_alert_integration", AsyncMock()
            ) as upsert_mock,
            patch(
                "api_svc.routers.alert_integrations.get_org_alert_integration_status",
                AsyncMock(
                    return_value={
                        "config": {"team_id": "team-1", "project_id": "proj-1"},
                        "enabled": True,
                    }
                ),
            ),
        ):
            result = await set_linear_integration(body, org_id="org-1")

        encrypt_mock.assert_called_once_with({"api_key": "lin_api_x", "webhook_secret": "whsec_y"})
        upsert_mock.assert_awaited_once_with(
            "org-1", "linear", "encrypted-token", {"team_id": "team-1", "project_id": "proj-1"}
        )
        self.assertTrue(result.configured)

    async def test_response_never_contains_credentials(self):
        body = LinearIntegrationRequest(
            api_key="lin_api_secret", webhook_secret="whsec_secret", team_id="team-1"
        )
        with (
            patch("api_svc.routers.alert_integrations.encrypt_credentials", return_value="tok"),
            patch("api_svc.routers.alert_integrations.upsert_org_alert_integration", AsyncMock()),
            patch(
                "api_svc.routers.alert_integrations.get_org_alert_integration_status",
                AsyncMock(return_value={"config": {"team_id": "team-1"}, "enabled": True}),
            ),
        ):
            result = await set_linear_integration(body, org_id="org-1")

        dumped = str(result.model_dump())
        self.assertNotIn("lin_api_secret", dumped)
        self.assertNotIn("whsec_secret", dumped)


class TestRemoveLinearIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.alert_integrations.delete_org_alert_integration",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_linear_integration(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)


class TestPreviewLinearTeams(unittest.IsolatedAsyncioTestCase):
    async def test_returns_teams_with_nested_projects(self):
        body = LinearPreviewTeamsRequest(api_key="lin_api_x")
        with (
            patch(
                "api_svc.routers.alert_integrations.fetch_teams",
                AsyncMock(return_value=[{"id": "team-1", "name": "Engineering"}]),
            ),
            patch(
                "api_svc.routers.alert_integrations.fetch_projects",
                AsyncMock(return_value=[{"id": "proj-1", "name": "Backend"}]),
            ),
        ):
            result = await preview_linear_teams(body)

        self.assertEqual(len(result["teams"]), 1)
        self.assertEqual(result["teams"][0]["id"], "team-1")
        self.assertEqual(result["teams"][0]["projects"], [{"id": "proj-1", "name": "Backend"}])

    async def test_credentials_never_persisted(self):
        """This is a preview-only endpoint — it must not touch any upsert/write path."""
        body = LinearPreviewTeamsRequest(api_key="lin_api_x")
        with (
            patch("api_svc.routers.alert_integrations.fetch_teams", AsyncMock(return_value=[])),
            patch(
                "api_svc.routers.alert_integrations.upsert_org_alert_integration", AsyncMock()
            ) as upsert_mock,
        ):
            await preview_linear_teams(body)

        upsert_mock.assert_not_called()

    async def test_linear_api_error_returns_502(self):
        body = LinearPreviewTeamsRequest(api_key="bad_key")
        with patch(
            "api_svc.routers.alert_integrations.fetch_teams",
            AsyncMock(side_effect=LinearApiError("invalid api key")),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await preview_linear_teams(body)
        self.assertEqual(ctx.exception.status_code, 502)

    async def test_project_fetch_failure_for_one_team_does_not_fail_whole_request(self):
        body = LinearPreviewTeamsRequest(api_key="lin_api_x")
        with (
            patch(
                "api_svc.routers.alert_integrations.fetch_teams",
                AsyncMock(return_value=[{"id": "team-1", "name": "Engineering"}]),
            ),
            patch(
                "api_svc.routers.alert_integrations.fetch_projects",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            result = await preview_linear_teams(body)

        self.assertEqual(result["teams"][0]["projects"], [])


if __name__ == "__main__":
    unittest.main()
