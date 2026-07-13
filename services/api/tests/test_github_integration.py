"""
Endpoint-level tests for Phase 4.3's per-org GitHub App config
(api_svc/routers/github_integration.py). Calls route functions directly
(this codebase's established pattern), mocked DB calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.github_integration import (
    GitHubConfigRequest,
    RepoConfig,
    get_github_config,
    get_install_url,
    install_callback,
    remove_github_config,
    set_github_config,
)


class TestGetInstallUrl(unittest.IsolatedAsyncioTestCase):
    async def test_returns_url_with_org_id_as_state(self):
        with patch(
            "api_svc.routers.github_integration.build_install_url",
            return_value="https://github.com/apps/dunetrace-fixit/installations/new?state=org-1",
        ) as build_mock:
            result = await get_install_url(org_id="org-1")

        build_mock.assert_called_once_with(state="org-1")
        self.assertIn("org-1", result["install_url"])

    async def test_not_configured_returns_503(self):
        with patch(
            "api_svc.routers.github_integration.build_install_url",
            side_effect=ValueError("GITHUB_APP_SLUG is not configured"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await get_install_url(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)


class TestInstallCallback(unittest.IsolatedAsyncioTestCase):
    async def test_state_round_trips_to_org_id(self):
        with patch(
            "api_svc.routers.github_integration.upsert_org_github_installation", AsyncMock()
        ) as upsert_mock:
            result = await install_callback(installation_id=999, state="org-42")

        upsert_mock.assert_awaited_once_with("org-42", 999)
        self.assertEqual(result, {"installed": True, "org_id": "org-42", "installation_id": 999})


class TestSetGithubConfig(unittest.IsolatedAsyncioTestCase):
    def _body(self):
        return GitHubConfigRequest(
            repos=[RepoConfig(repo="acme/bot", base_branch="main")],
            reviewers=["octocat"],
        )

    async def test_no_installation_returns_404(self):
        with patch(
            "api_svc.routers.github_integration.set_org_github_config",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_github_config(self._body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_success_returns_configured_status(self):
        with (
            patch(
                "api_svc.routers.github_integration.set_org_github_config",
                AsyncMock(return_value=True),
            ) as set_mock,
            patch(
                "api_svc.routers.github_integration.get_org_github_integration",
                AsyncMock(
                    return_value={
                        "installation_id": 555,
                        "repos": [{"repo": "acme/bot", "base_branch": "main"}],
                        "reviewers": ["octocat"],
                    }
                ),
            ),
        ):
            result = await set_github_config(self._body(), org_id="org-1")

        set_mock.assert_awaited_once_with(
            "org-1", [{"repo": "acme/bot", "base_branch": "main"}], ["octocat"]
        )
        self.assertTrue(result.configured)
        self.assertEqual(result.installation_id, 555)
        self.assertEqual(result.reviewers, ["octocat"])


class TestGetGithubConfig(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_returns_configured_false(self):
        with patch(
            "api_svc.routers.github_integration.get_org_github_integration",
            AsyncMock(return_value=None),
        ):
            result = await get_github_config(org_id="org-1")
        self.assertFalse(result.configured)
        self.assertIsNone(result.installation_id)

    async def test_configured_reports_repos_and_reviewers(self):
        with patch(
            "api_svc.routers.github_integration.get_org_github_integration",
            AsyncMock(
                return_value={
                    "installation_id": 555,
                    "repos": [{"repo": "acme/bot", "base_branch": "main"}],
                    "reviewers": ["octocat"],
                }
            ),
        ):
            result = await get_github_config(org_id="org-1")
        self.assertTrue(result.configured)
        self.assertEqual(result.installation_id, 555)


class TestRemoveGithubConfig(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.github_integration.delete_org_github_integration",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_github_config(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_found_deletes_without_raising(self):
        with patch(
            "api_svc.routers.github_integration.delete_org_github_integration",
            AsyncMock(return_value=True),
        ):
            await remove_github_config(org_id="org-1")  # must not raise


if __name__ == "__main__":
    unittest.main()
