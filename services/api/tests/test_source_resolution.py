"""
Tests for Phase 4.3's two-tier source mapping (api_svc/source_resolution.py).
Mocks DB queries and the GitHub tree fetch — no network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from api_svc.source_resolution import _suffix_match, resolve_source


class TestSuffixMatch(unittest.TestCase):
    def test_exact_match(self):
        result = _suffix_match("services/agents/bot.py", ["services/agents/bot.py", "other.py"])
        self.assertEqual(result, "services/agents/bot.py")

    def test_absolute_path_suffix_matches_repo_relative_file(self):
        result = _suffix_match(
            "/app/services/agents/bot.py", ["services/agents/bot.py", "services/other/thing.py"]
        )
        self.assertEqual(result, "services/agents/bot.py")

    def test_no_match_returns_none(self):
        result = _suffix_match("/app/nonexistent.py", ["services/agents/bot.py"])
        self.assertIsNone(result)

    def test_ambiguous_match_returns_none(self):
        """Two different files both end with '/x/bot.py' — must not guess
        which one the detected path refers to."""
        result = _suffix_match("x/bot.py", ["a/x/bot.py", "b/x/bot.py"])
        self.assertIsNone(result)

    def test_windows_style_path_normalized(self):
        result = _suffix_match(r"C:\app\services\agents\bot.py", ["services/agents/bot.py"])
        self.assertEqual(result, "services/agents/bot.py")


class TestResolveSource(unittest.IsolatedAsyncioTestCase):
    async def test_tier1_full_config_used_directly(self):
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": "src/bot.py"}),
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration", AsyncMock(return_value=None)
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload", AsyncMock()
            ) as run_started_mock,
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertEqual(result, {"repo": "acme/bot", "file_path": "src/bot.py"})
        # Tier-1 fully resolved — no need to even look at tier-2.
        run_started_mock.assert_not_called()

    async def test_no_tier1_no_tier2_returns_none(self):
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config", AsyncMock(return_value=None)
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration", AsyncMock(return_value=None)
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload",
                AsyncMock(return_value=None),
            ),
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertIsNone(result)

    async def test_tier1_repo_only_combines_with_tier2_path(self):
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": None}),
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration",
                AsyncMock(return_value={"installation_id": 1, "repos": [{"repo": "acme/bot"}]}),
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload",
                AsyncMock(return_value={"source_file": "/app/src/bot.py"}),
            ),
            patch(
                "api_svc.source_resolution._fetch_repo_tree_paths",
                AsyncMock(return_value=["src/bot.py", "src/other.py"]),
            ),
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertEqual(result, {"repo": "acme/bot", "file_path": "src/bot.py"})

    async def test_no_tier1_single_connected_repo_combines_with_tier2(self):
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config", AsyncMock(return_value=None)
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration",
                AsyncMock(
                    return_value={"installation_id": 1, "repos": [{"repo": "acme/only-repo"}]}
                ),
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload",
                AsyncMock(return_value={"source_file": "/app/src/bot.py"}),
            ),
            patch(
                "api_svc.source_resolution._fetch_repo_tree_paths",
                AsyncMock(return_value=["src/bot.py"]),
            ),
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertEqual(result, {"repo": "acme/only-repo", "file_path": "src/bot.py"})

    async def test_no_tier1_multiple_repos_no_resolution(self):
        """Multiple connected repos with no tier-1 hint — genuinely
        ambiguous, must not guess."""
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config", AsyncMock(return_value=None)
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration",
                AsyncMock(
                    return_value={
                        "installation_id": 1,
                        "repos": [{"repo": "acme/repo-a"}, {"repo": "acme/repo-b"}],
                    }
                ),
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload",
                AsyncMock(return_value={"source_file": "/app/src/bot.py"}),
            ),
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertIsNone(result)

    async def test_ambiguous_suffix_match_returns_none(self):
        """Two distinct files in the repo both end with '/x/bot.py' — must
        not guess which one the detected path resolves to."""
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": None}),
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration",
                AsyncMock(return_value={"installation_id": 1, "repos": [{"repo": "acme/bot"}]}),
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload",
                AsyncMock(return_value={"source_file": "x/bot.py"}),
            ),
            patch(
                "api_svc.source_resolution._fetch_repo_tree_paths",
                AsyncMock(return_value=["a/x/bot.py", "b/x/bot.py"]),
            ),
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertIsNone(result)

    async def test_tree_fetch_failure_returns_none_not_raises(self):
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": None}),
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration",
                AsyncMock(return_value={"installation_id": 1, "repos": [{"repo": "acme/bot"}]}),
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload",
                AsyncMock(return_value={"source_file": "/app/src/bot.py"}),
            ),
            patch(
                "api_svc.source_resolution._fetch_repo_tree_paths",
                AsyncMock(side_effect=RuntimeError("GitHub API down")),
            ),
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertIsNone(result)

    async def test_no_detected_path_no_github_integration_returns_none(self):
        with (
            patch(
                "api_svc.source_resolution.get_agent_source_config", AsyncMock(return_value=None)
            ),
            patch(
                "api_svc.source_resolution.get_org_github_integration", AsyncMock(return_value=None)
            ),
            patch(
                "api_svc.source_resolution.get_latest_run_started_payload",
                AsyncMock(return_value={"source_file": "/app/src/bot.py"}),
            ),
        ):
            result = await resolve_source("org-1", "agent-1")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
