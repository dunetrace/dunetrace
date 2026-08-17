"""
Tests for Phase 4.3's real-diff-application flow in
api_svc/routers/signals.py: _resolve_github_auth (per-org GitHub App vs
legacy PAT fallback), _attempt_real_diff (source resolution -> fetch ->
LLM rewrite -> security guardrail, all-or-nothing with a safe None
fallback), and open_pr's real-file vs summary-file branching.

Complements test_signals_explain.py's TestOpenPR (which covers the
pre-existing config-gating/404/403 cases) — this file is scoped to the
new GitHub App auth resolution and real-diff-application behavior only.
Calls route functions directly (this codebase's established pattern).
No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_svc.routers.signals import OpenPRRequest, _attempt_real_diff, _resolve_github_auth, open_pr


def _signal(failure_type="CONTEXT_BLOAT", agent_id="agent-1", run_id="run-1"):
    return {
        "id": 42,
        "failure_type": failure_type,
        "run_id": run_id,
        "agent_id": agent_id,
    }


def _open_pr_body():
    return OpenPRRequest(root_cause="because X", fix_content="add this", fix_patch="+ add this")


class TestResolveGithubAuth(unittest.IsolatedAsyncioTestCase):
    async def test_per_org_installation_used_when_present(self):
        with (
            patch(
                "api_svc.routers.signals.get_org_github_integration",
                AsyncMock(
                    return_value={
                        "installation_id": 555,
                        "repos": [{"repo": "acme/bot", "base_branch": "main"}],
                        "reviewers": ["octocat"],
                    }
                ),
            ),
            patch(
                "api_svc.github_app_auth.get_installation_token",
                AsyncMock(return_value="ghs_token"),
            ),
        ):
            result = await _resolve_github_auth("org-1")

        self.assertEqual(result["token"], "ghs_token")
        self.assertEqual(result["repos"], [{"repo": "acme/bot", "base_branch": "main"}])
        self.assertEqual(result["reviewers"], ["octocat"])

    async def test_falls_back_to_legacy_pat_when_no_installation(self):
        settings_mock = MagicMock()
        settings_mock.github_configured = True
        settings_mock.GITHUB_TOKEN = "ghp_legacy"
        settings_mock.GITHUB_REPO = "acme/legacy-repo"
        settings_mock.GITHUB_BASE_BRANCH = "main"

        with (
            patch(
                "api_svc.routers.signals.get_org_github_integration", AsyncMock(return_value=None)
            ),
            patch("api_svc.routers.signals.settings", settings_mock),
        ):
            result = await _resolve_github_auth("org-1")

        self.assertEqual(result["token"], "ghp_legacy")
        self.assertEqual(result["repos"], [{"repo": "acme/legacy-repo", "base_branch": "main"}])
        self.assertEqual(result["reviewers"], [])

    async def test_installation_with_no_repos_falls_back_to_legacy(self):
        """An org that installed the App but hasn't configured any repos
        yet must not be treated as 'configured' — falls through to the
        legacy PAT (or None) just like having no installation at all."""
        settings_mock = MagicMock()
        settings_mock.github_configured = False

        with (
            patch(
                "api_svc.routers.signals.get_org_github_integration",
                AsyncMock(return_value={"installation_id": 555, "repos": [], "reviewers": []}),
            ),
            patch("api_svc.routers.signals.settings", settings_mock),
        ):
            result = await _resolve_github_auth("org-1")

        self.assertIsNone(result)

    async def test_returns_none_when_neither_configured(self):
        settings_mock = MagicMock()
        settings_mock.github_configured = False

        with (
            patch(
                "api_svc.routers.signals.get_org_github_integration", AsyncMock(return_value=None)
            ),
            patch("api_svc.routers.signals.settings", settings_mock),
        ):
            result = await _resolve_github_auth("org-1")

        self.assertIsNone(result)


class TestAttemptRealDiff(unittest.IsolatedAsyncioTestCase):
    _repos = [{"repo": "acme/bot", "base_branch": "main"}]

    async def test_full_success_path(self):
        with (
            patch(
                "api_svc.source_resolution.resolve_source",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": "src/bot.py"}),
            ),
            patch("api_svc.fix_security.validate_target_path", return_value=(True, "")),
            patch(
                "api_svc.github_client.fetch_file_content",
                AsyncMock(return_value="def broken(): pass\n"),
            ),
            patch(
                "api_svc.diff_generation.generate_real_file_content",
                AsyncMock(return_value="def fixed(): return True\n"),
            ),
        ):
            result = await _attempt_real_diff(
                "org-1", "agent-1", "tok", self._repos, "root cause", "fix it"
            )

        self.assertEqual(result["repo"], "acme/bot")
        self.assertEqual(result["file_path"], "src/bot.py")
        self.assertEqual(result["base_branch"], "main")
        self.assertEqual(result["new_content"], "def fixed(): return True\n")

    async def test_no_source_resolution_returns_none(self):
        with patch("api_svc.source_resolution.resolve_source", AsyncMock(return_value=None)):
            result = await _attempt_real_diff(
                "org-1", "agent-1", "tok", self._repos, "root cause", "fix it"
            )
        self.assertIsNone(result)

    async def test_resolved_repo_not_in_connected_repos_returns_none(self):
        """Source resolution points at a repo this org hasn't connected the
        GitHub App to — must not attempt to write there."""
        with patch(
            "api_svc.source_resolution.resolve_source",
            AsyncMock(return_value={"repo": "acme/other-repo", "file_path": "src/bot.py"}),
        ):
            result = await _attempt_real_diff(
                "org-1", "agent-1", "tok", self._repos, "root cause", "fix it"
            )
        self.assertIsNone(result)

    async def test_security_guardrail_rejection_returns_none(self):
        with (
            patch(
                "api_svc.source_resolution.resolve_source",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": ".env"}),
            ),
            patch(
                "api_svc.fix_security.validate_target_path",
                return_value=(False, "sensitive path"),
            ),
        ):
            result = await _attempt_real_diff(
                "org-1", "agent-1", "tok", self._repos, "root cause", "fix it"
            )
        self.assertIsNone(result)

    async def test_file_not_found_in_repo_returns_none(self):
        with (
            patch(
                "api_svc.source_resolution.resolve_source",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": "src/bot.py"}),
            ),
            patch("api_svc.fix_security.validate_target_path", return_value=(True, "")),
            patch("api_svc.github_client.fetch_file_content", AsyncMock(return_value=None)),
        ):
            result = await _attempt_real_diff(
                "org-1", "agent-1", "tok", self._repos, "root cause", "fix it"
            )
        self.assertIsNone(result)

    async def test_llm_declines_to_produce_new_content_returns_none(self):
        with (
            patch(
                "api_svc.source_resolution.resolve_source",
                AsyncMock(return_value={"repo": "acme/bot", "file_path": "src/bot.py"}),
            ),
            patch("api_svc.fix_security.validate_target_path", return_value=(True, "")),
            patch(
                "api_svc.github_client.fetch_file_content",
                AsyncMock(return_value="def broken(): pass\n"),
            ),
            patch(
                "api_svc.diff_generation.generate_real_file_content",
                AsyncMock(return_value=None),
            ),
        ):
            result = await _attempt_real_diff(
                "org-1", "agent-1", "tok", self._repos, "root cause", "fix it"
            )
        self.assertIsNone(result)

    async def test_unexpected_exception_returns_none_not_raises(self):
        with patch(
            "api_svc.source_resolution.resolve_source",
            AsyncMock(side_effect=RuntimeError("GitHub API down")),
        ):
            result = await _attempt_real_diff(
                "org-1", "agent-1", "tok", self._repos, "root cause", "fix it"
            )
        self.assertIsNone(result)


class TestOpenPrRealDiffBranching(unittest.IsolatedAsyncioTestCase):
    _auth = {
        "token": "tok",
        "repos": [{"repo": "acme/bot", "base_branch": "main"}],
        "reviewers": [],
    }

    async def test_real_file_applied_when_attempt_real_diff_succeeds(self):
        signal = _signal()
        with (
            patch(
                "api_svc.routers.signals._resolve_github_auth", AsyncMock(return_value=self._auth)
            ),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch(
                "api_svc.routers.signals._attempt_real_diff",
                AsyncMock(
                    return_value={
                        "repo": "acme/bot",
                        "file_path": "src/bot.py",
                        "base_branch": "main",
                        "old_content": "def broken(): pass\n",
                        "new_content": "def fixed(): return True\n",
                    }
                ),
            ),
            patch(
                "api_svc.github_client.create_fix_pr",
                AsyncMock(
                    return_value={
                        "pr_url": "https://github.com/acme/bot/pull/1",
                        "pr_number": 1,
                        "branch": "b",
                        "applied_to_real_file": True,
                    }
                ),
            ) as create_pr_mock,
            patch("api_svc.routers.signals.record_fix", AsyncMock(return_value=5)),
        ):
            result = await open_pr(1, _open_pr_body(), org_id="org-1")

        self.assertTrue(result["applied_to_real_file"])
        call_kwargs = create_pr_mock.call_args.kwargs
        self.assertEqual(call_kwargs["repo"], "acme/bot")
        self.assertEqual(call_kwargs["real_file"]["file_path"], "src/bot.py")
        self.assertEqual(call_kwargs["real_file"]["new_content"], "def fixed(): return True\n")
        self.assertIn("+def fixed(): return True", call_kwargs["fix_patch"])

    async def test_falls_back_to_summary_file_when_real_diff_unavailable(self):
        signal = _signal()
        with (
            patch(
                "api_svc.routers.signals._resolve_github_auth", AsyncMock(return_value=self._auth)
            ),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals._attempt_real_diff", AsyncMock(return_value=None)),
            patch(
                "api_svc.github_client.create_fix_pr",
                AsyncMock(
                    return_value={
                        "pr_url": "https://github.com/acme/bot/pull/2",
                        "pr_number": 2,
                        "branch": "b",
                        "applied_to_real_file": False,
                    }
                ),
            ) as create_pr_mock,
            patch("api_svc.routers.signals.record_fix", AsyncMock(return_value=6)),
        ):
            result = await open_pr(1, _open_pr_body(), org_id="org-1")

        self.assertFalse(result["applied_to_real_file"])
        call_kwargs = create_pr_mock.call_args.kwargs
        self.assertIsNone(call_kwargs["real_file"])
        self.assertEqual(call_kwargs["repo"], "acme/bot")
        self.assertEqual(call_kwargs["fix_patch"], "+ add this")

    async def test_reviewers_from_auth_passed_through(self):
        signal = _signal()
        auth = {
            "token": "tok",
            "repos": [{"repo": "acme/bot", "base_branch": "main"}],
            "reviewers": ["octocat"],
        }
        with (
            patch("api_svc.routers.signals._resolve_github_auth", AsyncMock(return_value=auth)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals._attempt_real_diff", AsyncMock(return_value=None)),
            patch(
                "api_svc.github_client.create_fix_pr",
                AsyncMock(return_value={"pr_url": "u", "pr_number": 3, "branch": "b"}),
            ) as create_pr_mock,
            patch("api_svc.routers.signals.record_fix", AsyncMock(return_value=7)),
        ):
            await open_pr(1, _open_pr_body(), org_id="org-1")

        self.assertEqual(create_pr_mock.call_args.kwargs["reviewers"], ["octocat"])


if __name__ == "__main__":
    unittest.main()
