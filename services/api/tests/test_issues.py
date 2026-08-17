"""
Endpoint-level tests for api_svc/routers/issues.py. Covers the pre-existing
get_issues (agent-scoped list — had no test file before this) and Phase
4.2's three new endpoints (search_issues, get_issue, resolve_issue), added
for the MCP server's coding-agent-facing tools. Calls route functions
directly (this codebase's established pattern), mocked DB/explain calls. No
network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.issues import (
    ResolveIssueRequest,
    get_issue,
    get_issues,
    resolve_issue,
    search_issues,
)
from llm_test_utils import configured_llm


def _issue(**overrides):
    fields = {
        "id": 7,
        "agent_id": "support-bot",
        "failure_type": "TOOL_LOOP",
        "status": "open",
        "first_seen": 1_752_000_000.0,
        "last_seen": 1_752_003_600.0,
        "affected_runs": 12,
        "clean_runs_since": 0,
        "resolved_at": None,
        "resolution_notes": None,
        "manually_resolved": False,
    }
    fields.update(overrides)
    return fields


class TestGetIssues(unittest.IsolatedAsyncioTestCase):
    async def test_returns_list_and_total(self):
        with patch("api_svc.routers.issues.list_issues", AsyncMock(return_value=[_issue()])):
            result = await get_issues("support-bot", status="open", org_id="org-1")

        self.assertEqual(result.total, 1)
        self.assertEqual(result.issues[0].id, 7)


class TestSearchIssues(unittest.IsolatedAsyncioTestCase):
    async def test_returns_empty_when_no_matches(self):
        with patch("api_svc.routers.issues.search_issues_query", AsyncMock(return_value=([], 0))):
            result = await search_issues(offset=0, limit=20, org_id="org-1")

        self.assertEqual(result.issues, [])
        self.assertEqual(result.page.total, 0)

    async def test_filters_passed_through(self):
        with patch(
            "api_svc.routers.issues.search_issues_query", AsyncMock(return_value=([], 0))
        ) as mock:
            await search_issues(
                q="loop",
                status="open",
                agent_id="support-bot",
                failure_type="TOOL_LOOP",
                offset=10,
                limit=20,
                org_id="org-1",
            )

        mock.assert_called_once_with("org-1", "loop", "open", "support-bot", "TOOL_LOOP", 10, 20)

    async def test_maps_rows_to_issues(self):
        with patch(
            "api_svc.routers.issues.search_issues_query",
            AsyncMock(return_value=([_issue(), _issue(id=8)], 2)),
        ):
            result = await search_issues(offset=0, limit=20, org_id="org-1")

        self.assertEqual(len(result.issues), 2)
        self.assertEqual(result.issues[1].id, 8)


class TestGetIssue(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch("api_svc.routers.issues.get_issue_by_id", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as ctx:
                await get_issue(999, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_returns_metadata_and_affected_runs_without_llm_key(self):
        pattern = {
            "top_runs": [
                {
                    "run_id": "run-1",
                    "detected_at": 1_752_000_000.0,
                    "step_index": 3,
                    "confidence": 0.9,
                }
            ]
        }
        with (
            patch("api_svc.routers.issues.get_issue_by_id", AsyncMock(return_value=_issue())),
            patch("api_svc.routers.issues.agent_failure_pattern", AsyncMock(return_value=pattern)),
            patch("api_svc.routers.issues.settings") as mock_settings,
        ):
            mock_settings.ANTHROPIC_API_KEY = ""
            mock_settings.OPENAI_API_KEY = ""
            result = await get_issue(7, org_id="org-1")

        self.assertEqual(result.id, 7)
        self.assertEqual(len(result.affected_runs), 1)
        self.assertEqual(result.affected_runs[0].run_id, "run-1")
        self.assertIsNone(result.root_cause)
        self.assertEqual(result.code_references, [])

    async def test_code_references_always_empty(self):
        """Phase 4.3 (source mapping) doesn't exist yet — always empty,
        per explicit maintainer decision (see BACKLOG.md)."""
        with (
            patch("api_svc.routers.issues.get_issue_by_id", AsyncMock(return_value=_issue())),
            patch("api_svc.routers.issues.agent_failure_pattern", AsyncMock(return_value={})),
            patch("api_svc.routers.issues.settings") as mock_settings,
        ):
            mock_settings.ANTHROPIC_API_KEY = ""
            mock_settings.OPENAI_API_KEY = ""
            result = await get_issue(7, org_id="org-1")

        self.assertEqual(result.code_references, [])

    async def test_root_cause_populated_when_llm_key_present(self):
        with (
            patch("api_svc.routers.issues.get_issue_by_id", AsyncMock(return_value=_issue())),
            patch("api_svc.routers.issues.agent_failure_pattern", AsyncMock(return_value={})),
            # The gate is llm_provider now, not this router's own settings.
            configured_llm("anthropic"),
            patch(
                "api_svc.routers.issues.get_most_recent_signal_id",
                AsyncMock(return_value=42),
            ),
            patch(
                "api_svc.routers.signals.explain_signal",
                AsyncMock(
                    return_value={"root_cause": "Loop detected", "fix_content": "Add a limit"}
                ),
            ),
        ):
            result = await get_issue(7, org_id="org-1")

        self.assertEqual(result.root_cause, "Loop detected")
        self.assertEqual(result.suggested_fix, "Add a limit")

    async def test_explain_failure_does_not_block_issue_metadata(self):
        """A root-cause analysis failure must not prevent get_issue from
        returning the issue's core metadata/affected runs."""
        with (
            patch("api_svc.routers.issues.get_issue_by_id", AsyncMock(return_value=_issue())),
            patch("api_svc.routers.issues.agent_failure_pattern", AsyncMock(return_value={})),
            patch("api_svc.routers.issues.settings") as mock_settings,
            patch(
                "api_svc.routers.issues.get_most_recent_signal_id",
                AsyncMock(return_value=42),
            ),
            patch(
                "api_svc.routers.signals.explain_signal",
                AsyncMock(side_effect=RuntimeError("LLM down")),
            ),
        ):
            mock_settings.ANTHROPIC_API_KEY = "sk-x"
            mock_settings.OPENAI_API_KEY = ""
            result = await get_issue(7, org_id="org-1")

        self.assertEqual(result.id, 7)
        self.assertIsNone(result.root_cause)


class TestResolveIssue(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch("api_svc.routers.issues.resolve_issue_manually", AsyncMock(return_value=False)):
            with self.assertRaises(HTTPException) as ctx:
                await resolve_issue(
                    999, ResolveIssueRequest(resolution_notes="fixed"), org_id="org-1"
                )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_resolved_calls_db_with_notes(self):
        with patch(
            "api_svc.routers.issues.resolve_issue_manually", AsyncMock(return_value=True)
        ) as mock:
            result = await resolve_issue(
                7, ResolveIssueRequest(resolution_notes="Added a tool-call limit"), org_id="org-1"
            )

        mock.assert_called_once_with("org-1", 7, "Added a tool-call limit")
        self.assertTrue(result["resolved"])
        self.assertEqual(result["issue_id"], 7)


if __name__ == "__main__":
    unittest.main()
