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
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from api_svc.routers.issues import (
    ResolveIssueRequest,
    _describe_policy,
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


class TestGetIssueNativeFix(unittest.IsolatedAsyncioTestCase):
    """Regression: explain_signal's `dunetrace_native` branch returns NO
    `fix_content` key at all — the fix is a runtime policy, delivered as a
    dict under `suggested_policy`. `fix_content or suggested_policy` therefore
    put a dict into IssueDetail.suggested_fix (typed Optional[str]), and the
    IssueDetail(...) construction sits outside the try/except guarding the
    explain call, so the ValidationError escaped as an unhandled 500 for every
    TOOL_LOOP / RETRY_STORM / CASCADING_TOOL_FAILURE / STEP_COUNT_INFLATION
    issue whenever an LLM key was configured.
    """

    NATIVE_POLICY = {
        "name": "Auto-suggested: stop recurring TOOL_LOOP",
        "agent_id": "support-bot",
        "condition": {"trigger": "tool_call_count", "operator": "gte", "value": 6},
        "action": {"type": "stop"},
        "priority": 100,
        "enabled": True,
    }

    def _native_explain_result(self):
        # Exactly what routers/signals.py::explain_signal returns for
        # fix_category == "dunetrace_native" — note the absent fix_content.
        return {
            "signal_id": 42,
            "source": "native",
            "root_cause": "The agent re-called web_search with identical args.",
            "fix_category": "dunetrace_native",
            "suggested_policy": self.NATIVE_POLICY,
            "fix_type": "policy",
            "apply_blocked": False,
        }

    def _patches(self, explain_result):
        return (
            patch("api_svc.routers.issues.get_issue_by_id", AsyncMock(return_value=_issue())),
            patch("api_svc.routers.issues.agent_failure_pattern", AsyncMock(return_value={})),
            configured_llm("anthropic"),
            patch("api_svc.routers.issues.get_most_recent_signal_id", AsyncMock(return_value=42)),
            patch(
                "api_svc.routers.signals.explain_signal",
                AsyncMock(return_value=explain_result),
            ),
        )

    async def test_native_fix_returns_policy_and_text(self):
        a, b, c, d, e = self._patches(self._native_explain_result())
        with a, b, c, d, e:
            result = await get_issue(7, org_id="org-1")

        # 1. It no longer 500s, and suggested_fix is a real string.
        self.assertIsInstance(result.suggested_fix, str)
        self.assertNotIn("{", result.suggested_fix, "suggested_fix must not be a stringified dict")

        # 2. The caller learns a runtime policy is the fix, and what it does.
        self.assertIn("policy", result.suggested_fix.lower())
        self.assertIn("tool_call_count", result.suggested_fix)
        self.assertIn("6", result.suggested_fix)

        # 3. The machine-readable body survives intact, typed.
        self.assertEqual(result.suggested_policy, self.NATIVE_POLICY)
        self.assertEqual(result.fix_category, "dunetrace_native")
        self.assertEqual(result.root_cause, "The agent re-called web_search with identical args.")

    async def test_customer_code_fix_uses_fix_content_and_no_policy(self):
        customer_code = {
            "signal_id": 42,
            "source": "native",
            "root_cause": "The system prompt never mentions the tool.",
            "fix_category": "customer_code",
            "fix_content": "Add 'use search_docs before answering' to the prompt.",
            "fix_patch": "+ use search_docs before answering",
            "fix_type": "prompt_addition",
            "apply_blocked": True,
        }
        a, b, c, d, e = self._patches(customer_code)
        with a, b, c, d, e:
            result = await get_issue(7, org_id="org-1")

        self.assertEqual(
            result.suggested_fix, "Add 'use search_docs before answering' to the prompt."
        )
        self.assertIsNone(result.suggested_policy)
        self.assertEqual(result.fix_category, "customer_code")

    async def test_end_to_end_through_the_real_explain_signal(self):
        """The strongest form of this regression: run the actual
        explain_signal, so a future change to its dunetrace_native shape is
        caught here rather than in production."""
        tool_loop_signal = {
            "id": 42,
            "failure_type": "TOOL_LOOP",
            "severity": "HIGH",
            "run_id": "run-1",
            "agent_id": "support-bot",
            "agent_version": "v1",
            "step_index": 3,
            "confidence": 0.9,
            "detected_at": 0.0,
            "evidence": {"tool": "web_search", "count": 6},
            "what": "w",
            "why_it_matters": "y",
            "evidence_summary": "s",
        }
        signals_settings = MagicMock()
        signals_settings.github_configured = False

        with (
            patch("api_svc.routers.issues.get_issue_by_id", AsyncMock(return_value=_issue())),
            patch("api_svc.routers.issues.agent_failure_pattern", AsyncMock(return_value={})),
            configured_llm("anthropic"),
            patch("api_svc.routers.issues.get_most_recent_signal_id", AsyncMock(return_value=42)),
            patch("api_svc.routers.signals.settings", signals_settings),
            patch(
                "api_svc.routers.signals.get_signal_by_id",
                AsyncMock(return_value=tool_loop_signal),
            ),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch(
                "api_svc.routers.signals._call_llm",
                AsyncMock(
                    return_value={
                        "root_cause": "loop",
                        "fix_content": "unused on this branch",
                        "fix_patch": "",
                    }
                ),
            ),
        ):
            result = await get_issue(7, org_id="org-1")

        self.assertEqual(result.fix_category, "dunetrace_native")
        self.assertIsInstance(result.suggested_fix, str)
        self.assertIsInstance(result.suggested_policy, dict)
        self.assertEqual(
            result.suggested_policy["condition"],
            {"trigger": "tool_call_count", "operator": "gte", "value": 6},
        )

    async def test_unexpected_explain_shape_still_constructs(self):
        """Defence in depth: whatever explain_signal hands back, building the
        response must not raise. A non-str fix_content is coerced, not fatal."""
        weird = {
            "root_cause": {"not": "a string"},
            "fix_category": "customer_code",
            "fix_content": ["also", "not", "a", "string"],
        }
        a, b, c, d, e = self._patches(weird)
        with a, b, c, d, e:
            result = await get_issue(7, org_id="org-1")

        self.assertIsInstance(result.root_cause, str)
        self.assertIsInstance(result.suggested_fix, str)
        self.assertEqual(result.id, 7)


class TestDescribePolicy(unittest.TestCase):
    def test_renders_trigger_operator_value(self):
        text = _describe_policy(
            {
                "condition": {"trigger": "error_count", "operator": "gte", "value": 3},
                "action": {"type": "stop"},
            }
        )
        self.assertIn("Stop the run when error_count >= 3", text)
        self.assertIn("suggested_policy", text)

    def test_non_dict_returns_none(self):
        self.assertIsNone(_describe_policy(None))
        self.assertIsNone(_describe_policy("a string"))

    def test_unknown_shape_degrades_instead_of_raising(self):
        text = _describe_policy({"condition": "expression-form", "action": None})
        self.assertIsInstance(text, str)
        self.assertIn("runtime policy", text)


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
