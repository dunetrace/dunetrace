"""
Endpoint-level tests for explain_signal (services/api/api_svc/routers/signals.py)
— root-cause analysis is fully native, and fix handling splits into
dunetrace_native (policy, always auto-applicable) vs customer_code (diff,
manual apply except code_change fixes which can open a GitHub PR).

Calls the route functions directly (this codebase's established pattern —
see test_api.py) with mocked DB/LLM calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from api_svc.routers.signals import (
    OpenPRRequest,
    RecordCopyRequest,
    explain_signal,
    open_pr,
    record_copy,
)


def _signal(
    failure_type: str = "TOOL_AVOIDANCE",
    evidence: dict = None,
    run_id: str = "run-1",
    agent_id: str = "agent-1",
    detected_at: float = 0.0,
) -> dict:
    return {
        "id": 42,
        "failure_type": failure_type,
        "severity": "HIGH",
        "run_id": run_id,
        "agent_id": agent_id,
        "agent_version": "v1",
        "step_index": 3,
        "confidence": 0.9,
        "detected_at": detected_at,
        "evidence": evidence or {},
        "what": "The agent did a thing.",
        "why_it_matters": "It matters.",
        "evidence_summary": "summary",
    }


def _llm_result(root_cause="because X", fix_content="add this", fix_patch="+ add this"):
    return {"root_cause": root_cause, "fix_content": fix_content, "fix_patch": fix_patch}


def _settings_mock(*, anthropic_key="key", github_configured=False):
    s = MagicMock()
    s.ANTHROPIC_API_KEY = anthropic_key
    s.OPENAI_API_KEY = None
    s.github_configured = github_configured
    return s


class TestExplainSignalGating(unittest.IsolatedAsyncioTestCase):
    async def test_no_llm_key_returns_503(self):
        with patch("api_svc.routers.signals.settings", _settings_mock(anthropic_key=None)):
            with self.assertRaises(HTTPException) as ctx:
                await explain_signal(1, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_signal_not_found_returns_404(self):
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=None)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await explain_signal(1, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_explain_works_with_no_events(self):
        """Root-cause analysis never depends on any external system — only on
        this run's own stored events, which may be sparse or absent."""
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")
        self.assertEqual(result["root_cause"], "because X")
        self.assertNotIn("langfuse_prompt_name", result)


class TestExplainSignalDunetraceNative(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_returns_suggested_policy(self):
        signal = _signal(
            failure_type="TOOL_LOOP", evidence={"count": 5, "first_step": 1, "last_step": 5}
        )
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")

        self.assertEqual(result["fix_category"], "dunetrace_native")
        self.assertEqual(result["fix_type"], "policy")
        self.assertFalse(result["apply_blocked"])
        self.assertEqual(
            result["suggested_policy"]["condition"],
            {"trigger": "tool_call_count", "operator": "gte", "value": 5},
        )
        self.assertEqual(result["suggested_policy"]["agent_id"], "agent-1")

    async def test_dunetrace_native_always_unblocked(self):
        """A Policy is Dunetrace's own config — apply_blocked must be False,
        no external system involved at all."""
        signal = _signal(failure_type="STEP_COUNT_INFLATION", evidence={"current_steps": 12})
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")
        self.assertFalse(result["apply_blocked"])


class TestExplainSignalCustomerCode(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_addition_is_always_apply_blocked(self):
        """No external prompt store exists — a prompt_addition fix is always
        a diff to copy in manually."""
        signal = _signal(failure_type="TOOL_AVOIDANCE")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")

        self.assertEqual(result["fix_category"], "customer_code")
        self.assertEqual(result["fix_type"], "prompt_addition")
        self.assertTrue(result["apply_blocked"])

    async def test_code_change_blocked_when_github_not_configured(self):
        signal = _signal(failure_type="CONTEXT_BLOAT")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(github_configured=False)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")

        self.assertEqual(result["fix_type"], "code_change")
        self.assertTrue(result["apply_blocked"])

    async def test_code_change_unblocked_when_github_configured(self):
        """code_change has its own one-click path — POST .../open-pr — gated
        only on GitHub being configured."""
        signal = _signal(failure_type="CONTEXT_BLOAT")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(github_configured=True)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")

        self.assertEqual(result["fix_type"], "code_change")
        self.assertFalse(result["apply_blocked"])

    async def test_no_auto_apply_always_blocked(self):
        signal = _signal(failure_type="PROMPT_INJECTION_SIGNAL")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(github_configured=True)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")

        self.assertEqual(result["fix_type"], "no_auto_apply")
        self.assertTrue(result["apply_blocked"])

    async def test_source_is_native_when_events_exist(self):
        signal = _signal(failure_type="TOOL_AVOIDANCE")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch(
                "api_svc.routers.signals.get_run_detail",
                AsyncMock(
                    return_value={
                        "events": [{"event_type": "run.started", "step_index": 0, "payload": {}}]
                    }
                ),
            ),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")
        self.assertEqual(result["source"], "native")

    async def test_source_is_signal_only_when_no_native_events(self):
        signal = _signal(failure_type="TOOL_AVOIDANCE")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value=None)),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, org_id="org-1")
        self.assertEqual(result["source"], "signal_only")


class TestRecordCopy(unittest.IsolatedAsyncioTestCase):
    async def test_signal_not_found_returns_404(self):
        with patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as ctx:
                await record_copy(1, RecordCopyRequest(fix_content="x"), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_records_fix_with_clipboard_applied_via(self):
        with (
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch("api_svc.routers.signals.record_fix", AsyncMock(return_value=7)) as record_mock,
        ):
            result = await record_copy(1, RecordCopyRequest(fix_content="new text"), org_id="org-1")

        record_mock.assert_awaited_once_with(
            org_id="org-1",
            signal_id=1,
            run_id="run-1",
            fix_content="new text",
            applied_via="clipboard",
        )
        self.assertEqual(result["fix_id"], 7)


def _pr_settings_mock(*, github_configured=True):
    s = MagicMock()
    s.github_configured = github_configured
    return s


def _open_pr_body():
    return OpenPRRequest(root_cause="because X", fix_content="add this", fix_patch="+ add this")


class TestOpenPR(unittest.IsolatedAsyncioTestCase):
    async def test_github_not_configured_returns_503(self):
        with patch("api_svc.routers.signals.settings", _pr_settings_mock(github_configured=False)):
            with self.assertRaises(HTTPException) as ctx:
                await open_pr(1, _open_pr_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_signal_not_found_returns_404(self):
        with (
            patch("api_svc.routers.signals.settings", _pr_settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=None)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await open_pr(1, _open_pr_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_prompt_injection_blocked_even_when_github_configured(self):
        """open-pr is now the only auto-apply endpoint left — it must carry
        the same no_auto_apply guard the removed apply-fix endpoint used to."""
        signal = _signal(failure_type="PROMPT_INJECTION_SIGNAL")
        with (
            patch("api_svc.routers.signals.settings", _pr_settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await open_pr(1, _open_pr_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_creates_pr_and_records_fix(self):
        signal = _signal(failure_type="CONTEXT_BLOAT")
        pr_result = {"pr_url": "https://github.com/o/r/pull/1", "pr_number": 1, "branch": "b"}
        with (
            patch("api_svc.routers.signals.settings", _pr_settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.github_client.create_fix_pr", AsyncMock(return_value=pr_result)),
            patch("api_svc.routers.signals.record_fix", AsyncMock(return_value=5)),
        ):
            result = await open_pr(1, _open_pr_body(), org_id="org-1")
        self.assertEqual(result["pr_number"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
