"""
Endpoint-level tests for explain_signal and apply_fix (services/api/api_svc/
routers/signals.py) — the D10 rewrite that made root-cause analysis native
(no Langfuse required) and split fix handling into dunetrace_native
(policy, auto-apply) vs customer_code (diff, optional external-store push).

Calls the route functions directly (this codebase's established pattern —
see test_api.py) with mocked DB/LLM/Langfuse calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from api_svc.routers.signals import ApplyFixRequest, ExplainRequest, apply_fix, explain_signal


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


def _settings_mock(*, anthropic_key="key", langfuse_configured=False):
    s = MagicMock()
    s.ANTHROPIC_API_KEY = anthropic_key
    s.OPENAI_API_KEY = None
    s.langfuse_configured = langfuse_configured
    return s


class TestExplainSignalGating(unittest.IsolatedAsyncioTestCase):
    async def test_no_llm_key_returns_503(self):
        with patch("api_svc.routers.signals.settings", _settings_mock(anthropic_key=None)):
            with self.assertRaises(HTTPException) as ctx:
                await explain_signal(1, ExplainRequest(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_signal_not_found_returns_404(self):
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=None)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await explain_signal(1, ExplainRequest(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_no_langfuse_configured_does_not_block_explain(self):
        """The core D10 behavior change: explain works with zero Langfuse setup."""
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(langfuse_configured=False)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")
        self.assertEqual(result["root_cause"], "because X")
        self.assertIsNone(result["langfuse_prompt_name"])


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
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")

        self.assertEqual(result["fix_category"], "dunetrace_native")
        self.assertEqual(result["fix_type"], "policy")
        self.assertFalse(result["apply_blocked"])
        self.assertEqual(
            result["suggested_policy"]["condition"],
            {"trigger": "tool_call_count", "operator": "gte", "value": 5},
        )
        self.assertEqual(result["suggested_policy"]["agent_id"], "agent-1")

    async def test_dunetrace_native_never_blocked_regardless_of_langfuse(self):
        """A Policy is Dunetrace's own config — apply_blocked must be False
        even when no external store is connected at all."""
        signal = _signal(failure_type="STEP_COUNT_INFLATION", evidence={"current_steps": 12})
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(langfuse_configured=False)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")
        self.assertFalse(result["apply_blocked"])


class TestExplainSignalCustomerCode(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_addition_blocked_when_no_store_connected(self):
        # get_connected_prompt_store() reads api_svc.config.settings via its
        # own deferred import — a separate reference from
        # api_svc.routers.signals.settings (bound at that module's import
        # time), and NOT patched by mocking the latter alone. Both must be
        # patched or this test is at the mercy of the real environment's
        # .env (which has real Langfuse credentials configured for this
        # repo's dev stack).
        signal = _signal(failure_type="TOOL_AVOIDANCE")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(langfuse_configured=False)),
            patch("api_svc.config.settings", _settings_mock(langfuse_configured=False)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")

        self.assertEqual(result["fix_category"], "customer_code")
        self.assertEqual(result["fix_type"], "prompt_addition")
        self.assertTrue(result["apply_blocked"])  # no store connected

    async def test_prompt_addition_unblocked_when_store_connected(self):
        signal = _signal(failure_type="TOOL_AVOIDANCE")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(langfuse_configured=True)),
            patch("api_svc.config.settings", _settings_mock(langfuse_configured=True)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
            patch(
                "api_svc.langfuse_client.fetch_langfuse_trace",
                AsyncMock(return_value={"observations": []}),
            ),
        ):
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")

        self.assertFalse(result["apply_blocked"])

    async def test_code_change_type_always_blocked_even_with_store_connected(self):
        signal = _signal(failure_type="CONTEXT_BLOAT")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(langfuse_configured=True)),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value={"events": []})),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
            patch(
                "api_svc.langfuse_client.fetch_langfuse_trace",
                AsyncMock(return_value={"observations": []}),
            ),
        ):
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")

        self.assertEqual(result["fix_type"], "code_change")
        self.assertTrue(result["apply_blocked"])

    async def test_langfuse_trace_not_found_degrades_gracefully(self):
        """A missing Langfuse trace must not fail the whole request — explain
        already has native events regardless."""
        signal = _signal(failure_type="TOOL_AVOIDANCE")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock(langfuse_configured=True)),
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
            patch(
                "api_svc.langfuse_client.fetch_langfuse_trace",
                AsyncMock(side_effect=LookupError("not found")),
            ),
        ):
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")

        self.assertEqual(result["source"], "native")
        self.assertIsNone(result["langfuse_prompt_name"])

    async def test_source_is_signal_only_when_no_native_events(self):
        signal = _signal(failure_type="TOOL_AVOIDANCE")
        with (
            patch("api_svc.routers.signals.settings", _settings_mock()),
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)),
            patch("api_svc.routers.signals.get_run_detail", AsyncMock(return_value=None)),
            patch("api_svc.routers.signals._call_llm", AsyncMock(return_value=_llm_result())),
        ):
            result = await explain_signal(1, ExplainRequest(), org_id="org-1")
        self.assertEqual(result["source"], "signal_only")


class TestApplyFix(unittest.IsolatedAsyncioTestCase):
    async def test_no_store_connected_returns_503(self):
        req = ApplyFixRequest(fix_content="x", langfuse_prompt_name="p")
        with (
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch("api_svc.prompt_stores.get_connected_prompt_store", return_value=None),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await apply_fix(1, req, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_prompt_injection_blocked_regardless_of_store(self):
        signal = _signal(failure_type="PROMPT_INJECTION_SIGNAL")
        req = ApplyFixRequest(fix_content="x", langfuse_prompt_name="p")
        with patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=signal)):
            with self.assertRaises(HTTPException) as ctx:
                await apply_fix(1, req, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_store_connected_pushes_fix_and_records_it(self):
        req = ApplyFixRequest(fix_content="new text", langfuse_prompt_name="my-prompt")
        fake_store = MagicMock()
        fake_store.push_fix = AsyncMock(
            return_value={
                "new_version": 4,
                "prompt_url": "https://x/prompts/my-prompt",
                "old_text": "old",
                "new_text": "old\n\nnew text",
            }
        )
        with (
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch("api_svc.prompt_stores.get_connected_prompt_store", return_value=fake_store),
            patch("api_svc.routers.signals.record_fix", AsyncMock(return_value=99)),
        ):
            result = await apply_fix(1, req, org_id="org-1")

        fake_store.push_fix.assert_awaited_once_with("my-prompt", "new text")
        self.assertEqual(result["fix_id"], 99)
        self.assertEqual(result["new_version"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
