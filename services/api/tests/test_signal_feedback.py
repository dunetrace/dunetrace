"""
Endpoint-level tests for Phase 1.4.3's feedback capture
(services/api/api_svc/routers/signals.py::submit_signal_feedback) and the
org-level opt-in toggle (services/api/api_svc/routers/orgs.py).

Calls the route functions directly (this codebase's established pattern —
see test_signals_explain.py) with mocked DB calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.signals import SignalFeedbackRequest, submit_signal_feedback
from api_svc.routers.orgs import (
    SemanticFeedbackSettings,
    get_semantic_feedback_settings,
    set_semantic_feedback_settings,
)


def _signal(source: str = "semantic", failure_type: str = "HALLUCINATION") -> dict:
    return {
        "id": 42,
        "failure_type": failure_type,
        "severity": "HIGH",
        "run_id": "run-1",
        "agent_id": "agent-1",
        "agent_version": "v1",
        "step_index": 0,
        "confidence": 0.9,
        "evidence": {},
        "source": source,
    }


class TestSubmitSignalFeedback(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_verdict_returns_422(self):
        with self.assertRaises(HTTPException) as ctx:
            await submit_signal_feedback(
                1, SignalFeedbackRequest(verdict="not_a_real_verdict"), org_id="org-1"
            )
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_signal_not_found_returns_404(self):
        with patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as ctx:
                await submit_signal_feedback(
                    1, SignalFeedbackRequest(verdict="false_positive"), org_id="org-1"
                )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_structural_signal_rejected(self):
        with patch(
            "api_svc.routers.signals.get_signal_by_id",
            AsyncMock(return_value=_signal(source="structural")),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await submit_signal_feedback(
                    1, SignalFeedbackRequest(verdict="false_positive"), org_id="org-1"
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Slack", ctx.exception.detail)

    async def test_feedback_not_enabled_for_org_returns_403(self):
        with (
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch(
                "api_svc.routers.signals.get_organization_semantic_feedback",
                AsyncMock(return_value={"enabled": False, "auto_suppress": False}),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await submit_signal_feedback(
                    1, SignalFeedbackRequest(verdict="false_positive"), org_id="org-1"
                )
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_missing_org_settings_row_treated_as_disabled(self):
        with (
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch(
                "api_svc.routers.signals.get_organization_semantic_feedback",
                AsyncMock(return_value=None),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await submit_signal_feedback(
                    1, SignalFeedbackRequest(verdict="false_positive"), org_id="org-1"
                )
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_records_feedback_when_enabled(self):
        with (
            patch("api_svc.routers.signals.get_signal_by_id", AsyncMock(return_value=_signal())),
            patch(
                "api_svc.routers.signals.get_organization_semantic_feedback",
                AsyncMock(return_value={"enabled": True, "auto_suppress": False}),
            ),
            patch(
                "api_svc.routers.signals.record_signal_feedback", AsyncMock(return_value=99)
            ) as record_mock,
        ):
            result = await submit_signal_feedback(
                1,
                SignalFeedbackRequest(verdict="false_positive", notes="wrong call"),
                org_id="org-1",
            )

        record_mock.assert_awaited_once_with(1, "org-1", "false_positive", "wrong call")
        self.assertEqual(result["feedback_id"], 99)
        self.assertEqual(result["signal_id"], 1)


class TestSemanticFeedbackSettingsEndpoints(unittest.IsolatedAsyncioTestCase):
    async def test_get_returns_stored_settings(self):
        with patch(
            "api_svc.routers.orgs.get_organization_semantic_feedback",
            AsyncMock(return_value={"enabled": True, "auto_suppress": True}),
        ):
            result = await get_semantic_feedback_settings(org_id="org-1")
        self.assertTrue(result.enabled)
        self.assertTrue(result.auto_suppress)

    async def test_get_defaults_to_disabled_when_org_row_missing(self):
        with patch(
            "api_svc.routers.orgs.get_organization_semantic_feedback",
            AsyncMock(return_value=None),
        ):
            result = await get_semantic_feedback_settings(org_id="org-1")
        self.assertFalse(result.enabled)
        self.assertFalse(result.auto_suppress)

    async def test_patch_updates_settings(self):
        with patch(
            "api_svc.routers.orgs.update_organization_semantic_feedback", AsyncMock()
        ) as update_mock:
            result = await set_semantic_feedback_settings(
                SemanticFeedbackSettings(enabled=True, auto_suppress=False), org_id="org-1"
            )
        update_mock.assert_awaited_once_with("org-1", True, False)
        self.assertTrue(result.enabled)
        self.assertFalse(result.auto_suppress)


if __name__ == "__main__":
    unittest.main()
