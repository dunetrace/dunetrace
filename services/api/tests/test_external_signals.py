"""
Endpoint-level tests for Phase 2.4's generic external signal push
(api_svc/routers/external_signals.py). Calls the route function directly
(this codebase's established pattern), mocked DB calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api_svc.routers.external_signals import ExternalSignalRequest, push_external_signal


def _body(**overrides):
    fields = {
        "trace_id": "trace-abc",
        "provider": "ragas",
        "name": "hallucination",
        "external_id": "ragas-eval-1",
        "value": 0.9,
    }
    fields.update(overrides)
    return ExternalSignalRequest(**fields)


class TestExternalSignalRequestValidation(unittest.TestCase):
    def test_requires_at_least_one_of_value_or_string_value(self):
        with self.assertRaises(ValidationError):
            ExternalSignalRequest(
                trace_id="t",
                provider="ragas",
                name="hallucination",
                external_id="e1",
            )

    def test_string_value_alone_is_sufficient(self):
        body = ExternalSignalRequest(
            trace_id="t",
            provider="ragas",
            name="tone",
            external_id="e1",
            string_value="positive",
        )
        self.assertIsNone(body.value)


class TestPushExternalSignal(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_external_id_short_circuits_without_correlating(self):
        with (
            patch(
                "api_svc.routers.external_signals.has_processed_external",
                AsyncMock(return_value=True),
            ),
            patch(
                "api_svc.routers.external_signals.fetch_run_by_trace_id_for_org", AsyncMock()
            ) as fetch_mock,
            patch(
                "api_svc.routers.external_signals.write_pushed_external_signal", AsyncMock()
            ) as write_mock,
        ):
            result = await push_external_signal(_body(), org_id="org-1")

        self.assertTrue(result.duplicate)
        self.assertIsNone(result.signal_id)
        fetch_mock.assert_not_called()
        write_mock.assert_not_called()

    async def test_unmatched_trace_id_returns_404(self):
        with (
            patch(
                "api_svc.routers.external_signals.has_processed_external",
                AsyncMock(return_value=False),
            ),
            patch(
                "api_svc.routers.external_signals.fetch_run_by_trace_id_for_org",
                AsyncMock(return_value=None),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await push_external_signal(_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_matched_trace_id_writes_signal_and_marks_processed(self):
        with (
            patch(
                "api_svc.routers.external_signals.has_processed_external",
                AsyncMock(return_value=False),
            ),
            patch(
                "api_svc.routers.external_signals.fetch_run_by_trace_id_for_org",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                    }
                ),
            ),
            patch(
                "api_svc.routers.external_signals.write_pushed_external_signal",
                AsyncMock(return_value=99),
            ) as write_mock,
            patch(
                "api_svc.routers.external_signals.mark_processed_external", AsyncMock()
            ) as mark_mock,
        ):
            result = await push_external_signal(_body(), org_id="org-1")

        self.assertFalse(result.duplicate)
        self.assertEqual(result.signal_id, 99)
        self.assertEqual(result.run_id, "run-1")
        self.assertEqual(result.agent_id, "agent-1")

        write_mock.assert_called_once()
        kwargs = write_mock.call_args.kwargs
        self.assertEqual(kwargs["failure_type"], "RAGAS_HALLUCINATION")
        self.assertEqual(kwargs["confidence"], 0.9)
        self.assertEqual(kwargs["provider"], "ragas")
        mark_mock.assert_called_once_with("org-1", "ragas", "ragas-eval-1")

    async def test_out_of_range_value_falls_back_to_neutral_confidence(self):
        with (
            patch(
                "api_svc.routers.external_signals.has_processed_external",
                AsyncMock(return_value=False),
            ),
            patch(
                "api_svc.routers.external_signals.fetch_run_by_trace_id_for_org",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                    }
                ),
            ),
            patch(
                "api_svc.routers.external_signals.write_pushed_external_signal",
                AsyncMock(return_value=1),
            ) as write_mock,
            patch("api_svc.routers.external_signals.mark_processed_external", AsyncMock()),
        ):
            await push_external_signal(_body(value=42.0), org_id="org-1")

        self.assertEqual(write_mock.call_args.kwargs["confidence"], 0.5)
        self.assertEqual(write_mock.call_args.kwargs["evidence"]["raw_value"], 42.0)

    async def test_categorical_value_falls_back_to_neutral_confidence(self):
        with (
            patch(
                "api_svc.routers.external_signals.has_processed_external",
                AsyncMock(return_value=False),
            ),
            patch(
                "api_svc.routers.external_signals.fetch_run_by_trace_id_for_org",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                    }
                ),
            ),
            patch(
                "api_svc.routers.external_signals.write_pushed_external_signal",
                AsyncMock(return_value=1),
            ) as write_mock,
            patch("api_svc.routers.external_signals.mark_processed_external", AsyncMock()),
        ):
            await push_external_signal(_body(value=None, string_value="positive"), org_id="org-1")

        self.assertEqual(write_mock.call_args.kwargs["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
