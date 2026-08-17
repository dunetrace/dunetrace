"""
Tests for the integrations worker's orchestration: multi-provider dispatch,
correlation, dedup, confidence normalization, and failure/alert handling. DB
and provider clients are mocked — nothing running required.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import integrations_svc.worker  # must be imported before patch() resolves "integrations_svc.worker.*"
from integrations_svc.providers.base import ExternalEvaluation
from integrations_svc.worker import _poll_one, poll_once, run_worker


def _integration(
    id=1,
    org_id="org-1",
    last_success_at=None,
    first_failure_at=None,
    last_alerted_at=None,
    created_at=None,
):
    return {
        "id": id,
        "org_id": org_id,
        "endpoint_url": "https://cloud.langfuse.com",
        "encrypted_credentials": "encrypted-blob",
        "poll_interval_secs": 60,
        "last_success_at": last_success_at,
        "first_failure_at": first_failure_at,
        "last_alerted_at": last_alerted_at,
        "created_at": created_at or datetime.now(timezone.utc),
    }


def _evaluation(external_id="ev-1", trace_id="trace-1", value=0.9, name="hallucination"):
    return ExternalEvaluation(
        external_id=external_id,
        trace_id=trace_id,
        name=name,
        value=value,
        string_value=None,
        comment="c",
        timestamp=1_700_000_000.0,
        source_url="https://cloud.langfuse.com/project/p/traces/" + trace_id,
    )


def _mock_provider_cls(provider_instance):
    """A stand-in for a provider class — _poll_one calls
    provider_cls(endpoint_url, **creds), so this needs to be callable, not
    just an object with the right methods."""
    return MagicMock(side_effect=lambda *a, **kw: provider_instance)


class TestPollOne(unittest.IsolatedAsyncioTestCase):
    async def test_writes_signal_when_run_correlates(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation()])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"public_key": "pk", "secret_key": "sk"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"langfuse": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=False)),
            patch(
                "integrations_svc.worker.fetch_run_by_trace_id",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                        "org_id": "org-1",
                    }
                ),
            ),
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.mark_processed", AsyncMock()) as mark_mock,
            patch("integrations_svc.worker.record_poll_success", AsyncMock()) as success_mock,
        ):
            await _poll_one("langfuse", _integration())

        write_mock.assert_called_once()
        kwargs = write_mock.call_args.kwargs
        self.assertEqual(kwargs["run_id"], "run-1")
        self.assertEqual(kwargs["agent_id"], "agent-1")
        self.assertEqual(kwargs["failure_type"], "LANGFUSE_HALLUCINATION")
        self.assertEqual(kwargs["confidence"], 0.9)
        mark_mock.assert_called_once_with("org-1", "langfuse", "ev-1")
        success_mock.assert_called_once_with(1)

    async def test_failure_type_prefixed_by_provider_name(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation(name="task_success")])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"api_key": "k", "project_name": "p"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"langsmith": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=False)),
            patch(
                "integrations_svc.worker.fetch_run_by_trace_id",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                        "org_id": "org-1",
                    }
                ),
            ),
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.mark_processed", AsyncMock()),
            patch("integrations_svc.worker.record_poll_success", AsyncMock()),
        ):
            await _poll_one("langsmith", _integration())

        self.assertEqual(write_mock.call_args.kwargs["failure_type"], "LANGSMITH_TASK_SUCCESS")
        self.assertEqual(write_mock.call_args.kwargs["provider"], "langsmith")

    async def test_braintrust_failure_type_prefixed_by_provider_name(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation(name="relevance")])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"api_key": "k", "project_id": "p"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"braintrust": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=False)),
            patch(
                "integrations_svc.worker.fetch_run_by_trace_id",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                        "org_id": "org-1",
                    }
                ),
            ),
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.mark_processed", AsyncMock()),
            patch("integrations_svc.worker.record_poll_success", AsyncMock()),
        ):
            await _poll_one("braintrust", _integration())

        self.assertEqual(write_mock.call_args.kwargs["failure_type"], "BRAINTRUST_RELEVANCE")
        self.assertEqual(write_mock.call_args.kwargs["provider"], "braintrust")

    async def test_skips_already_processed_evaluation(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation()])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"public_key": "pk", "secret_key": "sk"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"langfuse": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=True)),
            patch("integrations_svc.worker.fetch_run_by_trace_id", AsyncMock()) as fetch_mock,
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.record_poll_success", AsyncMock()),
        ):
            await _poll_one("langfuse", _integration())

        fetch_mock.assert_not_called()
        write_mock.assert_not_called()

    async def test_unmatched_trace_id_marks_processed_without_writing(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation()])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"public_key": "pk", "secret_key": "sk"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"langfuse": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=False)),
            patch("integrations_svc.worker.fetch_run_by_trace_id", AsyncMock(return_value=None)),
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.mark_processed", AsyncMock()) as mark_mock,
            patch("integrations_svc.worker.record_poll_success", AsyncMock()),
        ):
            await _poll_one("langfuse", _integration())

        write_mock.assert_not_called()
        mark_mock.assert_called_once_with("org-1", "langfuse", "ev-1")

    async def test_out_of_range_numeric_value_falls_back_to_neutral_confidence(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation(value=42.0)])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"public_key": "pk", "secret_key": "sk"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"langfuse": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=False)),
            patch(
                "integrations_svc.worker.fetch_run_by_trace_id",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                        "org_id": "org-1",
                    }
                ),
            ),
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.mark_processed", AsyncMock()),
            patch("integrations_svc.worker.record_poll_success", AsyncMock()),
        ):
            await _poll_one("langfuse", _integration())

        self.assertEqual(write_mock.call_args.kwargs["confidence"], 0.5)
        self.assertEqual(write_mock.call_args.kwargs["evidence"]["raw_value"], 42.0)

    async def test_categorical_value_falls_back_to_neutral_confidence(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation(value=None)])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"public_key": "pk", "secret_key": "sk"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"langfuse": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=False)),
            patch(
                "integrations_svc.worker.fetch_run_by_trace_id",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                        "org_id": "org-1",
                    }
                ),
            ),
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.mark_processed", AsyncMock()),
            patch("integrations_svc.worker.record_poll_success", AsyncMock()),
        ):
            await _poll_one("langfuse", _integration())

        self.assertEqual(write_mock.call_args.kwargs["confidence"], 0.5)

    async def test_failure_below_alert_threshold_does_not_write_operational_signal(self):
        recent_failure = datetime.now(timezone.utc) - timedelta(minutes=5)
        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "integrations_svc.worker.record_poll_failure",
                AsyncMock(
                    return_value={
                        "consecutive_failures": 3,
                        "first_failure_at": recent_failure,
                        "last_alerted_at": None,
                    }
                ),
            ),
            patch(
                "integrations_svc.worker.write_integration_down_signal", AsyncMock()
            ) as down_mock,
            patch("integrations_svc.worker.record_alert_sent", AsyncMock()),
        ):
            await _poll_one("langfuse", _integration())

        down_mock.assert_not_called()

    async def test_failure_over_30min_writes_operational_signal(self):
        old_failure = datetime.now(timezone.utc) - timedelta(minutes=45)
        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "integrations_svc.worker.record_poll_failure",
                AsyncMock(
                    return_value={
                        "consecutive_failures": 10,
                        "first_failure_at": old_failure,
                        "last_alerted_at": None,
                    }
                ),
            ),
            patch(
                "integrations_svc.worker.write_integration_down_signal", AsyncMock()
            ) as down_mock,
            patch("integrations_svc.worker.record_alert_sent", AsyncMock()) as alert_mock,
        ):
            await _poll_one("langfuse", _integration(org_id="org-2"))

        down_mock.assert_called_once_with("org-2", "langfuse", "boom")
        alert_mock.assert_called_once_with(1)

    async def test_recently_alerted_does_not_re_alert(self):
        old_failure = datetime.now(timezone.utc) - timedelta(minutes=45)
        recent_alert = datetime.now(timezone.utc) - timedelta(minutes=10)
        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "integrations_svc.worker.record_poll_failure",
                AsyncMock(
                    return_value={
                        "consecutive_failures": 10,
                        "first_failure_at": old_failure,
                        "last_alerted_at": recent_alert,
                    }
                ),
            ),
            patch(
                "integrations_svc.worker.write_integration_down_signal", AsyncMock()
            ) as down_mock,
            patch("integrations_svc.worker.record_alert_sent", AsyncMock()),
        ):
            await _poll_one("langfuse", _integration())

        down_mock.assert_not_called()


class TestPollOnce(unittest.IsolatedAsyncioTestCase):
    async def test_returns_zero_when_no_due_integrations(self):
        with patch("integrations_svc.worker.fetch_due_integrations", AsyncMock(return_value=[])):
            count = await poll_once()
        self.assertEqual(count, 0)

    async def test_polls_every_due_integration_for_a_provider(self):
        integrations = [_integration(id=1, org_id="org-1"), _integration(id=2, org_id="org-2")]

        async def fake_fetch_due(provider_name):
            return integrations if provider_name == "langfuse" else []

        with (
            patch("integrations_svc.worker.fetch_due_integrations", side_effect=fake_fetch_due),
            patch("integrations_svc.worker._poll_one", AsyncMock()) as poll_mock,
        ):
            count = await poll_once()

        self.assertEqual(count, 2)
        self.assertEqual(poll_mock.call_count, 2)

    async def test_checks_every_registered_provider(self):
        """Every registered provider must each get its own
        fetch_due_integrations call — a single global fetch would miss
        provider-specific due-polling logic."""
        seen_providers = []

        async def fake_fetch_due(provider_name):
            seen_providers.append(provider_name)
            return []

        with patch("integrations_svc.worker.fetch_due_integrations", side_effect=fake_fetch_due):
            await poll_once()

        self.assertEqual(set(seen_providers), {"langfuse", "langsmith", "braintrust"})

    async def test_integrations_from_different_providers_both_polled(self):
        langfuse_integration = _integration(id=1, org_id="org-1")
        langsmith_integration = _integration(id=2, org_id="org-2")

        async def fake_fetch_due(provider_name):
            if provider_name == "langfuse":
                return [langfuse_integration]
            if provider_name == "langsmith":
                return [langsmith_integration]
            return []

        with (
            patch("integrations_svc.worker.fetch_due_integrations", side_effect=fake_fetch_due),
            patch("integrations_svc.worker._poll_one", AsyncMock()) as poll_mock,
        ):
            count = await poll_once()

        self.assertEqual(count, 2)
        called_providers = {c.args[0] for c in poll_mock.call_args_list}
        self.assertEqual(called_providers, {"langfuse", "langsmith"})


class TestRunWorkerDisabled(unittest.IsolatedAsyncioTestCase):
    async def test_exits_without_opening_pool_when_disabled(self):
        init_mock = AsyncMock()
        with (
            patch("integrations_svc.worker.settings") as mock_settings,
            patch("integrations_svc.worker.init_pool", init_mock),
        ):
            mock_settings.INTEGRATIONS_WORKER_ENABLED = False
            await run_worker()
        init_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
