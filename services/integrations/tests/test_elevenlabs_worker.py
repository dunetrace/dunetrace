"""Tests for the ElevenLabs worker's orchestration: fetch, store, high-water
mark advance, dedup, failure isolation, and the >30min operational alert. DB
and provider clients are mocked — nothing running required. Mirrors
test_worker.py's structure for the evaluation-provider worker.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import integrations_svc.elevenlabs_worker  # import before patch() resolves module paths
from integrations_svc.elevenlabs_worker import (
    _OVERLAP_SECS,
    _poll_one,
    poll_once,
    run_worker,
)
from integrations_svc.providers.elevenlabs import ElevenLabsGeneration


def _integration(
    id=1,
    org_id="org-1",
    last_seen_generation_at=None,
    first_failure_at=None,
    last_alerted_at=None,
    created_at=None,
):
    return {
        "id": id,
        "org_id": org_id,
        "encrypted_credentials": "encrypted-blob",
        "poll_interval_secs": 300,
        "last_seen_generation_at": last_seen_generation_at,
        "first_failure_at": first_failure_at,
        "last_alerted_at": last_alerted_at,
        "created_at": created_at or datetime.now(timezone.utc),
    }


def _gen(generation_id="hist-1", generated_at=1_752_000_000.0, character_count=42):
    return ElevenLabsGeneration(
        generation_id=generation_id,
        voice_id="voice-abc",
        voice_name="Rachel",
        model="eleven_multilingual_v2",
        character_count=character_count,
        text="your order shipped",
        source="TTS",
        generated_at=generated_at,
    )


def _provider_returning(generations):
    """A stand-in ElevenLabsProvider class: _poll_one calls
    ElevenLabsProvider(**creds), so this must be callable and return an instance
    whose fetch_generations is an AsyncMock."""
    instance = MagicMock()
    instance.fetch_generations = AsyncMock(return_value=generations)
    cls = MagicMock(side_effect=lambda *a, **kw: instance)
    cls._instance = instance  # exposed for assertions
    return cls


class TestPollOne(unittest.IsolatedAsyncioTestCase):
    async def test_stores_generations_and_advances_high_water_mark(self):
        gens = [_gen("h1", generated_at=100.0), _gen("h2", generated_at=250.0)]
        provider_cls = _provider_returning(gens)
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch("integrations_svc.elevenlabs_worker.ElevenLabsProvider", provider_cls),
            patch(
                "integrations_svc.elevenlabs_worker.store_generation",
                AsyncMock(return_value=True),
            ) as store_mock,
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()
            ) as success_mock,
        ):
            await _poll_one(_integration(id=7, last_seen_generation_at=50.0))

        self.assertEqual(store_mock.await_count, 2)
        # High-water mark advances to the newest generated_at seen.
        success_mock.assert_awaited_once_with(7, 250.0)

    async def test_since_uses_last_seen_minus_overlap(self):
        provider_cls = _provider_returning([])
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch("integrations_svc.elevenlabs_worker.ElevenLabsProvider", provider_cls),
            patch("integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()),
        ):
            await _poll_one(_integration(last_seen_generation_at=1_000_000.0))

        called_since = provider_cls._instance.fetch_generations.await_args.args[0]
        self.assertEqual(called_since, 1_000_000.0 - _OVERLAP_SECS)

    async def test_first_poll_since_uses_created_at_not_full_history(self):
        created = datetime(2026, 7, 1, tzinfo=timezone.utc)
        provider_cls = _provider_returning([])
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch("integrations_svc.elevenlabs_worker.ElevenLabsProvider", provider_cls),
            patch("integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()),
        ):
            await _poll_one(_integration(last_seen_generation_at=None, created_at=created))

        called_since = provider_cls._instance.fetch_generations.await_args.args[0]
        self.assertAlmostEqual(called_since, created.timestamp() - _OVERLAP_SECS, places=3)

    async def test_no_generations_leaves_high_water_mark_unchanged(self):
        provider_cls = _provider_returning([])
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch("integrations_svc.elevenlabs_worker.ElevenLabsProvider", provider_cls),
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()
            ) as success_mock,
        ):
            await _poll_one(_integration(id=3, last_seen_generation_at=999.0))

        success_mock.assert_awaited_once_with(3, None)  # None => leave mark unchanged

    async def test_deduped_generation_not_counted_but_poll_still_succeeds(self):
        provider_cls = _provider_returning([_gen("h1"), _gen("h2")])
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                return_value={"api_key": "k"},
            ),
            patch("integrations_svc.elevenlabs_worker.ElevenLabsProvider", provider_cls),
            # Both already stored -> store_generation returns False for each.
            patch(
                "integrations_svc.elevenlabs_worker.store_generation",
                AsyncMock(return_value=False),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_poll_success", AsyncMock()
            ) as success_mock,
        ):
            await _poll_one(_integration())

        success_mock.assert_awaited_once()  # success recorded regardless of dedup


class TestFailureIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_failure_below_threshold_records_failure_no_alert(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_poll_failure",
                AsyncMock(
                    return_value={
                        "consecutive_failures": 2,
                        "first_failure_at": recent,
                        "last_alerted_at": None,
                    }
                ),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.write_integration_down_signal", AsyncMock()
            ) as down_mock,
            patch("integrations_svc.elevenlabs_worker.record_elevenlabs_alert_sent", AsyncMock()),
        ):
            await _poll_one(_integration())  # must not raise — failure is isolated

        down_mock.assert_not_called()

    async def test_failure_over_30min_writes_operational_signal(self):
        old = datetime.now(timezone.utc) - timedelta(minutes=45)
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_poll_failure",
                AsyncMock(
                    return_value={
                        "consecutive_failures": 12,
                        "first_failure_at": old,
                        "last_alerted_at": None,
                    }
                ),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.write_integration_down_signal", AsyncMock()
            ) as down_mock,
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_alert_sent", AsyncMock()
            ) as alert_mock,
        ):
            await _poll_one(_integration(id=9, org_id="org-2"))

        down_mock.assert_awaited_once_with("org-2", "elevenlabs", "boom")
        alert_mock.assert_awaited_once_with(9)

    async def test_recently_alerted_does_not_re_alert(self):
        old = datetime.now(timezone.utc) - timedelta(minutes=45)
        recent_alert = datetime.now(timezone.utc) - timedelta(minutes=10)
        with (
            patch(
                "integrations_svc.elevenlabs_worker.decrypt_credentials",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.record_elevenlabs_poll_failure",
                AsyncMock(
                    return_value={
                        "consecutive_failures": 12,
                        "first_failure_at": old,
                        "last_alerted_at": recent_alert,
                    }
                ),
            ),
            patch(
                "integrations_svc.elevenlabs_worker.write_integration_down_signal", AsyncMock()
            ) as down_mock,
            patch("integrations_svc.elevenlabs_worker.record_elevenlabs_alert_sent", AsyncMock()),
        ):
            await _poll_one(_integration())

        down_mock.assert_not_called()


class TestPollOnce(unittest.IsolatedAsyncioTestCase):
    async def test_returns_zero_when_none_due(self):
        with patch(
            "integrations_svc.elevenlabs_worker.fetch_due_elevenlabs_integrations",
            AsyncMock(return_value=[]),
        ):
            self.assertEqual(await poll_once(), 0)

    async def test_polls_every_due_integration(self):
        integrations = [_integration(id=1, org_id="a"), _integration(id=2, org_id="b")]
        with (
            patch(
                "integrations_svc.elevenlabs_worker.fetch_due_elevenlabs_integrations",
                AsyncMock(return_value=integrations),
            ),
            patch("integrations_svc.elevenlabs_worker._poll_one", AsyncMock()) as poll_mock,
        ):
            count = await poll_once()
        self.assertEqual(count, 2)
        self.assertEqual(poll_mock.await_count, 2)


class TestRunWorkerDisabled(unittest.IsolatedAsyncioTestCase):
    async def test_exits_without_opening_pool_when_disabled(self):
        init_mock = AsyncMock()
        with (
            patch("integrations_svc.elevenlabs_worker.settings") as mock_settings,
            patch("integrations_svc.elevenlabs_worker.init_pool", init_mock),
        ):
            mock_settings.ELEVENLABS_WORKER_ENABLED = False
            await run_worker()
        init_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
