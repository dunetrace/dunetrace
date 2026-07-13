"""
Tests for Phase 1.0's pack activation: detector_svc/packs.py's TTL cache
and detectors.py::get_detectors()'s pack filtering. No DB — fetch_org_
enabled_packs is patched directly, same convention as every other DB-backed
function in this test suite.
"""

from __future__ import annotations

import unittest
from unittest import mock
from unittest.mock import AsyncMock, patch

import detector_svc.packs as packs_module
import detector_svc.detectors as detectors_module
from dunetrace.detectors import BaseDetector
from dunetrace.packs.base import DetectorPack, PACK_REGISTRY, register_pack


class TestGetEnabledPacksCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        packs_module._cache.clear()

    def tearDown(self):
        packs_module._cache.clear()

    async def test_fetches_from_db_on_first_call(self):
        with patch(
            "detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])
        ) as mock_fetch:
            result = await packs_module.get_enabled_packs("org-1")
        mock_fetch.assert_awaited_once_with("org-1")
        self.assertEqual(result, {"voice"})

    async def test_second_call_within_ttl_does_not_hit_db_again(self):
        with patch(
            "detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])
        ) as mock_fetch:
            await packs_module.get_enabled_packs("org-1")
            await packs_module.get_enabled_packs("org-1")
        self.assertEqual(mock_fetch.await_count, 1)

    async def test_expired_ttl_triggers_a_fresh_fetch(self):
        with patch(
            "detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])
        ) as mock_fetch:
            await packs_module.get_enabled_packs("org-1")
            packs_module._invalidate("org-1")
            await packs_module.get_enabled_packs("org-1")
        self.assertEqual(mock_fetch.await_count, 2)

    async def test_orgs_are_cached_independently(self):
        """Tenant isolation: org A's cached result must never leak to org B."""
        with patch(
            "detector_svc.packs.fetch_org_enabled_packs",
            AsyncMock(side_effect=lambda org_id: ["voice"] if org_id == "org-A" else []),
        ):
            result_a = await packs_module.get_enabled_packs("org-A")
            result_b = await packs_module.get_enabled_packs("org-B")
        self.assertEqual(result_a, {"voice"})
        self.assertEqual(result_b, set())

    async def test_no_packs_enabled_returns_empty_set(self):
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=[])):
            result = await packs_module.get_enabled_packs("org-1")
        self.assertEqual(result, set())


class _FakeDetector(BaseDetector):
    name = "FAKE_VOICE_DETECTOR"
    pack = "voice"

    def on_run_completion(self, state):
        return None


class TestGetDetectorsPackFiltering(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        packs_module._cache.clear()
        self._registry_patcher = mock.patch.dict(PACK_REGISTRY, clear=True)
        self._registry_patcher.start()

        class _FakeVoicePack(DetectorPack):
            name = "voice"
            description = "test pack"
            detectors = [_FakeDetector]

        register_pack(_FakeVoicePack())

    def tearDown(self):
        self._registry_patcher.stop()
        packs_module._cache.clear()

    async def test_pack_detector_absent_when_org_has_not_activated_it(self):
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=[])):
            result = await detectors_module.get_detectors("default", "org-not-activated")
        names = [d.name for d in result]
        self.assertNotIn("FAKE_VOICE_DETECTOR", names)
        self.assertGreater(len(names), 0)  # built-ins still present

    async def test_pack_detector_present_when_org_has_activated_it(self):
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])):
            result = await detectors_module.get_detectors("default", "org-activated")
        names = [d.name for d in result]
        self.assertIn("FAKE_VOICE_DETECTOR", names)

    async def test_built_ins_always_present_regardless_of_pack_activation(self):
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])):
            with_pack = await detectors_module.get_detectors("default", "org-a")
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=[])):
            without_pack = await detectors_module.get_detectors("default", "org-b")
        builtin_names_with = {d.name for d in with_pack} - {"FAKE_VOICE_DETECTOR"}
        builtin_names_without = {d.name for d in without_pack}
        self.assertEqual(builtin_names_with, builtin_names_without)

    async def test_tenant_isolation_org_a_activation_does_not_affect_org_b(self):
        async def fetch(org_id):
            return ["voice"] if org_id == "org-A" else []

        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(side_effect=fetch)):
            result_a = await detectors_module.get_detectors("default", "org-A")
            result_b = await detectors_module.get_detectors("default", "org-B")
        self.assertIn("FAKE_VOICE_DETECTOR", [d.name for d in result_a])
        self.assertNotIn("FAKE_VOICE_DETECTOR", [d.name for d in result_b])

    async def test_pack_detector_instantiation_failure_is_skipped_not_raised(self):
        class _BrokenPackDetector(BaseDetector):
            name = "BROKEN_PACK_DETECTOR"
            pack = "voice"

            def __init__(self, **overrides):
                raise RuntimeError("boom")

            def on_run_completion(self, state):
                return None

        class _BrokenPack(DetectorPack):
            name = "voice"
            description = "test pack"
            detectors = [_FakeDetector, _BrokenPackDetector]

        PACK_REGISTRY.clear()
        register_pack(_BrokenPack())

        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])):
            result = await detectors_module.get_detectors("default", "org-1")  # must not raise
        names = [d.name for d in result]
        self.assertIn("FAKE_VOICE_DETECTOR", names)
        self.assertNotIn("BROKEN_PACK_DETECTOR", names)

    async def test_named_category_also_gets_pack_detectors(self):
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])):
            result = await detectors_module.get_detectors("web-research", "org-1")
        names = [d.name for d in result]
        self.assertIn("FAKE_VOICE_DETECTOR", names)


# Voice detector names (Phase 1.2) — the actual shipped pack, not a fake.
_VOICE_DETECTOR_NAMES = {
    "VOICE_TRANSCRIPTION_CONFIDENCE_DROP",
    "VOICE_SILENCE_TIMEOUT",
    "VOICE_TURN_TAKING_COLLISION",
    "VOICE_LATENCY_INDUCED_HANGUP",
    "VOICE_AUDIO_QUALITY_DEGRADATION",
    "VOICE_SPEAKER_CONFUSION",
    "VOICE_BARGE_IN_FAILURE",
    "VOICE_TTS_TRUNCATION",
    "VOICE_VAD_FALSE_TRIGGER",
}


class TestRealVoicePackThroughGetDetectors(unittest.IsolatedAsyncioTestCase):
    """Exercises the actual Phase 1.2 voice pack (registered at import via
    dunetrace.packs) end-to-end through get_detectors — not a fake stand-in."""

    def setUp(self):
        packs_module._cache.clear()

    def tearDown(self):
        packs_module._cache.clear()

    async def test_all_nine_voice_detectors_present_when_activated(self):
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=["voice"])):
            result = await detectors_module.get_detectors("default", "org-1")
        names = {d.name for d in result}
        self.assertTrue(_VOICE_DETECTOR_NAMES.issubset(names))

    async def test_voice_detectors_absent_when_not_activated(self):
        with patch("detector_svc.packs.fetch_org_enabled_packs", AsyncMock(return_value=[])):
            result = await detectors_module.get_detectors("default", "org-2")
        names = {d.name for d in result}
        self.assertEqual(_VOICE_DETECTOR_NAMES & names, set())
        self.assertGreater(len(names), 0)  # built-ins still present


class TestWorkerResolvesPackDetectorClass(unittest.TestCase):
    """The worker's shadow-flag lookup must find pack detector classes, which
    are deliberately NOT in CUSTOM_DETECTOR_REGISTRY (Phase 1.0)."""

    def test_resolves_real_voice_detector(self):
        from detector_svc.worker import _resolve_custom_detector_class

        cls = _resolve_custom_detector_class("VOICE_TTS_TRUNCATION")
        self.assertIsNotNone(cls)
        self.assertEqual(cls.pack, "voice")

    def test_returns_none_for_unknown_name(self):
        from detector_svc.worker import _resolve_custom_detector_class

        self.assertIsNone(_resolve_custom_detector_class("NO_SUCH_DETECTOR"))

    def test_pack_detector_shadow_default_is_honored(self):
        from detector_svc.worker import _resolve_custom_detector_class

        cls = _resolve_custom_detector_class("VOICE_SILENCE_TIMEOUT")
        # SHADOW_BY_DEFAULT is inherited True — the whole point is this is
        # reachable at all, so a pack author's override would take effect.
        self.assertTrue(cls.SHADOW_BY_DEFAULT)


if __name__ == "__main__":
    unittest.main()
