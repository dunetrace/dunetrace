"""
Tests for Phase 1.0's pack registration mechanism
(dunetrace/packs/base.py). PACK_REGISTRY is module-level shared state, so
every test patches it to a fresh dict rather than mutating the real one.

Run: python -m unittest tests.test_packs -v
"""

from __future__ import annotations

import unittest
import unittest.mock

from dunetrace.detectors import BaseDetector
from dunetrace.packs.base import DetectorPack, PACK_REGISTRY, register_pack


class _FakeDetectorA(BaseDetector):
    name = "FAKE_DETECTOR_A"
    pack = "fake-pack"


class _FakeDetectorB(BaseDetector):
    name = "FAKE_DETECTOR_B"
    pack = "fake-pack"


class _FakePack(DetectorPack):
    name = "fake-pack"
    description = "A pack for testing registration."
    detectors = [_FakeDetectorA, _FakeDetectorB]


class TestRegisterPack(unittest.TestCase):
    def setUp(self):
        self._patcher = unittest.mock.patch.dict(PACK_REGISTRY, clear=True)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_register_pack_adds_to_registry(self):
        register_pack(_FakePack())
        self.assertIn("fake-pack", PACK_REGISTRY)

    def test_registered_pack_carries_its_detector_classes(self):
        register_pack(_FakePack())
        self.assertEqual(PACK_REGISTRY["fake-pack"].detectors, [_FakeDetectorA, _FakeDetectorB])

    def test_registered_pack_carries_its_description(self):
        register_pack(_FakePack())
        self.assertEqual(PACK_REGISTRY["fake-pack"].description, "A pack for testing registration.")

    def test_re_registering_the_same_name_is_last_write_wins(self):
        register_pack(_FakePack())

        class _FakePackV2(DetectorPack):
            name = "fake-pack"
            description = "Updated description."
            detectors = [_FakeDetectorA]

        register_pack(_FakePackV2())
        self.assertEqual(PACK_REGISTRY["fake-pack"].description, "Updated description.")
        self.assertEqual(PACK_REGISTRY["fake-pack"].detectors, [_FakeDetectorA])

    def test_two_distinct_packs_coexist(self):
        class _OtherPack(DetectorPack):
            name = "other-pack"
            description = "Another pack."
            detectors = []

        register_pack(_FakePack())
        register_pack(_OtherPack())
        self.assertEqual(set(PACK_REGISTRY.keys()), {"fake-pack", "other-pack"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
