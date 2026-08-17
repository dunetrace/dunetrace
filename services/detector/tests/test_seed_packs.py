"""
Tests that _seed_packs writes the registered packs (Phase 1.2 voice pack)
into the packs table. No DB — the asyncpg connection is a recording fake,
same convention the rest of this suite uses for DB-backed functions.

Run: PYTHONPATH=../../packages/sdk-py:. python -m unittest tests.test_seed_packs -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from detector_svc.db import _seed_packs


class TestSeedPacks(unittest.IsolatedAsyncioTestCase):
    async def test_seeds_voice_pack_with_nine_detector_names(self):
        conn = AsyncMock()
        await _seed_packs(conn)

        # Find the INSERT ... INTO packs call for "voice".
        voice_call = None
        for call in conn.execute.await_args_list:
            if len(call.args) >= 2 and call.args[1] == "voice":
                voice_call = call.args
                break

        self.assertIsNotNone(voice_call, "voice pack was not seeded")
        name, description, detector_names = voice_call[1], voice_call[2], voice_call[3]
        self.assertEqual(name, "voice")
        self.assertIsInstance(description, str)
        self.assertTrue(description)
        self.assertEqual(len(detector_names), 9)
        self.assertIn("VoiceTtsTruncationDetector", detector_names)

    async def test_detector_names_are_class_names(self):
        conn = AsyncMock()
        await _seed_packs(conn)
        for call in conn.execute.await_args_list:
            if len(call.args) >= 2 and call.args[1] == "voice":
                detector_names = call.args[3]
                # _seed_packs stores class __name__ values.
                self.assertIn("VoiceTranscriptionConfidenceDropDetector", detector_names)
                return
        self.fail("voice pack was not seeded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
