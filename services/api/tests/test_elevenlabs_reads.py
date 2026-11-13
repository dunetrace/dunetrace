"""Tests for Phase 5.1's ElevenLabs correlated-generation read endpoints
(api_svc/routers/elevenlabs.py): cost derivation via voice-pricing, the honest
uncertainty flag, filter passthrough, call-cost aggregation, and backward-compat
empties. Route functions called directly with mocked queries; no DB, no network.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from api_svc.routers.elevenlabs import (
    call_elevenlabs_cost,
    list_generations,
)

# Deterministic ElevenLabs TTS rate so cost_usd assertions are exact.
_PRICING = {"tts": {"default": "elevenlabs", "providers": {"elevenlabs": {"per_1k_chars": 0.30}}}}


def _gen_row(
    generation_id="hist-1",
    voice_id="voice-abc",
    voice_name="Rachel",
    model="eleven_multilingual_v2",
    character_count=1000,
    cost_credits=1000,
    method="generation_id",
    confidence=1.0,
    run_id="run-1",
    agent_id="agent-1",
):
    return {
        "generation_id": generation_id,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "model": model,
        "character_count": character_count,
        "cost_credits": cost_credits,
        "source": "TTS",
        "generated_at": 1_752_000_000.0,
        "run_id": run_id,
        "agent_id": agent_id,
        "correlation_method": method,
        "correlation_confidence": confidence,
    }


class TestListGenerationsForRun(unittest.IsolatedAsyncioTestCase):
    async def test_computes_cost_usd_from_characters(self):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.get_run_elevenlabs_generations",
                AsyncMock(return_value=[_gen_row(character_count=1000)]),
            ),
        ):
            out = await list_generations(run_id="run-1", org_id="org-1")

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].cost_usd, 0.30)  # 1000 chars * 0.30/1k
        self.assertEqual(out[0].cost_credits, 1000)
        self.assertEqual(out[0].voice_name, "Rachel")

    async def test_run_scoped_ignores_filter_params(self):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.get_run_elevenlabs_generations",
                AsyncMock(return_value=[]),
            ) as run_q,
            patch("api_svc.routers.elevenlabs.list_elevenlabs_generations", AsyncMock()) as list_q,
        ):
            await list_generations(run_id="run-1", voice_id="v", model="m", org_id="org-1")

        run_q.assert_awaited_once_with("org-1", "run-1")
        list_q.assert_not_awaited()  # run_id takes the run-scoped path

    async def test_empty_run_returns_empty_list(self):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.get_run_elevenlabs_generations",
                AsyncMock(return_value=[]),
            ),
        ):
            out = await list_generations(run_id="run-x", org_id="org-1")
        self.assertEqual(out, [])


class TestUncertaintyFlag(unittest.IsolatedAsyncioTestCase):
    async def _uncertain_for(self, method, confidence):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.get_run_elevenlabs_generations",
                AsyncMock(return_value=[_gen_row(method=method, confidence=confidence)]),
            ),
        ):
            out = await list_generations(run_id="run-1", org_id="org-1")
        return out[0].uncertain

    async def test_char_time_is_uncertain(self):
        self.assertTrue(await self._uncertain_for("char_time", 0.70))

    async def test_voice_char_time_is_confident(self):
        self.assertFalse(await self._uncertain_for("voice_char_time", 0.85))

    async def test_generation_id_is_confident(self):
        self.assertFalse(await self._uncertain_for("generation_id", 1.0))


class TestFilterListing(unittest.IsolatedAsyncioTestCase):
    async def test_passes_filters_through(self):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.list_elevenlabs_generations",
                AsyncMock(return_value=[_gen_row()]),
            ) as list_q,
        ):
            out = await list_generations(
                run_id=None,
                voice_id="voice-abc",
                model="eleven_multilingual_v2",
                min_credits=500,
                limit=50,
                org_id="org-1",
            )

        list_q.assert_awaited_once_with("org-1", "voice-abc", "eleven_multilingual_v2", 500, 50)
        self.assertEqual(out[0].run_id, "run-1")  # run link is present for the explorer


class TestCallCost(unittest.IsolatedAsyncioTestCase):
    async def test_aggregates_and_prices(self):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.get_call_elevenlabs_cost",
                AsyncMock(
                    return_value={
                        "generation_count": 3,
                        "character_count": 2000,
                        "cost_credits": 2000,
                    }
                ),
            ),
        ):
            out = await call_elevenlabs_cost(conversation_id=7, org_id="org-1")

        self.assertEqual(out.generation_count, 3)
        self.assertEqual(out.cost_credits, 2000)
        self.assertAlmostEqual(out.cost_usd, 0.60)  # 2000 chars * 0.30/1k

    async def test_no_data_returns_zeros(self):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.get_call_elevenlabs_cost",
                AsyncMock(
                    return_value={"generation_count": 0, "character_count": 0, "cost_credits": 0}
                ),
            ),
        ):
            out = await call_elevenlabs_cost(conversation_id=7, org_id="org-1")

        self.assertEqual(out.generation_count, 0)
        self.assertEqual(out.cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
