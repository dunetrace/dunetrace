"""Tests for Phase 6.1's cross-tool analytics. The pure summarizers are tested
against known rows (correct math) and edge cases (no data, samples too small).
The endpoints are tested with mocked queries to confirm they wire query -> pure
summarizer -> response.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from api_svc.elevenlabs_analytics import (
    summarize_cost_by_outcome,
    summarize_truncation_impact,
    summarize_voice_impact,
)
from api_svc.routers.elevenlabs import (
    analytics_cost_by_outcome_endpoint,
    analytics_truncation_impact_endpoint,
    analytics_voice_impact_endpoint,
)

# 0.30 per 1k chars, so 1000 chars -> $0.30.
_PRICING = {"tts": {"default": "elevenlabs", "providers": {"elevenlabs": {"per_1k_chars": 0.30}}}}


def _cost_row(conversation_id=1, chars=1000, credits=1000, gen_count=2, signal_count=0):
    return {
        "conversation_id": conversation_id,
        "external_id": f"call-{conversation_id}",
        "agent_id": "agent-1",
        "chars": chars,
        "credits": credits,
        "gen_count": gen_count,
        "signal_count": signal_count,
    }


class TestCostByOutcome(unittest.TestCase):
    def test_splits_spend_by_success(self):
        rows = [
            _cost_row(1, chars=1000, credits=1000, signal_count=0),  # successful, $0.30
            _cost_row(2, chars=2000, credits=2000, signal_count=3),  # unsuccessful, $0.60
        ]
        out = summarize_cost_by_outcome(rows, _PRICING)
        s = out["summary"]
        self.assertEqual(s["call_count"], 2)
        self.assertEqual(s["unsuccessful_call_count"], 1)
        self.assertAlmostEqual(s["total_cost_usd"], 0.90)
        self.assertAlmostEqual(s["unsuccessful_cost_usd"], 0.60)
        self.assertAlmostEqual(s["successful_cost_usd"], 0.30)
        self.assertAlmostEqual(s["wasted_share"], 0.60 / 0.90)
        # The unsuccessful call is flagged in the per-call list.
        self.assertTrue(next(c for c in out["calls"] if c["conversation_id"] == 2)["unsuccessful"])

    def test_no_data_is_clean_zeros(self):
        out = summarize_cost_by_outcome([], _PRICING)
        self.assertEqual(out["summary"]["call_count"], 0)
        self.assertEqual(out["summary"]["total_cost_usd"], 0.0)
        self.assertIsNone(out["summary"]["wasted_share"])  # no divide-by-zero
        self.assertEqual(out["calls"], [])


class TestVoiceImpact(unittest.TestCase):
    def test_signal_rate_and_sample_flag(self):
        rows = [
            {"voice_id": "A", "voice_name": "Rachel", "run_count": 100, "runs_with_signals": 10},
            {"voice_id": "B", "voice_name": "Domi", "run_count": 5, "runs_with_signals": 4},
        ]
        out = summarize_voice_impact(rows)
        a = next(v for v in out["voices"] if v["voice_id"] == "A")
        b = next(v for v in out["voices"] if v["voice_id"] == "B")
        self.assertAlmostEqual(a["signal_rate"], 0.10)
        self.assertFalse(a["insufficient_data"])  # 100 runs is enough
        self.assertAlmostEqual(b["signal_rate"], 0.80)
        self.assertTrue(b["insufficient_data"])  # 5 runs is too few to trust

    def test_zero_run_voice_has_none_rate(self):
        rows = [{"voice_id": "A", "voice_name": None, "run_count": 0, "runs_with_signals": 0}]
        out = summarize_voice_impact(rows)
        self.assertIsNone(out["voices"][0]["signal_rate"])  # no divide-by-zero
        self.assertTrue(out["voices"][0]["insufficient_data"])


class TestTruncationImpact(unittest.TestCase):
    def test_lift_when_truncation_correlates_with_frustration(self):
        row = {
            "truncated_runs": 50,
            "truncated_with_frustration": 20,  # 40%
            "clean_runs": 100,
            "clean_with_frustration": 10,  # 10%
        }
        out = summarize_truncation_impact(row)
        self.assertAlmostEqual(out["truncated_frustration_rate"], 0.40)
        self.assertAlmostEqual(out["clean_frustration_rate"], 0.10)
        self.assertAlmostEqual(out["lift"], 0.30)  # 30 points more frustration when truncated
        self.assertFalse(out["insufficient_data"])
        self.assertEqual(out["population"], 150)

    def test_insufficient_when_few_truncated_runs(self):
        row = {
            "truncated_runs": 3,
            "truncated_with_frustration": 2,
            "clean_runs": 100,
            "clean_with_frustration": 10,
        }
        out = summarize_truncation_impact(row)
        self.assertTrue(out["insufficient_data"])  # only 3 truncated runs

    def test_no_clean_runs_is_insufficient_and_no_crash(self):
        row = {
            "truncated_runs": 30,
            "truncated_with_frustration": 5,
            "clean_runs": 0,
            "clean_with_frustration": 0,
        }
        out = summarize_truncation_impact(row)
        self.assertIsNone(out["clean_frustration_rate"])
        self.assertIsNone(out["lift"])  # nothing to compare against
        self.assertTrue(out["insufficient_data"])


class TestEndpoints(unittest.IsolatedAsyncioTestCase):
    async def test_cost_endpoint_wires_query_to_summary(self):
        with (
            patch("api_svc.routers.elevenlabs.load_pricing", return_value=_PRICING),
            patch(
                "api_svc.routers.elevenlabs.analytics_cost_by_outcome",
                AsyncMock(return_value=[_cost_row(1, signal_count=1)]),
            ),
        ):
            out = await analytics_cost_by_outcome_endpoint(org_id="org-1")
        self.assertEqual(out["summary"]["unsuccessful_call_count"], 1)

    async def test_voice_endpoint_wires(self):
        with patch(
            "api_svc.routers.elevenlabs.analytics_voice_impact",
            AsyncMock(
                return_value=[
                    {"voice_id": "A", "voice_name": "R", "run_count": 30, "runs_with_signals": 3}
                ]
            ),
        ):
            out = await analytics_voice_impact_endpoint(org_id="org-1")
        self.assertAlmostEqual(out["voices"][0]["signal_rate"], 0.10)

    async def test_truncation_endpoint_passes_frustration_types(self):
        captured = {}

        async def fake_query(org_id, frustration_types):
            captured["types"] = frustration_types
            return {
                "truncated_runs": 0,
                "truncated_with_frustration": 0,
                "clean_runs": 0,
                "clean_with_frustration": 0,
            }

        with patch(
            "api_svc.routers.elevenlabs.analytics_truncation_impact", side_effect=fake_query
        ):
            out = await analytics_truncation_impact_endpoint(org_id="org-1")
        self.assertIn("GOAL_ABANDONMENT", captured["types"])
        # The cause under study is excluded from the frustration set.
        self.assertNotIn("VOICE_TTS_TRUNCATION", captured["types"])
        self.assertTrue(out["insufficient_data"])


if __name__ == "__main__":
    unittest.main()
