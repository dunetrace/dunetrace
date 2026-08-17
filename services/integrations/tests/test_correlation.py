"""Tests for correlation (Phase 4.4). The matching algorithm is pure, so most
of this exercises match_generation directly across every tier and every honest
non-match. correlate_once is tested with mocked DB reads/writes to prove the
pending-vs-give-up policy and failure isolation.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from integrations_svc.correlation import (
    CorrelationOutcome,
    correlate_once,
    match_generation,
)


def _event(
    id,
    timestamp=1000.0,
    text=None,
    voice_id=None,
    provider_generation_id=None,
    run_id="run-1",
    agent_id="agent-1",
):
    return {
        "id": id,
        "run_id": run_id,
        "agent_id": agent_id,
        "timestamp": timestamp,
        "text": text,
        "voice_id": voice_id,
        "provider_generation_id": provider_generation_id,
    }


def _match(
    candidates,
    *,
    generation_id="gen-1",
    text=None,
    voice_id=None,
    character_count=42,
    generated_at=1000.0,
    char_tolerance=0.10,
):
    return match_generation(
        generation_id=generation_id,
        text=text,
        voice_id=voice_id,
        character_count=character_count,
        generated_at=generated_at,
        candidates=candidates,
        char_tolerance=char_tolerance,
    )


class TestMatchGeneration(unittest.TestCase):
    def test_no_candidates_is_no_candidate_events(self):
        out = _match([])
        self.assertFalse(out.matched)
        self.assertEqual(out.reason, "no_candidate_events")

    def test_generation_id_is_deterministic_top_priority(self):
        # Even though several events sit in the window, the one carrying the
        # generation id wins outright at confidence 1.0.
        candidates = [
            _event(1, text="x" * 999),
            _event(2, provider_generation_id="gen-1", text="mismatched text"),
        ]
        out = _match(candidates, generation_id="gen-1", text="the real text", character_count=13)
        self.assertTrue(out.matched)
        self.assertEqual(out.event_id, 2)
        self.assertEqual(out.method, "generation_id")
        self.assertEqual(out.confidence, 1.0)

    def test_exact_text_single_match(self):
        candidates = [_event(1, text="your order shipped"), _event(2, text="different")]
        out = _match(candidates, text="your order shipped", character_count=18)
        self.assertTrue(out.matched)
        self.assertEqual(out.event_id, 1)
        self.assertEqual(out.method, "exact_text")
        self.assertEqual(out.confidence, 0.97)

    def test_exact_text_tie_breaks_to_nearest_timestamp(self):
        candidates = [
            _event(1, timestamp=1000.0, text="hi"),
            _event(2, timestamp=1005.0, text="hi"),
        ]
        out = _match(candidates, text="hi", character_count=2, generated_at=1004.0)
        self.assertTrue(out.matched)
        self.assertEqual(out.event_id, 2)  # 1005 is nearer to 1004 than 1000
        self.assertEqual(out.method, "exact_text")
        self.assertEqual(out.confidence, 0.90)

    def test_char_time_single_match_within_tolerance(self):
        # No generation id, no text on the event side — fall to char count.
        candidates = [_event(1, text="x" * 40)]  # 40 vs gen 42 -> within 10%
        out = _match(candidates, text=None, character_count=42)
        self.assertTrue(out.matched)
        self.assertEqual(out.method, "char_time")
        self.assertEqual(out.confidence, 0.70)

    def test_char_out_of_tolerance_is_no_char_match(self):
        candidates = [_event(1, text="x" * 10)]  # 10 vs 42 -> way outside 10%
        out = _match(candidates, text=None, character_count=42)
        self.assertFalse(out.matched)
        self.assertEqual(out.reason, "no_char_match")

    def test_voice_id_disambiguates_multiple_char_matches(self):
        candidates = [
            _event(1, text="x" * 42, voice_id="voice-A"),
            _event(2, text="x" * 42, voice_id="voice-B"),
        ]
        out = _match(candidates, text=None, voice_id="voice-B", character_count=42)
        self.assertTrue(out.matched)
        self.assertEqual(out.event_id, 2)
        self.assertEqual(out.method, "voice_char_time")
        self.assertEqual(out.confidence, 0.85)

    def test_ambiguous_when_char_matches_and_no_voice_to_separate(self):
        candidates = [_event(1, text="x" * 42), _event(2, text="x" * 42)]
        out = _match(candidates, text=None, voice_id=None, character_count=42)
        self.assertFalse(out.matched)
        self.assertEqual(out.reason, "ambiguous_multiple_matches")

    def test_ambiguous_when_voice_matches_multiple(self):
        candidates = [
            _event(1, text="x" * 42, voice_id="voice-A"),
            _event(2, text="x" * 42, voice_id="voice-A"),
        ]
        out = _match(candidates, text=None, voice_id="voice-A", character_count=42)
        self.assertFalse(out.matched)
        self.assertEqual(out.reason, "ambiguous_multiple_matches")

    def test_exact_text_beats_a_competing_char_match(self):
        # One event matches by text, another only by char count. Text wins.
        candidates = [_event(1, text="the exact reply"), _event(2, text="x" * 15)]
        out = _match(candidates, text="the exact reply", character_count=15)
        self.assertTrue(out.matched)
        self.assertEqual(out.event_id, 1)
        self.assertEqual(out.method, "exact_text")

    def test_zero_char_generation_matches_zero_char_event(self):
        out = _match([_event(1, text="")], text=None, character_count=0)
        self.assertTrue(out.matched)
        self.assertEqual(out.method, "char_time")


def _gen_row(
    id=1,
    org_id="org-1",
    generation_id="gen-1",
    character_count=42,
    text="hello",
    voice_id="v",
    generated_at=1000.0,
):
    return {
        "id": id,
        "org_id": org_id,
        "generation_id": generation_id,
        "character_count": character_count,
        "text": text,
        "voice_id": voice_id,
        "generated_at": generated_at,
        "fetched_at": None,
    }


class TestCorrelateOnce(unittest.IsolatedAsyncioTestCase):
    async def test_writes_match_when_found(self):
        with (
            patch(
                "integrations_svc.correlation.fetch_uncorrelated_generations",
                AsyncMock(return_value=[_gen_row()]),
            ),
            patch(
                "integrations_svc.correlation.fetch_candidate_tts_events",
                AsyncMock(
                    return_value=[_event(99, text="hello", run_id="run-9", agent_id="agent-9")]
                ),
            ),
            patch(
                "integrations_svc.correlation.mark_generation_correlated", AsyncMock()
            ) as corr_mock,
            patch("integrations_svc.correlation.mark_generation_unmatched", AsyncMock()) as un_mock,
        ):
            summary = await correlate_once()

        corr_mock.assert_awaited_once()
        args = corr_mock.await_args.args
        self.assertEqual(args[0], 1)  # generation row id
        self.assertEqual(args[1], 99)  # matched event id
        self.assertEqual(args[2], "run-9")  # denormalized run_id from matched event
        self.assertEqual(args[3], "agent-9")  # denormalized agent_id
        self.assertEqual(args[4], "exact_text")  # method
        un_mock.assert_not_awaited()
        self.assertEqual(summary["matched"], 1)

    async def test_recent_unmatched_stays_pending_not_given_up(self):
        import time

        recent = time.time() - 60  # 1 minute old, well under give-up
        with (
            patch(
                "integrations_svc.correlation.fetch_uncorrelated_generations",
                AsyncMock(return_value=[_gen_row(generated_at=recent, text="hello")]),
            ),
            patch(
                "integrations_svc.correlation.fetch_candidate_tts_events",
                AsyncMock(return_value=[]),  # no candidates yet
            ),
            patch("integrations_svc.correlation.mark_generation_correlated", AsyncMock()),
            patch("integrations_svc.correlation.mark_generation_unmatched", AsyncMock()) as un_mock,
        ):
            summary = await correlate_once()

        un_mock.assert_not_awaited()  # too young to give up — retry next pass
        self.assertEqual(summary["still_pending"], 1)

    async def test_old_unmatched_is_recorded_as_drift(self):
        import time

        old = time.time() - 7200  # 2 hours old, past the 1h give-up
        with (
            patch(
                "integrations_svc.correlation.fetch_uncorrelated_generations",
                AsyncMock(return_value=[_gen_row(generated_at=old, text="hello")]),
            ),
            patch(
                "integrations_svc.correlation.fetch_candidate_tts_events",
                AsyncMock(return_value=[]),
            ),
            patch("integrations_svc.correlation.mark_generation_correlated", AsyncMock()),
            patch("integrations_svc.correlation.mark_generation_unmatched", AsyncMock()) as un_mock,
        ):
            summary = await correlate_once()

        un_mock.assert_awaited_once_with(1, "no_candidate_events")
        self.assertEqual(summary["unmatched"], 1)

    async def test_per_row_error_is_isolated(self):
        with (
            patch(
                "integrations_svc.correlation.fetch_uncorrelated_generations",
                AsyncMock(return_value=[_gen_row(id=1), _gen_row(id=2)]),
            ),
            patch(
                "integrations_svc.correlation.fetch_candidate_tts_events",
                AsyncMock(side_effect=[RuntimeError("db blip"), [_event(5, text="hello")]]),
            ),
            patch(
                "integrations_svc.correlation.mark_generation_correlated", AsyncMock()
            ) as corr_mock,
            patch("integrations_svc.correlation.mark_generation_unmatched", AsyncMock()),
        ):
            summary = await correlate_once()

        # First row raised and was skipped; second row still correlated.
        self.assertEqual(summary["processed"], 2)
        self.assertEqual(summary["matched"], 1)
        corr_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
