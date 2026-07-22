"""
Unit tests for Phase 2.1 call-level metric derivation (api_svc.db.queries).

These test _compute_call_metrics directly with controlled event timestamps, so
every completion-status path and every derived metric is exercised without a
database. Live end-to-end shape is verified against the running stack via the
/v1/calls endpoint.

Run: make test-api  (or python -m pytest services/api/tests/test_calls.py)
"""

from __future__ import annotations

import datetime
import unittest

from api_svc.db.queries import _compute_call_metrics, _is_drop_reason

UTC = datetime.timezone.utc
T0 = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _at(seconds: float) -> datetime.datetime:
    return T0 + datetime.timedelta(seconds=seconds)


def _ev(event_type: str, at: float, **payload) -> dict:
    return {"event_type": event_type, "payload": payload, "received_at": _at(at)}


class TestCompletionStatus(unittest.TestCase):
    def test_natural_when_nothing_abnormal(self):
        events = [
            _ev("transcription.received", 0, text="hi"),
            _ev("tts.generated", 1, text="hello"),
        ]
        self.assertEqual(_compute_call_metrics(events, [])["completion_status"], "natural")

    def test_escalated_beats_everything(self):
        events = [
            _ev("policy.triggered", 1, action_type="escalate_to_human"),
            _ev("run.errored", 2),  # even with an error, escalation wins
        ]
        self.assertEqual(_compute_call_metrics(events, [])["completion_status"], "escalated")

    def test_dropped_on_run_errored(self):
        events = [_ev("transcription.received", 0, text="hi"), _ev("run.errored", 1)]
        self.assertEqual(_compute_call_metrics(events, [])["completion_status"], "dropped")

    def test_dropped_on_hangup_signal(self):
        events = [_ev("tts.generated", 0, text="one moment", latency_ms=6000)]
        signals = [{"failure_type": "VOICE_LATENCY_INDUCED_HANGUP"}]
        self.assertEqual(_compute_call_metrics(events, signals)["completion_status"], "dropped")

    def test_dropped_on_call_ended_drop_reason(self):
        events = [
            _ev("transcription.received", 0, text="hi"),
            _ev(
                "external.signal", 1, signal_name="call_ended", meta={"reason": "silence-timed-out"}
            ),
        ]
        self.assertEqual(_compute_call_metrics(events, [])["completion_status"], "dropped")

    def test_natural_on_call_ended_normal_reason(self):
        events = [
            _ev("transcription.received", 0, text="hi"),
            _ev(
                "external.signal",
                1,
                signal_name="call_ended",
                meta={"reason": "customer-ended-call"},
            ),
        ]
        self.assertEqual(_compute_call_metrics(events, [])["completion_status"], "natural")


class TestDerivedMetrics(unittest.TestCase):
    def test_duration_from_event_span(self):
        events = [_ev("transcription.received", 0, text="a"), _ev("tts.generated", 12.5, text="b")]
        self.assertEqual(_compute_call_metrics(events, [])["duration_seconds"], 12.5)

    def test_silence_pct(self):
        # 10s call, 3s of silence -> 30%.
        events = [
            _ev("transcription.received", 0, text="a"),
            _ev("voice_activity.detected", 2, type="silence", duration_ms=3000),
            _ev("tts.generated", 10, text="b"),
        ]
        self.assertEqual(_compute_call_metrics(events, [])["silence_pct"], 0.3)

    def test_talk_ratio_from_turn_taking_spans(self):
        # agent floor for 2s, then caller floor for 6s, final event at t=8.
        events = [
            _ev("turn_taking.changed", 0, action="agent_speaking"),
            _ev("turn_taking.changed", 2, action="user_speaking"),
            _ev("tts.generated", 8, text="done"),
        ]
        m = _compute_call_metrics(events, [])
        self.assertEqual(m["agent_talk_ms"], 2000)
        self.assertEqual(m["caller_talk_ms"], 6000)
        self.assertEqual(m["agent_talk_ratio"], 0.25)

    def test_ratio_none_when_no_turns(self):
        events = [_ev("transcription.received", 0, text="a"), _ev("tts.generated", 1, text="b")]
        self.assertIsNone(_compute_call_metrics(events, [])["agent_talk_ratio"])

    def test_signal_count_and_sentiment_placeholder(self):
        events = [_ev("transcription.received", 0, text="a"), _ev("tts.generated", 1, text="b")]
        signals = [
            {"failure_type": "VOICE_SILENCE_TIMEOUT"},
            {"failure_type": "VOICE_BARGE_IN_FAILURE"},
        ]
        m = _compute_call_metrics(events, signals)
        self.assertEqual(m["voice_signal_count"], 2)
        self.assertIsNone(m["sentiment_trend"])  # Phase 3 placeholder


class TestDropReasonMapping(unittest.TestCase):
    def test_drop_reasons(self):
        for r in (
            "silence-timed-out",
            "customer-did-not-answer",
            "pipeline-error-x",
            "provider-fault",
        ):
            self.assertTrue(_is_drop_reason(r), r)

    def test_natural_reasons(self):
        for r in ("customer-ended-call", "assistant-ended-call", None, ""):
            self.assertFalse(_is_drop_reason(r), r)


class TestCallCost(unittest.TestCase):
    """Phase 2.2 cost attribution against the built-in default rates."""

    def setUp(self):
        from api_svc.voice_pricing import DEFAULT_PRICING

        self.pricing = DEFAULT_PRICING

    def _cost(self, events, agent_id="voice-agent"):
        from api_svc.db.queries import _compute_call_cost

        return _compute_call_cost(events, agent_id, self.pricing)

    def test_stt_cost_from_audio_seconds(self):
        # 60s at deepgram_nova3 default $0.0048/min = $0.0048.
        events = [_ev("transcription.received", 0, text="hi", audio_seconds=60)]
        c = self._cost(events)
        self.assertAlmostEqual(c["cost_breakdown"]["stt"], 0.0048, places=6)

    def test_stt_zero_without_audio_seconds(self):
        # No audio_seconds -> not measured, never guessed.
        events = [_ev("transcription.received", 0, text="hi")]
        self.assertEqual(self._cost(events)["cost_breakdown"]["stt"], 0.0)

    def test_tts_cost_from_char_count(self):
        # 1000 chars at deepgram_aura2 default $0.030/1k = $0.030.
        events = [_ev("tts.generated", 0, text="x" * 1000)]
        c = self._cost(events)
        self.assertAlmostEqual(c["cost_breakdown"]["tts"], 0.030, places=6)

    def test_llm_cost_from_tokens(self):
        # gpt-4o-mini: $0.15/1M in, $0.60/1M out. 1000 in + 500 out = $0.00045.
        events = [
            _ev("llm.called", 0, model="gpt-4o-mini", prompt_tokens=1000),
            _ev("llm.responded", 1, completion_tokens=500),
        ]
        c = self._cost(events)
        self.assertAlmostEqual(c["cost_breakdown"]["llm"], 0.00045, places=6)

    def test_telephony_off_by_default(self):
        events = [
            _ev("transcription.received", 0, text="hi", audio_seconds=1),
            _ev("tts.generated", 30, text="bye"),
        ]
        self.assertEqual(self._cost(events)["cost_breakdown"]["telephony"], 0.0)

    def test_total_is_sum_of_categories(self):
        events = [
            _ev("transcription.received", 0, text="hi", audio_seconds=60),
            _ev("tts.generated", 1, text="x" * 1000),
            _ev("llm.called", 0, model="gpt-4o-mini", prompt_tokens=1000),
            _ev("llm.responded", 1, completion_tokens=500),
        ]
        c = self._cost(events)
        b = c["cost_breakdown"]
        self.assertAlmostEqual(
            c["cost_usd"], b["stt"] + b["llm"] + b["tts"] + b["telephony"], places=6
        )

    def test_agent_override_picks_provider(self):
        from api_svc.voice_pricing import providers_for

        pricing = dict(self.pricing)
        pricing["agent_overrides"] = {"billing-line": {"stt": "openai_gpt4o_transcribe"}}
        self.assertEqual(providers_for("billing-line", pricing)["stt"], "openai_gpt4o_transcribe")
        self.assertEqual(providers_for("other", pricing)["stt"], "deepgram_nova3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
