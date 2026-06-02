"""
Tests for RiskEngine.

No network, no external deps. Builds minimal RunState fixtures by hand.
"""

from __future__ import annotations

import time
import unittest

from dunetrace.models import (
    AgentEvent,
    EventType,
    ExternalSignal,
    FailureSignal,
    FailureType,
    LlmCall,
    RetrievalResult,
    RiskScore,
    RunState,
    Severity,
    ToolCall,
)
from dunetrace.risk_engine import RiskEngine


# ── Helpers ───────────────────────────────────────────────────────────────────


def _state(**kwargs) -> RunState:
    defaults = dict(run_id="r1", agent_id="a1", agent_version="v1")
    defaults.update(kwargs)
    return RunState(**defaults)


def _tool(
    name: str,
    step: int,
    *,
    success: bool | None = True,
    args_hash: str = "abc",
    error_hash: str | None = None,
) -> ToolCall:
    return ToolCall(
        tool_name=name,
        args_hash=args_hash,
        step_index=step,
        timestamp=time.time(),
        success=success,
        error_hash=error_hash,
    )


def _llm(
    step: int, prompt_tokens: int = 100, finish_reason: str = "stop", latency_ms: int = 200
) -> LlmCall:
    return LlmCall(
        model="gpt-4o",
        prompt_tokens=prompt_tokens,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        step_index=step,
        timestamp=time.time(),
    )


def _event(event_type: EventType, step: int) -> AgentEvent:
    return AgentEvent(
        event_type=event_type, run_id="r1", agent_id="a1", agent_version="v1", step_index=step
    )


def _signal(failure_type: FailureType = FailureType.TOOL_LOOP) -> FailureSignal:
    return FailureSignal(
        failure_type=failure_type,
        severity=Severity.HIGH,
        run_id="r1",
        agent_id="a1",
        agent_version="v1",
        step_index=1,
        confidence=0.9,
        evidence={},
    )


# ── Constructor ───────────────────────────────────────────────────────────────


class TestRiskEngineConstructor(unittest.TestCase):
    def test_unknown_param_raises(self):
        with self.assertRaises(TypeError):
            RiskEngine(UNKNOWN_PARAM=5)

    def test_known_param_overrides(self):
        engine = RiskEngine(HARD_LOOP_CALLS=4)
        self.assertEqual(engine.HARD_LOOP_CALLS, 4)

    def test_default_instance_works(self):
        engine = RiskEngine()
        result = engine.evaluate([], _state())
        self.assertIsInstance(result, RiskScore)


# ── Empty / no-signal runs ────────────────────────────────────────────────────


class TestEmptyRun(unittest.TestCase):
    def test_no_signals_no_state_returns_zero_confidence(self):
        score = RiskEngine().evaluate([], _state())
        self.assertEqual(score.confidence, 0.0)
        self.assertEqual(score.active_signals, 0)

    def test_scores_dict_has_all_five_keys(self):
        score = RiskEngine().evaluate([], _state())
        self.assertEqual(
            set(score.scores.keys()), {"loop", "stagnation", "token", "retry", "latency"}
        )

    def test_severity_is_none_for_normal_run(self):
        score = RiskEngine().evaluate([], _state())
        self.assertIsNone(score.severity)


# ── Feature: loop score ───────────────────────────────────────────────────────


class TestLoopScore(unittest.TestCase):
    def test_zero_tool_calls(self):
        engine = RiskEngine()
        self.assertEqual(engine._loop_score(_state()), 0.0)

    def test_one_tool_call(self):
        state = _state(tool_calls=[_tool("search", 1)])
        self.assertEqual(RiskEngine()._loop_score(state), 0.0)

    def test_all_same_tool_in_window_is_1(self):
        calls = [_tool("search", i) for i in range(5)]
        state = _state(tool_calls=calls)
        self.assertAlmostEqual(RiskEngine()._loop_score(state), 1.0)

    def test_all_different_tools_is_low(self):
        calls = [_tool(f"tool_{i}", i) for i in range(5)]
        state = _state(tool_calls=calls)
        score = RiskEngine()._loop_score(state)
        self.assertLessEqual(score, 0.25)

    def test_score_bounded_0_to_1(self):
        calls = [_tool("x", i) for i in range(20)]
        state = _state(tool_calls=calls)
        score = RiskEngine()._loop_score(state)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_only_last_window_considered(self):
        # First 5 calls are all different, last 5 are all "search"
        calls = [_tool(f"t{i}", i) for i in range(5)] + [_tool("search", i + 5) for i in range(5)]
        state = _state(tool_calls=calls)
        self.assertAlmostEqual(RiskEngine()._loop_score(state), 1.0)


# ── Feature: stagnation score ─────────────────────────────────────────────────


class TestStagnationScore(unittest.TestCase):
    def test_no_tool_calls_returns_zero(self):
        # Agent never used tools — stagnation doesn't apply
        events = [_event(EventType.LLM_CALLED, i) for i in range(4)]
        state = _state(events=events)
        self.assertEqual(RiskEngine()._stagnation_score(state), 0.0)

    def test_tail_all_llm_after_tool_use_is_1(self):
        events = [_event(EventType.TOOL_CALLED, 1)] + [
            _event(EventType.LLM_CALLED, i) for i in range(2, 6)
        ]
        calls = [_tool("search", 1)]
        state = _state(tool_calls=calls, events=events)
        self.assertAlmostEqual(RiskEngine()._stagnation_score(state), 1.0)

    def test_mixed_tail_gives_partial_score(self):
        events = [
            _event(EventType.TOOL_CALLED, 1),
            _event(EventType.LLM_CALLED, 2),
            _event(EventType.TOOL_CALLED, 3),
            _event(EventType.LLM_CALLED, 4),
        ]
        calls = [_tool("search", 1), _tool("search", 3)]
        state = _state(tool_calls=calls, events=events)
        score = RiskEngine()._stagnation_score(state)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


# ── Feature: token score ─────────────────────────────────────────────────────


class TestTokenScore(unittest.TestCase):
    def test_no_llm_calls_returns_zero(self):
        self.assertEqual(RiskEngine()._token_score(_state()), 0.0)

    def test_single_llm_call_returns_zero(self):
        state = _state(llm_calls=[_llm(1, prompt_tokens=500)])
        self.assertEqual(RiskEngine()._token_score(state), 0.0)

    def test_3x_growth_is_1(self):
        calls = [_llm(1, prompt_tokens=1000), _llm(2, prompt_tokens=3000)]
        state = _state(llm_calls=calls)
        self.assertAlmostEqual(RiskEngine()._token_score(state), 1.0)

    def test_no_growth_is_0(self):
        calls = [_llm(1, prompt_tokens=1000), _llm(2, prompt_tokens=1000)]
        state = _state(llm_calls=calls)
        self.assertAlmostEqual(RiskEngine()._token_score(state), 0.0)

    def test_2x_growth_is_half(self):
        calls = [_llm(1, prompt_tokens=1000), _llm(2, prompt_tokens=2000)]
        state = _state(llm_calls=calls)
        # growth=2.0, normalised = (2-1)/(3-1) = 0.5
        self.assertAlmostEqual(RiskEngine()._token_score(state), 0.5)

    def test_tiny_first_tokens_returns_zero(self):
        calls = [_llm(1, prompt_tokens=5), _llm(2, prompt_tokens=5000)]
        state = _state(llm_calls=calls)
        self.assertEqual(RiskEngine()._token_score(state), 0.0)

    def test_tiny_last_tokens_returns_zero(self):
        calls = [_llm(1, prompt_tokens=100), _llm(2, prompt_tokens=50)]
        state = _state(llm_calls=calls)
        self.assertEqual(RiskEngine()._token_score(state), 0.0)

    def test_capped_at_1(self):
        calls = [_llm(1, prompt_tokens=100), _llm(2, prompt_tokens=10_000)]
        state = _state(llm_calls=calls)
        self.assertLessEqual(RiskEngine()._token_score(state), 1.0)


# ── Feature: retry score ─────────────────────────────────────────────────────


class TestRetryScore(unittest.TestCase):
    def test_no_failures_is_zero(self):
        state = _state(tool_calls=[_tool("t", 1, success=True)])
        self.assertEqual(RiskEngine()._retry_score(state), 0.0)

    def test_five_consecutive_failures_is_1(self):
        calls = [_tool("t", i, success=False) for i in range(5)]
        state = _state(tool_calls=calls)
        self.assertAlmostEqual(RiskEngine()._retry_score(state), 1.0)

    def test_success_breaks_streak(self):
        calls = [
            _tool("t", 1, success=False),
            _tool("t", 2, success=True),  # breaks streak
            _tool("t", 3, success=False),
            _tool("t", 4, success=False),
        ]
        state = _state(tool_calls=calls)
        # Only 2 consecutive at tail
        self.assertAlmostEqual(RiskEngine()._retry_score(state), 2 / 5)

    def test_capped_at_1(self):
        calls = [_tool("t", i, success=False) for i in range(20)]
        state = _state(tool_calls=calls)
        self.assertLessEqual(RiskEngine()._retry_score(state), 1.0)


# ── Feature: latency score ────────────────────────────────────────────────────


class TestLatencyScore(unittest.TestCase):
    def test_no_durations_is_zero(self):
        self.assertEqual(RiskEngine()._latency_score(_state()), 0.0)

    def test_within_threshold_is_zero(self):
        events = [_event(EventType.TOOL_CALLED, 1)]
        state = _state(step_durations_ms={1: 10_000}, events=events)
        self.assertEqual(RiskEngine()._latency_score(state), 0.0)

    def test_tool_5x_threshold_is_1(self):
        # tool threshold = 15000ms; 5× = 75000ms → score = (5-1)/4 = 1.0
        events = [_event(EventType.TOOL_CALLED, 1)]
        state = _state(step_durations_ms={1: 75_000}, events=events)
        self.assertAlmostEqual(RiskEngine()._latency_score(state), 1.0)

    def test_llm_uses_higher_threshold(self):
        # llm threshold = 30000ms; 31000 barely over → score > 0
        events = [_event(EventType.LLM_CALLED, 1)]
        state = _state(step_durations_ms={1: 31_000}, events=events)
        score = RiskEngine()._latency_score(state)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.1)

    def test_capped_at_1(self):
        events = [_event(EventType.TOOL_CALLED, 1)]
        state = _state(step_durations_ms={1: 999_999}, events=events)
        self.assertLessEqual(RiskEngine()._latency_score(state), 1.0)


# ── Multi-signal boosting ─────────────────────────────────────────────────────


class TestMultiSignalBoosting(unittest.TestCase):
    def test_single_strong_signal_no_boost(self):
        # Only loop fires strongly; active=1 → multiplier=1.0
        calls = [_tool("search", i) for i in range(5)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertAlmostEqual(score.active_signals, 1)
        # No boost for single active feature
        expected = RiskEngine()._loop_score(state)  # multiplier=1.0
        self.assertAlmostEqual(score.confidence, expected, places=5)

    def test_two_strong_signals_boost_1_2x(self):
        # 3 same-tool calls → loop score high, then 4 different failures → retry high.
        # Total "t" calls = 4 (< HARD_LOOP_CALLS=8), so no hard rule.
        calls = [_tool("search", i) for i in range(3)] + [
            _tool("t", i + 3, success=False) for i in range(4)
        ]
        state = _state(tool_calls=calls)
        engine = RiskEngine()
        score = engine.evaluate([], state)
        self.assertGreaterEqual(score.active_signals, 2)
        # With 2 active features, multiplier=1.2
        base = max(engine._loop_score(state), engine._retry_score(state))
        self.assertAlmostEqual(score.confidence, min(1.0, base * 1.2), places=3)

    def test_confidence_never_exceeds_1(self):
        calls = [_tool("search", i) for i in range(10)] + [
            _tool("search", i + 10, success=False) for i in range(10)
        ]
        llm_calls = [_llm(1, prompt_tokens=100), _llm(2, prompt_tokens=500)]
        events = [_event(EventType.TOOL_CALLED, 1), _event(EventType.LLM_CALLED, 2)] * 5
        tool_calls_b = [_tool("t", i) for i in range(3)]
        state = _state(tool_calls=calls + tool_calls_b, llm_calls=llm_calls, events=events)
        score = RiskEngine().evaluate([], state)
        self.assertLessEqual(score.confidence, 1.0)


# ── Time escalation ───────────────────────────────────────────────────────────


class TestTimeEscalation(unittest.TestCase):
    def test_no_latency_no_escalation(self):
        # Use a state where loop score is well below 1.0 so latency can push higher.
        # 2 "search" out of 3 calls → loop = 2/3 ≈ 0.667, base ≈ 0.667
        calls = [_tool("search", 1), _tool("other", 2), _tool("search", 3)]
        state_fast = _state(tool_calls=calls)
        score_fast = RiskEngine().evaluate([], state_fast)

        # Same run but with a slow tool step → latency_score > 0 → time_mult > 1
        events = [_event(EventType.TOOL_CALLED, 4)]
        state_slow = _state(tool_calls=calls, step_durations_ms={4: 100_000}, events=events)
        score_slow = RiskEngine().evaluate([], state_slow)

        self.assertGreater(score_slow.confidence, score_fast.confidence)

    def test_latency_only_drives_confidence(self):
        # A very slow step with no other failure signals → confidence driven purely by latency.
        events = [_event(EventType.TOOL_CALLED, 1)]
        state = _state(step_durations_ms={1: 100_000}, events=events)
        score = RiskEngine().evaluate([], state)
        # latency_score = (100000/15000 - 1) / 4 = (6.67-1)/4 ≈ 1.0 → capped at 1.0
        self.assertAlmostEqual(score.scores["latency"], 1.0)
        # base_confidence = max(scores) = 1.0 → final confidence = 1.0
        self.assertAlmostEqual(score.confidence, 1.0)


# ── Hard rules ────────────────────────────────────────────────────────────────


class TestHardRules(unittest.TestCase):
    def test_extreme_loop_returns_critical(self):
        # 8 calls to "search", all with same args_hash
        calls = [_tool("search", i, args_hash="same") for i in range(8)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertEqual(score.severity, "CRITICAL")
        self.assertAlmostEqual(score.confidence, 0.98)

    def test_extreme_loop_below_threshold_no_override(self):
        # 7 calls (< 8) — hard rule should NOT fire
        calls = [_tool("search", i, args_hash="same") for i in range(7)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertIsNone(score.severity)

    def test_extreme_loop_low_similarity_no_override(self):
        # 8 calls but all different args_hash (similarity ≈ 0.125)
        calls = [_tool("search", i, args_hash=f"hash_{i}") for i in range(8)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertIsNone(score.severity)

    def test_extreme_failure_streak_returns_high(self):
        calls = [_tool("t", i, success=False) for i in range(5)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertEqual(score.severity, "HIGH")
        self.assertAlmostEqual(score.confidence, 0.95)

    def test_failure_streak_below_threshold_no_override(self):
        calls = [_tool("t", i, success=False) for i in range(4)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertIsNone(score.severity)

    def test_critical_overrides_high(self):
        # Both hard rules trigger — CRITICAL fires first, returned immediately
        calls = [_tool("search", i, args_hash="same", success=False) for i in range(8)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertEqual(score.severity, "CRITICAL")

    def test_hard_rule_skips_scoring(self):
        # Hard rule result should not have normal 5-key scores dict
        calls = [_tool("search", i, args_hash="same") for i in range(8)]
        state = _state(tool_calls=calls)
        score = RiskEngine().evaluate([], state)
        self.assertEqual(score.scores, {"loop": 1.0})


# ── Custom thresholds ─────────────────────────────────────────────────────────


class TestCustomThresholds(unittest.TestCase):
    def test_lower_hard_loop_threshold(self):
        engine = RiskEngine(HARD_LOOP_CALLS=4)
        calls = [_tool("search", i, args_hash="same") for i in range(4)]
        state = _state(tool_calls=calls)
        score = engine.evaluate([], state)
        self.assertEqual(score.severity, "CRITICAL")

    def test_custom_boost_multipliers(self):
        engine = RiskEngine(BOOST_MULTIPLIERS={0: 1.0, 1: 1.0, 2: 2.0, 3: 2.0})
        # 5 same-tool calls → loop > 0.6 → active=1 → no boost
        calls = [_tool("search", i) for i in range(5)]
        state = _state(tool_calls=calls)
        score = engine.evaluate([], state)
        # active=1, multiplier=1.0
        self.assertAlmostEqual(score.confidence, engine._loop_score(state), places=4)


# ── Integration ───────────────────────────────────────────────────────────────


class TestIntegration(unittest.TestCase):
    def test_evaluate_accepts_signals_list(self):
        signals = [_signal(FailureType.TOOL_LOOP), _signal(FailureType.RETRY_STORM)]
        score = RiskEngine().evaluate(signals, _state())
        self.assertIsInstance(score, RiskScore)

    def test_healthy_run_low_confidence(self):
        # Diverse tools, no failures, no token growth, fast steps
        calls = [_tool(f"t{i}", i) for i in range(4)]
        llm_calls = [_llm(1, prompt_tokens=200), _llm(2, prompt_tokens=220)]
        state = _state(tool_calls=calls, llm_calls=llm_calls)
        score = RiskEngine().evaluate([], state)
        self.assertLess(score.confidence, 0.5)

    def test_sick_run_high_confidence(self):
        # Looping tool + all failures + token bloat
        calls = [_tool("search", i, args_hash="same", success=False) for i in range(4)]
        llm_calls = [_llm(1, prompt_tokens=500), _llm(2, prompt_tokens=2000)]
        events = [_event(EventType.LLM_CALLED, i) for i in range(4)]
        state = _state(tool_calls=calls, llm_calls=llm_calls, events=events)
        score = RiskEngine().evaluate([], state)
        self.assertGreater(score.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
