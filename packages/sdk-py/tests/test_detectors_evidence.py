"""
Evidence field tests for enriched detectors, plus coverage for detectors
not yet tested (RetryStorm, GoalAbandonment, SlowStep, EmptyLlmResponse,
StepCountInflation, CascadingToolFailure, FirstStepFailure).

All tests are deterministic, no DB, no network.
Run: pytest packages/sdk-py/tests/test_detectors_evidence.py -v
"""

from __future__ import annotations

import time
import unittest

from dunetrace.models import (
    AgentEvent,
    EventType,
    ExternalSignal,
    FailureType,
    LlmCall,
    RetrievalResult,
    RunState,
    Severity,
    ToolCall,
)
from dunetrace.detectors import (
    CascadingToolFailureDetector,
    ContextBloatDetector,
    EmptyLlmResponseDetector,
    FirstStepFailureDetector,
    GoalAbandonmentDetector,
    LlmTruncationLoopDetector,
    ReasoningSpinDetector,
    RetryStormDetector,
    SlowStepDetector,
    StepCountInflationDetector,
    ToolLoopDetector,
)

# ── Factories ──────────────────────────────────────────────────────────────────


def make_state(**kwargs) -> RunState:
    defaults = dict(
        run_id="test-run",
        agent_id="test-agent",
        agent_version="abc12345",
        available_tools=["web_search", "calculator"],
    )
    defaults.update(kwargs)
    return RunState(**defaults)


def make_tool_call(
    name: str,
    step: int = 0,
    success=None,
    args_hash: str = "aaa",
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


def make_llm_call(
    finish_reason: str = "stop",
    prompt_tokens: int | None = 500,
    step: int = 1,
    model: str = "gpt-4o-mini",
    output_length: int = 100,
) -> LlmCall:
    return LlmCall(
        model=model,
        prompt_tokens=prompt_tokens,
        finish_reason=finish_reason,
        latency_ms=1200,
        step_index=step,
        timestamp=time.time(),
        output_length=output_length,
    )


def make_event(
    event_type: EventType, step: int = 1, ts: float | None = None
) -> AgentEvent:
    return AgentEvent(
        event_type=event_type,
        step_index=step,
        timestamp=ts or time.time(),
        run_id="test-run",
        agent_id="test-agent",
        agent_version="abc12345",
    )


# ── ToolLoopDetector — evidence fields ────────────────────────────────────────


class TestToolLoopEvidence(unittest.TestCase):
    detector = ToolLoopDetector()

    def _firing_state(self, args_hashes=None, successes=None):
        calls = []
        for i in range(5):
            h = args_hashes[i] if args_hashes else "same"
            s = successes[i] if successes else None
            calls.append(make_tool_call("web_search", step=i, args_hash=h, success=s))
        state = make_state()
        state.tool_calls = calls
        state.current_step = 5
        return state

    def test_step_indices_in_evidence(self):
        state = self._firing_state()
        signal = self.detector.check(state)
        assert signal is not None
        assert "step_indices" in signal.evidence
        assert signal.evidence["step_indices"] == [0, 1, 2, 3, 4]

    def test_args_identical_true_when_same_hash(self):
        state = self._firing_state(args_hashes=["aaa"] * 5)
        signal = self.detector.check(state)
        assert signal.evidence["args_identical"] is True

    def test_args_identical_false_when_different_hashes(self):
        state = self._firing_state(args_hashes=["aaa", "bbb", "ccc", "ddd", "eee"])
        signal = self.detector.check(state)
        assert signal.evidence["args_identical"] is False

    def test_args_similar_true_when_two_unique_hashes(self):
        state = self._firing_state(args_hashes=["aaa", "bbb", "aaa", "bbb", "aaa"])
        signal = self.detector.check(state)
        assert signal.evidence["args_similar"] is True
        assert signal.evidence["args_identical"] is False

    def test_success_rate_none_when_no_results_reported(self):
        state = self._firing_state()
        signal = self.detector.check(state)
        assert signal.evidence["success_rate"] is None

    def test_success_rate_computed_when_results_present(self):
        # 2 successes, 3 failures → 0.4
        state = self._firing_state(successes=[True, False, True, False, False])
        signal = self.detector.check(state)
        assert signal is not None
        assert round(signal.evidence["success_rate"], 2) == 0.4

    def test_success_rate_1_0_when_all_succeed(self):
        state = self._firing_state(successes=[True] * 5)
        signal = self.detector.check(state)
        assert signal.evidence["success_rate"] == 1.0

    def test_first_and_last_step_in_evidence(self):
        state = self._firing_state()
        signal = self.detector.check(state)
        assert signal.evidence["first_step"] == 0
        assert signal.evidence["last_step"] == 4

    def test_args_hashes_in_evidence(self):
        state = self._firing_state(args_hashes=["x", "y", "x", "y", "x"])
        signal = self.detector.check(state)
        assert "args_hashes" in signal.evidence
        assert len(signal.evidence["args_hashes"]) == 5


# ── GoalAbandonmentDetector ───────────────────────────────────────────────────


class TestGoalAbandonmentDetector(unittest.TestCase):
    detector = GoalAbandonmentDetector()

    def _make_stalling_state(
        self, stall_steps: int = 4, tool_step: int = 1, current_step: int = 10
    ):
        state = make_state()
        state.exit_reason = None
        state.tool_calls = [make_tool_call("search_db", step=tool_step)]
        state.current_step = current_step
        state.events = [
            make_event(EventType.LLM_CALLED, step=current_step - stall_steps + i)
            for i in range(stall_steps)
        ]
        return state

    def test_fires_when_stalled(self):
        state = self._make_stalling_state()
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.failure_type == FailureType.GOAL_ABANDONMENT

    def test_last_tool_used_in_evidence(self):
        state = self._make_stalling_state()
        signal = self.detector.check(state)
        assert signal.evidence["last_tool_used"] == "search_db"

    def test_steps_since_last_tool_in_evidence(self):
        state = self._make_stalling_state(tool_step=2, current_step=10)
        signal = self.detector.check(state)
        assert signal.evidence["steps_since_last_tool"] == 8  # 10 - 2

    def test_stall_event_sequence_in_evidence(self):
        state = self._make_stalling_state()
        signal = self.detector.check(state)
        seq = signal.evidence["stall_event_sequence"]
        assert isinstance(seq, list)
        assert len(seq) == 4
        assert all("llm" in s for s in seq)

    def test_no_signal_when_exit_reason_set(self):
        state = self._make_stalling_state()
        state.exit_reason = "final_answer"
        assert self.detector.check(state) is None

    def test_no_signal_when_no_tool_calls(self):
        state = self._make_stalling_state()
        state.tool_calls = []
        assert self.detector.check(state) is None

    def test_no_signal_when_not_enough_events(self):
        state = make_state()
        state.exit_reason = None
        state.tool_calls = [make_tool_call("search_db", step=1)]
        state.current_step = 3
        state.events = [make_event(EventType.LLM_CALLED, step=3)]  # only 1, need 4
        assert self.detector.check(state) is None

    def test_no_signal_when_recent_events_include_tool_call(self):
        state = make_state()
        state.exit_reason = None
        state.tool_calls = [make_tool_call("search_db", step=1)]
        state.current_step = 6
        state.events = [
            make_event(EventType.LLM_CALLED, step=3),
            make_event(EventType.TOOL_CALLED, step=4),  # tool event breaks stall
            make_event(EventType.LLM_CALLED, step=5),
            make_event(EventType.LLM_CALLED, step=6),
        ]
        assert self.detector.check(state) is None

    def test_custom_stall_steps(self):
        detector = GoalAbandonmentDetector(STALL_STEPS=3)
        state = make_state()
        state.exit_reason = None
        state.tool_calls = [make_tool_call("search_db", step=1)]
        state.current_step = 5
        state.events = [
            make_event(EventType.LLM_CALLED, step=3),
            make_event(EventType.LLM_CALLED, step=4),
            make_event(EventType.LLM_CALLED, step=5),
        ]
        signal = detector.check(state)
        assert signal is not None
        assert signal.evidence["stall_steps"] == 3


# ── LlmTruncationLoopDetector — enriched evidence ────────────────────────────


class TestLlmTruncationLoopEvidence(unittest.TestCase):
    detector = LlmTruncationLoopDetector()

    def test_token_counts_at_truncation_present(self):
        state = make_state()
        state.llm_calls = [
            make_llm_call("length", prompt_tokens=4000, step=1),
            make_llm_call("length", prompt_tokens=8000, step=2),
        ]
        state.current_step = 2
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["token_counts_at_truncation"] == [4000, 8000]

    def test_token_counts_excludes_none_tokens(self):
        """If prompt_tokens is None for some truncated calls, exclude them gracefully."""
        state = make_state()
        state.llm_calls = [
            make_llm_call("length", prompt_tokens=None, step=1),
            make_llm_call("length", prompt_tokens=5000, step=2),
        ]
        state.current_step = 2
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["token_counts_at_truncation"] == [5000]

    def test_token_counts_empty_when_all_none(self):
        state = make_state()
        state.llm_calls = [
            make_llm_call("length", prompt_tokens=None, step=1),
            make_llm_call("length", prompt_tokens=None, step=2),
        ]
        state.current_step = 2
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["token_counts_at_truncation"] == []

    def test_models_single_model(self):
        state = make_state()
        state.llm_calls = [
            make_llm_call("length", model="gpt-4o", step=1),
            make_llm_call("length", model="gpt-4o", step=2),
        ]
        state.current_step = 2
        signal = self.detector.check(state)
        assert signal.evidence["models"] == ["gpt-4o"]

    def test_models_multiple_models_sorted(self):
        """Different models in truncated calls → sorted list of unique models."""
        state = make_state()
        state.llm_calls = [
            make_llm_call("length", model="gpt-4o-mini", step=1),
            make_llm_call("length", model="gpt-4o", step=2),
        ]
        state.current_step = 2
        signal = self.detector.check(state)
        assert signal.evidence["models"] == sorted(["gpt-4o-mini", "gpt-4o"])

    def test_models_excludes_non_truncated_calls(self):
        """Only models from truncated calls appear in evidence."""
        state = make_state()
        state.llm_calls = [
            make_llm_call("stop", model="claude-3", step=1),
            make_llm_call("length", model="gpt-4o", step=2),
            make_llm_call("length", model="gpt-4o", step=3),
        ]
        state.current_step = 3
        signal = self.detector.check(state)
        assert signal.evidence["models"] == ["gpt-4o"]

    def test_total_llm_calls_matches_all_calls(self):
        state = make_state()
        state.llm_calls = [
            make_llm_call("stop", step=1),
            make_llm_call("length", step=2),
            make_llm_call("stop", step=3),
            make_llm_call("length", step=4),
            make_llm_call("stop", step=5),
        ]
        state.current_step = 5
        signal = self.detector.check(state)
        assert signal.evidence["total_llm_calls"] == 5
        assert signal.evidence["truncation_count"] == 2


# ── ContextBloatDetector — token_growth_sequence ──────────────────────────────


class TestContextBloatEvidence(unittest.TestCase):
    detector = ContextBloatDetector()

    def _bloating_state(self):
        state = make_state()
        state.llm_calls = [
            make_llm_call(prompt_tokens=400, step=1),
            make_llm_call(prompt_tokens=2000, step=2),
            make_llm_call(prompt_tokens=4000, step=3),
        ]
        state.current_step = 3
        return state

    def test_token_growth_sequence_in_evidence(self):
        state = self._bloating_state()
        signal = self.detector.check(state)
        assert signal is not None
        seq = signal.evidence["token_growth_sequence"]
        assert isinstance(seq, list)
        assert len(seq) == 3

    def test_token_growth_sequence_has_step_and_tokens(self):
        state = self._bloating_state()
        signal = self.detector.check(state)
        for entry in signal.evidence["token_growth_sequence"]:
            assert "step" in entry
            assert "tokens" in entry

    def test_token_growth_sequence_correct_values(self):
        state = self._bloating_state()
        signal = self.detector.check(state)
        seq = signal.evidence["token_growth_sequence"]
        assert seq[0] == {"step": 1, "tokens": 400}
        assert seq[1] == {"step": 2, "tokens": 2000}
        assert seq[2] == {"step": 3, "tokens": 4000}

    def test_token_growth_sequence_excludes_none_tokens(self):
        """Calls with None prompt_tokens are excluded from the sequence."""
        state = make_state()
        state.llm_calls = [
            make_llm_call(prompt_tokens=500, step=1),
            make_llm_call(prompt_tokens=None, step=2),  # excluded
            make_llm_call(prompt_tokens=2500, step=3),
            make_llm_call(prompt_tokens=6000, step=4),
        ]
        state.current_step = 4
        signal = self.detector.check(state)
        assert signal is not None
        seq = signal.evidence["token_growth_sequence"]
        # Step 2 (None) must not appear
        assert all(e["step"] != 2 for e in seq)
        assert len(seq) == 3


# ── RetryStormDetector ────────────────────────────────────────────────────────


class TestRetryStormDetector(unittest.TestCase):
    detector = RetryStormDetector()

    def _failing_streak(
        self, tool: str, count: int, args_hashes=None, error_hashes=None
    ) -> RunState:
        state = make_state()
        calls = []
        for i in range(count):
            ah = args_hashes[i] if args_hashes else f"arg{i}"
            eh = error_hashes[i] if error_hashes else None
            calls.append(
                make_tool_call(
                    tool, step=i + 1, success=False, args_hash=ah, error_hash=eh
                )
            )
        state.tool_calls = calls
        state.current_step = count
        return state

    def test_fires_at_threshold(self):
        state = self._failing_streak("search_api", 3)
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.failure_type == FailureType.RETRY_STORM

    def test_no_signal_below_threshold(self):
        state = self._failing_streak("search_api", 2)
        assert self.detector.check(state) is None

    def test_no_signal_when_all_succeed(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("tool", step=i, success=True) for i in range(5)
        ]
        state.current_step = 5
        assert self.detector.check(state) is None

    def test_no_signal_when_success_breaks_streak(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("tool", step=1, success=False),
            make_tool_call("tool", step=2, success=False),
            make_tool_call("tool", step=3, success=True),  # breaks the streak
            make_tool_call("tool", step=4, success=False),
        ]
        state.current_step = 4
        assert self.detector.check(state) is None

    def test_step_indices_in_evidence(self):
        state = self._failing_streak("api", 4)
        signal = self.detector.check(state)
        assert signal.evidence["step_indices"] == [1, 2, 3, 4]

    def test_error_hashes_in_evidence(self):
        state = self._failing_streak("api", 3, error_hashes=["err1", "err1", "err1"])
        signal = self.detector.check(state)
        assert signal.evidence["error_hashes"] == ["err1", "err1", "err1"]

    def test_reason_identical_true_when_same_error_hash(self):
        state = self._failing_streak("api", 3, error_hashes=["e1", "e1", "e1"])
        signal = self.detector.check(state)
        assert signal.evidence["reason_identical"] is True
        assert signal.evidence["failure_reason_hash"] == "e1"

    def test_reason_identical_false_when_different_error_hashes(self):
        state = self._failing_streak("api", 3, error_hashes=["e1", "e2", "e3"])
        signal = self.detector.check(state)
        assert signal.evidence["reason_identical"] is False
        assert signal.evidence["failure_reason_hash"] is None

    def test_reason_identical_false_when_error_hashes_are_none(self):
        state = self._failing_streak("api", 3, error_hashes=[None, None, None])
        signal = self.detector.check(state)
        assert signal.evidence["reason_identical"] is False

    def test_args_identical_true_when_same_args(self):
        state = self._failing_streak("api", 3, args_hashes=["same"] * 3)
        signal = self.detector.check(state)
        assert signal.evidence["args_identical"] is True

    def test_args_identical_false_when_different_args(self):
        state = self._failing_streak("api", 3, args_hashes=["a", "b", "c"])
        signal = self.detector.check(state)
        assert signal.evidence["args_identical"] is False

    def test_consecutive_fails_count(self):
        state = self._failing_streak("api", 5)
        signal = self.detector.check(state)
        assert signal.evidence["consecutive_fails"] == 5

    def test_first_fail_step(self):
        state = self._failing_streak("api", 4)
        signal = self.detector.check(state)
        assert signal.evidence["first_fail_step"] == 1

    def test_severity_is_high(self):
        state = self._failing_streak("api", 3)
        signal = self.detector.check(state)
        assert signal.severity.value == "HIGH"

    def test_custom_threshold(self):
        detector = RetryStormDetector(THRESHOLD=5)
        state = self._failing_streak("api", 4)
        assert detector.check(state) is None

    def test_fires_on_most_recent_streak_when_multiple_tools(self):
        """Detector picks the best streak — here api_b has 3 failures."""
        state = make_state()
        state.tool_calls = [
            make_tool_call("api_a", step=1, success=False),
            make_tool_call("api_a", step=2, success=False),
            make_tool_call("api_a", step=3, success=True),  # streak broken
            make_tool_call("api_b", step=4, success=False),
            make_tool_call("api_b", step=5, success=False),
            make_tool_call("api_b", step=6, success=False),
        ]
        state.current_step = 6
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["tool"] == "api_b"


# ── ReasoningSpinDetector — event_sequence ────────────────────────────────────


class TestReasoningSpinEvidence(unittest.TestCase):
    detector = ReasoningSpinDetector()

    def _spin_state(self, llm_count: int, tool_count: int) -> RunState:
        state = make_state()
        state.llm_calls = [make_llm_call(step=i) for i in range(llm_count)]
        state.tool_calls = [make_tool_call("search", step=i) for i in range(tool_count)]
        state.current_step = max(llm_count, tool_count)
        state.exit_reason = "final_answer"
        # Build realistic events
        state.events = []
        for i in range(llm_count):
            state.events.append(make_event(EventType.LLM_CALLED, step=i))
        for i in range(tool_count):
            state.events.append(make_event(EventType.TOOL_CALLED, step=i))
        state.events.sort(key=lambda e: e.step_index)
        return state

    def test_event_sequence_in_evidence(self):
        state = self._spin_state(llm_count=10, tool_count=1)
        signal = self.detector.check(state)
        assert signal is not None
        assert "event_sequence" in signal.evidence

    def test_event_sequence_contains_correct_labels(self):
        state = self._spin_state(llm_count=10, tool_count=1)
        signal = self.detector.check(state)
        seq = signal.evidence["event_sequence"]
        assert all(s in ("llm", "tool") for s in seq)

    def test_event_sequence_llm_dominates(self):
        state = self._spin_state(llm_count=10, tool_count=1)
        signal = self.detector.check(state)
        seq = signal.evidence["event_sequence"]
        llm_count = seq.count("llm")
        tool_count = seq.count("tool")
        assert llm_count > tool_count

    def test_event_sequence_empty_when_no_events(self):
        state = make_state()
        state.llm_calls = [make_llm_call(step=i) for i in range(10)]
        state.tool_calls = [make_tool_call("search", step=1)]
        state.current_step = 10
        state.exit_reason = "final_answer"
        state.events = []  # no events recorded
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["event_sequence"] == []


# ── SlowStepDetector ──────────────────────────────────────────────────────────


class TestSlowStepDetector(unittest.TestCase):
    detector = SlowStepDetector()

    def test_no_signal_when_no_durations(self):
        state = make_state()
        assert self.detector.check(state) is None

    def test_no_signal_when_no_events(self):
        state = make_state()
        state.step_durations_ms = {1: 60_001}
        assert self.detector.check(state) is None

    def test_fires_on_slow_tool_call(self):
        state = make_state()
        state.events = [make_event(EventType.TOOL_CALLED, step=1)]
        state.step_durations_ms = {1: 20_000}  # 20s > 15s threshold
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.failure_type == FailureType.SLOW_STEP

    def test_no_signal_when_tool_within_threshold(self):
        state = make_state()
        state.events = [make_event(EventType.TOOL_CALLED, step=1)]
        state.step_durations_ms = {1: 14_000}  # 14s < 15s threshold
        assert self.detector.check(state) is None

    def test_fires_on_slow_llm_call(self):
        state = make_state()
        state.events = [make_event(EventType.LLM_CALLED, step=1)]
        state.step_durations_ms = {1: 35_000}  # 35s > 30s threshold
        signal = self.detector.check(state)
        assert signal is not None

    def test_severity_medium_at_2x(self):
        state = make_state()
        state.events = [make_event(EventType.TOOL_CALLED, step=1)]
        state.step_durations_ms = {1: 30_000}  # 2× tool threshold (15s)
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.severity == Severity.MEDIUM

    def test_severity_high_at_5x(self):
        state = make_state()
        state.events = [make_event(EventType.TOOL_CALLED, step=1)]
        state.step_durations_ms = {1: 75_001}  # >5× tool threshold (15s)
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.severity == Severity.HIGH

    def test_picks_worst_step(self):
        """With two slow steps, the signal should point at the worse one."""
        state = make_state()
        state.events = [
            make_event(EventType.TOOL_CALLED, step=1),
            make_event(EventType.TOOL_CALLED, step=2),
        ]
        state.step_durations_ms = {1: 20_000, 2: 50_000}
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["step_index"] == 2

    def test_confidence_scales_with_excess(self):
        # 20s on a 15s threshold → ratio=1.33 → confidence≈0.63
        state = make_state()
        state.events = [make_event(EventType.TOOL_CALLED, step=1)]
        state.step_durations_ms = {1: 20_000}
        signal = self.detector.check(state)
        assert round(signal.confidence, 2) == 0.63
        # 60s on a 15s threshold → ratio=4.0 → confidence=1.0 (capped)
        state.step_durations_ms = {1: 60_000}
        high = self.detector.check(state)
        assert high.confidence == 1.0

    def test_coincident_external_signal_in_evidence(self):
        """An external signal coincident with the slow step appears in evidence."""
        ts = time.time()
        state = make_state()
        state.events = [make_event(EventType.TOOL_CALLED, step=1, ts=ts)]
        state.step_durations_ms = {1: 20_000}  # step runs ts → ts+20s
        state.external_signals = [
            ExternalSignal(
                signal_name="high_db_latency",
                step_index=1,
                source="datadog",
                timestamp=ts + 5,  # within the step window
            )
        ]
        signal = self.detector.check(state)
        assert signal is not None
        assert "coincident_signals" in signal.evidence
        assert (
            signal.evidence["coincident_signals"][0]["signal_name"] == "high_db_latency"
        )

    def test_external_signal_outside_window_not_coincident(self):
        ts = time.time()
        state = make_state()
        state.events = [make_event(EventType.TOOL_CALLED, step=1, ts=ts)]
        state.step_durations_ms = {1: 20_000}  # ts → ts+20s
        state.external_signals = [
            ExternalSignal(
                signal_name="unrelated_spike",
                step_index=1,
                source="pagerduty",
                timestamp=ts + 60,  # well outside the window
            )
        ]
        signal = self.detector.check(state)
        assert signal is not None
        assert "coincident_signals" not in signal.evidence


# ── EmptyLlmResponseDetector ──────────────────────────────────────────────────


class TestEmptyLlmResponseDetector(unittest.TestCase):
    detector = EmptyLlmResponseDetector()

    def test_fires_on_empty_stop_response(self):
        state = make_state()
        state.llm_calls = [make_llm_call("stop", step=1, output_length=0)]
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.failure_type == FailureType.EMPTY_LLM_RESPONSE

    def test_no_signal_when_output_present(self):
        state = make_state()
        state.llm_calls = [make_llm_call("stop", step=1, output_length=100)]
        assert self.detector.check(state) is None

    def test_no_signal_for_length_finish_reason(self):
        state = make_state()
        state.llm_calls = [make_llm_call("length", step=1, output_length=0)]
        assert self.detector.check(state) is None

    def test_no_signal_when_output_length_is_none(self):
        """output_length=None means not reported — should not fire."""
        state = make_state()
        call = LlmCall(
            model="gpt-4o",
            prompt_tokens=500,
            finish_reason="stop",
            latency_ms=100,
            step_index=1,
            timestamp=time.time(),
        )
        # output_length defaults to None — don't override
        state.llm_calls = [call]
        assert self.detector.check(state) is None

    def test_no_signal_with_no_llm_calls(self):
        state = make_state()
        assert self.detector.check(state) is None

    def test_occurrences_in_evidence(self):
        state = make_state()
        state.llm_calls = [
            make_llm_call("stop", step=1, output_length=0),
            make_llm_call("stop", step=2, output_length=0),
        ]
        signal = self.detector.check(state)
        assert signal.evidence["occurrences"] == 2

    def test_first_step_in_evidence(self):
        state = make_state()
        state.llm_calls = [
            make_llm_call("stop", step=3, output_length=0),
            make_llm_call("stop", step=7, output_length=0),
        ]
        signal = self.detector.check(state)
        assert signal.evidence["first_step"] == 3

    def test_severity_is_high(self):
        state = make_state()
        state.llm_calls = [make_llm_call("stop", step=1, output_length=0)]
        signal = self.detector.check(state)
        assert signal.severity == Severity.HIGH


# ── StepCountInflationDetector ────────────────────────────────────────────────


class TestStepCountInflationDetector(unittest.TestCase):
    detector = StepCountInflationDetector()

    def test_no_signal_without_baseline(self):
        state = make_state()
        state.current_step = 100
        state.baseline_p75_steps = None
        assert self.detector.check(state) is None

    def test_no_signal_below_factor(self):
        state = make_state()
        state.current_step = 15
        state.baseline_p75_steps = 10.0  # 1.5× < 2.0× factor
        assert self.detector.check(state) is None

    def test_fires_at_exactly_2x_plus_one(self):
        state = make_state()
        state.current_step = 21
        state.baseline_p75_steps = 10.0  # 21 > 10 * 2.0
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.failure_type == FailureType.STEP_COUNT_INFLATION

    def test_inflation_ratio_in_evidence(self):
        state = make_state()
        state.current_step = 30
        state.baseline_p75_steps = 10.0
        signal = self.detector.check(state)
        assert signal.evidence["inflation_ratio"] == 3.0

    def test_baseline_in_evidence(self):
        state = make_state()
        state.current_step = 25
        state.baseline_p75_steps = 10.0
        signal = self.detector.check(state)
        assert signal.evidence["baseline_p75"] == 10.0

    def test_severity_is_medium(self):
        state = make_state()
        state.current_step = 25
        state.baseline_p75_steps = 10.0
        signal = self.detector.check(state)
        assert signal.severity == Severity.MEDIUM

    def test_custom_inflation_factor(self):
        detector = StepCountInflationDetector(INFLATION_FACTOR=3.0)
        state = make_state()
        state.current_step = 25
        state.baseline_p75_steps = 10.0  # 25/10=2.5× < 3.0 factor
        assert detector.check(state) is None


# ── CascadingToolFailureDetector ──────────────────────────────────────────────


class TestCascadingToolFailureDetector(unittest.TestCase):
    detector = CascadingToolFailureDetector()

    def test_fires_on_multi_tool_consecutive_failures(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("search", step=1, success=False),
            make_tool_call("database", step=2, success=False),
            make_tool_call("cache", step=3, success=False),
        ]
        state.current_step = 3
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.failure_type == FailureType.CASCADING_TOOL_FAILURE

    def test_no_signal_when_only_one_tool(self):
        """Same tool failing ≥ threshold = RetryStorm, not CascadingToolFailure."""
        state = make_state()
        state.tool_calls = [
            make_tool_call("api", step=i, success=False) for i in range(4)
        ]
        state.current_step = 4
        assert self.detector.check(state) is None

    def test_no_signal_below_threshold(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("search", step=1, success=False),
            make_tool_call("database", step=2, success=False),
        ]
        state.current_step = 2
        assert self.detector.check(state) is None

    def test_success_breaks_cascade(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("search", step=1, success=False),
            make_tool_call("database", step=2, success=True),  # breaks cascade
            make_tool_call("cache", step=3, success=False),
            make_tool_call("search", step=4, success=False),
        ]
        state.current_step = 4
        # Only 2 failures after the success — below threshold
        assert self.detector.check(state) is None

    def test_distinct_tools_in_evidence(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("alpha", step=1, success=False),
            make_tool_call("beta", step=2, success=False),
            make_tool_call("gamma", step=3, success=False),
        ]
        state.current_step = 3
        signal = self.detector.check(state)
        assert sorted(signal.evidence["distinct_tools"]) == ["alpha", "beta", "gamma"]

    def test_first_fail_step_in_evidence(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("a", step=5, success=False),
            make_tool_call("b", step=6, success=False),
            make_tool_call("c", step=7, success=False),
        ]
        state.current_step = 7
        signal = self.detector.check(state)
        assert signal.evidence["first_fail_step"] == 5

    def test_severity_is_high(self):
        state = make_state()
        state.tool_calls = [
            make_tool_call("x", step=i, success=False) for i in range(3)
        ]
        # Need 2 distinct tools
        state.tool_calls[1] = make_tool_call("y", step=1, success=False)
        state.current_step = 3
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.severity == Severity.HIGH


# ── FirstStepFailureDetector ──────────────────────────────────────────────────


class TestFirstStepFailureDetector(unittest.TestCase):
    detector = FirstStepFailureDetector()

    def test_fires_on_early_run_error(self):
        state = make_state()
        state.exit_reason = "error"
        state.current_step = 1
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.failure_type == FailureType.FIRST_STEP_FAILURE
        assert signal.evidence["trigger"] == "run_errored"

    def test_no_signal_for_error_at_late_step(self):
        state = make_state()
        state.exit_reason = "error"
        state.current_step = 10  # beyond MAX_STEP=2
        assert self.detector.check(state) is None

    def test_fires_on_empty_llm_response_at_step_1(self):
        state = make_state()
        state.exit_reason = None
        state.current_step = 1
        state.llm_calls = [
            LlmCall(
                model="gpt-4o",
                prompt_tokens=100,
                finish_reason="stop",
                latency_ms=50,
                step_index=1,
                timestamp=time.time(),
                output_length=0,
            ),
        ]
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["trigger"] == "empty_llm_response"

    def test_no_signal_for_empty_llm_at_late_step(self):
        state = make_state()
        state.exit_reason = None
        state.current_step = 5
        state.llm_calls = [
            LlmCall(
                model="gpt-4o",
                prompt_tokens=100,
                finish_reason="stop",
                latency_ms=50,
                step_index=5,
                timestamp=time.time(),
                output_length=0,
            ),
        ]
        assert self.detector.check(state) is None

    def test_fires_on_tool_failure_at_step_2(self):
        state = make_state()
        state.exit_reason = None
        state.current_step = 2
        state.tool_calls = [make_tool_call("init_db", step=2, success=False)]
        signal = self.detector.check(state)
        assert signal is not None
        assert signal.evidence["trigger"] == "tool_failure"
        assert signal.evidence["tool"] == "init_db"

    def test_no_signal_for_successful_tool(self):
        state = make_state()
        state.exit_reason = None
        state.current_step = 2
        state.tool_calls = [make_tool_call("init_db", step=2, success=True)]
        assert self.detector.check(state) is None

    def test_severity_is_medium(self):
        state = make_state()
        state.exit_reason = "error"
        state.current_step = 1
        signal = self.detector.check(state)
        assert signal.severity == Severity.MEDIUM

    def test_custom_max_step(self):
        detector = FirstStepFailureDetector(MAX_STEP=5)
        state = make_state()
        state.exit_reason = "error"
        state.current_step = 5  # within extended max
        signal = detector.check(state)
        assert signal is not None

    def test_no_signal_for_normal_exit(self):
        state = make_state()
        state.exit_reason = "final_answer"
        state.current_step = 1
        assert self.detector.check(state) is None


if __name__ == "__main__":
    unittest.main(verbosity=2)
