"""
Tests for BaseDetector's shared contract: tunable SEVERITY, MAX_COST_NS budget
warnings, the on_event/on_run_completion split, and run_detectors()'s evidence
validation. Per-detector behavior/evidence shape is covered in test_detectors.py
and test_detectors_evidence.py — this file is about the base class itself.

Run: python -m unittest tests.test_base_detector -v
"""

from __future__ import annotations

import time
import unittest
from typing import Optional

from dunetrace.models import (
    AgentEvent,
    EventType,
    FailureSignal,
    FailureType,
    LlmCall,
    RunState,
    Severity,
    ToolCall,
)
from dunetrace.detectors import (
    BaseDetector,
    ToolLoopDetector,
    SlowStepDetector,
    ReasoningSpinDetector,
    run_detectors,
)


def make_state(**kwargs) -> RunState:
    defaults = dict(
        run_id="test-run-1",
        agent_id="test-agent",
        agent_version="abc12345",
        available_tools=["web_search"],
    )
    defaults.update(kwargs)
    return RunState(**defaults)


def make_tool_call(name: str, step: int = 0) -> ToolCall:
    return ToolCall(tool_name=name, args="aaa", step_index=step, timestamp=time.time())


def _looping_state() -> RunState:
    state = make_state()
    for i in range(5):
        state.tool_calls.append(make_tool_call("web_search", step=i))
    return state


# ── SEVERITY tunability ────────────────────────────────────────────────────────


class TestSeverityOverride(unittest.TestCase):
    def test_default_severity_matches_detector_design(self):
        detector = ToolLoopDetector()
        signal = detector.on_run_completion(_looping_state())
        self.assertEqual(signal.severity, Severity.HIGH)

    def test_severity_overridden_via_constructor_kwarg(self):
        detector = ToolLoopDetector(SEVERITY=Severity.CRITICAL)
        signal = detector.on_run_completion(_looping_state())
        self.assertEqual(signal.severity, Severity.CRITICAL)

    def _slow_step_state(self) -> RunState:
        state = make_state()
        state.step_durations_ms = {0: 90_000}  # >> the 60s catch-all threshold, ratio >= 5
        state.events = [
            AgentEvent(
                event_type=EventType.TOOL_CALLED,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=0,
            )
        ]
        return state

    def test_dynamic_severity_detector_default_still_dynamic(self):
        """SlowStepDetector computes HIGH vs MEDIUM from the excess ratio when SEVERITY is unset."""
        signal = SlowStepDetector().on_run_completion(self._slow_step_state())
        self.assertEqual(signal.severity, Severity.HIGH)  # ratio >= 5 -> HIGH by default

    def test_dynamic_severity_detector_override_forces_fixed_level(self):
        """Setting SEVERITY on a dynamic detector must short-circuit its own computation."""
        detector = SlowStepDetector(SEVERITY=Severity.LOW)
        signal = detector.on_run_completion(self._slow_step_state())
        self.assertEqual(signal.severity, Severity.LOW)  # override wins over ratio>=5 -> HIGH

    def test_reasoning_spin_severity_override(self):
        state = make_state(exit_reason="final_answer")
        state.tool_calls = [make_tool_call("web_search", step=0)]
        state.llm_calls = [
            LlmCall(
                model="gpt-4o",
                prompt_tokens=100,
                finish_reason="stop",
                latency_ms=10,
                step_index=i,
                timestamp=time.time(),
                output_length=10,
                completion_tokens=10,
            )
            for i in range(10)
        ]
        state.events = [
            AgentEvent(
                event_type=EventType.LLM_CALLED,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=i,
            )
            for i in range(10)
        ]
        detector = ReasoningSpinDetector(SEVERITY=Severity.CRITICAL, MIN_LLM_CALLS=1)
        signal = detector.on_run_completion(state)
        self.assertIsNotNone(signal)
        self.assertEqual(
            signal.severity, Severity.CRITICAL
        )  # override wins over final_answer -> MEDIUM


# ── on_event / on_run_completion split ─────────────────────────────────────────


class TestLifecycleMethods(unittest.TestCase):
    def test_on_event_default_is_a_noop(self):
        detector = ToolLoopDetector()
        result = detector.on_event(event=None, state=make_state())  # type: ignore[arg-type]
        self.assertIsNone(result)

    def test_base_on_run_completion_raises_not_implemented(self):
        detector = BaseDetector()
        with self.assertRaises(NotImplementedError):
            detector.on_run_completion(make_state())

    def test_run_detectors_calls_on_run_completion(self):
        signals = run_detectors(_looping_state(), detectors=[ToolLoopDetector()])
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].failure_type, FailureType.TOOL_LOOP)


# ── MAX_COST_NS budget warnings ─────────────────────────────────────────────────


class _SlowDetector(BaseDetector):
    name = "SLOW_TEST_DETECTOR"
    SEVERITY = Severity.LOW
    MAX_COST_NS = 1  # absurdly tight — guarantees the warning fires

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        return None


class _NormalDetector(BaseDetector):
    name = "NORMAL_TEST_DETECTOR"
    SEVERITY = Severity.LOW

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        return None


class TestMaxCostBudget(unittest.TestCase):
    def test_max_cost_ns_default(self):
        self.assertEqual(BaseDetector.MAX_COST_NS, 1_000_000)

    def test_max_cost_ns_tunable_via_constructor(self):
        detector = _NormalDetector(MAX_COST_NS=5_000_000)
        self.assertEqual(detector.MAX_COST_NS, 5_000_000)

    def test_exceeding_budget_logs_a_warning_not_an_exception(self):
        with self.assertLogs("dunetrace.detectors", level="WARNING") as cm:
            signals = run_detectors(make_state(), detectors=[_SlowDetector()])
        self.assertEqual(signals, [])  # returns normally, no exception
        self.assertTrue(any("exceeded its cost budget" in msg for msg in cm.output))

    def test_normal_detector_does_not_warn(self):
        with self.assertNoLogs("dunetrace.detectors", level="WARNING"):
            run_detectors(make_state(), detectors=[_NormalDetector()])


# ── Evidence validation ──────────────────────────────────────────────────────


class _BadEvidenceDetector(BaseDetector):
    name = "BAD_EVIDENCE_DETECTOR"
    SEVERITY = Severity.LOW

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        return FailureSignal(
            failure_type=FailureType.CUSTOM,
            severity=self.SEVERITY,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=0,
            confidence=0.5,
            evidence=None,  # type: ignore[arg-type]  -- deliberately malformed
        )


class TestEvidenceValidation(unittest.TestCase):
    def test_non_dict_evidence_logs_warning_but_signal_is_kept(self):
        with self.assertLogs("dunetrace.detectors", level="WARNING") as cm:
            signals = run_detectors(make_state(), detectors=[_BadEvidenceDetector()])
        self.assertEqual(len(signals), 1)  # not dropped
        self.assertTrue(any("non-dict evidence" in msg for msg in cm.output))

    def test_well_formed_evidence_does_not_warn(self):
        signals = run_detectors(_looping_state(), detectors=[ToolLoopDetector()])
        self.assertEqual(len(signals), 1)
        self.assertIsInstance(signals[0].evidence, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
