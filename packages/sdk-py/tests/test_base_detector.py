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
import unittest.mock
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
import dunetrace.detectors as detectors_module
from dunetrace.detectors import (
    BaseDetector,
    ToolLoopDetector,
    SlowStepDetector,
    ReasoningSpinDetector,
    reset_cost_downgrade,
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
    def setUp(self):
        detectors_module._cost_trackers.clear()

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


class TestCostBudgetTracking(unittest.TestCase):
    """Rate-limited warnings, P50/P95/P99 tracking, and runtime-path downgrade.
    _cost_trackers is module-level state keyed by detector name, so every test
    clears it first for isolation from other tests (and from each other)."""

    def setUp(self):
        detectors_module._cost_trackers.clear()

    def test_warning_is_rate_limited_to_once_per_minute(self):
        detector = _SlowDetector()
        with self.assertLogs("dunetrace.detectors", level="WARNING") as cm:
            run_detectors(make_state(), detectors=[detector])
        self.assertTrue(any("exceeded its cost budget" in msg for msg in cm.output))

        # Second call immediately after — same minute, should NOT log again.
        with self.assertNoLogs("dunetrace.detectors", level="WARNING"):
            run_detectors(make_state(), detectors=[detector])

    def test_warning_fires_again_after_rate_limit_window(self):
        detector = _SlowDetector()
        with self.assertLogs("dunetrace.detectors", level="WARNING"):
            run_detectors(make_state(), detectors=[detector])

        tracker = detectors_module._get_cost_tracker(detector.name)
        # Simulate 61 real seconds having passed since the last warning.
        tracker.last_warning_at = time.monotonic() - 61.0

        with self.assertLogs("dunetrace.detectors", level="WARNING") as cm:
            run_detectors(make_state(), detectors=[detector])
        self.assertTrue(any("exceeded its cost budget" in msg for msg in cm.output))

    def test_cost_stats_logged_alongside_rate_limited_warning(self):
        with self.assertLogs("dunetrace.detectors", level="INFO") as cm:
            run_detectors(make_state(), detectors=[_SlowDetector()])
        self.assertTrue(any("cost stats" in msg for msg in cm.output))

    def test_downgrade_after_sustained_p99_over_budget(self):
        detector = _SlowDetector()
        tracker = detectors_module._get_cost_tracker(detector.name)

        # Feed enough over-budget samples directly to cross the minimum-sample
        # floor without needing real rate-limit-window timing gymnastics.
        now = time.monotonic()
        for _ in range(10):
            tracker.samples.append((now, 999_999_999))

        with self.assertLogs("dunetrace.detectors", level="WARNING") as cm:
            run_detectors(make_state(), detectors=[detector])
        self.assertTrue(any("downgrading to analytics-only" in msg for msg in cm.output))
        self.assertIsNotNone(tracker.downgraded_at)

    def test_downgraded_detector_is_skipped_in_runtime_context_only(self):
        detector = _SlowDetector()
        tracker = detectors_module._get_cost_tracker(detector.name)
        tracker.downgraded_at = time.monotonic()

        # context="runtime" (default is "analytics") skips it entirely — no call,
        # no signal, no fresh cost sample recorded.
        sample_count_before = len(tracker.samples)
        run_detectors(make_state(), detectors=[detector], context="runtime")
        self.assertEqual(len(tracker.samples), sample_count_before)

    def test_analytics_context_never_skips_a_downgraded_detector(self):
        detector = _SlowDetector()
        tracker = detectors_module._get_cost_tracker(detector.name)
        tracker.downgraded_at = time.monotonic()

        sample_count_before = len(tracker.samples)
        run_detectors(make_state(), detectors=[detector], context="analytics")
        self.assertEqual(len(tracker.samples), sample_count_before + 1)

    def test_reset_cost_downgrade_re_enables_the_detector(self):
        detector = _SlowDetector()
        tracker = detectors_module._get_cost_tracker(detector.name)
        tracker.downgraded_at = time.monotonic()

        reset_cost_downgrade(detector.name)
        self.assertIsNone(tracker.downgraded_at)

        sample_count_before = len(tracker.samples)
        run_detectors(make_state(), detectors=[detector], context="runtime")
        self.assertEqual(len(tracker.samples), sample_count_before + 1)

    def test_reset_cost_downgrade_on_unknown_name_is_a_safe_no_op(self):
        reset_cost_downgrade("NEVER_SEEN_THIS_DETECTOR")  # must not raise

    def test_percentiles_only_include_samples_within_the_window(self):
        tracker = detectors_module._DetectorCostTracker()
        now = time.monotonic()
        tracker.samples.append((now - 400.0, 1))  # outside the 300s window
        tracker.samples.append((now, 100))
        stats = tracker.percentiles()
        self.assertIsNotNone(stats)
        p50, p95, p99, n = stats
        self.assertEqual(n, 1)
        self.assertEqual(p50, 100)

    def test_percentiles_returns_none_when_no_samples_in_window(self):
        tracker = detectors_module._DetectorCostTracker()
        self.assertIsNone(tracker.percentiles())


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


class TestCustomDetectorRegistration(unittest.TestCase):
    """A3: BaseDetector.__init_subclass__ auto-registers third-party detector
    classes into CUSTOM_DETECTOR_REGISTRY — the plugin surface
    detector_svc/custom_python_detectors.py loads from disk and merges into
    get_detectors(). CUSTOM_DETECTOR_REGISTRY is module-level shared state, so
    every test here patches it to a fresh dict rather than mutating the real
    one (which detector_svc's own tests, and this file's later tests, also
    read from)."""

    def setUp(self):
        self._registry_patcher = unittest.mock.patch.dict(
            detectors_module.CUSTOM_DETECTOR_REGISTRY, clear=True
        )
        self._registry_patcher.start()

    def tearDown(self):
        self._registry_patcher.stop()

    def test_subclass_defined_outside_detectors_module_registers(self):
        # __module__ is this test file, not dunetrace.detectors — must register.
        class _MyPlugin(BaseDetector):
            name = "MY_PLUGIN"

        self.assertIn("_MyPlugin", detectors_module.CUSTOM_DETECTOR_REGISTRY)
        self.assertIs(detectors_module.CUSTOM_DETECTOR_REGISTRY["_MyPlugin"], _MyPlugin)

    def test_builtin_detectors_do_not_register(self):
        # ToolLoopDetector etc. are defined inside dunetrace.detectors itself —
        # they're wired into detector_svc._DETECTOR_CLASSES explicitly, not
        # via this registry, and must not show up here.
        self.assertNotIn("ToolLoopDetector", detectors_module.CUSTOM_DETECTOR_REGISTRY)

    def test_default_category_and_shadow_metadata(self):
        class _MyPlugin(BaseDetector):
            name = "MY_PLUGIN"

        self.assertEqual(_MyPlugin.CATEGORY, "custom")
        self.assertTrue(_MyPlugin.SHADOW_BY_DEFAULT)

    def test_metadata_overridable_on_subclass(self):
        class _MyPlugin(BaseDetector):
            name = "MY_PLUGIN"
            CATEGORY = "security"
            SHADOW_BY_DEFAULT = False

        self.assertEqual(_MyPlugin.CATEGORY, "security")
        self.assertFalse(_MyPlugin.SHADOW_BY_DEFAULT)

    def test_metadata_fields_are_constructor_tunable(self):
        # CATEGORY/SHADOW_BY_DEFAULT are UPPERCASE class attributes, so the
        # existing BaseDetector.__init__ override mechanism already accepts
        # them — same as SEVERITY/MAX_COST_NS.
        class _MyPlugin(BaseDetector):
            name = "MY_PLUGIN"

        instance = _MyPlugin(CATEGORY="security", SHADOW_BY_DEFAULT=False)
        self.assertEqual(instance.CATEGORY, "security")
        self.assertFalse(instance.SHADOW_BY_DEFAULT)

    def test_nested_subclass_of_a_third_party_class_also_registers(self):
        class _MyPlugin(BaseDetector):
            name = "MY_PLUGIN"

        class _MyPluginV2(_MyPlugin):
            name = "MY_PLUGIN_V2"

        self.assertIn("_MyPlugin", detectors_module.CUSTOM_DETECTOR_REGISTRY)
        self.assertIn("_MyPluginV2", detectors_module.CUSTOM_DETECTOR_REGISTRY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
