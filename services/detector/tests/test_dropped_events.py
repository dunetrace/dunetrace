"""An incomplete run — one the SDK's buffer shed events from — never yields a
live verdict.

The SDK stamps ``dropped_events`` on run.completed / run.errored when its
outbound buffer shed part of the run. ``build_run_state`` reads it, and
``process_run`` then holds every signal in shadow with capped confidence and
severity, marks the evidence, keeps the risk engine off them, and treats the
run as *unknown* for issue tracking and baselines. DB is mocked.

Run:
    cd services/detector
    pytest tests/test_dropped_events.py -v
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from detector_svc.run_builder import build_run_state
from detector_svc.worker import _mark_incomplete, _mark_incomplete_custom
from dunetrace.models import FailureSignal, FailureType, Severity
import detector_svc.worker  # noqa: F401  # so patch() can resolve "detector_svc.worker.*"

# ── Event factories ────────────────────────────────────────────────────────────


def evt(event_type: str, step_index: int = 1, payload: dict | None = None) -> dict:
    return {
        "event_type": event_type,
        "run_id": "run-shed-1",
        "agent_id": "agent-test",
        "agent_version": "abc12345",
        "step_index": step_index,
        "timestamp": time.time(),
        "payload": payload or {},
        "parent_run_id": None,
    }


def run_started(step: int = 0) -> dict:
    return evt("run.started", step, {"input_text": "abc", "model": "gpt-4o", "tools": ["web"]})


def tool_evt(step: int) -> dict:
    return evt("tool.called", step, {"tool_name": "web", "args": "aa"})


def run_completed(step: int, dropped: int | None = None) -> dict:
    payload: dict = {"exit_reason": "final_answer", "total_steps": step}
    if dropped is not None:
        payload["dropped_events"] = dropped
    return evt("run.completed", step, payload)


def run_errored(step: int, dropped: int | None = None) -> dict:
    payload: dict = {"error_type": "ValueError", "error": "boom"}
    if dropped is not None:
        payload["dropped_events"] = dropped
    return evt("run.errored", step, payload)


def looping_run(dropped: int | None = None, n: int = 6) -> list[dict]:
    """Enough identical tool calls to fire TOOL_LOOP, a live detector."""
    return [run_started(), *[tool_evt(i) for i in range(1, n + 1)], run_completed(n + 1, dropped)]


def clean_run(dropped: int | None = None) -> list[dict]:
    return [run_started(), tool_evt(1), run_completed(2, dropped)]


# ── build_run_state ────────────────────────────────────────────────────────────


class TestRunBuilderDroppedEvents(unittest.TestCase):
    def test_defaults_to_zero_when_absent(self):
        self.assertEqual(build_run_state(clean_run()).dropped_events, 0)

    def test_reads_from_run_completed(self):
        self.assertEqual(build_run_state(clean_run(dropped=7)).dropped_events, 7)

    def test_reads_from_run_errored(self):
        events = [run_started(), tool_evt(1), run_errored(2, dropped=3)]
        state = build_run_state(events)
        self.assertEqual(state.dropped_events, 3)
        self.assertEqual(state.exit_reason, "error")

    def test_takes_the_max_when_both_terminals_are_present(self):
        events = [run_started(), run_completed(2, dropped=3), run_errored(3, dropped=5)]
        self.assertEqual(build_run_state(events).dropped_events, 5)
        events = [run_started(), run_errored(2, dropped=9), run_completed(3, dropped=4)]
        self.assertEqual(build_run_state(events).dropped_events, 9)

    def test_malformed_or_negative_count_reads_as_zero(self):
        self.assertEqual(build_run_state(clean_run(dropped="lots")).dropped_events, 0)
        self.assertEqual(build_run_state(clean_run(dropped=-4)).dropped_events, 0)
        self.assertEqual(build_run_state(clean_run(dropped=None)).dropped_events, 0)

    def test_numeric_string_is_accepted(self):
        self.assertEqual(build_run_state(clean_run(dropped="12")).dropped_events, 12)


# ── Marker helpers ─────────────────────────────────────────────────────────────


def _signal(severity=Severity.CRITICAL, confidence=0.95, failure_type=FailureType.TOOL_LOOP):
    return FailureSignal(
        failure_type=failure_type,
        severity=severity,
        run_id="run-shed-1",
        agent_id="agent-test",
        agent_version="abc12345",
        step_index=3,
        confidence=confidence,
        evidence={"first_step": 1, "last_step": 6},
    )


class TestMarkIncomplete(unittest.TestCase):
    def test_caps_confidence_and_severity_and_marks_evidence(self):
        sig = _signal(Severity.CRITICAL, 0.95)
        _mark_incomplete(sig, 12)
        self.assertEqual(sig.confidence, 0.5)
        self.assertEqual(sig.severity, Severity.MEDIUM)
        self.assertEqual(
            sig.evidence["incomplete_data"], {"dropped_events": 12, "reason": "sdk_buffer_shed"}
        )
        self.assertEqual(sig.evidence["first_step"], 1, "existing evidence is kept")

    def test_high_is_capped_but_lower_severities_are_left_alone(self):
        sig = _signal(Severity.HIGH, 0.3)
        _mark_incomplete(sig, 1)
        self.assertEqual(sig.severity, Severity.MEDIUM)
        self.assertEqual(sig.confidence, 0.3, "a confidence under the cap is not raised")
        low = _signal(Severity.LOW, 0.2)
        _mark_incomplete(low, 1)
        self.assertEqual(low.severity, Severity.LOW)

    def test_custom_result_dict_gets_the_same_treatment(self):
        result = {
            "failure_type": "MY_RULE",
            "severity": "HIGH",
            "step_index": 1,
            "confidence": 0.9,
            "evidence": {"detector_name": "MY_RULE"},
        }
        _mark_incomplete_custom(result, 4)
        self.assertEqual(result["severity"], "MEDIUM")
        self.assertEqual(result["confidence"], 0.5)
        self.assertEqual(
            result["evidence"]["incomplete_data"],
            {"dropped_events": 4, "reason": "sdk_buffer_shed"},
        )
        self.assertEqual(result["evidence"]["detector_name"], "MY_RULE")


# ── process_run ────────────────────────────────────────────────────────────────


class _Captured:
    """Everything process_run wrote, with the DB mocked out."""

    def __init__(self) -> None:
        self.signals: list = []
        self.shadow: list = []
        self.custom_calls: list = []

    async def write_signals(self, signals, shadow, org_id):
        self.signals.extend(signals)
        self.shadow.extend([shadow] * len(signals))
        return len(signals)

    async def write_custom_signal(self, **kw):
        self.custom_calls.append(kw)
        return 1


class TestProcessRunIncomplete(unittest.IsolatedAsyncioTestCase):
    def _patches(self, cap: _Captured, events: list, **extra):
        mocks = {
            "upsert": AsyncMock(),
            "advance": AsyncMock(),
            "baseline": AsyncMock(),
            "mark": AsyncMock(),
        }
        mocks.update(extra)
        ctx = [
            patch("detector_svc.worker.fetch_run_events", AsyncMock(return_value=events)),
            patch("detector_svc.worker.write_signals", cap.write_signals),
            patch("detector_svc.worker.write_custom_signal", cap.write_custom_signal),
            patch("detector_svc.worker.mark_run_processed", mocks["mark"]),
            patch("detector_svc.worker.upsert_fired_issues", mocks["upsert"]),
            patch("detector_svc.worker.advance_clean_runs", mocks["advance"]),
            patch("detector_svc.worker.write_baseline_metrics", mocks["baseline"]),
        ]
        return ctx, mocks

    async def _run(self, events: list, **extra):
        cap = _Captured()
        ctx, mocks = self._patches(cap, events, **extra)
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in ctx:
                stack.enter_context(p)
            from detector_svc.worker import process_run

            count = await process_run("run-shed-1", "agent-test", "abc1", "completed", "org-1")
        return count, cap, mocks

    async def test_control_complete_looping_run_is_live_and_tracked(self):
        """Without dropped_events the same run yields a LIVE TOOL_LOOP and
        counts for issue tracking — so the assertions below mean something."""
        count, cap, mocks = await self._run(looping_run())
        self.assertGreater(count, 0)
        types = [s.failure_type for s in cap.signals]
        self.assertIn(FailureType.TOOL_LOOP, types)
        loop_shadow = [
            sh for s, sh in zip(cap.signals, cap.shadow) if s.failure_type == FailureType.TOOL_LOOP
        ]
        self.assertEqual(loop_shadow, [False])
        self.assertNotIn("incomplete_data", cap.signals[0].evidence)
        mocks["upsert"].assert_called_once()
        mocks["advance"].assert_called_once()
        mocks["baseline"].assert_called_once()

    async def test_incomplete_run_writes_every_signal_shadow_capped_and_marked(self):
        count, cap, mocks = await self._run(looping_run(dropped=9))
        self.assertGreater(count, 0)
        self.assertIn(FailureType.TOOL_LOOP, [s.failure_type for s in cap.signals])
        self.assertTrue(
            all(cap.shadow), "every signal of a shed run is shadow, LIVE_DETECTORS or not"
        )
        for sig in cap.signals:
            self.assertLessEqual(sig.confidence, 0.5)
            self.assertNotIn(sig.severity, (Severity.HIGH, Severity.CRITICAL))
            self.assertEqual(
                sig.evidence["incomplete_data"],
                {"dropped_events": 9, "reason": "sdk_buffer_shed"},
            )
        mocks["mark"].assert_called_once()

    async def test_incomplete_run_is_neither_fired_nor_clean(self):
        _, _, mocks = await self._run(looping_run(dropped=9))
        mocks["upsert"].assert_not_called()
        mocks["advance"].assert_not_called()
        mocks["baseline"].assert_not_called()

    async def test_clean_looking_incomplete_run_is_not_counted_clean(self):
        """A run with no signals but dropped events is unknown: it must not
        advance the consecutive-clean counter or feed the baselines."""
        count, cap, mocks = await self._run(clean_run(dropped=2))
        self.assertEqual(count, 0)
        self.assertEqual(cap.signals, [])
        mocks["advance"].assert_not_called()
        mocks["upsert"].assert_not_called()
        mocks["baseline"].assert_not_called()
        mocks["mark"].assert_called_once()

    async def test_control_clean_complete_run_is_counted_clean(self):
        _, _, mocks = await self._run(clean_run())
        mocks["advance"].assert_called_once()
        mocks["baseline"].assert_called_once()

    async def test_risk_engine_sees_no_live_signals_for_an_incomplete_run(self):
        engine = MagicMock()
        engine.evaluate.return_value = MagicMock(
            severity=None, confidence=0.0, active_signals=0, scores={}
        )
        with patch("detector_svc.worker.RiskEngine", return_value=engine):
            _, cap, _ = await self._run(looping_run(dropped=3))
        self.assertIn(FailureType.TOOL_LOOP, [s.failure_type for s in cap.signals])
        engine.evaluate.assert_called_once()
        live_signals = engine.evaluate.call_args[0][0]
        self.assertEqual(live_signals, [])

    async def test_json_custom_detector_signal_is_shadowed_and_capped(self):
        fired = {
            "failure_type": "MY_RULE",
            "severity": "CRITICAL",
            "step_index": 1,
            "confidence": 0.99,
            "evidence": {"detector_name": "MY_RULE"},
        }
        custom_defs = [{"id": 7, "config": {}, "shadow": False}]  # an ACTIVE custom detector
        with (
            patch(
                "detector_svc.worker.fetch_custom_detectors", AsyncMock(return_value=custom_defs)
            ),
            patch("detector_svc.worker.evaluate_custom_detector", return_value=fired),
            patch("detector_svc.worker.record_custom_detector_results", AsyncMock()),
        ):
            _, cap, _ = await self._run(clean_run(dropped=5))
        self.assertEqual(len(cap.custom_calls), 1)
        call = cap.custom_calls[0]
        self.assertTrue(call["shadow"])
        self.assertEqual(call["severity"], "MEDIUM")
        self.assertEqual(call["confidence"], 0.5)
        self.assertEqual(
            call["evidence"]["incomplete_data"], {"dropped_events": 5, "reason": "sdk_buffer_shed"}
        )

    async def test_python_plugin_signal_is_shadowed_and_capped(self):
        plugin = FailureSignal(
            failure_type=FailureType.CUSTOM,
            severity=Severity.HIGH,
            run_id="run-shed-1",
            agent_id="agent-test",
            agent_version="abc12345",
            step_index=1,
            confidence=0.9,
            evidence={"detector_name": "MyPlugin"},
        )
        live_plugin_cls = MagicMock(SHADOW_BY_DEFAULT=False)
        with (
            patch("detector_svc.worker.run_detectors", return_value=[plugin]),
            patch(
                "detector_svc.worker._resolve_custom_detector_class", return_value=live_plugin_cls
            ),
        ):
            _, cap, _ = await self._run(clean_run(dropped=2))
        self.assertEqual(cap.signals, [], "plugin signals go through write_custom_signal")
        self.assertEqual(len(cap.custom_calls), 1)
        call = cap.custom_calls[0]
        self.assertEqual(call["failure_type"], "MyPlugin")
        self.assertTrue(call["shadow"])
        self.assertEqual(call["severity"], "MEDIUM")
        self.assertEqual(call["confidence"], 0.5)
        self.assertEqual(call["evidence"]["incomplete_data"]["dropped_events"], 2)

    async def test_incomplete_run_logs_one_info_line(self):
        with self.assertLogs("dunetrace.detector", level="INFO") as cm:
            await self._run(clean_run(dropped=2))
        lines = [line for line in cm.output if "incomplete" in line and "run-shed-1" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("shed 2", lines[0])


if __name__ == "__main__":
    unittest.main()
