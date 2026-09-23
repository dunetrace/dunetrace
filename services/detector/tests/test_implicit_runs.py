"""
Implicit runs in the detector.

A run the SDK opened on its own (``run.started`` payload ``implicit: true``)
has a guessed boundary. ``build_run_state`` reads the flag and ``process_run``
treats the run like a shed one: every signal in shadow, confidence and
severity capped, evidence marked with reason ``implicit_run``, and the run
excluded from issue tracking and baselines. An entry-point run that only
carries ``opened_by`` is an ordinary run.

Reuses the harness from test_dropped_events.py.
"""

from __future__ import annotations

import unittest

from dunetrace.models import FailureType, Severity
from detector_svc.run_builder import build_run_state
from detector_svc.worker import _incomplete_reason, _mark_incomplete

from test_dropped_events import TestProcessRunIncomplete, clean_run, looping_run


def _implicit(events: list, opened_by: str = "openai.chat.completions.create") -> list:
    out = [dict(e) for e in events]
    for e in out:
        if e["event_type"] == "run.started":
            e["payload"] = {**(e.get("payload") or {}), "implicit": True, "opened_by": opened_by}
    return out


def _entry_point(events: list, opened_by: str = "langgraph.invoke") -> list:
    out = [dict(e) for e in events]
    for e in out:
        if e["event_type"] == "run.started":
            e["payload"] = {**(e.get("payload") or {}), "opened_by": opened_by}
    return out


class TestBuildRunStateImplicit(unittest.TestCase):
    def test_flag_is_read_from_run_started(self):
        self.assertFalse(build_run_state(clean_run()).implicit)
        self.assertTrue(build_run_state(_implicit(clean_run())).implicit)

    def test_only_a_true_flag_counts(self):
        events = _implicit(clean_run())
        events[0]["payload"]["implicit"] = "yes"
        self.assertFalse(build_run_state(events).implicit)

    def test_an_entry_point_run_is_not_implicit(self):
        self.assertFalse(build_run_state(_entry_point(clean_run())).implicit)

    def test_incomplete_reason_prefers_shed_over_implicit(self):
        self.assertIsNone(_incomplete_reason(build_run_state(clean_run())))
        self.assertEqual(
            _incomplete_reason(build_run_state(_implicit(clean_run()))), "implicit_run"
        )
        self.assertEqual(
            _incomplete_reason(build_run_state(_implicit(clean_run(dropped=3)))), "sdk_buffer_shed"
        )

    def test_marker_carries_the_reason(self):
        from test_dropped_events import _signal

        sig = _signal(Severity.CRITICAL, 0.9)
        _mark_incomplete(sig, 0, "implicit_run")
        self.assertEqual(
            sig.evidence["incomplete_data"], {"dropped_events": 0, "reason": "implicit_run"}
        )
        self.assertEqual(sig.confidence, 0.5)
        self.assertEqual(sig.severity, Severity.MEDIUM)


class TestProcessRunImplicit(TestProcessRunIncomplete):
    """Drive process_run through the same mocked harness as the shed-run tests."""

    # The inherited tests stay valid controls; the ones below add the implicit case.

    async def test_implicit_run_signals_are_shadow_capped_and_marked(self):
        count, cap, mocks = await self._run(_implicit(looping_run()))
        self.assertGreater(count, 0)
        self.assertIn(FailureType.TOOL_LOOP, [s.failure_type for s in cap.signals])
        self.assertTrue(all(cap.shadow), "every signal of an implicit run is shadow")
        for sig in cap.signals:
            self.assertLessEqual(sig.confidence, 0.5)
            self.assertNotIn(sig.severity, (Severity.HIGH, Severity.CRITICAL))
            self.assertEqual(
                sig.evidence["incomplete_data"], {"dropped_events": 0, "reason": "implicit_run"}
            )
        mocks["mark"].assert_called_once()

    async def test_implicit_run_is_neither_fired_nor_clean(self):
        _, _, mocks = await self._run(_implicit(looping_run()))
        mocks["upsert"].assert_not_called()
        mocks["advance"].assert_not_called()
        mocks["baseline"].assert_not_called()

    async def test_clean_implicit_run_does_not_count_as_clean(self):
        count, cap, mocks = await self._run(_implicit(clean_run()))
        self.assertEqual(count, 0)
        mocks["advance"].assert_not_called()
        mocks["baseline"].assert_not_called()

    async def test_entry_point_run_is_an_ordinary_live_run(self):
        count, cap, mocks = await self._run(_entry_point(looping_run()))
        self.assertGreater(count, 0)
        loop_shadow = [
            sh for s, sh in zip(cap.signals, cap.shadow) if s.failure_type == FailureType.TOOL_LOOP
        ]
        self.assertEqual(loop_shadow, [False])
        mocks["upsert"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
