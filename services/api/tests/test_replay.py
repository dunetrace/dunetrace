"""
Regression tests for replay-simulation modifications (api_svc.routers.replay).

Two modifications shipped broken because they didn't actually move the detector
they targeted:
  - reduce_context scaled every prompt-token count uniformly, but CONTEXT_BLOAT
    fires on the last/first RATIO (scale-invariant), so it never resolved.
  - break_tool_loop deduped (tool_name, args), but TOOL_LOOP counts by tool_name,
    so an arg-varying loop was never thinned below threshold.

These tests apply the modification to synthetic events, rebuild the run state,
run the real detectors, and assert the targeted signal actually clears.

Run: make test-api  (or python -m pytest services/api/tests/test_replay.py)
"""

from __future__ import annotations

import unittest

from api_svc.routers.replay import _apply_modifications
from api_svc.run_builder import build_run_state
from dunetrace.detectors import run_detectors


def _ev(event_type: str, step: int, **payload) -> dict:
    return {
        "event_type": event_type,
        "run_id": "r1",
        "agent_id": "a1",
        "agent_version": "v1",
        "step_index": step,
        "timestamp": float(step),
        "payload": payload,
        "parent_run_id": None,
    }


def _fired(events: list) -> set:
    return {s.failure_type.value for s in run_detectors(build_run_state(events))}


class TestReduceContext(unittest.TestCase):
    def _bloated_run(self) -> list:
        # prompt_tokens on llm.responded (llm.called carries 0, as some frameworks
        # do). Growth 1000 -> 9000 = 9x, well over CONTEXT_BLOAT's 3x, last >= 2000.
        events = [_ev("run.started", 0)]
        pts = [1000, 3000, 9000]
        for i, pt in enumerate(pts):
            step = i + 1
            events.append(_ev("llm.called", step, model="gpt-4o", prompt_tokens=0))
            events.append(_ev("llm.responded", step, prompt_tokens=pt, completion_tokens=50))
        events.append(_ev("run.completed", len(pts) + 1, exit_reason="completed"))
        return events

    def test_context_bloat_fires_before_fix(self):
        self.assertIn("CONTEXT_BLOAT", _fired(self._bloated_run()))

    def test_reduce_context_resolves_context_bloat(self):
        modified = _apply_modifications(self._bloated_run(), ["reduce_context"])
        self.assertNotIn("CONTEXT_BLOAT", _fired(modified))


class TestBreakToolLoop(unittest.TestCase):
    def _loop_run(self) -> list:
        # Same tool, six times, each with DIFFERENT args (an arg-varying loop).
        events = [_ev("run.started", 0)]
        for i in range(6):
            step = i + 1
            events.append(_ev("tool.called", step, tool_name="search", args=f"q{i}"))
            events.append(_ev("tool.responded", step, tool_name="search", success=True))
        events.append(_ev("run.completed", 7, exit_reason="completed"))
        return events

    def test_tool_loop_fires_before_fix(self):
        self.assertIn("TOOL_LOOP", _fired(self._loop_run()))

    def test_break_tool_loop_resolves_arg_varying_loop(self):
        modified = _apply_modifications(self._loop_run(), ["break_tool_loop"])
        self.assertNotIn("TOOL_LOOP", _fired(modified))


if __name__ == "__main__":
    unittest.main(verbosity=2)
