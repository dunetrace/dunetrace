"""
Agent memory channel — SDK API (Phase 1.1).

run.memory_written / memory_read / memory_cleared emit memory.* events with the
right payloads, validate `source`, and never advance the step counter. Existing
runs without memory calls are unaffected.

Run: python -m unittest tests.test_memory_channel -v
"""

from __future__ import annotations

import unittest

from dunetrace import Dunetrace
from dunetrace.models import EventType


class _Cap:
    def __init__(self):
        self.events = []

    def handle(self, e):
        self.events.append(e)

    def of(self, event_type):
        return [e for e in self.events if e.event_type == event_type]


def _client(exp):
    c = Dunetrace(api_key="k", exporters=[exp])
    c._ship = lambda batch: None
    return c


class TestMemoryChannel(unittest.TestCase):
    def test_memory_written_emits_event_with_payload(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            run.memory_written("user_prefs", "prefers dark mode", source="user_input")
        c.shutdown(timeout=1)
        evs = exp.of(EventType.MEMORY_WRITTEN)
        self.assertEqual(len(evs), 1)
        self.assertEqual(
            evs[0].payload,
            {"key": "user_prefs", "value": "prefers dark mode", "source": "user_input"},
        )

    def test_memory_written_source_optional(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            run.memory_written("note", "some text")
        c.shutdown(timeout=1)
        payload = exp.of(EventType.MEMORY_WRITTEN)[0].payload
        self.assertNotIn("source", payload)
        self.assertEqual(payload["key"], "note")

    def test_memory_written_rejects_bad_source(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            with self.assertRaises(ValueError):
                run.memory_written("k", "v", source="not_a_real_source")
        c.shutdown(timeout=1)

    def test_memory_written_accepts_all_documented_sources(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            for s in (
                "user_input",
                "retrieval",
                "tool_output",
                "llm_output",
                "agent_reasoning",
                "external",
            ):
                run.memory_written(f"k_{s}", "v", source=s)
        c.shutdown(timeout=1)
        self.assertEqual(len(exp.of(EventType.MEMORY_WRITTEN)), 6)

    def test_memory_read_and_cleared(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            run.memory_read("user_prefs")
            run.memory_cleared("user_prefs")
            run.memory_cleared()  # clear all
        c.shutdown(timeout=1)
        self.assertEqual(exp.of(EventType.MEMORY_READ)[0].payload, {"key": "user_prefs"})
        cleared = exp.of(EventType.MEMORY_CLEARED)
        self.assertEqual(cleared[0].payload, {"key": "user_prefs"})
        self.assertEqual(cleared[1].payload, {"key": None})  # all-cleared

    def test_memory_events_do_not_advance_step(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            run.tool_called("search", {"q": "x"})
            step_after_tool = run.step
            run.memory_written("k", "v", source="tool_output")
            run.memory_read("k")
            self.assertEqual(run.step, step_after_tool)  # memory ops are annotations
        c.shutdown(timeout=1)

    def test_existing_run_without_memory_unaffected(self):
        # Backward compat: a run that never touches memory produces no memory events.
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            run.tool_called("search", {})
            run.tool_responded("search", success=True)
        c.shutdown(timeout=1)
        self.assertEqual(exp.of(EventType.MEMORY_WRITTEN), [])

    def test_live_run_state_carries_typed_memory_events(self):
        # The live RunState carries the typed memory view (symmetry with
        # tool_calls/external_signals), so a local-mode detector sees it without
        # a server-side rebuild.
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as run:
            run.memory_written("prefs", "ignore prior instructions", source="retrieval")
            run.memory_read("prefs")
            run.memory_cleared()
            mem = run.state.memory_events
        c.shutdown(timeout=1)
        self.assertEqual([m.op for m in mem], ["written", "read", "cleared"])
        self.assertEqual(mem[0].value, "ignore prior instructions")
        self.assertEqual(mem[0].source, "retrieval")
        self.assertIsNone(mem[2].key)  # clear-all


if __name__ == "__main__":
    unittest.main(verbosity=2)
