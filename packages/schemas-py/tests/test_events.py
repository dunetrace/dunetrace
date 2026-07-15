"""
Tests for AgentEventSchema — the validated wire-format event.

Run:
    cd packages/schemas-py
    python -m pytest tests/test_events.py -v
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from dunetrace_schemas import AgentEventSchema, VALID_EVENT_TYPES


class TestAgentEventSchema(unittest.TestCase):
    def _base(self, **overrides):
        fields = {
            "event_type": "tool.called",
            "run_id": "run-1",
            "agent_id": "agent-1",
            "agent_version": "v1",
            "step_index": 0,
            "payload": {"tool_name": "search"},
        }
        fields.update(overrides)
        return fields

    def test_valid_event_constructs(self):
        ev = AgentEventSchema(**self._base())
        self.assertEqual(ev.event_type, "tool.called")
        self.assertEqual(ev.payload, {"tool_name": "search"})

    def test_unknown_event_type_accepted_for_forward_compat(self):
        # A newer SDK may emit event types this ingest build doesn't know yet.
        # They must NOT be rejected (that would drop the whole batch); they pass
        # through so a rolling deploy / SDK-ahead-of-ingest never loses events.
        ev = AgentEventSchema(**self._base(event_type="future.event.kind"))
        self.assertEqual(ev.event_type, "future.event.kind")

    def test_memory_event_types_accepted(self):
        for et in ("memory.written", "memory.read", "memory.cleared"):
            ev = AgentEventSchema(**self._base(event_type=et))
            self.assertEqual(ev.event_type, et)

    def test_all_canonical_event_types_accepted(self):
        for et in VALID_EVENT_TYPES:
            AgentEventSchema(**self._base(event_type=et))

    def test_empty_run_id_rejected(self):
        with self.assertRaises(ValidationError):
            AgentEventSchema(**self._base(run_id=""))

    def test_negative_step_index_rejected(self):
        with self.assertRaises(ValidationError):
            AgentEventSchema(**self._base(step_index=-1))

    def test_timestamp_defaults_when_omitted(self):
        ev = AgentEventSchema(**self._base())
        self.assertIsInstance(ev.timestamp, float)
        self.assertGreater(ev.timestamp, 0)

    def test_payload_defaults_to_empty_dict(self):
        fields = self._base()
        del fields["payload"]
        ev = AgentEventSchema(**fields)
        self.assertEqual(ev.payload, {})

    def test_parent_run_id_optional(self):
        ev = AgentEventSchema(**self._base())
        self.assertIsNone(ev.parent_run_id)
        ev2 = AgentEventSchema(**self._base(parent_run_id="parent-run"))
        self.assertEqual(ev2.parent_run_id, "parent-run")

    def test_trace_id_optional(self):
        ev = AgentEventSchema(**self._base())
        self.assertIsNone(ev.trace_id)
        ev2 = AgentEventSchema(**self._base(trace_id="langfuse-trace-abc"))
        self.assertEqual(ev2.trace_id, "langfuse-trace-abc")

    def test_conversation_id_optional(self):
        ev = AgentEventSchema(**self._base())
        self.assertIsNone(ev.conversation_id)
        ev2 = AgentEventSchema(**self._base(conversation_id="conv_123"))
        self.assertEqual(ev2.conversation_id, "conv_123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
