"""
Fault-injection regression corpus for the structural detectors.

`test_replay.py` covers the repair direction — take a broken run, apply a
replay modification, assert the signal clears. This is the other direction:
take a run that fires NOTHING, inject one fault as events, assert the right
detector fires. Together they close the loop, and the round-trip test below
asserts exactly that: inject -> fires -> repair -> clears.

Three things here that the per-detector unit tests cannot do, because those
build a `RunState` by hand and never touch the wire format:

  * the negative control. `clean_run()` must produce zero signals. A detector
    that starts firing on well-behaved traffic is the expensive regression —
    it costs trust rather than coverage — and nothing asserted it before.
  * event-level coverage. Faults are injected as events and rebuilt through
    `build_run_state`, so a reconstruction bug shows up as a detector that
    stopped firing. The API and detector copies of that function had already
    silently drifted apart once.
  * the pairing. A replay modification is only correct relative to the fault
    it claims to repair; checking each against its own hand-written fixture is
    how `reduce_context` and `break_tool_loop` both shipped not moving their
    own detectors.

Writing this corpus immediately found two more of the same kind, both now
pinned below: dict-valued tool args silently disabled three detectors, and
`fix_rag_retrieval` did not clear RAG_EMPTY_RETRIEVAL.

No DB, no network, no credentials, no clock.

Run: make test-api
  or PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api \
       python -m pytest services/api/tests/test_fault_injection.py -v
"""

from __future__ import annotations

import copy
import unittest

from api_svc.routers.replay import _VALID_MODS, _apply_modifications
from api_svc.run_builder import build_run_state
from dunetrace.detectors import run_detectors

from fault_injection import INJECTORS, clean_run, inject_tool_loop


def fired(events: list) -> set:
    """Failure types the in-path battery produces for these events.

    TIER1_DETECTORS (31), not the worker's 34: PROMPT_INJECTION_SIGNAL reads
    the raw run.started input, and HANDOFF_CONTEXT_LOSS / DELEGATION_LOOP are
    evaluated against a cross-run delegation graph — none of the three is a
    function of one RunState, which is all a synthetic stream can offer.
    """
    return {s.failure_type.value for s in run_detectors(build_run_state(events))}


class TestBaselineIsClean(unittest.TestCase):
    """The negative control. If this fails, either a detector started firing
    on healthy traffic or the baseline stopped being healthy — and the rest of
    the file proves nothing until it is resolved."""

    def test_clean_run_fires_nothing(self):
        self.assertEqual(fired(clean_run()), set())

    def test_clean_run_is_clean_at_several_lengths(self):
        for steps in (1, 2, 3, 5, 8):
            with self.subTest(steps=steps):
                self.assertEqual(fired(clean_run(steps=steps)), set())

    def test_clean_run_is_deterministic(self):
        self.assertEqual(clean_run(), clean_run())

    def test_clean_run_actually_reconstructs(self):
        """A baseline that is silent because it rebuilt into an empty state
        would pass every other test here while testing nothing."""
        state = build_run_state(clean_run(steps=3))
        self.assertEqual(len(state.tool_calls), 3)
        self.assertEqual(len(state.llm_calls), 4)
        self.assertEqual(state.exit_reason, "final_answer")


class TestInjectorsFireTheirFault(unittest.TestCase):
    def test_each_injector_produces_its_failure_type(self):
        for name, (inject, expected, _) in INJECTORS.items():
            with self.subTest(injector=name):
                self.assertIn(expected, fired(inject(clean_run())))

    def test_each_injector_is_deterministic(self):
        for name, (inject, _, _) in INJECTORS.items():
            with self.subTest(injector=name):
                self.assertEqual(inject(clean_run()), inject(clean_run()))

    def test_injectors_do_not_mutate_their_input(self):
        """Injectors are composed in tests and must not alias the baseline."""
        for name, (inject, _, _) in INJECTORS.items():
            with self.subTest(injector=name):
                events = clean_run()
                before = copy.deepcopy(events)
                inject(events)
                self.assertEqual(events, before)

    def test_every_injector_names_a_real_failure_type(self):
        from dunetrace.models import FailureType

        valid = {f.value for f in FailureType}
        for name, (_, expected, _) in INJECTORS.items():
            with self.subTest(injector=name):
                self.assertIn(expected, valid)


class TestRoundTrip(unittest.TestCase):
    """inject -> fires -> repair -> clears. A replay modification is only
    meaningful against the fault it claims to undo."""

    def test_repair_clears_the_injected_fault(self):
        for name, (inject, expected, repair) in INJECTORS.items():
            if repair is None:
                continue
            with self.subTest(injector=name, repair=repair):
                broken = inject(clean_run())
                self.assertIn(expected, fired(broken), "fault did not reproduce")
                self.assertNotIn(expected, fired(_apply_modifications(broken, [repair])))

    def test_named_repairs_are_in_the_replay_allowlist(self):
        for name, (_, _, repair) in INJECTORS.items():
            if repair is None:
                continue
            with self.subTest(injector=name):
                self.assertIn(repair, _VALID_MODS)

    def test_repairs_leave_a_clean_run_clean(self):
        """A repair applied to a healthy run must not invent a signal."""
        for repair in sorted(_VALID_MODS - {"truncate_at_step"}):
            with self.subTest(repair=repair):
                self.assertEqual(fired(_apply_modifications(clean_run(), [repair])), set())


class TestDictToolArgsRegression(unittest.TestCase):
    """`ToolCall.args` is typed `str`, and the SDK and OTLP mapper both
    serialize before emitting — but `AgentEventSchema.payload` is
    `Dict[str, Any]` and validates nothing inside it, so a hand-rolled
    POST /v1/ingest can put a dict there.

    TOOL_LOOP and RETRY_STORM then hashed it into a set and
    TOOL_ARGUMENT_FABRICATION regexed over it; all three raised, and
    `run_detectors` catches per-detector exceptions, so the run produced no
    signal from any of them and nothing recorded that detection had been
    switched off. build_run_state coerces at the boundary now.
    """

    def test_dict_args_reconstruct_as_strings(self):
        state = build_run_state(clean_run())
        for call in state.tool_calls:
            self.assertIsInstance(call.args, str)

    def test_equal_dicts_serialize_identically(self):
        """TOOL_LOOP compares these for equality to decide whether a loop is
        arg-varying, so key ordering must not make two equal dicts differ."""
        a = build_run_state(
            [
                {
                    "event_type": "tool.called",
                    "run_id": "r",
                    "agent_id": "a",
                    "agent_version": "v",
                    "step_index": 1,
                    "timestamp": 1.0,
                    "payload": {"tool_name": "t", "args": {"x": 1, "y": 2}},
                }
            ]
        )
        b = build_run_state(
            [
                {
                    "event_type": "tool.called",
                    "run_id": "r",
                    "agent_id": "a",
                    "agent_version": "v",
                    "step_index": 1,
                    "timestamp": 1.0,
                    "payload": {"tool_name": "t", "args": {"y": 2, "x": 1}},
                }
            ]
        )
        self.assertEqual(a.tool_calls[0].args, b.tool_calls[0].args)

    def test_detectors_that_used_to_raise_now_fire(self):
        """The injected loop carries dict args. Before the coercion all three
        of these were skipped, so TOOL_LOOP silently never fired."""
        self.assertIn("TOOL_LOOP", fired(inject_tool_loop(clean_run())))

    def test_unserializable_args_do_not_break_reconstruction(self):
        state = build_run_state(
            [
                {
                    "event_type": "tool.called",
                    "run_id": "r",
                    "agent_id": "a",
                    "agent_version": "v",
                    "step_index": 1,
                    "timestamp": 1.0,
                    "payload": {"tool_name": "t", "args": {1, 2, 3}},
                }
            ]
        )
        self.assertIsInstance(state.tool_calls[0].args, str)

    def test_string_args_are_passed_through_unchanged(self):
        state = build_run_state(
            [
                {
                    "event_type": "tool.called",
                    "run_id": "r",
                    "agent_id": "a",
                    "agent_version": "v",
                    "step_index": 1,
                    "timestamp": 1.0,
                    "payload": {"tool_name": "t", "args": "q=hello"},
                }
            ]
        )
        self.assertEqual(state.tool_calls[0].args, "q=hello")


class TestRagRepairRegression(unittest.TestCase):
    """RAG_EMPTY_RETRIEVAL fires on `result_count < MIN_RESULTS` OR a non-None
    `top_score < MIN_SCORE`. fix_rag_retrieval repaired only the count and set
    the score only when it was None, so an empty retrieval reporting
    `top_score: 0.0` — the natural representation of "nothing matched" — was
    never repaired and the signal never cleared."""

    def _empty(self, top_score):
        return [
            {
                "event_type": "run.started",
                "run_id": "r",
                "agent_id": "a",
                "agent_version": "v",
                "step_index": 0,
                "timestamp": 0.0,
                "payload": {},
            },
            {
                "event_type": "retrieval.responded",
                "run_id": "r",
                "agent_id": "a",
                "agent_version": "v",
                "step_index": 1,
                "timestamp": 1.0,
                "payload": {"result_count": 0, "top_score": top_score},
            },
            {
                "event_type": "run.completed",
                "run_id": "r",
                "agent_id": "a",
                "agent_version": "v",
                "step_index": 2,
                "timestamp": 2.0,
                "payload": {"exit_reason": "final_answer"},
            },
        ]

    def test_zero_top_score_is_repaired(self):
        events = self._empty(0.0)
        self.assertIn("RAG_EMPTY_RETRIEVAL", fired(events))
        self.assertNotIn(
            "RAG_EMPTY_RETRIEVAL", fired(_apply_modifications(events, ["fix_rag_retrieval"]))
        )

    def test_null_top_score_is_still_repaired(self):
        events = self._empty(None)
        self.assertIn("RAG_EMPTY_RETRIEVAL", fired(events))
        self.assertNotIn(
            "RAG_EMPTY_RETRIEVAL", fired(_apply_modifications(events, ["fix_rag_retrieval"]))
        )

    def test_below_threshold_score_with_results_is_repaired(self):
        """The other half of the OR: results came back, but none relevant."""
        events = self._empty(0.1)
        events[1]["payload"]["result_count"] = 5
        self.assertIn("RAG_EMPTY_RETRIEVAL", fired(events))
        self.assertNotIn(
            "RAG_EMPTY_RETRIEVAL", fired(_apply_modifications(events, ["fix_rag_retrieval"]))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
