"""
Approval notes as detector trust-surface input.

This is the half of the feature that stops it being self-defeating. A human
denies an action and writes a correction; the agent obeys it and retries. If the
trust surfaces cannot see the note, our own detectors flag the *corrected*
behaviour — UNGROUNDED_DESTINATION on the address the human supplied,
UNRESOLVED_AMBIGUITY on the record the human named. A false positive on the run
where the human was obeyed is the worst possible place to spend operator trust,
so the surfaces and the note ship together.

Each grounding/warrant test is built so the note is the ONLY thing that can
silence it: the paired control removes the note and nothing else, and asserts the
detector fires. Without that control a passing test proves nothing — the run
might have been silent for some unrelated reason.

The last class pins the run-scoping limitation. It is a negative test on purpose:
it asserts today's behaviour so the cross-run extension has a written
specification to turn green.
"""

from __future__ import annotations

import json
import time
import unittest

from dunetrace.detectors import (
    UngroundedDestinationDetector,
    UnresolvedAmbiguityDetector,
    _approval_notes,
)
from dunetrace.models import AgentEvent, EventType, RunState, ToolCall

# Verbatim from dunetrace-demos/demo1/agent/customers.py.
SARAH = {
    "id": "CUST_8834",
    "name": "Sarah Chen",
    "email": "sarah.chen@example.com",
    "signup_date": "2024-03-15",
    "plan": "Enterprise",
    "monthly_value": 12500,
    "business_name": "Chen Manufacturing GmbH",
    "annual_value_eur": 340000,
    "customer_since": "2019",
    "tickets_resolved": 47,
    "primary_contact": "Sarah Chen",
}
EMILY = {
    "id": "CUST_1183",
    "name": "Emily Chen",
    "email": "emily.chen@example.com",
    "signup_date": "2023-12-05",
    "plan": "Standard",
    "monthly_value": 250,
    "business_name": "Chen Design Studio",
    "annual_value_eur": 3000,
    "customer_since": "2023",
    "tickets_resolved": 3,
    "primary_contact": "Emily Chen",
}
GRACE = {
    "id": "CUST_2867",
    "name": "Grace Chen",
    "email": "grace.chen@example.com",
    "signup_date": "2024-05-22",
    "plan": "Individual",
    "monthly_value": 20,
    "business_name": "Grace Chen",
    "annual_value_eur": 240,
    "customer_since": "2024",
    "tickets_resolved": 0,
    "primary_contact": "Grace Chen",
}
THREE_CHENS = [SARAH, EMILY, GRACE]


def new_state(run_id="run-1", input_text="close the account for the Chen family"):
    return RunState(
        run_id=run_id, agent_id="support-agent", agent_version="v1", input_text=input_text
    )


def add_tool(state, tool_name, step, args, output=None, success=True):
    tc = ToolCall(
        tool_name=tool_name,
        args=args if isinstance(args, str) else json.dumps(args),
        step_index=step,
        timestamp=time.time(),
        success=success,
        output=None
        if output is None
        else (output if isinstance(output, str) else json.dumps(output)),
    )
    state.tool_calls.append(tc)
    state.current_step = max(state.current_step, step)
    return tc


def deny_and_retry(
    state, tool_name, blocked_args, retry_args, *, step, note, decided_by="alice", event_type=None
):
    """The measured runtime topology of a denied call followed by a retry.

    Probed against the real RunContext rather than assumed, because the offsets
    are load-bearing and not obvious:

        ToolCall(blocked)      step = S      success=None   <- gate raised before
                                                               tool_responded
        approval.denied        step = S + 1                 <- advance=False, but
                                                               tool.called already
                                                               advanced the step
        ToolCall(retry)        step = S + 1

    So the note sits AFTER the call it rejected and AT the retry — which is
    exactly what makes step-ordered warrant work without a special case: the note
    cannot justify the call it was a response to, and does justify the retry.
    """
    # success=None, exactly as the runtime leaves it: the gate raises before
    # tool_responded ever runs, so a blocked call is never marked failed. A
    # fixture that says success=True here is testing a call that ran.
    blocked = add_tool(state, tool_name, step, blocked_args, success=None)
    payload = {"approval_id": 7, "tool_name": tool_name}
    if note:
        payload["note"] = note
    if decided_by:
        payload["decided_by"] = decided_by
    state.events.append(
        AgentEvent(
            event_type=event_type or EventType.APPROVAL_DENIED,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=step + 1,
            payload=payload,
        )
    )
    # The retry is success=None too: it has been called but not yet responded,
    # which is precisely the state a retry issued from the except block is in.
    # A fixture that marks it success=True hides the collision this helper's
    # step arithmetic exists to get right.
    retry = add_tool(state, tool_name, step + 1, retry_args, success=None)
    return blocked, retry


def add_approval_event(
    state,
    step,
    event_type,
    *,
    note=None,
    decided_by="alice",
    tool_name="delete_customer",
    approval_id=7,
):
    """A bare approval terminal event, as RunContext._finish_approval emits it —
    keys omitted when absent, not set to None."""
    payload = {"approval_id": approval_id, "tool_name": tool_name}
    if note:
        payload["note"] = note
    if decided_by:
        payload["decided_by"] = decided_by
    state.events.append(
        AgentEvent(
            event_type=event_type,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=step,
            payload=payload,
        )
    )


# ── Grounding: UNGROUNDED_DESTINATION ─────────────────────────────────────────


class TestGroundingByApprovalNote(unittest.TestCase):
    """The human supplies an address in the deny note; the agent retries with it.

    The address appears in NO other trusted surface — not the task input, not the
    system prompt, not any tool output — so the note is the only thing that can
    ground it.
    """

    ADDRESS = "escalations@corp-legal.example.com"

    def _state(self, *, note):
        state = new_state(input_text="email the closure notice to legal@corp.example.com")
        deny_and_retry(
            state,
            "send_email",
            {"to": "legal@corp.example.com"},  # grounded by the task input
            {"to": self.ADDRESS},  # grounded ONLY by the note
            step=1,
            note=(f"not that mailbox — send it to {self.ADDRESS}" if note else None),
        )
        return state

    def _fire(self, state):
        return UngroundedDestinationDetector().on_run_completion(state)

    def test_retry_grounded_solely_by_the_deny_note_is_silent(self):
        self.assertIsNone(self._fire(self._state(note=True)))

    def test_control_without_the_note_fires(self):
        """Same run, note removed and nothing else. Proves the note is what
        silenced the case above."""
        signal = self._fire(self._state(note=False))
        self.assertIsNotNone(signal, "control must fire, or the test above proves nothing")
        self.assertEqual(signal.evidence["destination"], self.ADDRESS)

    def test_grant_note_grounds_too(self):
        """ "approved — use X going forward" is as legitimate a grounding as a
        correction on a deny."""
        state = new_state(input_text="email the closure notice to our legal contact")
        add_approval_event(
            state,
            1,
            EventType.APPROVAL_GRANTED,
            note=f"approved — use {self.ADDRESS} going forward",
            tool_name="send_email",
        )
        add_tool(state, "send_email", 2, {"to": self.ADDRESS})
        self.assertIsNone(self._fire(state))

    def test_approval_note_is_named_among_the_grounded_surfaces(self):
        state = self._state(note=True)
        add_tool(state, "send_email", 4, {"to": "someone@unrelated.example.org"})
        signal = self._fire(state)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.evidence["destination"], "someone@unrelated.example.org")
        self.assertIn("approval_note", signal.evidence["grounded_surfaces"])

    def test_blocked_call_is_not_reported_as_a_send(self):
        """The gate stopped it, so nothing was sent. Alerting here means alerting
        on the run where the control worked."""
        state = new_state(input_text="email the closure notice to our legal contact")
        deny_and_retry(
            state,
            "send_email",
            {"to": "attacker@evil.test"},  # ungrounded, but BLOCKED
            {"to": "legal@corp.example.com"},
            step=1,
            note="no — use legal@corp.example.com",
        )
        self.assertIsNone(self._fire(state))

    def test_blocked_call_still_reported_when_nothing_gated_it(self):
        """The exclusion is scoped to gated calls; an ordinary ungrounded send is
        untouched."""
        state = new_state(input_text="email the closure notice")
        add_tool(state, "send_email", 1, {"to": "attacker@evil.test"})
        signal = self._fire(state)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.evidence["destination"], "attacker@evil.test")


# ── Warrant: UNRESOLVED_AMBIGUITY ─────────────────────────────────────────────


class TestWarrantByApprovalNote(unittest.TestCase):
    """The canonical case. lookup_customer("Chen") returns three Chens, the agent
    picks Emily, a human denies with "wrong Chen — it's Sarah, CUST_8834", and
    the agent retries with Sarah. The retry is warranted by the note and nothing
    else — the task input says only "the Chen family"."""

    IRREVERSIBLE = ["delete_customer"]
    NOTE = "wrong Chen — it's Sarah, CUST_8834"

    def _state(self, *, note, retry_to=SARAH):
        state = new_state()
        add_tool(state, "lookup_customer", 0, {"q": "Chen"}, {"customers": THREE_CHENS})
        deny_and_retry(
            state,
            "delete_customer",
            {"customer_id": EMILY["id"]},
            {"customer_id": retry_to["id"]},
            step=1,
            note=(self.NOTE if note else None),
        )
        return state

    def _fire(self, state):
        det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=self.IRREVERSIBLE)
        return det.on_run_completion(state)

    def test_retry_warranted_solely_by_the_deny_note_is_silent(self):
        self.assertIsNone(self._fire(self._state(note=True)))

    def test_control_without_the_note_fires(self):
        signal = self._fire(self._state(note=False))
        self.assertIsNotNone(signal, "control must fire, or the test above proves nothing")
        self.assertEqual(signal.evidence["tier"], "weak")
        self.assertEqual(signal.evidence["selected_id"], SARAH["id"])

    def test_retry_against_the_note_is_strong(self):
        """The human named Sarah; the agent went to Grace anyway. That is the
        provable-mismatch tier, and the note is what makes it provable."""
        signal = self._fire(self._state(note=True, retry_to=GRACE))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.evidence["tier"], "strong")
        self.assertIn("sarah", signal.evidence["sibling_matched_in_request"])
        self.assertEqual(signal.evidence["selected_id"], GRACE["id"])

    def test_the_blocked_selection_is_not_itself_reported(self):
        """Emily was never deleted — the gate stopped it. Without this the run
        fires WEAK on the blocked attempt and the feature never reads as silent,
        no matter how well the note grounds the retry."""
        state = self._state(note=True)
        blocked = [
            tc
            for tc in state.tool_calls
            if tc.tool_name == "delete_customer" and tc.step_index == 1
        ]
        self.assertEqual(len(blocked), 1, "fixture must contain the blocked attempt")
        self.assertIsNone(self._fire(state))

    def test_note_cannot_warrant_an_action_that_preceded_it(self):
        """Step ordering still applies — a note is a turn like any other, and the
        runtime places it after the call it rejected."""
        state = new_state()
        add_tool(state, "lookup_customer", 0, {"q": "Chen"}, {"customers": THREE_CHENS})
        add_tool(state, "delete_customer", 1, {"customer_id": SARAH["id"]})
        add_approval_event(state, 5, EventType.APPROVAL_GRANTED, note=self.NOTE)
        signal = self._fire(state)
        self.assertIsNotNone(signal, "a later note must not justify an earlier action")
        self.assertEqual(signal.evidence["tier"], "weak")


# ── Scoping ───────────────────────────────────────────────────────────────────


class TestRunScoping(unittest.TestCase):
    """A note grounds and warrants only inside the run it was issued in.

    RunState.events holds one run's events, so this is structural rather than
    enforced — but it is load-bearing enough to pin, in both directions.
    """

    def test_note_does_not_leak_into_another_runs_state(self):
        note_run = new_state(run_id="run-A")
        add_approval_event(
            note_run,
            1,
            EventType.APPROVAL_DENIED,
            note="send it to escalations@corp-legal.example.com",
        )
        self.assertEqual(len(_approval_notes(note_run)), 1)

        other = new_state(run_id="run-B")
        self.assertEqual(
            _approval_notes(other), [], "a note must not be visible from a different run's state"
        )

    def test_KNOWN_LIMITATION_cross_run_retry_is_flagged(self):
        """THIS TEST PINS A LIMITATION, NOT A FEATURE.

        An agent that ends the run on denial and retries as a NEW run gets its
        corrected behaviour flagged: the note lives in run N's events and run N+1
        cannot see it. The prescribed idiom is to retry inside the same run,
        which is why the docs show only that pattern.

        When the cross-run extension lands — the same shape as
        UNGROUNDED_DESTINATION's cross-run memory taint, fed by the worker from
        prior events — this assertion inverts, and that inversion is the
        specification of the work. Do not "fix" this test by silencing the
        detector run-locally; that would ground a value from a note the run never
        saw.
        """
        det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=["delete_customer"])

        # Run N: the agent picks Emily, is denied with the correction, and stops.
        run_n = new_state(run_id="run-N")
        add_tool(run_n, "lookup_customer", 0, {"q": "Chen"}, {"customers": THREE_CHENS})
        add_tool(run_n, "delete_customer", 1, {"customer_id": EMILY["id"]}, success=None)
        add_approval_event(
            run_n, 2, EventType.APPROVAL_DENIED, note="wrong Chen — it's Sarah, CUST_8834"
        )

        # Run N+1: the corrected retry, as a separate run. The note is not here.
        run_n1 = new_state(run_id="run-N+1")
        add_tool(run_n1, "lookup_customer", 0, {"q": "Chen"}, {"customers": THREE_CHENS})
        add_tool(run_n1, "delete_customer", 1, {"customer_id": SARAH["id"]})

        signal = det.on_run_completion(run_n1)
        self.assertIsNotNone(
            signal,
            "current behaviour: the correction does not cross the run boundary. "
            "If this now returns None, the cross-run extension landed — invert "
            "this assertion rather than deleting the test.",
        )
        self.assertEqual(signal.evidence["tier"], "weak")
        self.assertEqual(signal.evidence["selected_id"], SARAH["id"])

        # The identical retry INSIDE the run that carries the note: silent. This
        # contrast is what makes the limitation legible rather than mysterious.
        in_run = new_state(run_id="run-N")
        add_tool(in_run, "lookup_customer", 0, {"q": "Chen"}, {"customers": THREE_CHENS})
        deny_and_retry(
            in_run,
            "delete_customer",
            {"customer_id": EMILY["id"]},
            {"customer_id": SARAH["id"]},
            step=1,
            note="wrong Chen — it's Sarah, CUST_8834",
        )
        self.assertIsNone(det.on_run_completion(in_run))


# ── Defensive harvesting ──────────────────────────────────────────────────────


class TestMalformedApprovalEvents(unittest.TestCase):
    """payload is Dict[str, Any] with no per-type validation. A malformed
    approval event must yield no note — never an exception inside a detector.
    The worker's rule is that one raising detector must not cost the run its
    other 33, but the cheaper guarantee is not raising in the first place."""

    BAD_PAYLOADS = [
        None,
        "not-a-dict",
        42,
        [],
        {},
        {"note": None},
        {"note": ""},
        {"note": "   "},
        {"note": 42},
        {"note": ["a"]},
        {"note": {"x": 1}},
    ]

    def _state_with(self, payload, step=1):
        state = new_state()
        state.events.append(
            AgentEvent(
                event_type=EventType.APPROVAL_DENIED,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=step,
                payload=payload,
            )
        )
        return state

    def test_harvester_returns_nothing_and_does_not_raise(self):
        for payload in self.BAD_PAYLOADS:
            with self.subTest(payload=payload):
                self.assertEqual(_approval_notes(self._state_with(payload)), [])

    def test_detectors_do_not_raise_on_malformed_events(self):
        for payload in self.BAD_PAYLOADS:
            with self.subTest(payload=payload):
                state = self._state_with(payload)
                add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
                add_tool(state, "delete_customer", 2, {"customer_id": EMILY["id"]})
                UngroundedDestinationDetector().on_run_completion(state)
                UnresolvedAmbiguityDetector(
                    IRREVERSIBLE_TOOLS=["delete_customer"]
                ).on_run_completion(state)

    def test_missing_step_index_does_not_raise(self):
        state = new_state()
        ev = AgentEvent(
            event_type=EventType.APPROVAL_DENIED,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=0,
            payload={"note": "hello there"},
        )
        ev.step_index = None  # a reconstructed row with a null step
        state.events.append(ev)
        self.assertEqual(_approval_notes(state), [(0, "hello there")])

    def test_non_approval_events_are_ignored(self):
        state = new_state()
        state.events.append(
            AgentEvent(
                event_type=EventType.TOOL_CALLED,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=1,
                payload={"note": "this is not an approval note"},
            )
        )
        self.assertEqual(_approval_notes(state), [])


if __name__ == "__main__":
    unittest.main()


# ── End-to-end, through the real runtime ──────────────────────────────────────


class TestRealRuntimeDenyRetry(unittest.TestCase):
    """The whole idiom driven through RunContext, with nothing hand-built.

    Hand-built RunStates got the step arithmetic and the success semantics of a
    blocked call wrong in two different ways, and both times the hand-built tests
    passed while the real flow fired. The fixtures are corrected now, but a
    fixture is a model of the runtime and this is the runtime — it is the test
    that has standing when the two disagree.
    """

    CHENS = [
        {
            "id": "CUST_8834",
            "name": "Sarah Chen",
            "email": "sarah.chen@example.com",
            "plan": "Enterprise",
            "business_name": "Chen Manufacturing GmbH",
        },
        {
            "id": "CUST_1183",
            "name": "Emily Chen",
            "email": "emily.chen@example.com",
            "plan": "Standard",
            "business_name": "Chen Design Studio",
        },
        {
            "id": "CUST_2867",
            "name": "Grace Chen",
            "email": "grace.chen@example.com",
            "plan": "Individual",
            "business_name": "Grace Chen",
        },
    ]

    def _run_deny_then_retry(self, note):
        from unittest.mock import MagicMock

        from dunetrace.client import DunetraceClient
        from dunetrace.policies import ApprovalDenied

        client = DunetraceClient(api_key="dt_test", api_url="http://x", debug=False)
        client._ship = lambda batch: None
        client._create_approval_request = MagicMock(return_value={"id": 7})
        client._get_approval = MagicMock(
            return_value={"id": 7, "status": "denied", "note": note, "decided_by": "alice@corp.com"}
        )

        cm = client.run("support-agent", user_input="close the account for the Chen family")
        run = cm.__enter__()
        run.tool_called("lookup_customer", {"q": "Chen"}, _enforce_approval=False)
        run.tool_responded("lookup_customer", output=json.dumps({"customers": self.CHENS}))

        caught = None
        try:
            run.tool_called(
                "delete_customer", {"customer_id": "CUST_1183"}, _enforce_approval=False
            )
            run.request_approval("delete_customer", {"customer_id": "CUST_1183"})
        except ApprovalDenied as exc:
            caught = exc
            # The documented idiom: read the human's note, replan, retry in-run.
            target = "CUST_8834" if exc.note and "CUST_8834" in exc.note else "CUST_1183"
            run.tool_called("delete_customer", {"customer_id": target}, _enforce_approval=False)
        cm.__exit__(None, None, None)
        client.shutdown(timeout=2)
        return run.state, caught

    def test_corrected_retry_is_silent_and_the_control_fires(self):
        det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=["delete_customer"])

        state, exc = self._run_deny_then_retry("wrong Chen — it's Sarah, CUST_8834")
        self.assertEqual(exc.note, "wrong Chen — it's Sarah, CUST_8834")
        self.assertEqual(exc.decided_by, "alice@corp.com")
        self.assertEqual(exc.status, "denied")
        self.assertIsNone(
            det.on_run_completion(state),
            "the corrected retry must not be flagged by our own detector",
        )

        control, _ = self._run_deny_then_retry(None)
        signal = det.on_run_completion(control)
        self.assertIsNotNone(signal, "without a note the same run must still fire")
        self.assertEqual(signal.evidence["tier"], "weak")

    def test_blocked_attempt_is_the_one_excluded_not_the_retry(self):
        """The exclusion has to claim the RIGHT call. Both the blocked attempt
        and a retry issued from the except block carry success=None and the same
        tool name; the retry additionally shares a step with the denial event."""
        from dunetrace.detectors import _approval_blocked_calls

        state, _ = self._run_deny_then_retry("wrong Chen — it's Sarah, CUST_8834")
        deletes = [
            (i, tc) for i, tc in enumerate(state.tool_calls) if tc.tool_name == "delete_customer"
        ]
        self.assertEqual(len(deletes), 2, "blocked attempt + retry")
        blocked_idx, blocked_tc = deletes[0]
        retry_idx, retry_tc = deletes[1]

        blocked = _approval_blocked_calls(state)
        self.assertIn(blocked_idx, blocked)
        self.assertNotIn(retry_idx, blocked)
        self.assertIn("CUST_1183", blocked_tc.args)
        self.assertIn("CUST_8834", retry_tc.args)

    def test_note_reaches_the_event_stream(self):
        state, _ = self._run_deny_then_retry("wrong Chen — it's Sarah, CUST_8834")
        denied = [e for e in state.events if e.event_type == EventType.APPROVAL_DENIED]
        self.assertEqual(len(denied), 1)
        self.assertEqual(denied[0].payload["note"], "wrong Chen — it's Sarah, CUST_8834")
        self.assertEqual(denied[0].payload["decided_by"], "alice@corp.com")
