"""
UNRESOLVED_AMBIGUITY acceptance suite.

Every case is a hand-constructed RunState run against the real detector — no LLM,
no network, no backend. The candidate records are copied VERBATIM from
dunetrace-demos/demo1/agent/customers.py so the cases exercise the same data the
demo agent actually sees (three Chens, two Rodriguezes, two Alis), rather than a
tidied-up fixture that quietly removes the hard part.

Two record families are synthetic and marked as such below: there is no Sarah
Rodriguez and no Müller in customers.py, and the "two Sarahs" and unicode rows
need them. They are shaped field-for-field like the real ones.

The tables in this file ARE the specification. If the algorithm disagrees with a
row, the row wins — report the discrepancy rather than tuning the rule to fit it.
In particular the nickname / superlative / comparative rows are WEAK on purpose:
from the tokens' point of view those requests genuinely do not resolve to anyone,
and weakening the warrant test to silence them breaks all five misselection rows.
"""

from __future__ import annotations

import json
import time

import pytest

from dunetrace.detectors import UnresolvedAmbiguityDetector
from dunetrace.models import (
    AgentEvent,
    EventType,
    ExternalSignal,
    RunState,
    Severity,
    ToolCall,
)

# ── Records ───────────────────────────────────────────────────────────────────
# Verbatim from dunetrace-demos/demo1/agent/customers.py.

SARAH_CHEN = {
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
EMILY_CHEN = {
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
GRACE_CHEN = {
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
MIGUEL_RODRIGUEZ = {
    "id": "CUST_4521",
    "name": "Miguel Rodriguez",
    "email": "miguel.rodriguez@example.com",
    "signup_date": "2023-11-02",
    "plan": "Enterprise",
    "monthly_value": 9800,
    "business_name": "Rodriguez Logistics SA",
    "annual_value_eur": 210000,
    "customer_since": "2020",
    "tickets_resolved": 31,
    "primary_contact": "Miguel Rodriguez",
}
CARLOS_RODRIGUEZ = {
    "id": "CUST_5507",
    "name": "Carlos Rodriguez",
    "email": "carlos.rodriguez@example.com",
    "signup_date": "2024-04-17",
    "plan": "Standard",
    "monthly_value": 180,
    "business_name": "Rodriguez Trading",
    "annual_value_eur": 2160,
    "customer_since": "2024",
    "tickets_resolved": 2,
    "primary_contact": "Carlos Rodriguez",
}
FATIMA_ALI = {
    "id": "CUST_6675",
    "name": "Fatima Ali",
    "email": "fatima.ali@example.com",
    "signup_date": "2024-02-28",
    "plan": "Standard",
    "monthly_value": 300,
    "business_name": "Ali Consulting",
    "annual_value_eur": 3600,
    "customer_since": "2024",
    "tickets_resolved": 4,
    "primary_contact": "Fatima Ali",
}
AISHA_ALI = {
    "id": "CUST_6120",
    "name": "Aisha Ali",
    "email": "aisha.ali@example.com",
    "signup_date": "2023-07-07",
    "plan": "Individual",
    "monthly_value": 30,
    "business_name": "Aisha Ali",
    "annual_value_eur": 360,
    "customer_since": "2023",
    "tickets_resolved": 2,
    "primary_contact": "Aisha Ali",
}

# SYNTHETIC — not in customers.py. Same shape, same field set. Needed for the
# "two Sarahs" family (the surname, not the given name, is the discriminator)
# and the unicode family (the design partners are German).
SARAH_RODRIGUEZ = {
    "id": "CUST_7043",
    "name": "Sarah Rodriguez",
    "email": "sarah.rodriguez@example.com",
    "signup_date": "2024-01-09",
    "plan": "Standard",
    "monthly_value": 220,
    "business_name": "Rodriguez Trading",
    "annual_value_eur": 2640,
    "customer_since": "2024",
    "tickets_resolved": 2,
    "primary_contact": "Sarah Rodriguez",
}
JUERGEN_MUELLER = {
    "id": "CUST_3311",
    "name": "Jürgen Müller",
    "email": "juergen.mueller@example.de",
    "signup_date": "2022-09-30",
    "plan": "Enterprise",
    "monthly_value": 8100,
    "business_name": "Müller Werkzeugbau GmbH",
    "annual_value_eur": 175000,
    "customer_since": "2018",
    "tickets_resolved": 63,
    "primary_contact": "Jürgen Müller",
}
JOERG_MUELLER = {
    "id": "CUST_3312",
    "name": "Jörg Müller",
    "email": "joerg.mueller@example.de",
    "signup_date": "2023-04-11",
    "plan": "Standard",
    "monthly_value": 190,
    "business_name": "Müller Werkzeugbau GmbH",
    "annual_value_eur": 2280,
    "customer_since": "2023",
    "tickets_resolved": 5,
    "primary_contact": "Jörg Müller",
}

THREE_CHENS = [SARAH_CHEN, EMILY_CHEN, GRACE_CHEN]
TWO_CHENS = [SARAH_CHEN, EMILY_CHEN]
TWO_RODRIGUEZ = [MIGUEL_RODRIGUEZ, CARLOS_RODRIGUEZ]
TWO_ALIS = [FATIMA_ALI, AISHA_ALI]
TWO_SARAHS = [SARAH_CHEN, SARAH_RODRIGUEZ]
TWO_MUELLERS = [JUERGEN_MUELLER, JOERG_MUELLER]

IRREVERSIBLE = ["delete_customer", "cancel_subscription"]


# ── Builders ──────────────────────────────────────────────────────────────────


def new_state(request: str, run_id: str = "run-1") -> RunState:
    return RunState(
        run_id=run_id,
        agent_id="support-agent",
        agent_version="v1",
        input_text=request,
        # Present on purpose. demo1's system prompt names CUST_8834 as an
        # example, and it must NOT warrant a selection — it is byte-identical on
        # every run, so it cannot explain why THIS run chose THIS candidate.
        system_prompt=(
            "You are a support agent. Act decisively. Example: to close an account "
            "call delete_customer with the id, e.g. CUST_8834 for Sarah Chen."
        ),
    )


def add_tool(state: RunState, tool_name: str, step: int, args, output=None) -> ToolCall:
    tc = ToolCall(
        tool_name=tool_name,
        args=args if isinstance(args, str) else json.dumps(args),
        step_index=step,
        timestamp=time.time(),
        success=True,
        output=None
        if output is None
        else (output if isinstance(output, str) else json.dumps(output)),
    )
    if tc.output is not None:
        tc.output_length = len(tc.output)
    state.tool_calls.append(tc)
    state.current_step = max(state.current_step, step)
    return tc


def add_user_turn(state: RunState, step: int, text: str, *, via_events: bool = True) -> None:
    """A later user-authored turn.

    Emitted through the external_signal channel, which is how a mid-run reply
    reaches RunState — there is no dedicated user-message event type. Both views
    are populated by default because they are NOT equivalent across the wire:
    the SDK fills state.external_signals, but build_run_state() (every
    detector-worker run) only ever fills state.events.
    """
    state.external_signals.append(
        ExternalSignal(
            signal_name="user_message",
            step_index=step,
            timestamp=time.time(),
            source="chat",
            meta={"text": text},
        )
    )
    if via_events:
        state.events.append(
            AgentEvent(
                event_type=EventType.EXTERNAL_SIGNAL,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=step,
                payload={"signal_name": "user_message", "source": "chat", "meta": {"text": text}},
            )
        )


def run_case(
    request: str,
    candidates: list,
    selected_id: str,
    *,
    lookup_query: str = "Chen",
    consuming_tool: str = "delete_customer",
    irreversible_tools=None,
    later_turns=(),
    extra_lookups=(),
    also_delete=(),
):
    """Build the canonical shape — lookup, then act — and evaluate it.

    extra_lookups: (query, records) pairs appended AFTER the first lookup, which
    is how the narrowing-lookup case is expressed.
    also_delete: further identifiers deleted in the same run (plural suppression).
    """
    state = new_state(request)
    step = 1
    add_tool(state, "lookup_customer", step, {"name": lookup_query}, candidates)
    for query, records in extra_lookups:
        step += 1
        add_tool(state, "lookup_customer", step, {"name": query}, records)
    for text in later_turns:
        add_user_turn(state, step, text)
    step += 1
    add_tool(state, consuming_tool, step, {"customer_id": selected_id})
    for other in also_delete:
        step += 1
        add_tool(state, consuming_tool, step, {"customer_id": other})

    detector = UnresolvedAmbiguityDetector(
        IRREVERSIBLE_TOOLS=IRREVERSIBLE if irreversible_tools is None else irreversible_tools
    )
    return detector.on_run_completion(state)


def tier_of(signal) -> str:
    return "silent" if signal is None else signal.evidence["tier"]


# ── MISSELECTION — must be STRONG ─────────────────────────────────────────────

MISSELECTION = [
    ("Sarah Chen", THREE_CHENS, EMILY_CHEN["id"], "Chen"),
    ("Miguel Rodriguez", TWO_RODRIGUEZ, CARLOS_RODRIGUEZ["id"], "Rodriguez"),
    ("Fatima Ali", TWO_ALIS, AISHA_ALI["id"], "Ali"),
    ("delete Sarah Chen", TWO_SARAHS, SARAH_RODRIGUEZ["id"], "Sarah"),
    ("delete Sarah", THREE_CHENS, EMILY_CHEN["id"], "Chen"),
]


@pytest.mark.parametrize("request_text,candidates,selected,query", MISSELECTION)
def test_misselection_is_strong(request_text, candidates, selected, query):
    signal = run_case(request_text, candidates, selected, lookup_query=query)
    assert tier_of(signal) == "strong", f"{request_text!r} -> {selected}"
    assert signal.severity == Severity.HIGH
    assert signal.confidence == 0.9
    assert signal.evidence["sibling_matched_in_request"], "STRONG must name the matched tokens"
    assert signal.evidence["candidate_count"] == len(candidates)
    assert signal.evidence["selected_id"] == selected


# ── CORRECT + WARRANTED — must be SILENT ──────────────────────────────────────

WARRANTED = [
    ("Sarah Chen", THREE_CHENS, SARAH_CHEN["id"], "Chen"),
    ("Miguel Rodriguez", TWO_RODRIGUEZ, MIGUEL_RODRIGUEZ["id"], "Rodriguez"),
    ("Chen Design Studio", THREE_CHENS, EMILY_CHEN["id"], "Chen"),
    ("the Enterprise Chen account", THREE_CHENS, SARAH_CHEN["id"], "Chen"),
    ("sarah.chen@example.com", THREE_CHENS, SARAH_CHEN["id"], "Chen"),
    ("delete Sarah", THREE_CHENS, SARAH_CHEN["id"], "Chen"),
    ("delete Sarah Chen", TWO_SARAHS, SARAH_CHEN["id"], "Sarah"),
]


@pytest.mark.parametrize("request_text,candidates,selected,query", WARRANTED)
def test_warranted_selection_is_silent(request_text, candidates, selected, query):
    assert run_case(request_text, candidates, selected, lookup_query=query) is None


# ── GENUINE AMBIGUITY — must be WEAK ──────────────────────────────────────────

AMBIGUOUS = [
    ("the Chen family", THREE_CHENS, EMILY_CHEN["id"], "Chen"),
    # Fires on a lucky-correct guess. This is the point: an agent that guesses
    # right still guessed, and the trace cannot tell the two apart.
    ("the Chen family", THREE_CHENS, SARAH_CHEN["id"], "Chen"),
    ("delete Sarah", TWO_SARAHS, SARAH_CHEN["id"], "Sarah"),
    ("delete Sarah", TWO_SARAHS, SARAH_RODRIGUEZ["id"], "Sarah"),
    ("Mike Rodriguez", TWO_RODRIGUEZ, MIGUEL_RODRIGUEZ["id"], "Rodriguez"),
    ("the big Chen account", THREE_CHENS, SARAH_CHEN["id"], "Chen"),
    ("the older Ali account", TWO_ALIS, AISHA_ALI["id"], "Ali"),
]


@pytest.mark.parametrize("request_text,candidates,selected,query", AMBIGUOUS)
def test_genuine_ambiguity_is_weak(request_text, candidates, selected, query):
    signal = run_case(request_text, candidates, selected, lookup_query=query)
    assert tier_of(signal) == "weak", f"{request_text!r} -> {selected}"
    assert signal.severity == Severity.MEDIUM
    assert signal.confidence == 0.6
    assert "sibling_matched_in_request" not in signal.evidence
    assert signal.evidence["discriminators_unused"], "WEAK must list what would have disambiguated"


# ── MUST NOT FIRE ─────────────────────────────────────────────────────────────


def test_single_candidate_is_silent():
    assert run_case("the Chen family", [EMILY_CHEN], EMILY_CHEN["id"]) is None


def test_narrowing_lookup_anchors_to_the_most_recent_set():
    """Widened, then narrowed to one, then acted. The set that mattered is the
    narrow one, so there was no ambiguity at the moment of the choice."""
    signal = run_case(
        "the Chen family",
        THREE_CHENS,
        SARAH_CHEN["id"],
        extra_lookups=[("sarah.chen@example.com", [SARAH_CHEN])],
    )
    assert signal is None


def test_narrowing_lookup_that_excludes_the_target_falls_back_to_the_wider_set():
    """The anchor is the most recent result CONTAINING the selected id, not
    simply the most recent result — an unrelated lookup in between must not
    launder the choice."""
    signal = run_case(
        "the Chen family",
        THREE_CHENS,
        SARAH_CHEN["id"],
        extra_lookups=[("Ali", TWO_ALIS)],
    )
    assert tier_of(signal) == "weak"
    assert signal.evidence["candidate_count"] == 3


def test_consuming_tool_out_of_scope_is_silent():
    assert (
        run_case(
            "the Chen family",
            THREE_CHENS,
            EMILY_CHEN["id"],
            consuming_tool="preview_account_closure",
        )
        is None
    )


def test_identifier_never_seen_is_silent():
    """TOOL_ARGUMENT_FABRICATION's job. Never double-report it."""
    assert run_case("the Chen family", THREE_CHENS, "CUST_9999") is None


def test_empty_irreversible_tools_is_inert():
    assert run_case("the Chen family", THREE_CHENS, EMILY_CHEN["id"], irreversible_tools=[]) is None


def test_default_detector_is_inert():
    """The shipped default declares nothing irreversible, so it cannot fire."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"name": "Chen"}, THREE_CHENS)
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    assert UnresolvedAmbiguityDetector().on_run_completion(state) is None


def test_plural_suppression():
    """ "close the accounts for the Chen family", both Chens deleted — no
    selection occurred, so there is nothing to warrant."""
    signal = run_case(
        "close the accounts for the Chen family",
        TWO_CHENS,
        SARAH_CHEN["id"],
        also_delete=[EMILY_CHEN["id"]],
    )
    assert signal is None


def test_plural_suppression_does_not_apply_to_a_partial_sweep():
    """Two of three deleted is still a selection — against the third."""
    signal = run_case(
        "close the accounts for the Chen family",
        THREE_CHENS,
        SARAH_CHEN["id"],
        also_delete=[EMILY_CHEN["id"]],
    )
    assert tier_of(signal) == "weak"


# ── Unicode ───────────────────────────────────────────────────────────────────


def test_umlaut_correct_selection_is_silent():
    assert (
        run_case("delete Jürgen", TWO_MUELLERS, JUERGEN_MUELLER["id"], lookup_query="Müller")
        is None
    )


def test_umlaut_misselection_is_strong():
    signal = run_case("Jürgen", TWO_MUELLERS, JOERG_MUELLER["id"], lookup_query="Müller")
    assert tier_of(signal) == "strong"
    assert "jürgen" in signal.evidence["sibling_matched_in_request"]


def test_umlaut_decomposed_form_matches_precomposed():
    """A request typed with a combining diaeresis must resolve the same record as
    one typed precomposed — the two are indistinguishable on screen, and a German
    name entered one way in the request and stored the other way in the CRM would
    otherwise never match.

    Written with an explicit escape rather than a literal: an editor or a
    pre-commit hook that normalises source files would silently turn a literal
    into its precomposed twin and make this test vacuous.
    """
    decomposed = "delete Ju\u0308rgen"  # u + U+0308 COMBINING DIAERESIS
    precomposed = "delete J\u00fcrgen"  # U+00FC LATIN SMALL LETTER U WITH DIAERESIS
    assert decomposed != precomposed
    assert run_case(decomposed, TWO_MUELLERS, JUERGEN_MUELLER["id"], lookup_query="Müller") is None
    assert run_case(precomposed, TWO_MUELLERS, JUERGEN_MUELLER["id"], lookup_query="Müller") is None


def test_casefold_handles_eszett():
    """casefold(), not lower(): "Straße" folds to "strasse" and matches a request
    typed the ASCII way.

    The fixture must not leak an ASCII "strasse" through any other field — an
    email of "emil.strasse@example.de" makes the case pass under lower() too, and
    the test then proves nothing about the fold it exists to cover.
    """
    strasse = {
        "id": "CUST_7777",
        "name": "Emil Straße",
        "email": "emil.b@example.de",
        "plan": "Standard",
        "business_name": "Straße Design",
    }
    other = {
        "id": "CUST_7778",
        "name": "Emil Chen",
        "email": "emil.c@example.de",
        "plan": "Standard",
        "business_name": "Chen Design",
    }
    # lower(), deliberately: casefold() would fold "Straße" itself to "strasse"
    # and flag the very value the test is built on. What must not appear is a
    # literal ASCII "strasse" anywhere in the record.
    assert not any("strasse" in str(v).lower() for v in strasse.values()), (
        "fixture leaks an ASCII 'strasse'; the test would pass under lower() too"
    )

    from dunetrace.detectors import _ua_tokens

    assert "strasse" in _ua_tokens("Straße", 3)
    assert (
        run_case("delete the STRASSE account", [strasse, other], strasse["id"], lookup_query="Emil")
        is None
    )
    # ...and its mismatch twin, so the fold is load-bearing in both directions.
    assert (
        tier_of(
            run_case(
                "delete the STRASSE account", [strasse, other], other["id"], lookup_query="Emil"
            )
        )
        == "strong"
    )


# ── Warrant surface ───────────────────────────────────────────────────────────


def test_later_user_turn_is_warrant():
    """Agent asks "which Chen?", user replies "Emily", agent deletes Emily."""
    assert run_case("the Chen family", THREE_CHENS, EMILY_CHEN["id"], later_turns=["Emily"]) is None


def test_later_user_turn_via_events_only():
    """build_run_state() never repopulates state.external_signals, so the
    server-side path sees the reply only as a raw EXTERNAL_SIGNAL event."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"name": "Chen"}, THREE_CHENS)
    state.events.append(
        AgentEvent(
            event_type=EventType.EXTERNAL_SIGNAL,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=1,
            payload={"signal_name": "user_message", "meta": {"text": "Emily"}},
        )
    )
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None


def test_later_user_turn_naming_a_sibling_is_strong():
    signal = run_case("the Chen family", THREE_CHENS, GRACE_CHEN["id"], later_turns=["Emily"])
    assert tier_of(signal) == "strong"
    assert signal.evidence["sibling_matched_in_request"] == ["emily"]


def test_system_prompt_is_not_warrant():
    """demo1's system prompt names CUST_8834. A lucky-correct guess must still
    fire — a constant cannot explain a per-run choice. A maintainer who "fixes"
    this makes the detector silent on exactly the case it exists for."""
    signal = run_case("the Chen family", THREE_CHENS, SARAH_CHEN["id"])
    assert tier_of(signal) == "weak"
    state = new_state("the Chen family")
    assert "CUST_8834" in state.system_prompt and "Sarah Chen" in state.system_prompt


def test_tool_output_is_not_warrant():
    """The candidate records ARE tool output. If tool output counted as warrant,
    every discriminator would be trivially present and the detector would be
    silent on every run it exists for."""
    signal = run_case("the Chen family", THREE_CHENS, EMILY_CHEN["id"])
    assert tier_of(signal) == "weak"


def test_first_user_turn_surface_ignores_later_replies():
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"name": "Chen"}, THREE_CHENS)
    add_user_turn(state, 1, "Emily")
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(
        IRREVERSIBLE_TOOLS=IRREVERSIBLE, WARRANT_SURFACES=["first_user_turn"]
    )
    signal = det.on_run_completion(state)
    assert tier_of(signal) == "weak"


# ── Token-level discrimination ────────────────────────────────────────────────


def test_token_level_not_whole_value():
    """ "delete Sarah" resolves Sarah Chen even though the whole value
    "sarah chen" is not a substring of the request."""
    assert "sarah chen" not in "delete Sarah".lower()
    assert run_case("delete Sarah", THREE_CHENS, SARAH_CHEN["id"]) is None


def test_identifier_token_is_a_discriminating_request():
    assert run_case("delete CUST_1183", THREE_CHENS, EMILY_CHEN["id"]) is None


def test_possessive_is_stripped():
    assert run_case("close Sarah's account", THREE_CHENS, SARAH_CHEN["id"]) is None


def test_min_token_len_drops_short_tokens():
    """ "Ali" is 3 characters and shared by both Alis; it discriminates nothing
    either way, and the default drops it before it can."""
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    from dunetrace.detectors import _ua_tokens

    assert _ua_tokens("the older Ali account", det.MIN_TOKEN_LEN) == {"older", "account"}


# ── Evidence and cost controls ────────────────────────────────────────────────


def test_evidence_shape():
    signal = run_case("the Chen family", THREE_CHENS, EMILY_CHEN["id"])
    ev = signal.evidence
    assert ev["tier"] == "weak"
    assert ev["candidate_count"] == 3
    assert ev["selected_id"] == "CUST_1183"
    assert ev["tool_name"] == "delete_customer"
    assert ev["source_tool"] == "lookup_customer"
    assert ev["warrant_surfaces"] == ["user_turns"]
    assert ev["warrant_turn_count"] == 1
    # The tokens that would have made the request unambiguous.
    assert {"emily", "design", "studio"} <= set(ev["discriminators_unused"])
    # Tokens all three Chens share must not be offered as disambiguators.
    assert "chen" not in ev["discriminators_unused"]
    assert "example" not in ev["discriminators_unused"]


def test_strong_evidence_names_both_sides():
    signal = run_case("Sarah Chen", THREE_CHENS, EMILY_CHEN["id"])
    ev = signal.evidence
    assert ev["sibling_matched_in_request"] == ["sarah"]
    assert ev["sibling_id"] == "CUST_8834"
    assert ev["selected_id"] == "CUST_1183"
    assert "emily" in ev["discriminators_unused"]


def test_strong_outranks_weak_within_one_run():
    state = new_state("Sarah Chen")
    add_tool(state, "lookup_customer", 1, {"name": "Chen"}, THREE_CHENS)
    add_tool(state, "lookup_customer", 2, {"name": "Ali"}, TWO_ALIS)
    add_tool(state, "delete_customer", 3, {"customer_id": AISHA_ALI["id"]})  # weak
    add_tool(state, "delete_customer", 4, {"customer_id": EMILY_CHEN["id"]})  # strong
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    signal = det.on_run_completion(state)
    assert signal.evidence["tier"] == "strong"
    assert signal.step_index == 4


def test_oversized_candidate_set_fails_open():
    many = [dict(EMILY_CHEN, id=f"CUST_{9000 + i}", name=f"Person{i} Chen") for i in range(60)]
    signal = run_case("the Chen family", many, "CUST_9001")
    assert signal is None


def test_oversized_tool_output_fails_open():
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE, MAX_OUTPUT_CHARS=50)
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"name": "Chen"}, THREE_CHENS)
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    assert det.on_run_completion(state) is None


def test_unparseable_output_is_silent():
    state = new_state("the Chen family")
    add_tool(
        state,
        "lookup_customer",
        1,
        {"name": "Chen"},
        "Sarah Chen (CUST_8834), Emily Chen (CUST_1183)",
    )
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None


def test_python_repr_output_parses():
    """The Python SDK emits str(obj), not JSON."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"name": "Chen"}, str(THREE_CHENS))
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert tier_of(det.on_run_completion(state)) == "weak"


def test_records_nested_under_a_key():
    state = new_state("the Chen family")
    add_tool(
        state,
        "lookup_customer",
        1,
        {"name": "Chen"},
        {"count": 3, "tags": ["crm", "search"], "results": THREE_CHENS},
    )
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    signal = det.on_run_completion(state)
    assert signal.evidence["candidate_count"] == 3, "a list of dicts must beat a list of scalars"


def test_unknown_tunable_raises():
    with pytest.raises(TypeError):
        UnresolvedAmbiguityDetector(IRREVERSABLE_TOOLS=["delete_customer"])


# ── Fixture fidelity ──────────────────────────────────────────────────────────


def test_records_still_match_the_real_customers_py():
    """The records above are copies, and a copy can go stale.

    This repo cannot depend on ../dunetrace-demos, so the acceptance cases carry
    their own copies of customers.py's records. That is a real drift risk: edit
    the demo data and this suite keeps passing against records the demo agent no
    longer returns. When the demos checkout IS present, hold the copies to it.
    Skipped, not failed, when it is absent — CI has no demos checkout.
    """
    import importlib.util
    import os
    import pathlib

    src = (
        pathlib.Path(os.environ.get("DUNETRACE_DEMOS", pathlib.Path.home() / "dunetrace-demos"))
        / "demo1/agent/customers.py"
    )
    if not src.is_file():
        pytest.skip(f"no demos checkout at {src}")

    spec = importlib.util.spec_from_file_location("_demo_customers", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    real = {c["id"]: c for c in mod.CUSTOMERS}

    for copy in (
        SARAH_CHEN,
        EMILY_CHEN,
        GRACE_CHEN,
        MIGUEL_RODRIGUEZ,
        CARLOS_RODRIGUEZ,
        FATIMA_ALI,
        AISHA_ALI,
    ):
        assert copy["id"] in real, f"{copy['id']} no longer exists in customers.py"
        assert copy == real[copy["id"]], (
            f"{copy['id']} has drifted from customers.py — update the copy above, "
            "then re-check every acceptance row that uses it"
        )

    # The synthetic two must stay synthetic: if customers.py ever gains a real
    # Sarah Rodriguez or Müller, these ids would collide and the "two Sarahs"
    # and unicode families would silently stop testing what they claim to.
    for synthetic in (SARAH_RODRIGUEZ, JUERGEN_MUELLER, JOERG_MUELLER):
        assert synthetic["id"] not in real, (
            f"{synthetic['id']} now exists in customers.py — pick a fresh id"
        )

    # The candidate sets are what the demo's own lookup returns, not a
    # hand-picked subset.
    assert mod.lookup_customer_by_name("Chen") == THREE_CHENS
    assert mod.lookup_customer_by_name("Rodriguez") == TWO_RODRIGUEZ
    assert mod.lookup_customer_by_name("Ali") == TWO_ALIS


# ── Regressions ───────────────────────────────────────────────────────────────
#
# Every case below is a defect an adversarial review reproduced against the first
# working draft. They are grouped here rather than scattered because each one is
# a way the detector can be quietly WRONG — either inventing evidence it does not
# have, or discarding a real finding — as opposed to the acceptance rows above,
# which are what it is supposed to do.


def real_lookup_output(query: str, records: list):
    """The shape dunetrace-demos' own `lookup_customer` actually returns.

    A NAME match comes back as `{"customers": [...]}`; an ID or email match comes
    back as a BARE record dict, never a one-element list. The first draft only
    recognised lists, so the narrowing lookup the spec calls out by name walked
    straight past and the run was judged against the older, wider set.
    """
    for r in records:
        if query in (r["id"], r["email"]):
            return dict(r)
    matches = [r for r in records if query.lower() in r["name"].lower()]
    return {"customers": matches} if matches else {}


def test_narrowing_lookup_returning_a_bare_record_object():
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, real_lookup_output("Chen", THREE_CHENS))
    add_tool(
        state,
        "lookup_customer",
        2,
        {"q": "CUST_8834"},
        real_lookup_output("CUST_8834", THREE_CHENS),
    )
    add_tool(state, "delete_customer", 3, {"customer_id": SARAH_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None, (
        "a single record returned as an object is still a one-record set"
    )


def test_narrowing_lookup_by_email_returning_a_bare_record_object():
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, real_lookup_output("Chen", THREE_CHENS))
    add_tool(
        state,
        "lookup_customer",
        2,
        {"q": EMILY_CHEN["email"]},
        real_lookup_output(EMILY_CHEN["email"], THREE_CHENS),
    )
    add_tool(state, "delete_customer", 3, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None


def test_depth_truncated_record_fails_open():
    """A record whose values are nested past MAX_DEPTH is read INCOMPLETELY, so a
    token it really shares with a sibling vanishes and becomes a discriminator of
    the sibling — the detector would manufacture the evidence it is weighing, and
    can reach a HIGH-severity STRONG that way. Abort instead."""
    deep = {
        "id": "CUST_AAA1",
        "name": "Emil Chen",
        "meta": {"a": {"b": {"c": {"d": {"e": {"tier": "platinum"}}}}}},
    }
    flat = {"id": "CUST_BBB2", "name": "Emil Chen", "tier": "platinum"}
    signal = run_case("delete the platinum account", [deep, flat], "CUST_AAA1")
    assert signal is None, "both records are platinum; only the depth limit hid one"


def test_collateral_identifier_argument_does_not_discard_the_finding():
    """`effective_date` here equals a SIBLING's signup_date. Matching a call
    against every identifier-shaped field value made the call look like it named
    two records, and the exactly-one guard then threw the finding away."""
    state = new_state("Sarah Chen")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    add_tool(
        state,
        "cancel_subscription",
        2,
        {"customer_id": EMILY_CHEN["id"], "effective_date": SARAH_CHEN["signup_date"]},
    )
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    signal = det.on_run_completion(state)
    assert tier_of(signal) == "strong"
    assert signal.evidence["selected_id"] == EMILY_CHEN["id"]


def test_tool_acting_on_a_non_id_field_still_matches():
    """The narrow (id-keyed) reading is tried first, but a tool that acts on an
    email must still resolve — the broad reading is the fallback, not the
    default."""
    records = [{"email": c["email"], "name": c["name"], "plan": c["plan"]} for c in THREE_CHENS]
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": records})
    add_tool(state, "delete_customer", 2, {"email": EMILY_CHEN["email"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert tier_of(det.on_run_completion(state)) == "weak"


@pytest.mark.parametrize("facets_first", [False, True])
def test_candidate_set_is_not_decided_by_dict_key_order(facets_first):
    """A result carrying facets/aggregations beside the records must anchor on
    the records. Picking the first list-of-dicts in iteration order let the
    serializer decide, and byte-identical data gave opposite verdicts."""
    facets = [{"field": "plan", "count": 3}, {"field": "city", "count": 2}]
    output = (
        {"facets": facets, "results": THREE_CHENS}
        if facets_first
        else {"results": THREE_CHENS, "facets": facets}
    )
    state = new_state("Sarah Chen")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, output)
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    signal = det.on_run_completion(state)
    assert tier_of(signal) == "strong"
    assert signal.evidence["candidate_count"] == 3


def test_a_reply_after_the_action_is_not_warrant():
    """Warrant has to precede the action. A turn recorded later in the run cannot
    justify a choice the agent had already committed to — and reading the run's
    turns as one undifferentiated bag silences act-first-ask-afterwards, which is
    the behaviour this detector exists to catch."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    add_user_turn(state, 9, "Emily")
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert tier_of(det.on_run_completion(state)) == "weak"


def test_a_reply_before_the_action_is_warrant():
    state = new_state("which Chen?")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    add_user_turn(state, 2, "Emily")
    add_tool(state, "delete_customer", 3, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None


def test_a_rejected_irreversible_call_performed_no_action():
    """`success=False` means the call was refused. Nothing irreversible happened,
    so there is no action whose warrant could be in question — and the signal
    would have said "acted irreversibly" about a call that did not."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    tc = add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    tc.success = False
    tc.error = "403 Forbidden"
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None


def test_unknown_success_still_evaluates():
    """Only an explicit False is a non-action. Most instrumentation never sets
    `success`, and treating None as failed would make the detector inert on the
    majority of real traffic."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    tc = add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    tc.success = None
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert tier_of(det.on_run_completion(state)) == "weak"


def test_candidate_cap_is_per_run_not_per_acting_call():
    """The same cached candidate set was charged against the cap once per
    irreversible call, so the documented per-run cap of 50 became 50/N and runs
    well under it went silent."""
    many = [
        {
            "id": f"CUST_{9000 + i}",
            "name": "Sarah Chen" if i == 0 else f"Person{i} Chen",
            "email": f"person{i}.chen@example.com",
            "plan": "Standard",
        }
        for i in range(30)
    ]
    many[0]["email"] = "sarah.chen@example.com"
    for deletes in (1, 2, 3):
        state = new_state("Sarah Chen")
        add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": many})
        for j in range(deletes):
            add_tool(state, "delete_customer", 2 + j, {"customer_id": f"CUST_{9005 + j}"})
        det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
        assert tier_of(det.on_run_completion(state)) == "strong", f"{deletes} delete(s)"


def test_scan_budget_is_polled_inside_the_anchor_walk():
    """All the expensive work — the reverse walk, parsing every prior output —
    happens after the per-acting-call budget check. Polling only there let the
    in-path instance run orders of magnitude over its 1ms budget."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    for i in range(2, 402):
        add_tool(state, "search_notes", i, {"q": "x"}, {"note": "y" * 8000})
    add_tool(state, "delete_customer", 500, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(
        IRREVERSIBLE_TOOLS=IRREVERSIBLE, MAX_SCAN_NS=1_000_000, MAX_OUTPUT_CHARS=20_000
    )
    started = time.perf_counter_ns()
    signal = det.on_run_completion(state)
    elapsed_ns = time.perf_counter_ns() - started
    assert signal is None, "must fail open, not report on a partial scan"
    assert elapsed_ns < 20_000_000, f"aborted after {elapsed_ns / 1e6:.1f}ms — budget is 1ms"


def test_min_candidates_guard_is_load_bearing():
    """`test_single_candidate_is_silent` is over-determined: with one record and
    one delete, plural suppression silences the run independently, so it does not
    actually exercise the MIN_CANDIDATES guard or the operator-facing
    `min_candidates` knob. Raise the knob above the set size and act on only one
    of the candidates, and the guard is the only thing that can be doing it."""
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE, MIN_CANDIDATES=3)
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": TWO_CHENS})
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    assert det.on_run_completion(state) is None, "2 < min_candidates=3"

    # The same run with the default knob does fire, so the set-up is otherwise
    # sound and the guard is what silenced it above.
    default = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert tier_of(default.on_run_completion(state)) == "weak"


def test_warrant_reaches_the_sdk_in_path_channel():
    """In-process, the SDK fills `state.external_signals`; server-side,
    build_run_state() fills only `state.events`. Both have to work, and a test
    that always populates both proves neither."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    add_user_turn(state, 1, "Emily", via_events=False)
    assert not state.events, "this case must exercise the typed view alone"
    add_tool(state, "delete_customer", 2, {"customer_id": EMILY_CHEN["id"]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None


def test_a_call_naming_two_records_is_not_a_single_selection():
    """The algorithm requires the arguments to use EXACTLY ONE record's
    identifier. A call carrying two of them is a bulk action, not a choice
    between siblings, and there is no single selection to weigh."""
    state = new_state("the Chen family")
    add_tool(state, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    add_tool(state, "delete_customer", 2, {"customer_ids": [EMILY_CHEN["id"], GRACE_CHEN["id"]]})
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE)
    assert det.on_run_completion(state) is None

    # One identifier in the same shape does select, so the list is not what
    # silenced it above.
    single = new_state("the Chen family")
    add_tool(single, "lookup_customer", 1, {"q": "Chen"}, {"customers": THREE_CHENS})
    add_tool(single, "delete_customer", 2, {"customer_ids": [EMILY_CHEN["id"]]})
    assert tier_of(det.on_run_completion(single)) == "weak"
