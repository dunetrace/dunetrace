"""
The UNRESOLVED_AMBIGUITY explanation's wording is a CORRECTNESS requirement, not
a style preference, so it gets its own test.

The detector is outcome-blind: it cannot know which candidate the principal
intended, and it fires on a lucky-correct guess as readily as on a bad one. Any
sentence asserting the agent acted on the *wrong* record therefore states
something the system does not know. STRONG is the one tier where a mismatch is
provable, and even there the provable claim is about TOKENS — "a token unique to
a sibling appeared in the request" — not about outcome.

The interesting case is a signal whose evidence has been SCRUBBED. Every token
key this detector emits is content lifted out of a tool result, so all of them
are in `CONTENT_EVIDENCE_KEYS` and are stripped once the run's events pass
`EVENT_RETENTION_DAYS`. The row survives (the dashboard's fix-verification
comparison reads it), the tokens do not. A template that reads `absent` as
`empty` then produces two different lies — a bare record-level wrongness claim on
STRONG, and a false statement about the record's tokens on WEAK. Both shipped in
the first draft and are what this file exists to keep out.
"""

from __future__ import annotations

import pytest

from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.templates import TEMPLATES

TEMPLATE = TEMPLATES[FailureType.UNRESOLVED_AMBIGUITY]

# Phrases that assert an outcome the detector cannot know, plus the two
# self-contradictory fallbacks the first draft emitted once evidence was gone.
FORBIDDEN = [
    "wrong record",
    "wrong customer",
    "wrong account",
    "deleted the wrong",
    "incorrect record",
    "should have",
    "actually meant",
    "the correct record",
    "identifies a different record",
    "names a different record",
    "none of those",
]

STRONG_EVIDENCE = {
    "tier": "strong",
    "candidate_count": 3,
    "selected_id": "CUST_1183",
    "discriminators_unused": ["design", "emily", "studio"],
    "sibling_matched_in_request": ["sarah"],
    "sibling_id": "CUST_8834",
    "tool_name": "delete_customer",
    "tool_step": 2,
    "source_tool": "lookup_customer",
    "source_step": 1,
    "warrant_surfaces": ["user_turns"],
    "warrant_turn_count": 1,
}
WEAK_EVIDENCE = {
    k: v
    for k, v in {**STRONG_EVIDENCE, "tier": "weak"}.items()
    if k not in ("sibling_matched_in_request", "sibling_id")
}


def scrubbed(evidence: dict) -> dict:
    """Exactly what ingest_svc's retention pass leaves behind.

    Imports CONTENT_EVIDENCE_KEYS rather than hardcoding the list: if a future
    key is added to the detector and to that tuple, this test starts covering it
    for free — and if a key is added to the detector but NOT to that tuple, the
    ingest suite's TestContentEvidenceKeyCoverage is the one that fails.
    """
    try:
        from ingest_svc.db.postgres import CONTENT_EVIDENCE_KEYS
    except ImportError:  # ingest_svc isn't on the explainer's test PYTHONPATH
        CONTENT_EVIDENCE_KEYS = (
            "selected_id",
            "sibling_id",
            "discriminators_unused",
            "sibling_matched_in_request",
        )
    return {k: v for k, v in evidence.items() if k not in CONTENT_EVIDENCE_KEYS}


def render(evidence: dict, severity: Severity, confidence: float):
    signal = FailureSignal(
        failure_type=FailureType.UNRESOLVED_AMBIGUITY,
        severity=severity,
        run_id="run-1",
        agent_id="support-agent",
        agent_version="v1",
        step_index=2,
        confidence=confidence,
        evidence=evidence,
    )
    return TEMPLATE(signal)


def prose(explanation) -> str:
    parts = [
        explanation.title,
        explanation.what,
        explanation.why_it_matters,
        explanation.evidence_summary,
    ]
    parts += [f.description for f in explanation.suggested_fixes]
    return " ".join(parts).lower()


CASES = [
    ("strong, full evidence", STRONG_EVIDENCE, Severity.HIGH, 0.9),
    ("strong, evidence scrubbed", scrubbed(STRONG_EVIDENCE), Severity.HIGH, 0.9),
    ("weak, full evidence", WEAK_EVIDENCE, Severity.MEDIUM, 0.6),
    ("weak, evidence scrubbed", scrubbed(WEAK_EVIDENCE), Severity.MEDIUM, 0.6),
    (
        "weak, no discriminators at all",
        {**WEAK_EVIDENCE, "discriminators_unused": []},
        Severity.MEDIUM,
        0.6,
    ),
    ("empty evidence", {}, Severity.MEDIUM, 0.6),
]


@pytest.mark.parametrize("label,evidence,severity,confidence", CASES)
def test_never_claims_the_agent_acted_on_the_wrong_record(label, evidence, severity, confidence):
    text = prose(render(evidence, severity, confidence))
    hits = [phrase for phrase in FORBIDDEN if phrase in text]
    assert not hits, f"{label}: explanation asserts {hits}, which the detector cannot know"


@pytest.mark.parametrize("label,evidence,severity,confidence", CASES)
def test_renders_without_raising_and_is_populated(label, evidence, severity, confidence):
    e = render(evidence, severity, confidence)
    assert e.title and e.what and e.why_it_matters and e.evidence_summary
    assert e.suggested_fixes


def test_weak_carries_the_contracted_sentence():
    e = render(WEAK_EVIDENCE, Severity.MEDIUM, 0.6)
    assert "Acted irreversibly on 1 of 3 candidates with no distinguishing token" in e.what


def test_strong_names_both_token_sets():
    e = render(STRONG_EVIDENCE, Severity.HIGH, 0.9)
    assert "The request identifies `sarah`" in e.what
    assert "distinguishing tokens are `design`, `emily` and `studio`" in e.what


def test_weak_distinguishes_scrubbed_from_genuinely_empty():
    """An absent key means the tokens expired; an empty list means the record
    really had none. Conflating them makes the second sentence a falsehood."""
    expired = render(scrubbed(WEAK_EVIDENCE), Severity.MEDIUM, 0.6).what
    none_existed = render({**WEAK_EVIDENCE, "discriminators_unused": []}, Severity.MEDIUM, 0.6).what

    assert "content-retention horizon" in expired
    assert "has no token that its siblings lack" not in expired

    assert "has no token that its siblings lack" in none_existed
    assert "content-retention horizon" not in none_existed


def test_strong_degrades_to_the_mechanism_not_to_a_record_claim():
    what = render(scrubbed(STRONG_EVIDENCE), Severity.HIGH, 0.9).what
    assert "A token unique to one of the other candidates appeared in the request" in what
    assert "content-retention horizon" in what


def test_severity_and_confidence_pass_through():
    strong = render(STRONG_EVIDENCE, Severity.HIGH, 0.9)
    weak = render(WEAK_EVIDENCE, Severity.MEDIUM, 0.6)
    assert (strong.severity, strong.confidence) == ("HIGH", 0.9)
    assert (weak.severity, weak.confidence) == ("MEDIUM", 0.6)
