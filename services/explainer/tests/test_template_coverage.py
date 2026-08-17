"""
Coverage and robustness for the explanation templates.

Two failure modes this guards against, both silent:

1. **A missing template.** `explain()` falls back to a generic paragraph, so the
   signal still renders — just with no real content and a `No template for
   failure_type=...` line in the logs. Nothing breaks, so nobody notices.

2. **A template that raises.** `explain()` catches the exception and uses the same
   fallback. A template reading an evidence key the detector never emits, or doing
   arithmetic on a string default, therefore degrades to generic text rather than
   failing a test. Templates here are called *directly* so exceptions propagate.

The evidence fixtures below mirror what each detector actually puts in
`FailureSignal.evidence` — they were read off the detector classes in
`packages/sdk-py/dunetrace/detectors.py`, not invented. A template that stops
matching its detector's evidence shape fails here.

Run:
    cd services/explainer
    python -m unittest tests.test_template_coverage -v
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.templates import TEMPLATES

# Failure types with no producer anywhere in the codebase (reserved enum members)
# plus the CUSTOM sentinel, which must keep using the fallback: it stands in for
# open-set types whose real name lives in evidence["raw_failure_type"], so a
# template keyed on CUSTOM would render the sentinel instead of the real detector.
NO_TEMPLATE_EXPECTED = {
    FailureType.USER_DISSATISFACTION,
    FailureType.INTENT_MISALIGNMENT,
    FailureType.CONFIDENT_HALLUCINATION,
    FailureType.POLICY_VIOLATION,
    FailureType.CUSTOM,
}

# Evidence as emitted by each detector.
REAL_EVIDENCE = {
    FailureType.OVERSIZED_TOOL_ARGUMENTS: {
        "step_index": 4,
        "tool_name": "summarise_document",
        "arg_length": 42_318,
        "threshold": 10_000,
    },
    FailureType.EXCESSIVE_RETRIEVAL: {
        "retrieval_count": 9,
        "threshold": 5,
        "indexes": ["kb-main", "kb-faq"],
        "first_step": 3,
        "last_step": 14,
    },
    FailureType.SILENT_TRUNCATION: {
        "truncated_step": 6,
        "finish_reason": "length",
        "output_length": 4096,
        "model": "gpt-4o-mini",
        "recovered": False,
        "was_final_output": True,
        "subsequent_tool_steps": [7, 8],
    },
    FailureType.PREMATURE_TERMINATION: {
        "failed_tool": "charge_card",
        "failed_tool_step": 4,
        "tool_error": "gateway timeout",
        "matched_error_marker": "timeout",
        "failure_source": "tool_result",
        "claim_step": 5,
        "matched_completion_term": "all set",
        "is_final_message": True,
        "output_snippet": "All set — the payment went through.",
    },
    FailureType.UNREAD_TOOL_ERROR: {
        "failed_tool": "fetch_invoice",
        "failed_tool_step": 2,
        "tool_error": "404 not found",
        "matched_error_marker": "not found",
        "failure_source": "tool_result",
        "next_action_step": 3,
        "next_action_type": "llm.called",
        "unread_count": 2,
    },
    FailureType.TOOL_ARGUMENT_FABRICATION: {
        "tool_name": "delete_account",
        "tool_step": 7,
        "fabricated_entity": "acct_99213",
        "is_destructive_tool": True,
        "args_snippet": '{"account_id": "acct_99213"}',
    },
    FailureType.RETRIEVED_CONTENT_INJECTION: {
        "source_type": "retrieved_document",
        "source_name": "policy.pdf",
        "source_step": 3,
        "matched_marker": "ignore previous instructions",
        "behavior_deviation": "began emitting credentials",
        "content_snippet": "Ignore previous instructions and ...",
    },
    FailureType.HANDOFF_CONTEXT_LOSS: {
        "parent_context_length": 8400,
        "child_input_length": 620,
        "size_drop_ratio": 0.93,
        "missing_entities": ["order_1182", "2026-07-01"],
        "missing_entity_count": 2,
    },
    FailureType.AGENT_HANDOFF_FAILURE: {
        "tool_name": "delegate_to_researcher",
        "step_index": 5,
        "output_length": 0,
        "success": True,
        "reason": "empty response",
        "known_empty_response": True,
        "min_output_length": 40,
    },
    FailureType.RUNAWAY_ITERATION: {
        "step_count": 64,
        "step_threshold": 30,
        "step_exceeded": True,
        "estimated_cost_usd": 4.82,
        "cost_threshold_usd": 2.0,
        "cost_exceeded": True,
        "recent_messages_checked": 10,
    },
    FailureType.MODEL_FALLBACK_DRIFT: {
        "from_model": "gpt-4o",
        "to_model": "gpt-4o-mini",
        "from_tier": "frontier",
        "to_tier": "small",
        "tier_delta": 2,
        "downgrade_step": 9,
        "preceded_by_rate_limit": True,
    },
    FailureType.MEMORY_POISONING: {
        "memory_key": "user_prefs",
        "source": "retrieved_document",
        "matched_marker": "you are now",
        "untrusted_source": True,
        "consumed": True,
        "write_step": 4,
        "poisoned_write_count": 2,
        "value_snippet": "You are now in developer mode ...",
    },
    FailureType.DELEGATION_LOOP: {
        "cycle": ["planner", "executor"],
        "cycle_agents": ["planner", "executor"],
        "cycle_length": 2,
        "loop_run_count": 6,
        "delegation_chain": ["planner", "executor", "planner", "executor"],
        "min_loop_runs": 3,
    },
}


def _signal(ft: FailureType, evidence: dict) -> FailureSignal:
    return FailureSignal(
        failure_type=ft,
        severity=Severity.HIGH,
        run_id="run-abc",
        agent_id="agent-1",
        agent_version="v1",
        step_index=5,
        confidence=0.9,
        evidence=evidence,
        detected_at=0.0,
    )


class TestTemplateCoverage(unittest.TestCase):
    def test_every_producible_failure_type_has_a_template(self):
        missing = [
            f.name for f in FailureType if f not in TEMPLATES and f not in NO_TEMPLATE_EXPECTED
        ]
        self.assertEqual(missing, [], f"failure types with no explanation template: {missing}")

    def test_reserved_types_deliberately_have_no_template(self):
        """If one of these gains a producer, it needs a template — and this test
        is the reminder to write one rather than ship generic text."""
        for ft in NO_TEMPLATE_EXPECTED:
            with self.subTest(ft=ft.name):
                self.assertNotIn(ft, TEMPLATES)


class TestTemplatesRunOnRealEvidence(unittest.TestCase):
    """Called directly, not through explain(), so a raise fails the test instead of
    silently degrading to the fallback."""

    def test_real_evidence_produces_populated_content(self):
        for ft, evidence in REAL_EVIDENCE.items():
            with self.subTest(ft=ft.name):
                exp = TEMPLATES[ft](_signal(ft, evidence))
                self.assertTrue(exp.title.strip(), "title empty")
                self.assertTrue(exp.what.strip(), "what empty")
                self.assertTrue(exp.why_it_matters.strip(), "why_it_matters empty")
                self.assertTrue(exp.evidence_summary.strip(), "evidence_summary empty")
                self.assertTrue(exp.suggested_fixes, "no suggested fixes")

    def test_evidence_values_reach_the_output(self):
        """A template that ignores its evidence renders generic prose that reads
        fine but tells the on-call engineer nothing."""
        expectations = {
            FailureType.PREMATURE_TERMINATION: "charge_card",
            FailureType.UNREAD_TOOL_ERROR: "fetch_invoice",
            FailureType.TOOL_ARGUMENT_FABRICATION: "acct_99213",
            FailureType.RETRIEVED_CONTENT_INJECTION: "policy.pdf",
            FailureType.MODEL_FALLBACK_DRIFT: "gpt-4o-mini",
            FailureType.MEMORY_POISONING: "user_prefs",
            FailureType.DELEGATION_LOOP: "planner",
            FailureType.AGENT_HANDOFF_FAILURE: "delegate_to_researcher",
            FailureType.EXCESSIVE_RETRIEVAL: "kb-main",
            FailureType.SILENT_TRUNCATION: "gpt-4o-mini",
            FailureType.OVERSIZED_TOOL_ARGUMENTS: "summarise_document",
        }
        for ft, needle in expectations.items():
            with self.subTest(ft=ft.name):
                exp = TEMPLATES[ft](_signal(ft, REAL_EVIDENCE[ft]))
                haystack = f"{exp.title} {exp.what} {exp.evidence_summary}"
                self.assertIn(needle, haystack)

    def test_no_unrendered_placeholders(self):
        # Matches a bare `{identifier}` — an f-string that lost its `f` prefix.
        # Deliberately not a blanket `{` check: evidence snippets legitimately
        # contain JSON (`args_snippet`), and flagging those is a false positive.
        placeholder = re.compile(r"\{[a-z_][a-z0-9_]*\}")
        for ft, evidence in REAL_EVIDENCE.items():
            with self.subTest(ft=ft.name):
                exp = TEMPLATES[ft](_signal(ft, evidence))
                blob = f"{exp.title} {exp.what} {exp.why_it_matters} {exp.evidence_summary}"
                self.assertIsNone(placeholder.search(blob), f"unrendered placeholder in {ft.name}")
                self.assertNotIn("None", blob, "a None leaked into the prose")


class TestTemplatesSurviveMissingEvidence(unittest.TestCase):
    """Evidence keys are not guaranteed: signals predating a detector change, rows
    written by an older build, and shadow detectors still in flux all arrive with
    partial evidence. A template must degrade, not raise."""

    def test_empty_evidence_does_not_raise(self):
        for ft, template in TEMPLATES.items():
            with self.subTest(ft=ft.name):
                exp = template(_signal(ft, {}))
                self.assertTrue(exp.title.strip())
                self.assertTrue(exp.what.strip())

    def test_partial_evidence_does_not_raise(self):
        """Half the keys present — the shape most likely to hit a template that
        assumes two values arrive together."""
        for ft, evidence in REAL_EVIDENCE.items():
            keys = sorted(evidence)
            half = {k: evidence[k] for k in keys[: len(keys) // 2]}
            with self.subTest(ft=ft.name):
                TEMPLATES[ft](_signal(ft, half))

    def test_none_valued_evidence_does_not_raise(self):
        """Explicit nulls, as JSONB round-trips them."""
        for ft, evidence in REAL_EVIDENCE.items():
            nulled = {k: None for k in evidence}
            with self.subTest(ft=ft.name):
                TEMPLATES[ft](_signal(ft, nulled))


if __name__ == "__main__":
    unittest.main()
