"""
Reconstructs a ConversationEvaluationInput from a list of runs' raw event
lists — one list of event dicts per sibling run, same flat shape
run_extract.py already consumes. Reuses build_evaluation_input() per run
rather than re-implementing input_text/actual_output extraction.
"""

from __future__ import annotations

from semantic_svc.evaluators.base import ConversationEvaluationInput, ConversationTurn
from semantic_svc.evaluators.run_extract import build_evaluation_input


def build_conversation_evaluation_input(
    runs: list[tuple[str, list[dict]]],
) -> ConversationEvaluationInput | None:
    """runs: [(run_id, events), ...] ordered oldest -> newest.

    Runs with no extractable input_text/actual_output are skipped (same
    "nothing meaningful to score" case build_evaluation_input already
    handles for a single run) rather than aborting the whole conversation —
    a conversation with a few thin/incomplete turns can still be evaluated
    on the runs that do have content. Returns None only when NO run in the
    window has anything usable.
    """
    turns: list[ConversationTurn] = []
    for run_id, events in runs:
        extracted = build_evaluation_input(events)
        if extracted is None:
            continue
        turns.append(
            ConversationTurn(
                run_id=run_id,
                input_text=extracted.input_text,
                actual_output=extracted.actual_output,
            )
        )

    if not turns:
        return None

    return ConversationEvaluationInput(turns=turns)
