"""
services/explainer/explainer_svc/explainer.py

Public API for the explain layer.

    from explainer_svc.explainer import explain

    explanation = explain(signal)
    print(explanation.title)
    print(explanation.as_slack_text())

The explain layer is intentionally stateless and dependency-free.
It can be imported directly by the detector worker, the REST API,
or any future alert handler — no network calls, no DB, no LLM.
"""
from __future__ import annotations

import logging

from dunetrace.models import FailureSignal, FailureType
from explainer_svc.models import CodeFix, Explanation
from explainer_svc.templates import TEMPLATES

logger = logging.getLogger("dunetrace.explainer")


def explain(signal: FailureSignal) -> Explanation:
    """
    Produce a human-readable Explanation from a FailureSignal.

    Returns a fallback explanation for unknown failure types
    rather than raising — the caller should never crash on explain().
    """
    template = TEMPLATES.get(signal.failure_type)

    if template is None:
        logger.warning("No template for failure_type=%s — using fallback",
                       signal.failure_type)
        return _fallback(signal)

    try:
        return template(signal)
    except Exception as exc:
        logger.error("Template failed for %s: %s — using fallback",
                     signal.failure_type, exc)
        return _fallback(signal)


def _fallback(signal: FailureSignal) -> Explanation:
    """
    Generic explanation for failure types without a template.
    Used for Tier 2 / Tier 3 types not yet implemented,
    and as a safety net if a template raises.
    """
    return Explanation(
        failure_type=signal.failure_type.value,
        severity=signal.severity.value,
        run_id=signal.run_id,
        agent_id=signal.agent_id,
        agent_version=signal.agent_version,
        confidence=signal.confidence,
        step_index=signal.step_index,
        detected_at=signal.detected_at,
        evidence=signal.evidence,
        title=f"{signal.failure_type.value.replace('_', ' ').title()} detected",
        what=(
            f"The detector identified a {signal.failure_type.value} condition "
            f"in run `{signal.run_id}` at step {signal.step_index}."
        ),
        why_it_matters=(
            "This condition may indicate the agent is not behaving as intended. "
            "Review the run events for more context."
        ),
        evidence_summary=str(signal.evidence),
        suggested_fixes=[
            CodeFix(
                description="Review the run events in the Dunetrace dashboard",
                language="text",
                code=f"Run ID: {signal.run_id}",
            )
        ],
    )
