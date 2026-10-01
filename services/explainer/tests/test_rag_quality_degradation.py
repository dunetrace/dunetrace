"""Trend explanations include evidence and tolerate historical partial rows."""

from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.templates import explain_rag_quality_degradation


def test_explanation_uses_evidence_and_supports_missing_fields():
    for evidence in (
        {},
        {
            "early_avg_score": 0.9,
            "late_avg_score": 0.3,
            "drop_pct": 66.7,
            "retrieval_count": 4,
            "index_name": "docs",
        },
    ):
        signal = FailureSignal(
            failure_type=FailureType.RAG_QUALITY_DEGRADATION,
            severity=Severity.MEDIUM,
            run_id="run",
            agent_id="agent",
            agent_version="test",
            step_index=3,
            confidence=0.8,
            evidence=evidence,
        )
        explanation = explain_rag_quality_degradation(signal)
        assert explanation.title
        assert explanation.suggested_fixes
        if evidence:
            assert "0.9" in explanation.what
            assert "0.3" in explanation.what
            assert "66.7" in explanation.evidence_summary
