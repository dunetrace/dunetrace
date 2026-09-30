"""Offline trend detection, evidence and registration checks."""

import pytest

from dunetrace.detectors import DETECTOR_KEYS, TIER1_DETECTORS, RagQualityDegradationDetector
from dunetrace.models import FailureType, RetrievalResult, RunState


def state(scores, index="docs"):
    return RunState(
        run_id="rag-run",
        agent_id="rag-agent",
        agent_version="test",
        retrievals=[RetrievalResult(index, 4, score, step) for step, score in enumerate(scores)],
    )


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [0.9],
        [0.9, 0.1],
        [0.8] * 4,
        [0.2, 0.3, 0.8],
        [0, 0, 0],
        [None, 0.9, 0.1],
        [float("nan"), 0.9, 0.1],
        [float("inf"), 0.9, 0.1],
        [-1, 0.9, 0.1],
    ],
)
def test_no_signal_without_meaningful_scored_decline(scores):
    assert RagQualityDegradationDetector().on_run_completion(state(scores)) is None


def test_decline_evidence_and_order():
    run = state([1.0, 1.0, 0.4, 0.4])
    run.retrievals.reverse()
    signal = RagQualityDegradationDetector().on_run_completion(run)
    assert signal.failure_type == FailureType.RAG_QUALITY_DEGRADATION
    assert signal.step_index == 3
    assert signal.evidence["early_avg_score"] == 1.0
    assert signal.evidence["late_avg_score"] == 0.4
    assert signal.evidence["drop_pct"] == pytest.approx(60)
    assert signal.evidence["retrieval_count"] == 4


def test_odd_count_and_tunable():
    run = state([1.0, 0.5, 0.5])
    assert RagQualityDegradationDetector().on_run_completion(run) is not None
    assert RagQualityDegradationDetector(MIN_RETRIEVALS=4).on_run_completion(run) is None
    assert RagQualityDegradationDetector(DROP_THRESHOLD=0.6).on_run_completion(run) is None


def test_exact_threshold_does_not_fire():
    assert (
        RagQualityDegradationDetector(DROP_THRESHOLD=0.5).on_run_completion(state([1, 1, 0.5, 0.5]))
        is None
    )


def test_scores_from_different_indexes_are_not_compared():
    run = state([1.0, 1.0])
    run.retrievals += [RetrievalResult("other", 4, 0.1, step) for step in (2, 3)]
    assert RagQualityDegradationDetector().on_run_completion(run) is None


def test_confidence_increases_with_decline():
    detector = RagQualityDegradationDetector()
    mild = detector.on_run_completion(state([1, 1, 0.6, 0.6]))
    severe = detector.on_run_completion(state([1, 1, 0, 0]))
    assert 0 <= mild.confidence < severe.confidence <= 1


def test_registered_for_default_and_configured_execution():
    assert DETECTOR_KEYS["rag_quality_degradation"] is RagQualityDegradationDetector
    assert any(isinstance(detector, RagQualityDegradationDetector) for detector in TIER1_DETECTORS)
