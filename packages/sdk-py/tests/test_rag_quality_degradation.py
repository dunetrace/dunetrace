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
        [0.8, 0.75, 0.85],
        [0.2, 0.3, 0.8],
        [0, 0, 0],
        [None, 0.9, 0.1],
        [float("nan"), 0.9, 0.1],
        [float("inf"), 0.9, 0.1],
        [-1, 0.9, 0.1],
        [True, 0.9, 0.1],
        [False, 0.9, 0.1],
        ["bad", 0.9, 0.1],
        [{}, 0.9, 0.1],
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


def test_low_scores_can_signal_despite_a_small_absolute_gap():
    signal = RagQualityDegradationDetector().on_run_completion(state([0.15, 0.10, 0.05]))
    assert signal is not None
    assert signal.evidence["early_avg_score"] == pytest.approx(0.15)
    assert signal.evidence["late_avg_score"] == pytest.approx(0.075)
    assert signal.evidence["drop_pct"] == pytest.approx(50)


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


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "3", None])
def test_invalid_minimum_rejected(value):
    with pytest.raises(ValueError, match="MIN_RETRIEVALS"):
        RagQualityDegradationDetector(MIN_RETRIEVALS=value)


@pytest.mark.parametrize("value", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.3", None])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError, match="DROP_THRESHOLD"):
        RagQualityDegradationDetector(DROP_THRESHOLD=value)


def test_mixed_invalid_scores_do_not_change_evidence():
    signal = RagQualityDegradationDetector().on_run_completion(
        state([True, "bad", 1.0, None, 0.5, False, 0.5])
    )
    assert signal.evidence["retrieval_count"] == 3
    assert signal.evidence["drop_pct"] == pytest.approx(50)
