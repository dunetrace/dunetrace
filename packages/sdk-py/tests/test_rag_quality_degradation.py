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


@pytest.mark.parametrize("scores", [[1.0, 0.7, 0.7], [1.0, 1.0, 0.7, 0.7]])
def test_exactly_thirty_percent_does_not_fire_at_the_default(scores):
    """(1.0 - 0.7) / 1.0 is 0.30000000000000004 in floating point."""
    assert RagQualityDegradationDetector().on_run_completion(state(scores)) is None


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


@pytest.mark.parametrize("value", [True, False, 0, 1, 2, -1, 1.5, "3", None])
def test_invalid_minimum_rejected(value):
    with pytest.raises(ValueError, match="MIN_RETRIEVALS"):
        RagQualityDegradationDetector(MIN_RETRIEVALS=value)


@pytest.mark.parametrize(
    "value", [True, False, 0, -0.1, 1.1, float("nan"), float("inf"), "0.3", None]
)
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError, match="DROP_THRESHOLD"):
        RagQualityDegradationDetector(DROP_THRESHOLD=value)


def test_mixed_invalid_scores_do_not_change_evidence():
    signal = RagQualityDegradationDetector().on_run_completion(
        state([True, "bad", 1.0, None, 0.5, False, 0.5])
    )
    assert signal.evidence["retrieval_count"] == 3
    assert signal.evidence["drop_pct"] == pytest.approx(50)


def test_empty_retrievals_are_not_a_decline():
    """result_count 0 with top_score 0.0 is RAG_EMPTY_RETRIEVAL's signal."""
    run = state([0.8, 0.8])
    run.retrievals.append(RetrievalResult("docs", 0, 0.0, 5))
    assert RagQualityDegradationDetector().on_run_completion(run) is None


class _NumpyLike:
    def __init__(self, v):
        self.v = v

    def __float__(self):
        return float(self.v)


def test_numeric_scores_that_are_not_python_floats_count():
    run = state([_NumpyLike(0.9), "0.9", 0.3, 0.3])
    signal = RagQualityDegradationDetector().on_run_completion(run)
    assert signal is not None
    assert signal.evidence["retrieval_count"] == 4


def test_unhashable_index_name_does_not_raise():
    run = state([0.9, 0.9, 0.2, 0.2], index=["kb"])
    signal = RagQualityDegradationDetector().on_run_completion(run)
    assert signal is not None
    assert signal.evidence["index_name"] == "unknown"


def test_evidence_is_rounded_for_display():
    signal = RagQualityDegradationDetector().on_run_completion(
        state([0.9, 0.9, 0.8, 0.3, 0.3, 0.2])
    )
    assert signal.evidence["early_avg_score"] == 0.867
    assert signal.evidence["late_avg_score"] == 0.267
    assert signal.evidence["drop_pct"] == 69.2


def _events(*retrievals):
    out = []
    for step, (called_index, responded_index, score) in enumerate(retrievals, start=1):
        called = {"query": "q"}
        if called_index is not None:
            called["index_name"] = called_index
        responded = {"result_count": 3, "top_score": score}
        if responded_index is not None:
            responded["index_name"] = responded_index
        for et, payload in (("retrieval.called", called), ("retrieval.responded", responded)):
            out.append(
                {
                    "event_type": et,
                    "run_id": "r",
                    "agent_id": "a",
                    "agent_version": "v",
                    "step_index": step,
                    "timestamp": float(step),
                    "payload": payload,
                }
            )
    return out


def test_run_builder_carries_the_index_from_retrieval_called():
    """LangChain names the retriever on retrieval.called only. Without carrying it,
    every auto-instrumented retriever landed in one "unknown" group and two
    retrievers on different score scales read as a decline."""
    from dunetrace.run_builder import build_run_state

    st = build_run_state(
        _events(
            ("cosine", None, 0.8), ("cosine", None, 0.8), ("bm25", None, 0.1), ("bm25", None, 0.1)
        )
    )
    assert [r.index_name for r in st.retrievals] == ["cosine", "cosine", "bm25", "bm25"]
    assert RagQualityDegradationDetector().on_run_completion(st) is None


def test_run_builder_stringifies_a_non_string_index():
    from dunetrace.run_builder import build_run_state

    st = build_run_state(_events((None, ["kb"], 0.8)))
    assert st.retrievals[0].index_name == "['kb']"
