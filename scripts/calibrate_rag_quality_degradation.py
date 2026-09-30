"""Synthetic calibration; deliberately includes intent changes and missed decline."""

from dunetrace.detectors import RagQualityDegradationDetector
from dunetrace.models import RetrievalResult, RunState

CORPUS = [
    ("steep reformulation decline", [0.9, 0.85, 0.4, 0.3], True),
    ("three calls declining", [0.9, 0.4, 0.3], True),
    ("zero relevance late", [0.8, 0.8, 0, 0], True),
    ("long decline", [0.9, 0.9, 0.8, 0.3, 0.3, 0.2], True),
    ("gradual degradation below default", [0.8, 0.75, 0.65, 0.6], True),
    ("stable high", [0.9, 0.91, 0.9, 0.92], False),
    ("stable low", [0.1, 0.1, 0.1, 0.1], False),
    ("improving", [0.2, 0.3, 0.8, 0.9], False),
    ("alternating queries", [0.9, 0.3, 0.9, 0.3], False),
    ("small drift", [0.9, 0.85, 0.8, 0.78], False),
    ("outlier early then recovery", [0.9, 0.3, 0.5, 0.7], False),
    ("zero scores", [0, 0, 0, 0], False),
    ("intent legitimately changed", [0.9, 0.9, 0.4, 0.4], False),
]


def main():
    detector = RagQualityDegradationDetector()
    tp = fp = fn = tn = 0
    for name, scores, positive in CORPUS:
        state = RunState(
            run_id=name,
            agent_id="synthetic",
            agent_version="test",
            retrievals=[
                RetrievalResult("docs", 4, score, step) for step, score in enumerate(scores)
            ],
        )
        fired = detector.on_run_completion(state) is not None
        tp += fired and positive
        fp += fired and not positive
        fn += not fired and positive
        tn += not fired and not positive
        print(f"{name}: label={positive} fired={fired}")
    print(
        f"TP={tp} FP={fp} FN={fn} TN={tn}; recall={tp / (tp + fn):.1%}; FP rate={fp / (fp + tn):.1%}"
    )
    assert fp / (fp + tn) < 0.15
    assert tp / (tp + fn) >= 0.5


if __name__ == "__main__":
    main()
