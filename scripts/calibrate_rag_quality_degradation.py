"""Synthetic boundary checks and seeded benign-distribution sensitivity checks."""

import argparse
import random
from pathlib import Path

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
    ("three noisy flat high scores", [0.8, 0.75, 0.85], False),
    ("three benign mixed queries", [0.9, 0.3, 0.8], False),
    ("improving", [0.2, 0.3, 0.8, 0.9], False),
    ("alternating queries", [0.9, 0.3, 0.9, 0.3], False),
    ("small drift", [0.9, 0.85, 0.8, 0.78], False),
    ("outlier early then recovery", [0.9, 0.3, 0.5, 0.7], False),
    ("zero scores", [0, 0, 0, 0], False),
    ("intent legitimately changed", [0.9, 0.9, 0.4, 0.4], False),
]


def fires(detector, name, scores):
    state = RunState(
        run_id=name,
        agent_id="synthetic",
        agent_version="test",
        retrievals=[RetrievalResult("docs", 4, score, step) for step, score in enumerate(scores)],
    )
    return detector.on_run_completion(state) is not None


def sensitivity(detector, trials, seed):
    rng = random.Random(seed)
    distributions = {
        "uniform(0,1)": lambda: rng.random(),
        "Beta(8,2)": lambda: rng.betavariate(8, 2),
        "Beta(2,8)": lambda: rng.betavariate(2, 8),
        "Beta(9,1)": lambda: rng.betavariate(9, 1),
        "70/30 easy/hard query mix": lambda: (
            rng.betavariate(8, 2) if rng.random() < 0.7 else rng.betavariate(2, 8)
        ),
    }
    rows = ["| Benign IID score model | n=3 | n=4 | n=6 | n=10 |", "|---|---:|---:|---:|---:|"]
    for name, draw in distributions.items():
        rates = []
        for count in (3, 4, 6, 10):
            positives = sum(
                fires(detector, "iid-benign", [draw() for _ in range(count)]) for _ in range(trials)
            )
            rates.append(f"{positives / trials:.1%}")
        rows.append(f"| {name} | {' | '.join(rates)} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    detector = RagQualityDegradationDetector()
    tp = fp = fn = tn = 0
    for name, scores, positive in CORPUS:
        fired = fires(detector, name, scores)
        tp += fired and positive
        fp += fired and not positive
        fn += not fired and positive
        tn += not fired and not positive
        print(f"{name}: label={positive} fired={fired}")
    print(
        f"TP={tp} FP={fp} FN={fn} TN={tn}; recall={tp / (tp + fn):.1%}; FP rate={fp / (fp + tn):.1%}"
    )
    table = sensitivity(detector, args.trials, args.seed)
    print(table)
    if args.report:
        args.report.write_text(
            "# RAG quality degradation: synthetic sensitivity assessment\n\n"
            "Run `PYTHONPATH=packages/sdk-py python scripts/calibrate_rag_quality_degradation.py "
            "--report scripts/calibration/rag_quality_degradation_calibration.md`.\n"
            "No network, models, or infrastructure are required.\n\n"
            f"## Fixed boundary corpus\n\n{len(CORPUS)} hand-labeled traces: "
            f"TP {tp}, FP {fp}, FN {fn}, TN {tn}; recall {tp / (tp + fn):.1%}, "
            f"false-positive rate {fp / (fp + tn):.1%}, precision {tp / (tp + fp):.1%}.\n\n"
            "The labels describe intended query scenarios, not information available to the detector. "
            "Both a legitimate intent change and a three-call benign mixed-query trace fire. "
            "A noisy but flat three-call high-score trace does not. A gradual decline below "
            "the default threshold is missed. These fixed hand-assigned scores are boundary checks, "
            "not a representative scorer population or production precision estimate.\n\n"
            f"## Stationary benign-distribution stress test\n\nSeed {args.seed}; "
            f"{args.trials} independent traces per cell; default threshold 0.3. "
            "Each cell invokes the actual detector on IID scores with no temporal degradation. "
            "All signals here are false positives by construction. The mixed model changes query "
            "difficulty independently at each retrieval, not over time.\n\n" + table + "\n\n"
            "Rates are finite Monte Carlo estimates, not production rates. At 2,000 trials "
            "the maximum binomial standard error is about 1.1 percentage points per cell. "
            "Increasing the trial count reduces sampling noise, not the synthetic-population bias.\n\n"
            "## Operational interpretation\n\n"
            "The same relative-drop threshold behaves very differently across score distributions. "
            "Three-call traces compare one early sample with two late samples and are especially noisy. "
            "Raising min_retrievals alone does not establish an acceptable false-positive rate. "
            "The 0.3 threshold is a heuristic default, not a universally calibrated operating point. "
            "An absolute-gap floor would also depend on the scorer's scale and is not added here.\n\n"
            "Keep this detector shadow-only. Before alerts or interventions, evaluate labeled real "
            "traffic for each index/scorer and query mix, select a threshold against the actual "
            "false-positive budget, and measure recall separately. Query intent changes cannot be "
            "distinguished from degradation using scores alone. Use consistent higher-is-better "
            "scores; raw distances and cross-index comparisons are unsupported. No LLM judge is used.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
