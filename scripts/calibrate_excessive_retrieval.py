#!/usr/bin/env python3
"""Calibration harness for the EXCESSIVE_RETRIEVAL detector.

Usage:
    PYTHONPATH=packages/sdk-py python scripts/calibrate_excessive_retrieval.py
"""

from __future__ import annotations

from dunetrace.detectors import ExcessiveRetrievalDetector
from dunetrace.models import RetrievalResult, RunState


_CORPUS = [
    # (retrieval count, expected to fire)
    (0, False),
    (2, False),
    (5, False),
    (7, False),
    (8, True),
    (9, True),
    (12, True),
    (20, True),
]


def _fires(retrieval_count: int) -> bool:
    state = RunState(run_id="r", agent_id="a", agent_version="v")
    state.retrievals = [
        RetrievalResult(
            index_name="docs" if step % 2 else "tickets",
            result_count=3,
            top_score=0.8,
            step_index=step,
        )
        for step in range(1, retrieval_count + 1)
    ]
    return ExcessiveRetrievalDetector().on_run_completion(state) is not None


def main() -> None:
    correct = sum(_fires(count) == expected for count, expected in _CORPUS)
    print(f"EXCESSIVE_RETRIEVAL calibration: {correct}/{len(_CORPUS)} cases correct")
    if correct != len(_CORPUS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
