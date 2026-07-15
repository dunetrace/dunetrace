#!/usr/bin/env python3
"""
Calibration harness for the DELEGATION_LOOP detector (Capability 2, Phase 2.3).

Structural (deterministic) detector — no LLM, no API key. It runs the real graph
pipeline (build the ancestor chain -> agent-delegation edges -> DFS cycle
detection -> detector) over a labeled corpus of multi-agent delegation chains,
then sweeps MIN_LOOP_RUNS to pick the threshold.

The precision tension this corpus is built to expose: a pathological
delegation *loop* (A -> B -> A -> B -> ...) and a legitimate iterative
*supervisor* exchange (A delegates, B returns, A delegates again, then finishes)
look identical in the chain — the only structural difference is how many times it
goes around. So the negatives deliberately include short, legitimate
supervisor/hand-back exchanges (2-4 runs), and the sweep finds the lowest
MIN_LOOP_RUNS that keeps them below the fire line while still catching the
sustained loops.

Each sample is a root-first agent sequence (the order delegation happened).

Usage:
    PYTHONPATH=packages/sdk-py:services/detector \
      python scripts/calibrate_delegation_loop.py [--write]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# ── Positives: sustained, pathological delegation loops ────────────────────────
# Root-first agent order. These went around enough times that they are clearly
# not converging.
_POSITIVES = [
    ["planner", "executor"] * 4,  # 8 runs, 2-agent oscillation
    ["planner", "executor"] * 3,  # 6 runs
    ["researcher", "writer"] * 3,  # 6 runs
    ["manager", "worker"] * 4,  # 8 runs
    ["supervisor", "coder"] * 5,  # 10 runs
    ["planner", "critic"] * 3,  # 6 runs
    ["a", "b", "c"] * 2,  # 6 runs, 3-agent cycle
    ["a", "b", "c"] * 3,  # 9 runs, 3-agent cycle
    ["orchestrator", "sub"] * 3,  # 6 runs
    ["router", "solver"] * 4,  # 8 runs
    ["lead", "ic"] * 3,  # 6 runs
    ["triage", "specialist"] * 4,  # 8 runs
    # Borderline-but-real loops (exactly at/just above threshold) — kept as
    # positives so recall is measured honestly at each sweep point.
    ["planner", "executor", "planner", "executor", "planner"],  # 5 runs
    ["manager", "worker", "manager", "worker"],  # 4 runs
]

# ── Negatives: legitimate multi-agent patterns ────────────────────────────────
_NEGATIVES = [
    # Deep linear hierarchies — every agent distinct, no cycle.
    ["orchestrator", "researcher", "analyst", "writer", "editor"],
    ["orchestrator", "planner", "executor", "reviewer"],
    ["lead", "ic1", "ic2", "ic3"],
    ["router", "retriever", "ranker", "summarizer"],
    ["coordinator", "specialist", "validator"],
    # Single hand-backs — one round trip, legitimate clarification.
    ["manager", "worker", "manager"],
    ["triage", "billing", "triage"],
    ["coordinator", "specialist", "coordinator"],
    ["supervisor", "coder", "supervisor"],
    # Legitimate two-iteration supervisor exchange (the hard negatives): the
    # supervisor delegates twice and then the run finishes. 4 runs — right at the
    # loop-vs-iteration boundary.
    ["supervisor", "worker", "supervisor", "worker"],
    ["manager", "analyst", "manager", "analyst"],
    # A repeated agent that is not a sustained loop (B -> C -> B once).
    ["orchestrator", "worker", "helper", "worker"],
    # Simple two-agent, no repeat.
    ["router", "tool_agent"],
    ["planner", "executor"],
]


def _chain_from_sequence(seq):
    """Build a leaf-first RunNode chain from a root-first agent sequence."""
    from detector_svc.run_graph import RunNode

    nodes = []
    for i, agent in enumerate(seq):
        run_id = f"r{i}"
        parent = f"r{i - 1}" if i > 0 else None
        nodes.append(
            RunNode(run_id=run_id, agent_id=agent, agent_version="v1", parent_run_id=parent)
        )
    return list(reversed(nodes))  # leaf-first


def _fires(seq, min_loop_runs):
    from dunetrace.detectors import DelegationLoopDetector
    from detector_svc.run_graph import (
        agent_sequence,
        build_agent_delegation_edges,
        find_cycle,
    )

    chain = _chain_from_sequence(seq)
    cycle = find_cycle(build_agent_delegation_edges(chain))
    d = DelegationLoopDetector(MIN_LOOP_RUNS=min_loop_runs)
    sig = d.evaluate_delegation_cycle(cycle, agent_sequence(chain), "r", seq[-1], "v1")
    return sig is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print(
        f"\nDELEGATION_LOOP calibration — {len(_POSITIVES)} loops, {len(_NEGATIVES)} legit patterns\n"
    )
    header = f"{'MIN_LOOP_RUNS':>13} {'FP rate':>9} {'TP recall':>10} {'ship?':>6}"
    print(header)
    print("-" * len(header))
    sweep = {}
    for thresh in range(3, 9):
        fp = sum(1 for s in _NEGATIVES if _fires(s, thresh)) / len(_NEGATIVES)
        tp = sum(1 for s in _POSITIVES if _fires(s, thresh)) / len(_POSITIVES)
        ship = fp < 0.15
        sweep[thresh] = {"fp_rate": round(fp, 4), "tp_recall": round(tp, 4), "ship": ship}
        print(f"{thresh:>13} {fp:>8.0%} {tp:>9.0%} {'yes' if ship else '':>6}")
    print("-" * len(header))

    # Precision-first: among ship-eligible thresholds, prefer the lowest false
    # positive rate, then the highest recall. Loop-vs-legit-iteration is
    # genuinely ambiguous at the boundary, so we do not trade FP for one extra
    # borderline positive (mirrors the SYCOPHANCY precision-first choice).
    eligible = [t for t, r in sweep.items() if r["ship"]]
    best = min(eligible, key=lambda t: (sweep[t]["fp_rate"], -sweep[t]["tp_recall"]), default=None)
    if best is not None:
        print(
            f"\nRECOMMENDED MIN_LOOP_RUNS = {best} "
            f"(FP {sweep[best]['fp_rate']:.0%} < 15%, TP recall {sweep[best]['tp_recall']:.0%})\n"
        )
    else:
        print("\nNo threshold kept FP < 15% — revisit the corpus / detector before shipping.\n")

    if args.write:
        cache = pathlib.Path("scripts/calibration/delegation_loop_scores.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"sweep": sweep, "recommended_min_loop_runs": best}, indent=2))
        print(f"Cached scores -> {cache}")

    if best is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
