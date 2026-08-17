#!/usr/bin/env python3
"""
Micro-benchmark for the policy condition-expression evaluator (Phase 2).

Reports per-call latency (mean / p50 / p95 / p99 / max) for a spread of
representative expressions against a fixed context. Deterministic workload; no
network, no DB. Phase 7 does the 1000-policy × 10k-events/s load test — this is
the single-evaluation micro-benchmark the <100µs target is stated against.

    PYTHONPATH=packages/sdk-py python scripts/bench_policy_expressions.py
"""

from __future__ import annotations

import gc
import statistics
import time

from dunetrace.policies import parse_match_block
from dunetrace.policies.evaluator import EvaluationContext, evaluate

CASES = {
    "single_comparison": {"args.amount": {"gt": 10000}},
    "and_two_fields": {"args.amount": {"gt": 10000}, "args.currency": {"eq": "USD"}},
    "high_value_refund (AND + 2-way OR)": {
        "args.amount": {"gt": 10000},
        "or": [
            {"agent.tier": {"eq": "trial"}},
            {"org.plan": {"in": ["free", "starter"]}},
        ],
    },
    "regex_match": {"args.id": {"matches": r"^ord_\d+$"}},
    "deep_nested (depth 3)": {
        "args.amount": {"gt": 10000},
        "and": [
            {"or": [{"org.plan": {"eq": "free"}}, {"org.plan": {"eq": "starter"}}]},
            {"or": [{"agent.tier": {"eq": "trial"}}, {"args.destructive": {"eq": True}}]},
        ],
    },
}

CTX = EvaluationContext(
    args={"amount": 15000, "currency": "USD", "id": "ord_12345", "destructive": True},
    run={"agent_id": "billing", "duration_ms": 4200, "event_count": 12},
    agent={"tier": "paid"},
    org={"plan": "free"},
    event={"type": "tool.called", "severity": "high"},
)

WARMUP = 2000
BLOCKS = 3000  # outer samples
BATCH = 500  # evaluations timed per sample, to amortize timer overhead


def bench(expr) -> dict:
    """Amortized-batch timing: each of BLOCKS samples times BATCH evaluations and
    divides, so a single sample isn't dominated by the ~ns timer-call overhead.
    GC is disabled during measurement — GC pauses are real but are not the
    evaluator's own per-call cost, which is what the <100µs target is about."""
    for _ in range(WARMUP):
        evaluate(expr, CTX)
    per_call = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(BLOCKS):
            t0 = time.perf_counter_ns()
            for _ in range(BATCH):
                evaluate(expr, CTX)
            per_call.append((time.perf_counter_ns() - t0) / BATCH)
    finally:
        if gc_was_enabled:
            gc.enable()
    per_call.sort()
    return {
        "mean": statistics.mean(per_call) / 1000.0,
        "p50": per_call[int(0.50 * BLOCKS)] / 1000.0,
        "p95": per_call[int(0.95 * BLOCKS)] / 1000.0,
        "p99": per_call[int(0.99 * BLOCKS)] / 1000.0,
        "max": per_call[-1] / 1000.0,
    }


def main() -> None:
    print(
        f"blocks={BLOCKS} batch={BATCH} warmup={WARMUP}  (µs per evaluation, "
        f"amortized-batch, GC off)\n"
    )
    header = f"{'case':<38} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>8}"
    print(header)
    print("-" * len(header))
    worst_p99 = 0.0
    for name, block in CASES.items():
        expr = parse_match_block(block, policy_name=name)
        r = bench(expr)
        worst_p99 = max(worst_p99, r["p99"])
        print(
            f"{name:<38} {r['mean']:>7.2f} {r['p50']:>7.2f} {r['p95']:>7.2f} "
            f"{r['p99']:>7.2f} {r['max']:>8.2f}"
        )
    print("-" * len(header))
    status = "PASS" if worst_p99 < 100.0 else "FAIL"
    print(f"\nworst-case p99 = {worst_p99:.2f} µs  →  <100µs target: {status}")


if __name__ == "__main__":
    main()
