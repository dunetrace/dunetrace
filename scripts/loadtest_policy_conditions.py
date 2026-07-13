#!/usr/bin/env python3
"""
Load test for expression-condition policy evaluation (Phase 7).

Loads 1000 policies, all with expression conditions, and measures:
  * per-policy evaluation latency (the <100µs target), p50/p95/p99
  * full-battery (1000-policy) per-event latency
  * achievable single-thread throughput, and a 10k-event burst
  * memory: RSS growth + Python-heap growth over a sustained run (leak check)

Deterministic workload; no network, no DB. Run:
    PYTHONPATH=packages/sdk-py python scripts/loadtest_policy_conditions.py
"""

from __future__ import annotations

import gc
import resource
import statistics
import sys
import time
import tracemalloc

from dunetrace.policies import EvaluationContext, Policy, PolicyEngine

N_POLICIES = 1000

# A representative evaluation context (billing-style agent mid-run).
CTX = EvaluationContext(
    args={"amount": 15000, "currency": "USD", "id": "ord_12345", "destructive": True},
    run={
        "agent_id": "billing",
        "duration_ms": 4200,
        "event_count": 12,
        "tool_call_count": 6,
        "error_count": 1,
        "cost_usd": 0.12,
        "step_count": 6,
    },
    agent={"tier": "paid"},
    org={"plan": "free"},
    event={"type": "tool.called", "tool_name": "refund_customer", "hour": 14},
)

# Five expression shapes, cycled across the 1000 policies so the battery is
# realistically varied (thresholds/regex/membership/nesting). Most are tuned NOT
# to match CTX, so a no-match run exercises the full battery (worst case).
_SHAPES = [
    lambda i: {"args.amount": {"gt": 10_000_000 + i}},  # numeric, no match
    lambda i: {"args.currency": {"eq": f"XX{i}"}},  # eq, no match
    lambda i: {"org.plan": {"in": ["enterprise", f"tier{i}"]}},  # membership, no match
    lambda i: {"args.id": {"matches": r"^zzz_\d+$"}},  # regex, no match
    lambda i: {
        "run.tool_call_count": {"gt": 1000},  # nested AND/OR, no match
        "or": [{"agent.tier": {"eq": f"t{i}"}}, {"org.plan": {"eq": "x"}}],
    },
]


def build_engine(all_match: bool = False, n_agents: int = 1) -> PolicyEngine:
    """1000 pure-expression policies. If all_match, the LAST policy matches CTX
    (short-circuit near the end). ``n_agents`` scopes policies across N agents
    (agent_id cycled) — with N>1 a given event only evaluates its agent's share,
    the realistic case; N=1 (all wildcard) is the pathological worst case."""
    eng = PolicyEngine()
    for i in range(N_POLICIES):
        match = _SHAPES[i % len(_SHAPES)](i)
        if all_match and i == N_POLICIES - 1:
            match = {"args.amount": {"gt": 10000}}  # matches CTX
        eng.add(
            Policy(
                name=f"p{i}",
                condition={"trigger": "expression", "match": match},
                action={"type": "log"},
                agent_id="*" if n_agents == 1 else f"agent{i % n_agents}",
                priority=i,
            )
        )
    return eng


def _rss_mb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KB.
    return v / (1024.0 * 1024.0) if sys.platform == "darwin" else v / 1024.0


def _pct(sorted_us, q):
    return sorted_us[min(len(sorted_us) - 1, int(q * len(sorted_us)))]


def bench_battery(
    eng: PolicyEngine, blocks=1000, label="", agent_id="billing", policies_evaluated=N_POLICIES
) -> None:
    """Time evaluate() with no observer. Amortized: each sample times one
    evaluate() call. Reports per-event, per-evaluated-policy, and throughput.
    ``policies_evaluated`` is how many actually reach expression eval for this
    agent (the rest are cheap agent_id skips)."""
    triggered: set = set()
    for _ in range(200):
        eng.evaluate(agent_id, {}, triggered, CTX)  # warm up
    # Amortized batching: time INNER evaluate() calls per sample so a single OS
    # scheduling hiccup is spread across the batch, not attributed to one event.
    inner = 20
    per_event_us = []
    gc.disable()
    try:
        for _ in range(blocks):
            t0 = time.perf_counter_ns()
            for _ in range(inner):
                eng.evaluate(agent_id, {}, triggered, CTX)
            per_event_us.append((time.perf_counter_ns() - t0) / 1000.0 / inner)
    finally:
        gc.enable()
    per_event_us.sort()
    mean = statistics.mean(per_event_us)
    p50, p95, p99 = _pct(per_event_us, 0.50), _pct(per_event_us, 0.95), _pct(per_event_us, 0.99)
    per_policy_mean = mean / policies_evaluated * 1000  # ns per evaluated policy
    print(f"\n[{label}]  ({policies_evaluated} of {N_POLICIES} policies reach eval for this agent)")
    print(f"  per-event:  mean {mean:8.1f}µs  p50 {p50:8.1f}µs  p95 {p95:8.1f}µs  p99 {p99:8.1f}µs")
    print(f"  per-evaluated-policy: ~{per_policy_mean:6.1f}ns mean")
    print(f"  throughput: {1e6 / mean:,.0f} events/s single-thread")


def burst_10k(eng: PolicyEngine) -> None:
    triggered: set = set()
    gc.disable()
    try:
        t0 = time.perf_counter()
        for _ in range(10_000):
            eng.evaluate("billing", {}, triggered, CTX)
        dt = time.perf_counter() - t0
    finally:
        gc.enable()
    print(
        f"\n[burst] 10,000 events × {N_POLICIES}-policy battery: {dt * 1000:.1f} ms "
        f"({10_000 / dt:,.0f} events/s single-thread)"
    )


def memory_leak_check(eng: PolicyEngine, events=15_000) -> None:
    triggered: set = set()
    gc.collect()
    tracemalloc.start()
    base_py, _ = tracemalloc.get_traced_memory()
    rss_before = _rss_mb()
    for _ in range(events):
        eng.evaluate("billing", {}, triggered, CTX)
    gc.collect()
    cur_py, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_mb()
    grew_py = (cur_py - base_py) / 1024.0
    print(
        f"\n[memory] {events:,} events × {N_POLICIES}-policy battery "
        f"({events * N_POLICIES:,} policy-evals)"
    )
    print(
        f"  Python heap: {base_py / 1024:,.0f}KB → {cur_py / 1024:,.0f}KB "
        f"(Δ {grew_py:+,.1f}KB, peak {peak_py / 1024:,.0f}KB)"
    )
    print(f"  RSS: {rss_before:,.0f}MB → {rss_after:,.0f}MB (Δ {rss_after - rss_before:+,.1f}MB)")
    verdict = "PASS (no leak)" if abs(grew_py) < 512 else "INVESTIGATE"
    print(f"  leak verdict: {verdict}  (Python-heap Δ < 512KB over {events * N_POLICIES:,} evals)")


def main() -> None:
    print(f"Loading {N_POLICIES} expression policies…")
    t0 = time.perf_counter()
    eng_worst = build_engine(all_match=False)  # none match → full battery every event
    print(
        f"  loaded + parsed in {(time.perf_counter() - t0) * 1000:.0f}ms; engine RSS {_rss_mb():,.0f}MB"
    )

    bench_battery(eng_worst, label="WORST CASE — 1000 wildcard policies, none match, all evaluated")

    eng_best = build_engine(all_match=True)
    bench_battery(eng_best, label="a policy matches — short-circuits at first hit")

    # Realistic distribution: 1000 policies spread across 50 agents → a given
    # event only evaluates its agent's ~20 policies (the rest are agent_id skips).
    eng_scoped = build_engine(all_match=False, n_agents=50)
    bench_battery(
        eng_scoped,
        label="REALISTIC — 1000 policies across 50 agents",
        agent_id="agent0",
        policies_evaluated=N_POLICIES // 50,
    )

    if "--latency-only" not in sys.argv:
        burst_10k(eng_worst)
        memory_leak_check(eng_worst)


if __name__ == "__main__":
    main()
