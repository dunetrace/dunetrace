#!/usr/bin/env python3
"""
Calibration harness for the AGENT_HANDOFF_FAILURE detector.

Structural (deterministic) detector — no LLM, no API key. Runs the real
detector over a labeled corpus of handoff tool calls and reports precision /
recall. The corpus deliberately includes the disclosed false-positive surface
(convention-based name matching: `transfer_funds`, `user_agent`) so the number
reflects reality, not a best case.

Usage:
    PYTHONPATH=packages/sdk-py python scripts/calibrate_agent_handoff_failure.py [--write]
"""

from __future__ import annotations

import argparse
import json
import pathlib

# Each sample: (tool_name, success, output, label). output_length is derived from
# output. label "pos" = a real handoff failure (should fire), "neg" = should not.
_CORPUS = [
    # ── Positives: handoff tool failed or returned an empty/short payload ──────
    ("research_agent", False, "connection error", "pos"),
    ("delegate_planner", False, None, "pos"),
    ("handoff_billing", False, "timeout", "pos"),
    ("transfer_to_support", False, "", "pos"),
    ("summarizer_agent", True, "done", "pos"),
    ("delegate_writer", True, "ok", "pos"),
    ("handoff_legal", True, "complete", "pos"),
    ("billing_agent", True, "finished", "pos"),
    ("research_agent", True, "", "pos"),
    ("delegate_analyst", True, "n/a", "pos"),
    ("handoff_ops", True, "short", "pos"),  # 5 chars < 10
    ("triage_agent", True, "see log", "pos"),  # 7 chars < 10
    # ── Negatives: legitimate handoffs with a real payload ────────────────────
    (
        "research_agent",
        True,
        "Found 3 sources on Q3 revenue; summary attached with figures.",
        "neg",
    ),
    (
        "delegate_planner",
        True,
        "Plan: 1) scope 2) design 3) build 4) test. Owner assigned each.",
        "neg",
    ),
    ("handoff_billing", True, "Refund of $42.00 issued to order 12345; confirmation sent.", "neg"),
    ("transfer_to_support", True, "Escalated with customer context and order details.", "neg"),
    ("writer_agent", True, "Draft complete: 800-word post covering all three points.", "neg"),
    # ── Negatives: non-handoff tools (must never be considered) ───────────────
    ("web_search", True, "", "neg"),
    ("calculator", True, "42", "neg"),
    ("db_query", False, "syntax error", "neg"),
    ("fetch_page", True, "ok", "neg"),
    # ── Hard negatives: convention collisions the tightening now excludes ──────
    ("transfer_funds", True, "sent", "neg"),  # bare transfer_, not transfer_to_ → excluded
    ("user_agent", True, "Mozilla", "neg"),  # in EXCLUDED_TOOL_NAMES → excluded
]


def _fires(tool_name, success, output):
    from dunetrace.detectors import AgentHandoffFailureDetector
    from dunetrace.models import RunState, ToolCall

    tc = ToolCall(
        tool_name=tool_name,
        args="",
        step_index=1,
        timestamp=1.0,
        success=success,
        output=output,
        output_length=(len(output) if output else 0),
    )
    state = RunState(run_id="r", agent_id="a", agent_version="v")
    state.tool_calls = [tc]
    return AgentHandoffFailureDetector().on_run_completion(state) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    pos = [s for s in _CORPUS if s[3] == "pos"]
    neg = [s for s in _CORPUS if s[3] == "neg"]
    tp = sum(1 for n, s, o, _ in pos if _fires(n, s, o))
    fp = sum(1 for n, s, o, _ in neg if _fires(n, s, o))

    recall = tp / len(pos)
    fp_rate = fp / len(neg)
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"\nAGENT_HANDOFF_FAILURE calibration — {len(pos)} failures, {len(neg)} legit\n")
    print(f"  Recall     : {recall:.0%}  ({tp}/{len(pos)} failures caught)")
    print(f"  False-pos  : {fp_rate:.0%}  (target < 15%)")
    print(f"  Precision  : {precision:.0%}")
    print("\n  False positives (disclosed convention-FP surface):")
    for n, s, o, _ in neg:
        if _fires(n, s, o):
            print(f"    - {n!r} (success={s}, output={o!r})")
    live_ready = fp_rate < 0.15
    print(
        "\n  Verdict: "
        + (
            "LIVE-READY (FP < 15%)"
            if live_ready
            else "SHADOW-ONLY — FP > 15% from name collisions; tighten HANDOFF_PATTERNS before promoting"
        )
        + "\n"
    )

    if args.write:
        cache = pathlib.Path("scripts/calibration/agent_handoff_failure_scores.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {
                    "recall": round(recall, 4),
                    "fp_rate": round(fp_rate, 4),
                    "precision": round(precision, 4),
                    "positives": len(pos),
                    "negatives": len(neg),
                    "live_ready": live_ready,
                },
                indent=2,
            )
        )
        print(f"Cached scores -> {cache}")

    # Informational (documents an already-merged, shadow-by-default detector) —
    # does not gate, unlike the live detectors' calibration scripts.


if __name__ == "__main__":
    main()
