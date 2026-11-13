#!/usr/bin/env python3
"""
Calibration harness for the SCATTERSHOT_TOOL_USE detector.

Structural and deterministic: no LLM, API key, network, or third-party package.

What this measures, and what it does not
----------------------------------------
An earlier version of this harness reported "0/20 false positives, 87.5% recall"
and that number was an artifact of the corpus, not evidence about the boundary.
Only 2 of its 20 negatives reached 6+ distinct tools, and both had ZERO repeated
calls — they were excluded solely by being one or two calls short of
MIN_TOTAL_CALLS, not by any structural property. Every positive had exactly 8 or
9 calls, so recall collapsed 88% -> 19% when the total threshold moved 8 -> 9.
The two classes were separated by a single integer at exactly the threshold
being recommended, which means the sweep was scoring its own construction.

The corpus below is built to break that:

  * Negatives that CLEAR the volume and breadth floors outright — broad
    pipelines of 8-12 distinct tools and 8-14 calls — so they can only be
    excluded by structure (a low repeat ratio), never by being short.
  * Pipelines WITH retries, which repeat a stage or two, sitting just under the
    ratio boundary rather than far below it.
  * Positives spanning 8-16 calls, so recall is not a step function at one
    integer.

The sweep is therefore three-dimensional: distinct, total, and repeat ratio.
Ratio is the axis that actually separates the classes; the other two only
exclude runs too small to judge.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Hoisted: the previous version re-imported the module and re-instantiated the
# detector inside the inner loop, on all corpus x threshold evaluations.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "packages" / "sdk-py"))
from dunetrace.detectors import ScattershotToolUseDetector  # noqa: E402
from dunetrace.models import RunState, ToolCall  # noqa: E402

# Pathological: the agent keeps widening its approach AND retrying, instead of
# settling on a coherent sequence. Deliberately varied in length (8-16 calls) so
# recall does not hinge on a single integer.
_POSITIVES = [
    ["search", "browser", "shell", "python", "database", "email", "search", "shell"],
    ["read", "grep", "edit", "format", "lint", "test", "read", "test"],
    ["search", "fetch", "browser", "python", "shell", "database", "api", "search"],
    ["crm", "email", "calendar", "search", "browser", "notes", "crm", "search"],
    ["read", "search", "browser", "shell", "python", "database", "write", "test"],
    ["search", "fetch", "extract", "python", "database", "spreadsheet", "email", "search"],
    ["browser", "api", "shell", "python", "git", "editor", "test", "browser", "git"],
    ["search", "maps", "weather", "calendar", "email", "notes", "browser", "search"],
    ["read", "grep", "search", "browser", "shell", "python", "edit", "test", "git"],
    ["database", "api", "browser", "shell", "python", "search", "email", "calendar"],
    ["search", "fetch", "browser", "ocr", "python", "spreadsheet", "database", "email"],
    ["read", "grep", "search", "shell", "python", "editor", "lint", "test", "git"],
    ["crm", "database", "search", "browser", "calendar", "email", "notes", "crm"],
    ["search", "browser", "api", "shell", "python", "database", "search", "api"],
    ["search", "browser", "shell", "python", "database", "search", "shell", "python", "database"],
    ["read", "grep", "edit", "test", "git", "read", "edit", "test", "git"],
    # Longer flailing runs — the same six-to-eight tools cycled repeatedly.
    ["search", "browser", "shell", "python", "database", "email"] * 2,
    ["read", "grep", "edit", "lint", "test", "git", "read", "grep", "edit", "lint", "test", "git"],
    ["api", "browser", "shell", "python", "search", "database", "email", "notes"] * 2,
    ["crm", "email", "calendar", "search", "browser", "notes"] * 2 + ["crm", "search"],
]

# Legitimate. The first block is the one that matters: every entry clears BOTH
# the breadth and volume floors, so a threshold pair can only exclude them
# structurally. Under the old total-calls-only rule all of these fired.
_NEGATIVES = [
    # Broad one-pass pipelines — 8-12 distinct tools, each used once.
    ["ingest", "validate", "transform", "enrich", "persist", "index", "notify", "audit"],
    ["trigger", "lookup", "validate", "transform", "persist", "notify", "audit", "archive"],
    ["fetch", "parse", "normalise", "dedupe", "score", "rank", "store", "publish", "log"],
    ["auth", "fetch", "decrypt", "parse", "validate", "transform", "load", "verify", "report"],
    ["scrape", "clean", "embed", "index", "summarise", "translate", "store", "notify"],
    [
        "receive",
        "authenticate",
        "authorise",
        "validate",
        "enrich",
        "route",
        "persist",
        "acknowledge",
        "emit",
        "audit",
    ],
    # Broad pipelines WITH a retry or two — just under the ratio boundary, which
    # is the genuinely hard case rather than a run that is merely short.
    ["ingest", "validate", "transform", "persist", "index", "notify", "audit", "validate"],
    ["fetch", "parse", "score", "rank", "store", "publish", "log", "parse", "store"],
    ["auth", "fetch", "parse", "validate", "load", "verify", "report", "fetch", "verify"],
    ["scrape", "clean", "embed", "index", "summarise", "store", "notify", "clean", "index"],
    # Original corpus: narrower loops and shorter runs.
    ["search", "fetch", "extract", "summarize", "write"],
    ["read", "grep", "edit", "test", "git", "read", "test", "git", "read"],
    ["browser", "login", "navigate", "download", "parse", "browser", "parse", "download", "parse"],
    [
        "crm",
        "calendar",
        "email",
        "notes",
        "search",
        "crm",
        "calendar",
        "email",
        "notes",
        "search",
        "crm",
        "email",
    ],
    ["read", "grep", "edit", "format", "test", "read", "test"],
    ["search", "fetch", "extract", "rank", "summarize", "write"],
    ["read", "grep", "edit", "test", "git", "read", "grep", "test"],
    ["search", "browser", "notes", "search", "browser", "notes"],
    ["database", "transform", "database", "transform", "database", "transform"],
    ["read", "edit", "test", "read", "edit", "test", "read", "test"],
    ["calendar", "email", "calendar", "email", "calendar"],
    ["search", "fetch", "extract", "summarize"],
    ["read", "grep", "edit", "test", "read", "test"],
    ["browser", "download", "parse", "store", "browser", "parse"],
    ["crm", "lookup", "email", "notes", "crm", "email"],
    ["search", "maps", "calendar", "email", "notes", "search", "calendar", "email", "notes"],
    ["read", "grep", "edit", "format", "lint", "read", "edit", "format", "lint", "read"],
    ["api", "validate", "transform", "persist", "notify", "api", "validate", "persist"],
    ["search", "fetch", "rank", "summarize", "write", "search", "rank"],
]


def _state(names: list[str]) -> RunState:
    state = RunState(run_id="calibration", agent_id="agent", agent_version="v1")
    state.tool_calls = [
        ToolCall(tool_name=name, args="{}", step_index=i + 1, timestamp=0.0)
        for i, name in enumerate(names)
    ]
    state.current_step = len(names)
    # Not "final_answer": the convergence veto would exclude the whole corpus
    # and the sweep would measure nothing.
    state.exit_reason = "completed"
    return state


_POS_STATES = [_state(n) for n in _POSITIVES]
_NEG_STATES = [_state(n) for n in _NEGATIVES]


def _ratio(names: list[str]) -> float:
    return len(names) / len(set(names))


def _separation() -> None:
    """Print the class distributions, so a reader can see whether the corpus
    actually separates — rather than taking an FP rate on trust."""
    pos = sorted(_ratio(n) for n in _POSITIVES)
    neg = sorted(_ratio(n) for n in _NEGATIVES)
    broad_neg = [n for n in _NEGATIVES if len(set(n)) >= 6 and len(n) >= 8]
    print("Corpus separation (repeat ratio = total calls / distinct tools):")
    print(f"  positives : min {pos[0]:.2f}  median {pos[len(pos) // 2]:.2f}  max {pos[-1]:.2f}")
    print(f"  negatives : min {neg[0]:.2f}  median {neg[len(neg) // 2]:.2f}  max {neg[-1]:.2f}")
    print(
        f"  {len(broad_neg)} of {len(_NEGATIVES)} negatives clear BOTH floors "
        f"(>=6 distinct, >=8 calls) and so can only be excluded structurally.\n"
    )


def _fires(state: RunState, distinct: int, total: int, ratio: float) -> bool:
    return (
        ScattershotToolUseDetector(
            MIN_DISTINCT_TOOLS=distinct,
            MIN_TOTAL_CALLS=total,
            MIN_REPEAT_RATIO=ratio,
        ).on_run_completion(state)
        is not None
    )


# A configuration that fires on almost nothing trivially scores 0% FP. Both
# bars have to be cleared for a recommendation to mean anything.
_MAX_FP = 0.15
_MIN_RECALL = 0.50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print(
        f"\nSCATTERSHOT_TOOL_USE calibration — {len(_POSITIVES)} scattershot, "
        f"{len(_NEGATIVES)} legitimate runs\n"
    )
    _separation()

    header = (
        f"{'DISTINCT':>8} {'TOTAL':>6} {'RATIO':>6} {'FP rate':>9} {'TP recall':>10} {'ship?':>6}"
    )
    print(header)
    print("-" * len(header))

    sweep: dict[str, dict[str, float | bool]] = {}
    for distinct in range(4, 9):
        for total in range(max(5, distinct), 13):
            for ratio_x10 in range(10, 26, 5):  # 1.0, 1.5, 2.0, 2.5
                ratio = ratio_x10 / 10
                fp = sum(_fires(s, distinct, total, ratio) for s in _NEG_STATES) / len(_NEG_STATES)
                tp = sum(_fires(s, distinct, total, ratio) for s in _POS_STATES) / len(_POS_STATES)
                # A gate on FP alone is how you end up recommending a
                # configuration that never fires: 0% false positives is trivially
                # achievable by not detecting anything. Recall has to clear a bar
                # too, or the harness cannot tell "precise" from "inert".
                ship = fp < _MAX_FP and tp >= _MIN_RECALL
                sweep[f"{distinct}/{total}/{ratio}"] = {
                    "fp_rate": round(fp, 4),
                    "tp_recall": round(tp, 4),
                    "ship": ship,
                }
                if ratio in (1.0, 1.5, 2.0):
                    print(
                        f"{distinct:>8} {total:>6} {ratio:>6.1f} {fp:>8.0%} "
                        f"{tp:>9.0%} {'yes' if ship else '':>6}"
                    )

    eligible = [(tuple(key.split("/")), result) for key, result in sweep.items() if result["ship"]]
    best = min(
        eligible,
        key=lambda item: (
            -float(item[1]["tp_recall"]),
            float(item[1]["fp_rate"]),
            float(item[0][0]),
            float(item[0][1]),
            float(item[0][2]),
        ),
        default=None,
    )

    if best is None:
        print(
            f"\nNo threshold triple reached FP < {_MAX_FP:.0%} AND recall >= "
            f"{_MIN_RECALL:.0%}.\n\n"
            "This is a result, not a harness failure. On a corpus whose negatives\n"
            "actually clear the breadth and volume floors, repeat ratio does not\n"
            "separate the two classes — their distributions overlap almost\n"
            "entirely, and the negative median sits ABOVE the positive median\n"
            "(see the separation summary above). Precision is reachable only by\n"
            "setting the ratio high enough that the detector rarely fires.\n\n"
            "The conclusion is about the heuristic, not the thresholds: tool\n"
            "breadth plus repetition does not identify scattershot behaviour.\n"
            "The more promising axis is non-convergence — a run that fanned out\n"
            "AND never produced a final answer — which this corpus cannot score,\n"
            "since it carries tool-name sequences only and no exit_reason or\n"
            "per-call success data. Extending the corpus to carry those is the\n"
            "next step; tuning these three numbers is not.\n\n"
            "The detector stays in shadow (it is absent from LIVE_DETECTORS).\n"
        )
        sys.exit(1)

    (b_distinct, b_total, b_ratio), result = best
    print(
        "\nRECOMMENDED "
        f"MIN_DISTINCT_TOOLS = {b_distinct}, MIN_TOTAL_CALLS = {b_total}, "
        f"MIN_REPEAT_RATIO = {b_ratio} "
        f"(FP {result['fp_rate']:.0%} < 15%, TP recall {result['tp_recall']:.0%})\n"
    )

    if args.write:
        # Derived from __file__, not the cwd: the previous relative path combined
        # with mkdir(parents=True) meant running from scripts/ silently created
        # scripts/scripts/calibration/ and reported success.
        output = (
            pathlib.Path(__file__).resolve().parent
            / "calibration"
            / "scattershot_tool_use_scores.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "sweep": sweep,
                    "recommended_min_distinct_tools": int(b_distinct),
                    "recommended_min_total_calls": int(b_total),
                    "recommended_min_repeat_ratio": float(b_ratio),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Cached scores -> {output}")


if __name__ == "__main__":
    main()
