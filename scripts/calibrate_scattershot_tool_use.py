#!/usr/bin/env python3
"""
Calibration harness for the SCATTERSHOT_TOOL_USE detector.

Structural and deterministic: no LLM, API key, network, or third-party package.
The corpus emphasizes the detector's false-positive boundary — legitimate
multi-tool agents that use a broad fixed workflow, often once per tool — and
sweeps both MAX_DISTINCT_TOOLS and MIN_TOTAL_CALLS.

Usage:
    PYTHONPATH=packages/sdk-py \
      python scripts/calibrate_scattershot_tool_use.py [--write]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


# Pathological runs: the agent keeps widening its approach instead of settling
# on a coherent tool sequence. Two five-tool cases remain in the positives to
# expose the recall deliberately lost by a precision-first six-tool threshold.
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
]

# Legitimate runs include pipelines that are broad by design. The hard negatives
# make it impossible to eliminate false positives structurally; that is why the
# detector ships shadowed even after calibration.
_NEGATIVES = [
    ["search", "fetch", "extract", "summarize", "write"],
    ["read", "grep", "edit", "test", "git", "read", "test", "git", "read"],
    [
        "browser",
        "login",
        "navigate",
        "download",
        "parse",
        "browser",
        "parse",
        "download",
        "parse",
        "download",
    ],
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
    ["trigger", "lookup", "validate", "transform", "persist", "notify", "audit"],
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


def _fires(names: list[str], max_distinct_tools: int, min_total_calls: int) -> bool:
    """Return whether one labeled tool sequence fires for the supplied thresholds."""
    from dunetrace.detectors import ScattershotToolUseDetector
    from dunetrace.models import RunState, ToolCall

    state = RunState(run_id="calibration", agent_id="agent", agent_version="v1")
    state.tool_calls = [
        ToolCall(tool_name=name, args="{}", step_index=i + 1, timestamp=0.0)
        for i, name in enumerate(names)
    ]
    state.current_step = len(names)
    detector = ScattershotToolUseDetector(
        MAX_DISTINCT_TOOLS=max_distinct_tools,
        MIN_TOTAL_CALLS=min_total_calls,
    )
    return detector.on_run_completion(state) is not None


def main() -> None:
    """Sweep threshold pairs, print their scores, and optionally cache the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print(
        f"\nSCATTERSHOT_TOOL_USE calibration — {len(_POSITIVES)} scattershot, "
        f"{len(_NEGATIVES)} legitimate runs\n"
    )
    header = f"{'DISTINCT':>8} {'TOTAL':>7} {'FP rate':>9} {'TP recall':>10} {'ship?':>6}"
    print(header)
    print("-" * len(header))

    sweep: dict[str, dict[str, float | bool]] = {}
    for distinct_threshold in range(4, 9):
        for total_threshold in range(max(5, distinct_threshold), 13):
            fp = sum(
                _fires(sample, distinct_threshold, total_threshold) for sample in _NEGATIVES
            ) / len(_NEGATIVES)
            tp = sum(
                _fires(sample, distinct_threshold, total_threshold) for sample in _POSITIVES
            ) / len(_POSITIVES)
            ship = fp < 0.15
            key = f"{distinct_threshold}/{total_threshold}"
            sweep[key] = {
                "fp_rate": round(fp, 4),
                "tp_recall": round(tp, 4),
                "ship": ship,
            }
            print(
                f"{distinct_threshold:>8} {total_threshold:>7} {fp:>8.0%} "
                f"{tp:>9.0%} {'yes' if ship else '':>6}"
            )

    eligible = [
        (tuple(map(int, key.split("/"))), result) for key, result in sweep.items() if result["ship"]
    ]
    best = min(
        eligible,
        key=lambda item: (
            -float(item[1]["tp_recall"]),
            float(item[1]["fp_rate"]),
            item[0][0],
            item[0][1],
        ),
        default=None,
    )

    if best is None:
        print("\nNo threshold pair kept FP < 15% — revisit the heuristic before shipping.\n")
        sys.exit(1)

    (best_distinct, best_total), result = best
    print(
        "\nRECOMMENDED "
        f"MAX_DISTINCT_TOOLS = {best_distinct}, MIN_TOTAL_CALLS = {best_total} "
        f"(FP {result['fp_rate']:.0%} < 15%, TP recall {result['tp_recall']:.0%})\n"
    )

    if args.write:
        output = pathlib.Path("scripts/calibration/scattershot_tool_use_scores.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "sweep": sweep,
                    "recommended_max_distinct_tools": best_distinct,
                    "recommended_min_total_calls": best_total,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Cached scores -> {output}")


if __name__ == "__main__":
    main()
