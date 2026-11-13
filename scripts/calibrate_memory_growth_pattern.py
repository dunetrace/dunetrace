#!/usr/bin/env python3
"""Calibration harness for MEMORY_GROWTH_PATTERN.

Structural and deterministic: no LLM, API key, network, or third-party package.

Three slices, reported separately so one cannot hide another:

  * boundary -- memory-event runs that check the ratio and the write floor.
    These test the rule against itself (labels follow the rule), so they are a
    regression gate, not a precision estimate. The script exits non-zero on a
    boundary miss.
  * realistic -- memory-event shapes from real agents, labeled by whether the
    agent is actually misbehaving: LangGraph agents that read with
    store.search, a background extraction run that only writes, an agent that
    writes now and reads in a later run. The last two are false positives by
    construction: the detector compares within one run. They are why it ships
    in shadow.
  * uninstrumented -- tool calls with no memory events. The detector does not
    consult tool names (substring matching misread add_numbers and
    lookup_address as writes), so nothing here fires, including real
    memory-heavy runs. Coverage needs memory instrumentation.

Usage:
    PYTHONPATH=packages/sdk-py python scripts/calibrate_memory_growth_pattern.py \
        --report scripts/calibration/memory_growth_pattern_calibration.md
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "packages" / "sdk-py"))

from dunetrace.detectors import MemoryGrowthPatternDetector  # noqa: E402
from dunetrace.models import MemoryEvent, RunState, ToolCall  # noqa: E402

# (slice, name, human label, memory ops, tool names)
# label True means this run is the failure the detector exists to catch.
_CASES = [
    # ── memory_events positives: ratio at and above 3, including zero reads ──
    ("boundary", "3 writes / 1 read", True, [("written", "notes")] * 3 + [("read", "notes")], []),
    (
        "boundary",
        "6 writes / 2 reads",
        True,
        [("written", "notes")] * 6 + [("read", "notes")] * 2,
        [],
    ),
    (
        "boundary",
        "9 writes / 3 reads",
        True,
        [("written", "notes")] * 9 + [("read", "notes")] * 3,
        [],
    ),
    ("boundary", "4 writes / 1 read", True, [("written", "notes")] * 4 + [("read", "notes")], []),
    (
        "boundary",
        "8 writes / 2 reads",
        True,
        [("written", "notes")] * 8 + [("read", "notes")] * 2,
        [],
    ),
    (
        "boundary",
        "7 writes / 2 reads",
        True,
        [("written", "notes")] * 7 + [("read", "notes")] * 2,
        [],
    ),
    ("boundary", "3 writes / 0 reads", True, [("written", "notes")] * 3, []),
    ("boundary", "5 writes / 0 reads", True, [("written", "notes")] * 5, []),
    ("boundary", "10 writes / 0 reads", True, [("written", "notes")] * 10, []),
    (
        "boundary",
        "4 writes / 1 read, plus clears",
        True,
        [("written", "notes")] * 4 + [("cleared", "notes")] * 3 + [("read", "notes")],
        [],
    ),
    # ── memory_events negatives that clear the write floor ──
    (
        "boundary",
        "6 writes / 3 reads (ratio 2)",
        False,
        [("written", "k")] * 6 + [("read", "k")] * 3,
        [],
    ),
    (
        "boundary",
        "8 writes / 3 reads (ratio 2.67)",
        False,
        [("written", "k")] * 8 + [("read", "k")] * 3,
        [],
    ),
    (
        "boundary",
        "5 writes / 2 reads (ratio 2.5)",
        False,
        [("written", "k")] * 5 + [("read", "k")] * 2,
        [],
    ),
    (
        "boundary",
        "4 writes / 2 reads (ratio 2)",
        False,
        [("written", "k")] * 4 + [("read", "k")] * 2,
        [],
    ),
    ("boundary", "10 writes / 10 reads", False, [("written", "k")] * 10 + [("read", "k")] * 10, []),
    ("boundary", "6 writes / 6 reads", False, [("written", "k")] * 6 + [("read", "k")] * 6, []),
    ("boundary", "3 writes / 3 reads", False, [("written", "k")] * 3 + [("read", "k")] * 3, []),
    ("boundary", "3 writes / 5 reads", False, [("written", "k")] * 3 + [("read", "k")] * 5, []),
    # ── below MIN_WRITES, including the zero-read shape that would otherwise fire ──
    ("boundary", "2 writes / 0 reads", False, [("written", "k")] * 2, []),
    ("boundary", "2 writes / 1 read", False, [("written", "k")] * 2 + [("read", "k")], []),
    ("boundary", "1 write / 0 reads", False, [("written", "k")], []),
    ("boundary", "0 writes / 4 reads", False, [("read", "k")] * 4, []),
    ("boundary", "cleared only", False, [("cleared", "k")] * 4, []),
    ("boundary", "no memory use", False, [], []),
    # Non-empty memory_events must not consult tool names.
    (
        "boundary",
        "clears plus save_memory tools",
        False,
        [("cleared", "k")] * 2,
        ["save_memory"] * 4,
    ),
    (
        "boundary",
        "one memory write plus write_file tools",
        False,
        [("written", "k")],
        ["write_file"] * 5,
    ),
    # ── fallback on names that are memory tools ──
    (
        "boundary",
        "3 save_memory / 1 search_memory",
        True,
        [],
        ["save_memory"] * 3 + ["search_memory"],
    ),
    ("boundary", "4 add_memory / 1 recall_fact", True, [], ["add_memory"] * 4 + ["recall_fact"]),
    ("boundary", "3 store_note / 0 reads", True, [], ["store_note"] * 3),
    (
        "boundary",
        "6 insert_memory / 2 query_memory",
        True,
        [],
        ["insert_memory"] * 6 + ["query_memory"] * 2,
    ),
    ("boundary", "3 SAVE_NOTE / 1 Search_Memory", True, [], ["SAVE_NOTE"] * 3 + ["Search_Memory"]),
    ("boundary", "calculator and summarize", False, [], ["calculator", "summarize"] * 3),
    # ── substring collisions. Ordinary tools, empty memory_events. ──
    # Write-side markers fire. Read-side markers do not, unless a write marker
    # is also present at a 3x ratio.
    ("collision", "3 write_file", False, [], ["write_file"] * 3),
    ("collision", "3 write_file / 1 read_file", False, [], ["write_file"] * 3 + ["read_file"]),
    ("collision", "6 write_file / 2 read_file", False, [], ["write_file"] * 6 + ["read_file"] * 2),
    ("collision", "3 write_file / 1 web_search", False, [], ["write_file"] * 3 + ["web_search"]),
    ("collision", "3 write_file / 1 get_weather", False, [], ["write_file"] * 3 + ["get_weather"]),
    ("collision", "3 write_file / 1 fetch_url", False, [], ["write_file"] * 3 + ["fetch_url"]),
    ("collision", "3 write_file / 3 read_file", False, [], ["write_file"] * 3 + ["read_file"] * 3),
    ("collision", "2 write_file", False, [], ["write_file"] * 2),
    ("collision", "4 read_file", False, [], ["read_file"] * 4),
    ("collision", "6 web_search", False, [], ["web_search"] * 6),
    ("collision", "4 get_weather", False, [], ["get_weather"] * 4),
    ("collision", "4 fetch_url", False, [], ["fetch_url"] * 4),
    # Positive memory behavior, diluted by an unrelated read-marker tool.
    (
        "collision",
        "3 save_memory diluted by 2 web_search",
        True,
        [],
        ["save_memory"] * 3 + ["web_search"] * 2,
    ),
]


# Realistic memory-event shapes (labels: is the agent actually misbehaving?).
# store.search is a read since LangGraph's BaseStore.search is instrumented.
_CASES += [
    (
        "realistic",
        "LangGraph: 3 puts, 3 store.search reads",
        False,
        [("written", "user/1:prefs")] * 3 + [("read", "user/1:*")] * 3,
        [],
    ),
    (
        "realistic",
        "LangGraph: 4 puts, 2 store.search reads",
        False,
        [("written", "user/1:prefs")] * 4 + [("read", "user/1:*")] * 2,
        [],
    ),
    (
        "realistic",
        "write-only loop: 12 saves of one fact, no reads",
        True,
        [("written", "notes")] * 12,
        [],
    ),
    (
        "realistic",
        "background extraction run: 5 writes, reads elsewhere",
        False,
        [("written", f"fact:{i}") for i in range(5)],
        [],
    ),
    ("realistic", "writes now, reads in the next run", False, [("written", "todo")] * 3, []),
]


def _slice(slice_name: str, ops: list) -> str:
    # A run with no memory events is uninstrumented, whatever it was filed as.
    return slice_name if ops else "uninstrumented"


def _state(ops: list[tuple[str, str]], tools: list[str]) -> RunState:
    state = RunState(run_id="r", agent_id="a", agent_version="v")
    state.memory_events = [
        MemoryEvent(op=op, key=key, step_index=i, timestamp=float(i))
        for i, (op, key) in enumerate(ops, start=1)
    ]
    state.tool_calls = [
        ToolCall(tool_name=name, args="{}", step_index=i, timestamp=float(i), success=True)
        for i, name in enumerate(tools, start=1)
    ]
    return state


def _scores(rows: list[tuple[bool, bool]]) -> dict[str, float]:
    tp = sum(label and fired for label, fired in rows)
    fp = sum((not label) and fired for label, fired in rows)
    fn = sum(label and not fired for label, fired in rows)
    tn = sum((not label) and not fired for label, fired in rows)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "n": len(rows),
    }


def _fmt(title: str, scores: dict[str, float]) -> None:
    print(
        f"{title}: n={scores['n']}  "
        f"TP={scores['tp']} FP={scores['fp']} FN={scores['fn']} TN={scores['tn']}  "
        f"precision={scores['precision']:.1%} recall={scores['recall']:.1%} "
        f"FPR={scores['fpr']:.1%}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()

    detector = MemoryGrowthPatternDetector()
    evaluated = []
    print(f"{'slice':<15} {'label':<5} {'fired':<5} name")
    for slice_name, name, label, ops, tools in _CASES:
        slice_name = _slice(slice_name, ops)
        fired = detector.on_run_completion(_state(ops, tools)) is not None
        evaluated.append((slice_name, name, label, fired))
        flag = "OK" if fired == label else "MISS"
        print(f"{slice_name:<15} {str(label):<5} {str(fired):<5} {flag:<4} {name}")

    print()
    lines = []
    for slice_name in ("boundary", "realistic", "uninstrumented"):
        rows = [(lab, f) for sl, _, lab, f in evaluated if sl == slice_name]
        sc = _scores(rows)
        _fmt(slice_name, sc)
        misses = [n for sl, n, lab, f in evaluated if sl == slice_name and f != lab]
        lines.append(
            f"| {slice_name} | {sc['n']} | {sc['tp']} | {sc['fp']} | {sc['fn']} | {sc['tn']} | "
            f"{'; '.join(misses) or '-'} |"
        )

    if args.report:
        args.report.write_text(
            "# MEMORY_GROWTH_PATTERN: calibration\n\n"
            "Generated by `PYTHONPATH=packages/sdk-py python "
            "scripts/calibrate_memory_growth_pattern.py --report "
            "scripts/calibration/memory_growth_pattern_calibration.md`. Deterministic, offline.\n\n"
            "| slice | n | TP | FP | FN | TN | disagreements |\n|---|---:|---:|---:|---:|---:|---|\n"
            + "\n".join(lines)
            + "\n\n"
            "**boundary** checks the rule against itself (its labels follow the rule), so it "
            "is a regression gate, not evidence of precision. **realistic** labels runs by "
            "whether the agent is misbehaving: the false positives there are the within-run "
            "limitation (a background extraction run, an agent that reads in a later run), "
            "and they are why the detector ships in shadow. **uninstrumented** runs have no "
            "memory events and never fire: tool names are not consulted.\n\n"
            "No production precision estimate exists yet. Measure the fire rate on real "
            "traffic, per agent, before adding it to LIVE_DETECTORS.\n",
            encoding="utf-8",
        )

    boundary_misses = [
        name
        for slice_name, name, label, fired in evaluated
        if slice_name == "boundary" and fired != label
    ]
    if boundary_misses:
        print("boundary regressions:", ", ".join(boundary_misses))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
