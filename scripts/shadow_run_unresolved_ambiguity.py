#!/usr/bin/env python3
"""
UNRESOLVED_AMBIGUITY shadow-mode exit data.

Runs the real detector over every recorded run corpus in ../dunetrace-demos and
reports fire rate per 100 runs, split by tier. No LLM, no network, no database —
each corpus is replayed through the canonical run_builder and handed to
`UnresolvedAmbiguityDetector.on_run_completion`.

This is NOT a threshold sweep like scripts/calibrate_*.py. This detector has no
threshold to sweep: its verdict is a set operation on tokens, and the only
operator-facing knob (`irreversible_tools`) is a *declaration*, not a tuning
parameter. What needs measuring before promotion is the base RATE on real
traffic, which is what this prints.

Because the detector is inert until an operator declares which tools are
irreversible, this script has to make that declaration on the corpora's behalf.
The per-corpus lists below are deliberately GENEROUS — every tool whose name
suggests a state change the customer would notice. Being generous is the
conservative choice here: it can only over-report, so a low rate measured this
way is a real result and a high one is an upper bound.

    python scripts/shadow_run_unresolved_ambiguity.py
    python scripts/shadow_run_unresolved_ambiguity.py --demos ~/dunetrace-demos
    python scripts/shadow_run_unresolved_ambiguity.py --verbose
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "sdk-py"))

from dunetrace.detectors import (  # noqa: E402
    UnresolvedAmbiguityDetector,
    _parse_tool_args,
    _ua_identifiers,
    _ua_record_lists,
    _ua_scalars,
    _ua_targeted,
)
from dunetrace.run_builder import build_run_state  # noqa: E402

# Every tool in these corpora that mutates something a customer would notice.
# Read-only lookups (lookup_customer, search_knowledge_base, fetch_orders,
# get_billing_summary) are deliberately absent — declaring a lookup irreversible
# would be a category error, not a conservative choice.
IRREVERSIBLE_TOOLS = [
    "delete_customer",
    "refund_customer",
    "issue_refund",
    "send_email",
    "cancel_subscription",
    "cancel_order",
    "update_customer",
]


def load_jsonl_runs(path: pathlib.Path) -> list[list[dict]]:
    """One file == one run's event stream."""
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [events] if events else []


def load_trace_json(path: pathlib.Path) -> list[list[dict]]:
    """demo1's backup traces: {"scenario": ..., "events": [...]}."""
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    events = blob.get("events") if isinstance(blob, dict) else None
    return [events] if isinstance(events, list) and events else []


def load_recording(path: pathlib.Path) -> list[list[dict]]:
    """demo4 recordings/run.json — may hold one run or a list of them."""
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(blob, dict):
        for key in ("events", "dunetrace_events"):
            if isinstance(blob.get(key), list) and blob[key]:
                return [blob[key]]
        return []
    if isinstance(blob, list) and blob and isinstance(blob[0], dict) and "event_type" in blob[0]:
        return [blob]
    return []


CORPORA = [
    ("demo4/runs", "*.events.jsonl", load_jsonl_runs),
    ("demo4/runs", "*.json", load_recording),
    ("demo4/recordings", "**/run.json", load_recording),
    ("demo1/backup-traces", "**/trace_*.json", load_trace_json),
    ("demo1", "demo_tool_log.jsonl", load_jsonl_runs),
    ("demo2", "*.jsonl", load_jsonl_runs),
    ("demo/corpus", "*.jsonl", load_jsonl_runs),
]


def positive_control() -> list[str]:
    """Replay the worked example through the SAME path the corpora take.

    A fire rate of zero is only informative if the harness could have reported a
    non-zero one. This pushes the Chen case through build_run_state() — raw event
    dicts, exactly as a corpus file supplies them — so a zero above is a
    statement about the corpora, not about whether the pipeline works.
    """
    chens = [
        {"id": "CUST_8834", "name": "Sarah Chen", "plan": "Enterprise", "annual_value_eur": 340000},
        {"id": "CUST_1183", "name": "Emily Chen", "plan": "Standard", "annual_value_eur": 3000},
    ]
    base = {"run_id": "control-1", "agent_id": "control", "agent_version": "v1"}
    events = [
        {
            **base,
            "event_type": "run.started",
            "step_index": 0,
            "timestamp": 0.0,
            "payload": {"input_text": "close the account for the Chen family", "tools": []},
        },
        {
            **base,
            "event_type": "tool.called",
            "step_index": 1,
            "timestamp": 1.0,
            "payload": {"tool_name": "lookup_customer", "args": json.dumps({"name": "Chen"})},
        },
        {
            **base,
            "event_type": "tool.responded",
            "step_index": 1,
            "timestamp": 2.0,
            "payload": {
                "tool_name": "lookup_customer",
                "success": True,
                "output": json.dumps({"customers": chens}),
            },
        },
        {
            **base,
            "event_type": "tool.called",
            "step_index": 2,
            "timestamp": 3.0,
            "payload": {
                "tool_name": "delete_customer",
                "args": json.dumps({"customer_id": "CUST_1183"}),
            },
        },
        {**base, "event_type": "run.completed", "step_index": 3, "timestamp": 4.0, "payload": {}},
    ]
    det = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE_TOOLS)
    weak = det.on_run_completion(build_run_state(events))

    named = [dict(e) for e in events]
    named[0]["payload"] = dict(named[0]["payload"], input_text="close Sarah Chen's account")
    strong = det.on_run_completion(build_run_state(named))

    out = []
    for label, sig in (("ambiguous request", weak), ("request names the sibling", strong)):
        if sig is None:
            out.append(f"FAIL — {label}: no signal (the harness cannot detect anything)")
        else:
            ev = sig.evidence
            out.append(
                f"{ev['tier']:>6}  {label}: {sig.severity.value} conf={sig.confidence} "
                f"selected={ev['selected_id']} of {ev['candidate_count']} "
                f"unused={ev['discriminators_unused']}"
                + (f" matched={ev['sibling_matched_in_request']}" if ev["tier"] == "strong" else "")
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=str(pathlib.Path.home() / "dunetrace-demos"))
    ap.add_argument("--verbose", action="store_true", help="print every signal")
    args = ap.parse_args()

    demos = pathlib.Path(args.demos).expanduser()
    if not demos.is_dir():
        print(f"No demos directory at {demos}", file=sys.stderr)
        return 2

    detector = UnresolvedAmbiguityDetector(IRREVERSIBLE_TOOLS=IRREVERSIBLE_TOOLS)

    seen_runs: set[str] = set()
    per_corpus: dict[str, Counter] = defaultdict(Counter)
    signals: list = []
    # Why each run could not fire, so a rate of zero is EXPLAINED rather than
    # merely observed. The stages mirror the detector's own order, and each run
    # is charged to the first stage it fails.
    reasons: Counter = Counter()
    declared = {t.casefold() for t in IRREVERSIBLE_TOOLS}

    def classify(state) -> str:
        """The furthest stage this run reaches. "evaluated" means the detector
        genuinely had a multi-record candidate set containing the id the call
        used, and returned a verdict on it. Mirrors the detector's own order so
        a rate of zero can be attributed to a stage rather than guessed at."""
        acting = [
            tc
            for tc in state.tool_calls
            if tc.tool_name.casefold() in declared and tc.success is not False
        ]
        if not acting:
            return "no irreversible tool call"
        if not any(tc.output for tc in state.tool_calls):
            return "irreversible call, but no tool output captured anywhere in the run"
        sets_seen = False
        for tc in acting:
            args = detector._arg_values(tc.args)
            for prior in state.tool_calls:
                if prior.step_index >= tc.step_index or not prior.output:
                    continue
                parsed = _parse_tool_args(prior.output)
                if parsed is None:
                    continue
                for members in _ua_record_lists(parsed, detector.MAX_DEPTH):
                    if len(members) < detector.MIN_CANDIDATES:
                        continue
                    sets_seen = True
                    values = []
                    for m in members:
                        leaves: list[str] = []
                        _ua_scalars(m, detector.MAX_DEPTH, leaves)
                        values.append(leaves)
                    narrow = [_ua_identifiers(m, v) for m, v in zip(members, values)]
                    if _ua_targeted(args, narrow) or _ua_targeted(args, values):
                        return "evaluated"
        if not sets_seen:
            return "no prior result held >= 2 records — nothing to choose between"
        return "multi-record set present, but the id acted on came from somewhere else"

    for rel, glob, loader in CORPORA:
        root = demos / rel
        if not root.is_dir():
            continue
        label = rel
        for path in sorted(root.glob(glob)):
            for events in loader(path):
                try:
                    state = build_run_state(events)
                except Exception:
                    per_corpus[label]["unparseable"] += 1
                    continue
                if state.run_id in seen_runs:
                    continue
                seen_runs.add(state.run_id)
                per_corpus[label]["runs"] += 1

                stage = classify(state)
                reasons[stage] += 1
                if stage == "evaluated":
                    per_corpus[label]["evaluated"] += 1

                signal = detector.on_run_completion(state)
                if signal is None:
                    continue
                tier = signal.evidence.get("tier", "?")
                per_corpus[label][tier] += 1
                signals.append((label, state.run_id, state.agent_id, signal))

    total_runs = sum(c["runs"] for c in per_corpus.values())
    if not total_runs:
        print("No runs found.", file=sys.stderr)
        return 2

    strong = sum(c["strong"] for c in per_corpus.values())
    weak = sum(c["weak"] for c in per_corpus.values())
    evaluated = sum(c["evaluated"] for c in per_corpus.values())

    print(f"UNRESOLVED_AMBIGUITY shadow run — {total_runs} distinct recorded runs")
    print(f"declared irreversible_tools: {', '.join(IRREVERSIBLE_TOOLS)}\n")
    print(f"{'corpus':<26} {'runs':>6} {'evaluated':>10} {'strong':>7} {'weak':>6}")
    for label in sorted(per_corpus):
        c = per_corpus[label]
        if not c["runs"]:
            continue
        print(f"{label:<26} {c['runs']:>6} {c['evaluated']:>10} {c['strong']:>7} {c['weak']:>6}")
    print(f"{'TOTAL':<26} {total_runs:>6} {evaluated:>10} {strong:>7} {weak:>6}\n")

    per100 = lambda n: n * 100.0 / total_runs  # noqa: E731
    print(
        f"signals per 100 runs — strong: {per100(strong):.2f}   weak: {per100(weak):.2f}   "
        f"total: {per100(strong + weak):.2f}"
    )
    if evaluated:
        print(
            f"per 100 EVALUATED runs — strong: {strong * 100.0 / evaluated:.2f}   "
            f"weak: {weak * 100.0 / evaluated:.2f}"
        )
    else:
        print("per 100 EVALUATED runs — n/a: no recorded run reached a candidate set")
    print()

    print("how far each run got (charged to the first stage it failed):")
    for reason, n in reasons.most_common():
        print(f"  {n:>4}  {reason}")

    print()
    print("positive control (synthetic, NOT part of the rate above):")
    for line in positive_control():
        print(f"  {line}")

    if args.verbose and signals:
        print("\nsignals:")
        for label, run_id, agent_id, signal in signals:
            ev = signal.evidence
            print(
                f"  [{ev['tier']:>6}] {label} {run_id[:8]} {agent_id} "
                f"{ev['tool_name']}({ev['selected_id']}) of {ev['candidate_count']} "
                f"unused={ev['discriminators_unused'][:6]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
