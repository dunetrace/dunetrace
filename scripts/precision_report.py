#!/usr/bin/env python3
"""
scripts/precision_report.py

Queries shadow signals and prints a precision report to help decide
which detectors are ready to graduate.

Usage:
    cd dunetrace
    python scripts/precision_report.py

    # Filter to a specific agent
    python scripts/precision_report.py --agent my-agent

    # Show raw signal evidence for a specific detector
    python scripts/precision_report.py --inspect TOOL_LOOP

    # Show how many signals each detector would fire if graduated
    python scripts/precision_report.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://dunetrace:dunetrace@localhost:5432/dunetrace",
)

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
SEVERITY_COLOR = {
    "CRITICAL": "\033[91m",  # red
    "HIGH":     "\033[93m",  # yellow
    "MEDIUM":   "\033[94m",  # blue
    "LOW":      "\033[92m",  # green
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def connect():
    return psycopg2.connect(DATABASE_URL)


def colored(text, color):
    return f"{color}{text}{RESET}"


def fmt_ts(ts) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "timestamp"):
        ts = ts.timestamp()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_confidence(c: float) -> str:
    pct = int(c * 100)
    if pct >= 90:
        color = "\033[91m"
    elif pct >= 75:
        color = "\033[93m"
    else:
        color = "\033[92m"
    return colored(f"{pct}%", color)


# ── Summary view ───────────────────────────────────────────────────────────────

def print_summary(cur, agent_filter: str | None):
    agent_clause = "AND agent_id = %s" if agent_filter else ""
    params = [agent_filter] if agent_filter else []

    cur.execute(f"""
        SELECT
            failure_type,
            COUNT(*)                                          AS total,
            COUNT(*) FILTER (WHERE severity = 'CRITICAL')    AS critical,
            COUNT(*) FILTER (WHERE severity = 'HIGH')        AS high,
            COUNT(*) FILTER (WHERE severity = 'MEDIUM')      AS medium,
            COUNT(*) FILTER (WHERE severity = 'LOW')         AS low,
            ROUND(AVG(confidence)::numeric, 2)               AS avg_conf,
            COUNT(DISTINCT run_id)                           AS runs_affected,
            MAX(detected_at)                                 AS last_seen
        FROM failure_signals
        WHERE shadow = TRUE
        {agent_clause}
        GROUP BY failure_type
        ORDER BY total DESC
    """, params)

    rows = cur.fetchall()

    if not rows:
        print("No shadow signals found.")
        if agent_filter:
            print(f"(filtered to agent: {agent_filter})")
        return

    print(f"\n{BOLD}Shadow Signal Summary{RESET}" +
          (f"  {DIM}agent: {agent_filter}{RESET}" if agent_filter else ""))
    print(f"{DIM}{'─' * 90}{RESET}")
    print(f"{'DETECTOR':<32} {'TOTAL':>6} {'CRIT':>5} {'HIGH':>5} {'MED':>5} {'LOW':>5}  {'AVG CONF':>9}  {'RUNS':>5}  LAST SEEN")
    print(f"{DIM}{'─' * 90}{RESET}")

    for r in rows:
        last = fmt_ts(r["last_seen"])
        conf = fmt_confidence(float(r["avg_conf"]))
        crit_str = colored(str(r["critical"]), "\033[91m") if r["critical"] else DIM + "0" + RESET
        high_str = colored(str(r["high"]),     "\033[93m") if r["high"]     else DIM + "0" + RESET
        print(
            f"{r['failure_type']:<32} {r['total']:>6} {crit_str:>14} {high_str:>14} "
            f"{r['medium']:>5} {r['low']:>5}  {conf:>18}  {r['runs_affected']:>5}  {DIM}{last}{RESET}"
        )

    print(f"{DIM}{'─' * 90}{RESET}")
    total_signals = sum(r["total"] for r in rows)
    total_runs    = len(set())
    print(f"{BOLD}{total_signals} total shadow signals across {len(rows)} detector type(s){RESET}")
    print()

    # Graduation recommendations
    print(f"{BOLD}Graduation readiness:{RESET}")
    for r in rows:
        conf = float(r["avg_conf"])
        total = r["total"]
        if total < 5:
            status = colored("⚠  Too few samples (<5) — keep monitoring", "\033[93m")
        elif conf >= 0.85:
            status = colored("✓  Ready to graduate (avg conf ≥ 85%)", "\033[92m")
        elif conf >= 0.70:
            status = colored("~  Borderline — spot-check before graduating", "\033[94m")
        else:
            status = colored("✗  Not ready (avg conf < 70%) — review evidence", "\033[91m")
        print(f"  {r['failure_type']:<32} {status}")
    print()


# ── Recent signals view ────────────────────────────────────────────────────────

def print_recent(cur, agent_filter: str | None, limit: int = 20):
    agent_clause = "AND s.agent_id = %s" if agent_filter else ""
    params = [agent_filter] if agent_filter else []

    cur.execute(f"""
        SELECT
            s.id, s.failure_type, s.severity, s.run_id,
            s.agent_id, s.step_index, s.confidence,
            s.detected_at, s.evidence,
            p.trigger AS exit_reason
        FROM failure_signals s
        LEFT JOIN processed_runs p ON p.run_id = s.run_id
        WHERE s.shadow = TRUE
        {agent_clause}
        ORDER BY s.detected_at DESC
        LIMIT %s
    """, params + [limit])

    rows = cur.fetchall()

    if not rows:
        print("No shadow signals found.")
        return

    print(f"\n{BOLD}Recent Shadow Signals{RESET} {DIM}(latest {limit}){RESET}")
    print(f"{DIM}{'─' * 100}{RESET}")

    for r in rows:
        sev_color = SEVERITY_COLOR.get(r["severity"], "")
        evidence = r["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)

        # One-line evidence summary
        ev_parts = []
        if "tool" in evidence:
            ev_parts.append(f"tool={evidence['tool']}")
        if "count" in evidence:
            ev_parts.append(f"count={evidence['count']}")
        if "tool_a" in evidence:
            ev_parts.append(f"{evidence['tool_a']}↔{evidence['tool_b']}")
        if "matched_patterns" in evidence:
            ev_parts.append(f"patterns={evidence['matched_patterns'][:2]}")
        if "index_name" in evidence:
            ev_parts.append(f"index={evidence['index_name']}")
        if "available_tools" in evidence:
            ev_parts.append(f"tools={evidence['available_tools']}")
        ev_str = "  ".join(ev_parts) if ev_parts else str(evidence)[:60]

        sev_str = f"{r['severity']:>8}"
        print(
            f"  {DIM}#{r['id']:>6}{RESET}  "
            f"{colored(sev_str, sev_color)}  "
            f"{r['failure_type']:<32}  "
            f"conf={fmt_confidence(r['confidence'])}  "
            f"step={r['step_index']:>3}  "
            f"{DIM}{r['run_id'][:16]}…{RESET}"
        )
        print(f"          {DIM}exit={r['exit_reason'] or '?':<12}  {ev_str}{RESET}")

    print()


# ── Inspect a specific detector ────────────────────────────────────────────────

def print_inspect(cur, failure_type: str, agent_filter: str | None, limit: int = 10):
    agent_clause = "AND agent_id = %s" if agent_filter else ""
    params = [failure_type.upper()]
    if agent_filter:
        params.append(agent_filter)

    cur.execute(f"""
        SELECT id, severity, run_id, agent_id, step_index,
               confidence, detected_at, evidence
        FROM failure_signals
        WHERE shadow = TRUE
        AND failure_type = %s
        {agent_clause}
        ORDER BY detected_at DESC
        LIMIT %s
    """, params + [limit])

    rows = cur.fetchall()

    if not rows:
        print(f"No shadow signals found for {failure_type}.")
        return

    print(f"\n{BOLD}Inspecting: {failure_type}{RESET}  {DIM}(latest {len(rows)}){RESET}")
    print(f"{DIM}Run through these manually and mark each as TP (true positive) or FP (false positive){RESET}")
    print(f"{DIM}If ≥80% are TP → ready to graduate.{RESET}\n")

    for i, r in enumerate(rows, 1):
        evidence = r["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)

        sev_color = SEVERITY_COLOR.get(r["severity"], "")
        print(f"{BOLD}[{i}/{len(rows)}] Signal #{r['id']}{RESET}")
        print(f"  Severity:   {colored(r['severity'], sev_color)}")
        print(f"  Confidence: {fmt_confidence(r['confidence'])}")
        print(f"  Run ID:     {r['run_id']}")
        print(f"  Agent:      {r['agent_id']}")
        print(f"  Step:       {r['step_index']}")
        print(f"  Detected:   {fmt_ts(r['detected_at'])}")
        print(f"  Evidence:")
        for k, v in evidence.items():
            print(f"    {k}: {v}")
        print()

    print(f"{DIM}To see the full run events for any signal:{RESET}")
    print(f"  curl -s -H 'Authorization: Bearer dt_dev_test' \\")
    print(f"    http://localhost:8002/v1/runs/<run_id> | python -m json.tool")
    print()

    # Graduate hint
    print(f"{BOLD}To graduate this detector:{RESET}")
    print(f"  Edit services/detector/detector_svc/db.py:")
    print(f'  LIVE_DETECTORS = {{"{failure_type}"}}')
    print(f"  Then restart: PYTHONPATH=.:../../packages/sdk-py SHADOW_MODE=false python -m detector_svc.worker")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dunetrace precision report — evaluate shadow signals before graduating detectors"
    )
    parser.add_argument("--agent",   help="Filter to a specific agent_id")
    parser.add_argument("--inspect", help="Show detailed evidence for a specific detector type (e.g. TOOL_LOOP)")
    parser.add_argument("--recent",  action="store_true", help="Show recent individual signals")
    parser.add_argument("--limit",   type=int, default=20, help="Max signals to show (default 20)")
    args = parser.parse_args()

    try:
        conn = connect()
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        print(f"DATABASE_URL: {DATABASE_URL}")
        sys.exit(1)

    print_summary(cur, args.agent)

    if args.inspect:
        print_inspect(cur, args.inspect, args.agent, args.limit)
    elif args.recent:
        print_recent(cur, args.agent, args.limit)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
