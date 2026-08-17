"""
Pure math for Phase 3.3's cross-run state analytics — per-state aggregates
(average / p50 / p95 / total time), daily trends, and outlier detection. No DB,
no I/O; the caller (queries.py::agent_state_analytics) fetches run_state_metrics
rows and hands them here. Mirrors performance_trends.py's "pure math so the
arithmetic is unit-testable without Postgres" convention.

A row is one (run, state) measurement:
    {"run_id": str, "state": str, "total_ms": int, "run_started_at": datetime|None}
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

VALID_WINDOWS = (7, 30, 90)
STATES = ("thinking", "acting", "retrieving", "waiting_approval")


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 100]). Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _median(values: Sequence[float]) -> float:
    return percentile(values, 50)


def aggregate_by_state(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-state summary across all runs: run_count, avg_ms, p50_ms, p95_ms,
    total_ms. Only states that actually appear are included."""
    by_state: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        by_state[r["state"]].append(int(r.get("total_ms", 0)))

    out: Dict[str, Dict[str, Any]] = {}
    for state, durs in by_state.items():
        total = sum(durs)
        out[state] = {
            "run_count": len(durs),
            "total_ms": total,
            "avg_ms": round(total / len(durs)) if durs else 0,
            "p50_ms": round(_median(durs)),
            "p95_ms": round(percentile(durs, 95)),
        }
    return out


def detect_outliers(
    rows: List[Dict[str, Any]], factor: float = 3.0, min_baseline_ms: int = 100
) -> List[Dict[str, Any]]:
    """Runs that spent anomalously long in a state: total_ms > factor × the
    state's median, with the median above a floor (so a state that's normally
    near-zero doesn't flag every run as a 3× outlier on noise). Sorted
    worst-first."""
    by_state: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_state[r["state"]].append(r)

    outliers: List[Dict[str, Any]] = []
    for state, state_rows in by_state.items():
        durs = [int(r.get("total_ms", 0)) for r in state_rows]
        med = _median(durs)
        if med < min_baseline_ms:
            continue
        threshold = med * factor
        for r in state_rows:
            dur = int(r.get("total_ms", 0))
            if dur > threshold:
                outliers.append(
                    {
                        "run_id": r["run_id"],
                        "state": state,
                        "total_ms": dur,
                        "baseline_ms": round(med),
                        "ratio": round(dur / med, 1) if med else None,
                    }
                )
    outliers.sort(key=lambda o: o["total_ms"], reverse=True)
    return outliers


def _bucket_dates(window_days: int, now: Optional[datetime]) -> List[str]:
    now = now or datetime.now(timezone.utc)
    today = now.date()
    return [str(today - timedelta(days=i)) for i in range(window_days - 1, -1, -1)]


def daily_trend(
    rows: List[Dict[str, Any]], window_days: int, now: Optional[datetime] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Per state, an average-ms-per-day series over the window (oldest→newest),
    with zero-filled gaps so the chart x-axis is continuous."""
    buckets = _bucket_dates(window_days, now)
    bucket_set = set(buckets)
    # state -> day -> list of durations
    acc: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        started = r.get("run_started_at")
        if started is None:
            continue
        day = (
            started.date().isoformat()
            if isinstance(started, (datetime, date))
            else str(started)[:10]
        )
        if day in bucket_set:
            acc[r["state"]][day].append(int(r.get("total_ms", 0)))

    trend: Dict[str, List[Dict[str, Any]]] = {}
    for state, per_day in acc.items():
        series = []
        for day in buckets:
            durs = per_day.get(day, [])
            series.append(
                {
                    "date": day,
                    "avg_ms": round(sum(durs) / len(durs)) if durs else 0,
                    "run_count": len(durs),
                }
            )
        trend[state] = series
    return trend


def compute_state_analytics(
    rows: List[Dict[str, Any]],
    window_days: int = 30,
    now: Optional[datetime] = None,
    outlier_factor: float = 3.0,
) -> Dict[str, Any]:
    """Full analytics payload for an agent's run_state_metrics rows."""
    distinct_runs = {r["run_id"] for r in rows}
    return {
        "window_days": window_days,
        "run_count": len(distinct_runs),
        "by_state": aggregate_by_state(rows),
        "trend": daily_trend(rows, window_days, now),
        "outliers": detect_outliers(rows, factor=outlier_factor),
    }
