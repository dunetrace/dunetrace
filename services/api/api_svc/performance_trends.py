"""
Pure math for Phase 4.4's per-agent performance trends — day bucketing,
rate/delta computation, and self-baseline comparison. No DB, no I/O, mirrors
semantic_usage.py's "pure math" convention so aggregation correctness is
unit-testable without a live Postgres (queries.py's other insight functions
have zero test coverage today precisely because their arithmetic lives
inline in SQL/Python mixed with DB calls — this keeps the two apart).

Three independent time ranges are computed by the caller (services/api/
api_svc/db/queries.py::agent_performance_trends) and handed to the functions
here:

  - points              — the selected window_days (7/30/90), daily buckets
  - failure_mode_deltas — current window_days vs the immediately preceding
                          window_days (e.g. last 7 vs previous 7)
  - baseline_comparisons — ALWAYS last 30 days vs this agent's own rate
                          90-30 days ago, independent of window_days. This
                          is deliberately NOT parameterized: if it used
                          window_days as the "current" period too, a 90-day
                          selection would make the current window overlap
                          the 90-30-day-ago baseline reference. Pinning both
                          to fixed values (matching get_agent_health_score's
                          existing baseline windows exactly) avoids that and
                          keeps the baseline stable as you flip between chart
                          windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

VALID_WINDOWS = (7, 30, 90)
MIN_BASELINE_RUNS = 30  # same gate as get_agent_health_score


def build_day_buckets(window_days: int, now: Optional[datetime] = None) -> List[str]:
    """ISO date strings (UTC), oldest -> newest, window_days entries ending today."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    return [str(today - timedelta(days=i)) for i in range(window_days - 1, -1, -1)]


def run_prompt_tokens(responded: Optional[int], called: Optional[int]) -> int:
    """Prompt tokens for one run: the llm.responded total, else the llm.called one.

    OVERRIDE, never add. The same call reports prompt_tokens twice — llm.called
    carries the SDK's chars//4 estimate at call time, llm.responded the exact
    count once the provider reports it (see the run-builder note in CLAUDE.md).
    Summing the two overstated one seeded agent's tokens by 152% here while the
    run-detail page, which already overrides, showed the correct figure for the
    very same runs.

    The fallback is load-bearing, not defensive: direct-SDK traces report tokens
    only on llm.called, LangChain-shaped ones only on llm.responded. Taking
    `responded` unconditionally would zero every direct-SDK agent's cost.
    """
    return int(responded or 0) or int(called or 0)


def compute_cost_by_day(
    token_rows: Sequence[dict], price_fn: "Callable[[str, int, int], float]"
) -> Dict[str, float]:
    """Sum estimated cost per day over per-run token rows.

    ``price_fn`` is explainer_svc.cost.estimate_cost, injected so this stays
    importable and testable without the explainer package. Each row needs
    ``day``, ``model``, ``prompt_tokens_responded``, ``prompt_tokens_called``
    and ``completion_tokens``.
    """
    cost_by_day: Dict[str, float] = {}
    for row in token_rows:
        day = str(row["day"])
        cost = price_fn(
            row.get("model") or "unknown",
            run_prompt_tokens(row.get("prompt_tokens_responded"), row.get("prompt_tokens_called")),
            int(row.get("completion_tokens") or 0),
        )
        cost_by_day[day] = cost_by_day.get(day, 0.0) + cost
    return cost_by_day


def compute_daily_points(
    buckets: Sequence[str],
    runs_by_day: Dict[str, int],
    structural_by_day: Dict[str, int],
    semantic_by_day: Dict[str, int],
    cost_by_day: Dict[str, float],
    latency_by_day: Dict[str, Optional[float]],
) -> List[dict]:
    """Zero-fills every bucket day so a day with no runs shows 0, not a
    missing point. semantic_by_day is every non-'structural' source lumped
    together (Dunetrace's own DeepEval evaluators and external Langfuse/
    LangSmith/Braintrust/custom-push signals alike) — there's no separate
    'external' bucket in this metric, see BACKLOG.md."""
    points = []
    for day in buckets:
        total = runs_by_day.get(day, 0)
        structural = structural_by_day.get(day, 0)
        semantic = semantic_by_day.get(day, 0)
        points.append(
            {
                "day": day,
                "total_runs": total,
                "structural_signal_rate": round(structural / total, 4) if total else 0.0,
                "semantic_signal_rate": round(semantic / total, 4) if total else 0.0,
                "cost_usd": round(cost_by_day.get(day, 0.0), 4),
                "avg_latency_ms": latency_by_day.get(day),
            }
        )
    return points


def compute_failure_mode_deltas(
    current_counts: Dict[str, int],
    previous_counts: Dict[str, int],
    current_total: int,
    previous_total: int,
) -> List[dict]:
    """current_counts/previous_counts: failure_type -> affected_runs within
    each period. Only includes failure types seen in either period. Sorted
    by |delta| descending — the biggest movers first, matching the brief's
    "your Tool Loop rate went from 5% to 12%" framing."""
    failure_types = set(current_counts) | set(previous_counts)
    deltas = []
    for ft in sorted(failure_types):
        cur_affected = current_counts.get(ft, 0)
        prev_affected = previous_counts.get(ft, 0)
        cur_rate = round(cur_affected / current_total, 4) if current_total else 0.0
        prev_rate = round(prev_affected / previous_total, 4) if previous_total else 0.0
        deltas.append(
            {
                "failure_type": ft,
                "current_rate": cur_rate,
                "previous_rate": prev_rate,
                "delta": round(cur_rate - prev_rate, 4),
                "current_affected_runs": cur_affected,
                "previous_affected_runs": prev_affected,
            }
        )
    deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return deltas


def compute_baseline_comparisons(
    current_rates: Dict[str, float],
    baseline_counts: Dict[str, int],
    baseline_total_runs: int,
    min_baseline_runs: int = MIN_BASELINE_RUNS,
) -> List[dict]:
    """current_rates: failure_type -> rate over the fixed last-30-day window.
    baseline_counts: failure_type -> affected_runs over the fixed 90-30-day-
    ago reference window. baseline_total_runs is a single run count for that
    same reference window (shared across every failure type, exactly like
    get_agent_health_score's baseline_sample). ratio is None when the
    baseline rate is 0 (division by zero would otherwise be undefined, and
    "infinitely worse than a 0% baseline" isn't a meaningful number)."""
    baseline_ready = baseline_total_runs >= min_baseline_runs
    comparisons = []
    for ft in sorted(current_rates):
        cur_rate = current_rates[ft]
        baseline_affected = baseline_counts.get(ft, 0)
        baseline_rate = (
            round(baseline_affected / baseline_total_runs, 4) if baseline_total_runs else 0.0
        )
        ratio = round(cur_rate / baseline_rate, 2) if baseline_rate > 0 else None
        comparisons.append(
            {
                "failure_type": ft,
                "current_rate": cur_rate,
                "baseline_rate": baseline_rate,
                "ratio": ratio,
                "baseline_sample_runs": baseline_total_runs,
                "baseline_ready": baseline_ready,
            }
        )
    return comparisons
