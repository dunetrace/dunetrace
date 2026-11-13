"""
Per-agent P75 baselines — the SQL the detector worker and the ingest service
share so the two cannot disagree about what "normal" means for an agent.

The detector (``detector_svc/db.py::fetch_metric_baseline``) reads one column
of ``run_baseline_metrics`` per adaptive detector; ingest's
``GET /v1/detector-config`` serves every column in one round trip so the SDK's
client-side pass can use the same thresholds. Both build their statements
here. This module holds SQL *strings* only — no asyncpg, no pool — so it costs
``dunetrace_schemas`` nothing to import.

Placeholder contract (positional, fixed): ``$1`` org_id, ``$2`` agent_id,
``$3`` agent_version, ``$4`` run_id to exclude (the run being detected; pass
``""`` when there is none), ``$5`` lookback (how many recent clean runs to
sample).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# Minimum completed runs required before a P75 baseline is considered reliable.
# Below this every baseline is None and detectors fall back to their static
# defaults — a P75 from fewer runs is too sensitive to single-run outliers.
MIN_BASELINE_RUNS = 20

# How many recent clean runs (that carry the metric) feed one percentile.
BASELINE_LOOKBACK = 50

# Columns a baseline may aggregate. A closed set, not a formatting convenience:
# the column name is interpolated into SQL (asyncpg cannot bind an identifier),
# so anything outside this tuple must never reach the f-strings below.
BASELINE_COLUMNS: tuple[str, ...] = (
    "step_count",
    "gap_p75_tool_ms",
    "gap_p75_llm_ms",
    "token_growth",
    "llm_tool_ratio",
    "total_tokens",
    "duration_s",
)

# Column -> the key GET /v1/detector-config reports it under. The response
# names follow the detector each baseline feeds (STEP_COUNT_INFLATION,
# SLOW_STEP tool/llm, CONTEXT_BLOAT, REASONING_STALL, COST_SPIKE,
# SESSION_LATENCY) rather than the storage column.
BASELINE_RESPONSE_KEYS: dict[str, str] = {
    "step_count": "p75_steps",
    "gap_p75_tool_ms": "p75_latency_tool_ms",
    "gap_p75_llm_ms": "p75_latency_llm_ms",
    "token_growth": "p75_token_growth",
    "llm_tool_ratio": "p75_llm_tool_ratio",
    "total_tokens": "p75_total_tokens",
    "duration_s": "p75_duration_s",
}


# Per-tool SLOW_STEP latency. Not one of BASELINE_COLUMNS: it is a map
# {tool_name: P75 ms}, stored per run in the JSONB column below and reported
# under its own response key, so the scalar machinery above stays scalar.
PER_TOOL_LATENCY_COLUMN = "gap_p75_by_tool_ms"
PER_TOOL_LATENCY_RESPONSE_KEY = "p75_latency_by_tool_ms"

# Upper bound on tools per baseline map — both per stored run and per response.
# An agent with dynamically named tools would otherwise grow an unbounded JSONB
# row per run and an unbounded /v1/detector-config payload. Tools past the cap
# fall back to the agent-wide tool baseline, which is the pre-per-tool
# behaviour, not a loss of detection.
MAX_TOOLS_PER_BASELINE = 64


def per_tool_latency_baselines_sql() -> str:
    """Per-tool P75 of each run's own per-tool P75 step latency.

    Same placeholder contract as the scalar baselines (``$1``-``$5``) plus
    ``$6`` min_runs and ``$7`` the tool cap. Returns one row per tool with at
    least `$6` samples: ``tool_name``, ``sample_size``, ``p75``. The window is
    the newest `$5` clean runs that recorded any per-tool latency, so a tool
    needs to appear in `$6` of those to get a baseline of its own; rarer tools
    fall back to the agent-wide tool baseline.

    P75-of-per-run-P75s, like the stored scalar ``gap_p75_tool_ms``: one long
    run calling a tool 200 times contributes one sample, not 200.
    """
    return f"""
            WITH recent AS (
                SELECT {PER_TOOL_LATENCY_COLUMN} AS m
                FROM run_baseline_metrics
                WHERE org_id        = $1
                  AND agent_id      = $2
                  AND agent_version = $3
                  AND run_id       != $4
                  AND clean
                  AND {PER_TOOL_LATENCY_COLUMN} IS NOT NULL
                ORDER BY recorded_at DESC
                LIMIT $5
            ),
            per_tool AS (
                SELECT kv.key AS tool_name, kv.value::double precision AS v
                FROM recent, jsonb_each_text(recent.m) AS kv
                WHERE jsonb_typeof(recent.m -> kv.key) = 'number'
            )
            SELECT tool_name,
                   COUNT(*)                                          AS sample_size,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY v)   AS p75
            FROM per_tool
            GROUP BY tool_name
            HAVING COUNT(*) >= $6
            ORDER BY COUNT(*) DESC, tool_name
            LIMIT $7
            """


def per_tool_from_rows(rows: Any) -> Optional[dict[str, float]]:
    """Rows of per_tool_latency_baselines_sql() -> {tool_name: p75_ms}, or None
    when no tool has enough history."""
    out = {
        str(r["tool_name"]): float(r["p75"])
        for r in rows or ()
        if r["tool_name"] is not None and r["p75"] is not None
    }
    return out or None


def _check_column(column: str) -> None:
    if column not in BASELINE_COLUMNS:
        raise ValueError(f"{column!r} is not a baseline column: {BASELINE_COLUMNS}")


def recent_clean_metric_cte(column: str, alias: str) -> str:
    """The run-selection body every baseline shares: the newest `$5` clean
    runs of this agent+version that carry `column`, excluding `$4`.

    Note the LIMIT applies to runs that HAVE the metric — this yields up to
    `lookback` usable samples rather than however many of the last `lookback`
    runs happened to qualify. `clean` is a run that fired no LIVE signal; a
    misbehaving run cannot also define what normal looks like (shadow signals
    are deliberately not disqualifying — see detector_svc/db.py).
    """
    _check_column(column)
    return f"""{alias} AS (
                SELECT {column} AS v
                FROM run_baseline_metrics
                WHERE org_id        = $1
                  AND agent_id      = $2
                  AND agent_version = $3
                  AND run_id       != $4
                  AND clean
                  AND {column} IS NOT NULL
                ORDER BY recorded_at DESC
                LIMIT $5
            )"""


def metric_baseline_sql(column: str) -> str:
    """One column's sample size and P75. Returns a single row with
    ``sample_size`` and ``p75`` (NULL when there are no samples)."""
    return f"""
            WITH {recent_clean_metric_cte(column, "recent")}
            SELECT COUNT(*)                                          AS sample_size,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY v)   AS p75
            FROM recent
            """


def all_metric_baselines_sql() -> str:
    """Every baseline column in one statement, one row: ``<column>_n`` and
    ``<column>_p75`` for each entry of BASELINE_COLUMNS. Each column samples
    its own newest-`$5` window, exactly as the per-column query does, so the
    numbers match what the detector would compute one column at a time."""
    ctes = ",\n            ".join(recent_clean_metric_cte(c, f"r_{c}") for c in BASELINE_COLUMNS)
    selects = ",\n                   ".join(
        f"(SELECT COUNT(*) FROM r_{c}) AS {c}_n,\n                   "
        f"(SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY v) FROM r_{c}) AS {c}_p75"
        for c in BASELINE_COLUMNS
    )
    return f"""
            WITH {ctes}
            SELECT {selects}
            """


def latest_agent_version_sql() -> str:
    """The agent_version of the agent's most recently recorded run — what a
    caller uses when it has no version of its own to ask about. ``$1`` org_id,
    ``$2`` agent_id."""
    return """
            SELECT agent_version
            FROM run_baseline_metrics
            WHERE org_id = $1 AND agent_id = $2
            ORDER BY recorded_at DESC
            LIMIT 1
            """


def baselines_from_row(
    row: Optional[Mapping[str, Any]],
    min_runs: int = MIN_BASELINE_RUNS,
) -> Optional[dict[str, Optional[float]]]:
    """Turn an all_metric_baselines_sql() row into the response mapping.

    A column with fewer than `min_runs` samples reports None, independently of
    the others — one run can feed the token baseline while sitting out the
    growth one, so the sample sizes differ per column and each is judged on its
    own, as the detector does. Returns None (not a dict of Nones) when no
    column has enough history: "insufficient history" as a single value the
    SDK can test for.
    """
    if row is None:
        return None
    out: dict[str, Optional[float]] = {}
    for column, key in BASELINE_RESPONSE_KEYS.items():
        n = row.get(f"{column}_n") or 0
        p75 = row.get(f"{column}_p75")
        out[key] = float(p75) if (n >= min_runs and p75 is not None) else None
    if all(v is None for v in out.values()):
        return None
    return out
