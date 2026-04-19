"""
Database queries for the customer API. Reads from events, failure_signals,
processed_runs, and api_keys. This service never writes.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

from api_svc.config import settings
from dunetrace.models import FailureSignal, FailureType, Severity
from explainer_svc.explainer import explain

logger = logging.getLogger("dunetrace.api.db")
_pool = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

async def init_pool() -> None:
    global _pool
    if asyncpg is None:
        return
    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=15,
    )
    logger.info("DB pool ready")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def check_db() -> str:
    if not _pool:
        return "no_pool"
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return "ok"
    except Exception as exc:
        return str(exc)


async def verify_api_key(key: str) -> Optional[str]:
    """Returns customer_id if valid, None otherwise. Dev mode accepts anything."""
    if settings.is_dev:
        return "dev_customer"
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT customer_id FROM api_keys WHERE key = $1 AND active = TRUE",
            key,
        )
    return row["customer_id"] if row else None


# ── Agents ────────────────────────────────────────────────────────────────────

async def list_agents(customer_id: str, offset: int, limit: int) -> tuple[list, int]:
    """Returns (rows, total_count). Each row has: agent_id, last_seen, run_count, signal_count, critical_count, high_count."""
    if not _pool:
        return [], 0

    async with _pool.acquire() as conn:
        total = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT agent_id) FROM events
            WHERE ($1 = 'dev_customer' OR agent_id IN (
                SELECT agent_id FROM api_keys WHERE customer_id = $1 AND active = TRUE
            ))
            """,
            customer_id,
        )

        rows = await conn.fetch(
            """
            SELECT
                e.agent_id,
                MAX(e.received_at)                                          AS last_seen,
                COUNT(DISTINCT e.run_id)                                    AS run_count,
                COUNT(DISTINCT s.id) FILTER (WHERE s.shadow = FALSE)        AS signal_count,
                COUNT(DISTINCT s.id) FILTER (
                    WHERE s.shadow = FALSE AND s.severity = 'CRITICAL'
                )                                                            AS critical_count,
                COUNT(DISTINCT s.id) FILTER (
                    WHERE s.shadow = FALSE AND s.severity = 'HIGH'
                )                                                            AS high_count
            FROM events e
            LEFT JOIN failure_signals s ON s.agent_id = e.agent_id
            WHERE ($1 = 'dev_customer' OR e.agent_id IN (
                SELECT agent_id FROM api_keys WHERE customer_id = $1 AND active = TRUE
            ))
            GROUP BY e.agent_id
            ORDER BY MAX(e.received_at) DESC
            LIMIT $2 OFFSET $3
            """,
            customer_id, limit, offset,
        )

    return [dict(r) for r in rows], total or 0


# ── Failure type breakdown ────────────────────────────────────────────────────

async def agent_failure_type_counts(customer_id: str) -> dict:
    """Live signal counts per agent broken down by failure type: { agent_id: { "TOOL_LOOP": 3, ... } }."""
    if not _pool:
        return {}

    from collections import defaultdict

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT agent_id, failure_type, COUNT(*) AS cnt
            FROM failure_signals
            WHERE shadow = FALSE
              AND ($1 = 'dev_customer' OR agent_id IN (
                  SELECT agent_id FROM api_keys WHERE customer_id = $1 AND active = TRUE
              ))
            GROUP BY agent_id, failure_type
            """,
            customer_id,
        )

    result: dict = defaultdict(dict)
    for r in rows:
        result[r["agent_id"]][r["failure_type"]] = int(r["cnt"])
    return dict(result)


# ── Sparklines ────────────────────────────────────────────────────────────────

async def agent_signal_sparklines(customer_id: str) -> dict:
    """7-day daily live signal counts per agent, oldest→newest: { agent_id: [day-6, ..., today] }."""
    if not _pool:
        return {}

    import datetime
    from collections import defaultdict

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                agent_id,
                DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC') AS day,
                COUNT(*)                                            AS cnt
            FROM failure_signals
            WHERE shadow = FALSE
              AND detected_at >= NOW() - INTERVAL '7 days'
              AND ($1 = 'dev_customer' OR agent_id IN (
                  SELECT agent_id FROM api_keys WHERE customer_id = $1 AND active = TRUE
              ))
            GROUP BY agent_id, day
            ORDER BY agent_id, day
            """,
            customer_id,
        )

    # Build map: agent_id → {date: count}
    day_counts: dict = defaultdict(dict)
    for r in rows:
        d = r["day"].date() if hasattr(r["day"], "date") else r["day"]
        day_counts[r["agent_id"]][d] = int(r["cnt"])

    # Produce exactly 7 values per agent, oldest first
    today = datetime.date.today()
    return {
        agent_id: [
            day_counts[agent_id].get(today - datetime.timedelta(days=offset), 0)
            for offset in range(6, -1, -1)   # 6 days ago … today
        ]
        for agent_id in day_counts
    }


# ── Runs ──────────────────────────────────────────────────────────────────────

async def list_runs(
    agent_id: str,
    offset: int,
    limit: int,
    has_signals: Optional[bool] = None,
) -> tuple[list, int]:
    """List runs for an agent. Optionally filter to only runs with signals."""
    if not _pool:
        return [], 0

    signal_filter = ""
    if has_signals is True:
        signal_filter = "AND EXISTS (SELECT 1 FROM failure_signals s WHERE s.run_id = pr.run_id AND s.shadow = FALSE)"
    elif has_signals is False:
        signal_filter = "AND NOT EXISTS (SELECT 1 FROM failure_signals s WHERE s.run_id = pr.run_id AND s.shadow = FALSE)"

    async with _pool.acquire() as conn:
        total = await conn.fetchval(
            f"""
            SELECT COUNT(*) FROM processed_runs pr
            WHERE pr.agent_id = $1
            {signal_filter}
            """,
            agent_id,
        )

        rows = await conn.fetch(
            f"""
            SELECT
                pr.run_id,
                pr.agent_id,
                pr.agent_version,
                pr.trigger                                              AS exit_reason,
                pr.processed_at,
                -- started_at from SDK timestamp on run.started event
                (SELECT e.timestamp FROM events e
                 WHERE e.run_id = pr.run_id AND e.event_type = 'run.started'
                 LIMIT 1) AS started_at,
                -- completed_at from SDK timestamp on terminal event
                (SELECT e.timestamp FROM events e
                 WHERE e.run_id = pr.run_id AND e.event_type IN ('run.completed', 'run.errored')
                 LIMIT 1) AS completed_at,
                -- step_count
                (SELECT MAX(e.step_index) FROM events e WHERE e.run_id = pr.run_id) AS step_count,
                -- live signal count
                (SELECT COUNT(*) FROM failure_signals s
                 WHERE s.run_id = pr.run_id AND s.shadow = FALSE)      AS signal_count
            FROM processed_runs pr
            WHERE pr.agent_id = $1
            {signal_filter}
            ORDER BY pr.processed_at DESC
            LIMIT $2 OFFSET $3
            """,
            agent_id, limit, offset,
        )

    return [dict(r) for r in rows], total or 0


async def get_run_detail(run_id: str) -> Optional[dict]:
    """Full run detail: metadata + events + signals with explanations."""
    if not _pool:
        return None

    import json

    async with _pool.acquire() as conn:
        pr = await conn.fetchrow(
            "SELECT run_id, agent_id, agent_version, trigger, processed_at FROM processed_runs WHERE run_id = $1",
            run_id,
        )
        if not pr:
            return None

        events = await conn.fetch(
            """
            SELECT event_type, step_index, timestamp, payload, parent_run_id
            FROM events WHERE run_id = $1
            ORDER BY step_index ASC, timestamp ASC
            """,
            run_id,
        )

        signals = await conn.fetch(
            """
            SELECT id, failure_type, severity, step_index, confidence,
                   detected_at, evidence
            FROM failure_signals
            WHERE run_id = $1 AND shadow = FALSE
            ORDER BY step_index ASC
            """,
            run_id,
        )

    started_at = next(
        (e["timestamp"] for e in events if e["event_type"] == "run.started"), None
    )
    completed_at = next(
        (e["timestamp"] for e in events if e["event_type"] in ("run.completed", "run.errored")), None
    )

    # Build event list
    event_list = []
    for e in events:
        payload = e["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        event_list.append({
            "event_type":    e["event_type"],
            "step_index":    e["step_index"],
            "timestamp":     e["timestamp"],
            "payload":       dict(payload) if payload else {},
            "parent_run_id": e["parent_run_id"],
        })

    # Build signal list with explanations
    signal_list = []
    for s in signals:
        evidence = s["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)

        detected_at = s["detected_at"]
        if hasattr(detected_at, "timestamp"):
            detected_at = detected_at.timestamp()

        try:
            fs = FailureSignal(
                failure_type=FailureType(s["failure_type"]),
                severity=Severity(s["severity"]),
                run_id=run_id,
                agent_id=dict(pr)["agent_id"],
                agent_version=dict(pr)["agent_version"],
                step_index=s["step_index"],
                confidence=s["confidence"],
                evidence=dict(evidence) if evidence else {},
                detected_at=detected_at,
            )
            exp = explain(fs)
        except Exception as exc:
            logger.error("Explain failed for signal %d: %s", s["id"], exc)
            exp = None

        signal_list.append({
            "id":              s["id"],
            "failure_type":    s["failure_type"],
            "severity":        s["severity"],
            "step_index":      s["step_index"],
            "confidence":      s["confidence"],
            "detected_at":     detected_at,
            "evidence":        dict(evidence) if evidence else {},
            "title":           exp.title if exp else s["failure_type"],
            "what":            exp.what if exp else "",
            "why_it_matters":  exp.why_it_matters if exp else "",
            "evidence_summary": exp.evidence_summary if exp else "",
            "suggested_fixes": [
                {"description": f.description, "language": f.language, "code": f.code}
                for f in (exp.suggested_fixes if exp else [])
            ],
        })

    pr_dict = dict(pr)
    return {
        "run_id":        run_id,
        "agent_id":      pr_dict["agent_id"],
        "agent_version": pr_dict["agent_version"],
        "exit_reason":   pr_dict["trigger"],
        "started_at":    started_at,
        "completed_at":  completed_at,
        "step_count":    max((e["step_index"] for e in event_list), default=0) if event_list else 0,
        "events":        event_list,
        "signals":       signal_list,
    }


# ── Signals ───────────────────────────────────────────────────────────────────

async def list_signals(
    agent_id: str,
    offset: int,
    limit: int,
    severity: Optional[str] = None,
    failure_type: Optional[str] = None,
    include_shadow: bool = False,
) -> tuple[list, int]:
    """List signals for an agent with optional filters. By default only live (non-shadow) signals are returned; pass include_shadow=True to include shadow signals too."""
    if not _pool:
        return [], 0

    import json

    where = ["agent_id = $1"]
    if not include_shadow:
        where.append("shadow = FALSE")
    params: list = [agent_id]

    if severity:
        params.append(severity.upper())
        where.append(f"severity = ${len(params)}")
    if failure_type:
        params.append(failure_type.upper())
        where.append(f"failure_type = ${len(params)}")

    where_clause = " AND ".join(where)

    async with _pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM failure_signals WHERE {where_clause}",
            *params,
        )

        params_paged = params + [limit, offset]
        rows = await conn.fetch(
            f"""
            SELECT id, failure_type, severity, run_id, agent_id, agent_version,
                   step_index, confidence, detected_at, evidence, alerted, shadow
            FROM failure_signals
            WHERE {where_clause}
            ORDER BY detected_at DESC
            LIMIT ${len(params)+1} OFFSET ${len(params)+2}
            """,
            *params_paged,
        )

    results = []
    for s in rows:
        evidence = s["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)

        detected_at = s["detected_at"]
        if hasattr(detected_at, "timestamp"):
            detected_at = detected_at.timestamp()

        try:
            fs = FailureSignal(
                failure_type=FailureType(s["failure_type"]),
                severity=Severity(s["severity"]),
                run_id=s["run_id"],
                agent_id=s["agent_id"],
                agent_version=s["agent_version"],
                step_index=s["step_index"],
                confidence=s["confidence"],
                evidence=dict(evidence) if evidence else {},
                detected_at=detected_at,
            )
            exp = explain(fs)
        except Exception as exc:
            logger.error("Explain failed signal %d: %s", s["id"], exc)
            exp = None

        results.append({
            "id":              s["id"],
            "failure_type":    s["failure_type"],
            "severity":        s["severity"],
            "run_id":          s["run_id"],
            "agent_id":        s["agent_id"],
            "agent_version":   s["agent_version"],
            "step_index":      s["step_index"],
            "confidence":      s["confidence"],
            "detected_at":     detected_at,
            "evidence":        dict(evidence) if evidence else {},
            "alerted":         s["alerted"],
            "shadow":          s["shadow"],
            "title":           exp.title if exp else s["failure_type"],
            "what":            exp.what if exp else "",
            "why_it_matters":  exp.why_it_matters if exp else "",
            "evidence_summary": exp.evidence_summary if exp else "",
            "suggested_fixes": [
                {"description": f.description, "language": f.language, "code": f.code}
                for f in (exp.suggested_fixes if exp else [])
            ],
        })

    return results, total or 0


# ── Insights ───────────────────────────────────────────────────────────────────

async def agent_input_hash_patterns(agent_id: str) -> list:
    """Input hashes that consistently produce specific failure types. Only hashes seen ≥2 times so a single bad run doesn't dominate. Returns: [{input_hash, failure_type, triggered_count, total_runs, rate}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH run_inputs AS (
                SELECT e.run_id, e.payload->>'input_hash' AS input_hash
                FROM events e
                WHERE e.agent_id = $1
                  AND e.event_type = 'run.started'
                  AND e.payload->>'input_hash' IS NOT NULL
            ),
            hash_totals AS (
                SELECT input_hash, COUNT(DISTINCT run_id) AS total_runs
                FROM run_inputs
                GROUP BY input_hash
                HAVING COUNT(DISTINCT run_id) >= 2
            ),
            hash_signals AS (
                SELECT ri.input_hash, fs.failure_type,
                       COUNT(DISTINCT fs.run_id) AS triggered_count
                FROM run_inputs ri
                JOIN failure_signals fs ON fs.run_id = ri.run_id
                WHERE fs.shadow = FALSE AND fs.agent_id = $1
                GROUP BY ri.input_hash, fs.failure_type
            )
            SELECT
                hs.input_hash,
                hs.failure_type,
                hs.triggered_count::int,
                ht.total_runs::int,
                ROUND(hs.triggered_count::numeric / ht.total_runs, 2) AS rate
            FROM hash_signals hs
            JOIN hash_totals ht ON ht.input_hash = hs.input_hash
            ORDER BY rate DESC, triggered_count DESC
            LIMIT 20
            """,
            agent_id,
        )
    return [dict(r) for r in rows]


async def agent_signal_recurrence(agent_id: str) -> list:
    """Signal counts by failure_type × agent_version × day for the last 30 days. Returns: [{failure_type, agent_version, day (ISO str), count}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                failure_type,
                agent_version,
                DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC')::date AS day,
                COUNT(*) AS count
            FROM failure_signals
            WHERE agent_id = $1
              AND shadow = FALSE
              AND detected_at >= NOW() - INTERVAL '30 days'
            GROUP BY failure_type, agent_version, day
            ORDER BY day DESC, failure_type, agent_version
            LIMIT 300
            """,
            agent_id,
        )
    return [
        {**dict(r), "day": str(r["day"])}
        for r in rows
    ]


async def agent_version_stats(agent_id: str) -> list:
    """Signal rate per version (runs_with_signals / total_runs), newest first. Returns: [{agent_version, run_count, runs_with_signals, signal_count, signal_rate, first_seen, last_seen}]."""
    if not _pool:
        return []

    def _ts(v):
        if v is None:
            return None
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                pr.agent_version,
                COUNT(DISTINCT pr.run_id)                                              AS run_count,
                COUNT(DISTINCT fs.run_id) FILTER (WHERE fs.id IS NOT NULL)             AS runs_with_signals,
                COUNT(fs.id)                                                           AS signal_count,
                ROUND(
                    COUNT(DISTINCT fs.run_id) FILTER (WHERE fs.id IS NOT NULL)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0),
                    3
                )                                                                      AS signal_rate,
                MIN(pr.processed_at) AS first_seen,
                MAX(pr.processed_at) AS last_seen
            FROM processed_runs pr
            LEFT JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.agent_id = pr.agent_id AND fs.shadow = FALSE
            WHERE pr.agent_id = $1
            GROUP BY pr.agent_version
            ORDER BY MAX(pr.processed_at) DESC
            LIMIT 10
            """,
            agent_id,
        )
    return [
        {
            "agent_version":     r["agent_version"],
            "run_count":         int(r["run_count"]),
            "runs_with_signals": int(r["runs_with_signals"]),
            "signal_count":      int(r["signal_count"]),
            "signal_rate":       float(r["signal_rate"] or 0),
            "first_seen":        _ts(r["first_seen"]),
            "last_seen":         _ts(r["last_seen"]),
        }
        for r in rows
    ]


async def agent_time_to_first_tool(agent_id: str) -> dict:
    """Steps before the first tool call — overall P25/P50/P75 plus a 14-day daily trend. Returns: {p25, p50, p75, avg_steps, runs_with_tool, total_runs, daily_trend}."""
    if not _pool:
        return {
            "p25": None, "p50": None, "p75": None,
            "avg_steps": None, "runs_with_tool": 0,
            "total_runs": 0, "daily_trend": [],
        }
    async with _pool.acquire() as conn:
        overall = await conn.fetchrow(
            """
            WITH first_tool AS (
                SELECT run_id, MIN(step_index) AS first_tool_step
                FROM events
                WHERE agent_id = $1 AND event_type = 'tool.called'
                GROUP BY run_id
            )
            SELECT
                COUNT(pr.run_id)                                                      AS total_runs,
                COUNT(ft.run_id)                                                      AS runs_with_tool,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ft.first_tool_step)      AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ft.first_tool_step)      AS p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ft.first_tool_step)      AS p75,
                ROUND(AVG(ft.first_tool_step), 1)                                     AS avg_steps
            FROM processed_runs pr
            LEFT JOIN first_tool ft ON ft.run_id = pr.run_id
            WHERE pr.agent_id = $1
            """,
            agent_id,
        )
        daily = await conn.fetch(
            """
            WITH first_tool AS (
                SELECT run_id, MIN(step_index) AS first_tool_step
                FROM events
                WHERE agent_id = $1 AND event_type = 'tool.called'
                GROUP BY run_id
            )
            SELECT
                DATE_TRUNC('day', pr.processed_at AT TIME ZONE 'UTC')::date AS day,
                COUNT(pr.run_id)              AS run_count,
                COUNT(ft.run_id)              AS runs_with_tool,
                ROUND(AVG(ft.first_tool_step), 1) AS avg_first_tool_step
            FROM processed_runs pr
            LEFT JOIN first_tool ft ON ft.run_id = pr.run_id
            WHERE pr.agent_id = $1
              AND pr.processed_at >= NOW() - INTERVAL '14 days'
            GROUP BY day
            ORDER BY day
            """,
            agent_id,
        )
    return {
        "p25":            float(overall["p25"])      if overall["p25"]      is not None else None,
        "p50":            float(overall["p50"])      if overall["p50"]      is not None else None,
        "p75":            float(overall["p75"])      if overall["p75"]      is not None else None,
        "avg_steps":      float(overall["avg_steps"]) if overall["avg_steps"] is not None else None,
        "runs_with_tool": int(overall["runs_with_tool"]),
        "total_runs":     int(overall["total_runs"]),
        "daily_trend": [
            {
                "day":                 str(r["day"]),
                "run_count":           int(r["run_count"]),
                "runs_with_tool":      int(r["runs_with_tool"]),
                "avg_first_tool_step": float(r["avg_first_tool_step"])
                                       if r["avg_first_tool_step"] is not None else None,
            }
            for r in daily
        ],
    }


async def agent_hourly_pattern(agent_id: str) -> list:
    """Signal rate by UTC hour of day over the last 30 days. Sparse — only hours with ≥1 run are returned; the UI fills gaps. Returns: [{hour_of_day, run_count, signal_count, signal_rate}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                EXTRACT(HOUR FROM pr.processed_at AT TIME ZONE 'UTC')::int AS hour_of_day,
                COUNT(DISTINCT pr.run_id)                                   AS run_count,
                COUNT(DISTINCT fs.run_id)                                   AS signal_count,
                ROUND(
                    COUNT(DISTINCT fs.run_id)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0),
                    3
                )                                                           AS signal_rate
            FROM processed_runs pr
            LEFT JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.agent_id = pr.agent_id AND fs.shadow = FALSE
            WHERE pr.agent_id = $1
              AND pr.processed_at >= NOW() - INTERVAL '30 days'
            GROUP BY hour_of_day
            ORDER BY hour_of_day
            """,
            agent_id,
        )
    return [
        {
            "hour_of_day":  int(r["hour_of_day"]),
            "run_count":    int(r["run_count"]),
            "signal_count": int(r["signal_count"]),
            "signal_rate":  float(r["signal_rate"] or 0),
        }
        for r in rows
    ]


async def list_issues(agent_id: str, status: Optional[str] = None) -> list:
    """Return persistent issues for an agent, ordered: open → reopened → resolved, then by last_seen desc."""
    if not _pool:
        return []

    def _ts(v):
        if v is None:
            return None
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    where = "WHERE agent_id = $1"
    params: list = [agent_id]
    if status:
        params.append(status.lower())
        where += f" AND status = ${len(params)}"

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, agent_id, failure_type, status,
                   first_seen, last_seen, resolved_at,
                   affected_runs, clean_runs_since
            FROM issues
            {where}
            ORDER BY
                CASE status WHEN 'open' THEN 0 WHEN 'reopened' THEN 1 ELSE 2 END,
                last_seen DESC
            """,
            *params,
        )
    return [
        {
            "id":               r["id"],
            "agent_id":         r["agent_id"],
            "failure_type":     r["failure_type"],
            "status":           r["status"],
            "first_seen":       _ts(r["first_seen"]),
            "last_seen":        _ts(r["last_seen"]),
            "resolved_at":      _ts(r["resolved_at"]),
            "affected_runs":    int(r["affected_runs"]),
            "clean_runs_since": int(r["clean_runs_since"]),
        }
        for r in rows
    ]


async def agent_failure_rates(agent_id: str) -> list:
    """Daily failure rate per failure_type over 30 days — affected_runs / total_runs.
    Returns: [{failure_type, day (ISO str), total_runs, affected_runs, rate}]."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                fs.failure_type,
                DATE_TRUNC('day', pr.processed_at AT TIME ZONE 'UTC')::date AS day,
                COUNT(DISTINCT pr.run_id)::int  AS total_runs,
                COUNT(DISTINCT fs.run_id)::int  AS affected_runs,
                ROUND(
                    COUNT(DISTINCT fs.run_id)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0),
                    3
                ) AS rate
            FROM processed_runs pr
            JOIN failure_signals fs
                ON fs.run_id    = pr.run_id
                AND fs.agent_id = pr.agent_id
                AND fs.shadow   = FALSE
            WHERE pr.agent_id = $1
              AND pr.processed_at >= NOW() - INTERVAL '30 days'
            GROUP BY fs.failure_type, day
            ORDER BY day DESC, affected_runs DESC
            LIMIT 300
            """,
            agent_id,
        )
    return [
        {
            "failure_type":  r["failure_type"],
            "day":           str(r["day"]),
            "total_runs":    int(r["total_runs"]),
            "affected_runs": int(r["affected_runs"]),
            "rate":          float(r["rate"] or 0),
        }
        for r in rows
    ]


async def agent_systemic_patterns(agent_id: str, rate_threshold: float = 0.10) -> list:
    """Failure types firing on >= rate_threshold of runs in the last 7 days.
    Returns: [{failure_type, total_runs, affected_runs, rate, first_seen, last_seen, is_systemic}]."""
    if not _pool:
        return []

    def _ts(v):
        if v is None:
            return None
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                fs.failure_type,
                COUNT(DISTINCT pr.run_id)::int  AS total_runs,
                COUNT(DISTINCT fs.run_id)::int  AS affected_runs,
                ROUND(
                    COUNT(DISTINCT fs.run_id)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0),
                    3
                )                               AS rate,
                MIN(fs.detected_at)             AS first_seen,
                MAX(fs.detected_at)             AS last_seen
            FROM processed_runs pr
            JOIN failure_signals fs
                ON fs.run_id    = pr.run_id
                AND fs.agent_id = pr.agent_id
                AND fs.shadow   = FALSE
            WHERE pr.agent_id = $1
              AND pr.processed_at >= NOW() - INTERVAL '7 days'
            GROUP BY fs.failure_type
            ORDER BY rate DESC
            """,
            agent_id,
        )
    return [
        {
            "failure_type":  r["failure_type"],
            "total_runs":    int(r["total_runs"]),
            "affected_runs": int(r["affected_runs"]),
            "rate":          float(r["rate"] or 0),
            "first_seen":    _ts(r["first_seen"]),
            "last_seen":     _ts(r["last_seen"]),
            "is_systemic":   float(r["rate"] or 0) >= rate_threshold,
        }
        for r in rows
    ]


async def agent_failure_pattern(agent_id: str, failure_type: str) -> dict:
    """
    Cross-run deep-dive for one failure type.

    Returns overview stats, step distribution, evidence aggregates,
    14-day daily trend, co-occurring failure types, and top affected runs.
    """
    if not _pool:
        return {}

    async with _pool.acquire() as conn:
        # 1. Overview
        overview_row = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT run_id)                          AS affected_runs,
                ROUND(AVG(confidence)::numeric, 3)              AS avg_confidence,
                MIN(detected_at)                                AS first_seen,
                MAX(detected_at)                                AS last_seen,
                COUNT(*) FILTER (WHERE severity = 'CRITICAL')  AS critical_count,
                COUNT(*) FILTER (WHERE severity = 'HIGH')       AS high_count,
                COUNT(*) FILTER (WHERE severity = 'MEDIUM')     AS medium_count,
                COUNT(*) FILTER (WHERE severity = 'LOW')        AS low_count
            FROM failure_signals
            WHERE agent_id    = $1
              AND failure_type = $2
              AND shadow       = FALSE
            """,
            agent_id, failure_type,
        )

        total_row = await conn.fetchrow(
            "SELECT COUNT(DISTINCT run_id) AS total FROM processed_runs WHERE agent_id = $1",
            agent_id,
        )

        # 2. Step distribution (where in the run does this fire)
        step_row = await conn.fetchrow(
            """
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY step_index) AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY step_index) AS p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY step_index) AS p75,
                ROUND(AVG(step_index)::numeric, 1)                        AS avg_step
            FROM failure_signals
            WHERE agent_id    = $1
              AND failure_type = $2
              AND shadow       = FALSE
            """,
            agent_id, failure_type,
        )

        # 3. Evidence aggregates — extract the most useful keys per failure type
        evidence_rows = await conn.fetch(
            """
            SELECT
                -- Tool-based failures (TOOL_LOOP, RETRY_STORM, CASCADING_TOOL_FAILURE)
                evidence->>'tool'                                        AS tool,

                -- Loop / repetition
                ROUND(AVG((evidence->>'count')::numeric), 1)            AS avg_count,
                ROUND(AVG((evidence->>'consecutive_fails')::numeric), 1) AS avg_consecutive_fails,
                ROUND(AVG(
                    CASE WHEN evidence->>'args_identical' = 'true' THEN 1.0 ELSE 0.0 END
                ), 2)                                                    AS args_identical_rate,

                -- Context / token growth
                ROUND(AVG((evidence->>'growth_factor')::numeric), 2)    AS avg_growth_factor,
                ROUND(AVG((evidence->>'first_tokens')::numeric), 0)     AS avg_first_tokens,
                ROUND(AVG((evidence->>'last_tokens')::numeric), 0)      AS avg_last_tokens,

                -- Slow step
                ROUND(AVG((evidence->>'duration_ms')::numeric), 0)      AS avg_duration_ms,
                ROUND(AVG((evidence->>'threshold_ms')::numeric), 0)     AS avg_threshold_ms,
                ROUND(AVG((evidence->>'ratio')::numeric), 2)            AS avg_ratio,

                -- RAG
                evidence->>'index_name'                                  AS index_name,
                ROUND(AVG((evidence->>'top_score')::numeric), 3)        AS avg_top_score,
                ROUND(AVG((evidence->>'result_count')::numeric), 1)     AS avg_result_count,

                -- Step count inflation
                ROUND(AVG((evidence->>'inflation_ratio')::numeric), 2)  AS avg_inflation_ratio,
                ROUND(AVG((evidence->>'baseline_p75')::numeric), 1)     AS avg_baseline_p75,

                -- Reasoning / goal abandonment
                ROUND(AVG((evidence->>'stall_steps')::numeric), 1)      AS avg_stall_steps,

                COUNT(*) AS sample_count
            FROM failure_signals
            WHERE agent_id    = $1
              AND failure_type = $2
              AND shadow       = FALSE
            GROUP BY
                evidence->>'tool',
                evidence->>'index_name'
            ORDER BY sample_count DESC
            LIMIT 10
            """,
            agent_id, failure_type,
        )

        # 4. Daily trend (14 days)
        trend_rows = await conn.fetch(
            """
            WITH days AS (
                SELECT generate_series(
                    DATE_TRUNC('day', NOW() - INTERVAL '13 days'),
                    DATE_TRUNC('day', NOW()),
                    INTERVAL '1 day'
                )::date AS day
            ),
            daily_affected AS (
                SELECT
                    DATE_TRUNC('day', detected_at AT TIME ZONE 'UTC')::date AS day,
                    COUNT(DISTINCT run_id) AS affected_runs
                FROM failure_signals
                WHERE agent_id    = $1
                  AND failure_type = $2
                  AND shadow       = FALSE
                  AND detected_at >= NOW() - INTERVAL '14 days'
                GROUP BY 1
            ),
            daily_total AS (
                SELECT
                    DATE_TRUNC('day', processed_at AT TIME ZONE 'UTC')::date AS day,
                    COUNT(DISTINCT run_id) AS total_runs
                FROM processed_runs
                WHERE agent_id    = $1
                  AND processed_at >= NOW() - INTERVAL '14 days'
                GROUP BY 1
            )
            SELECT
                d.day,
                COALESCE(da.affected_runs, 0)  AS affected_runs,
                COALESCE(dt.total_runs, 0)     AS total_runs,
                CASE WHEN COALESCE(dt.total_runs, 0) > 0
                     THEN ROUND(COALESCE(da.affected_runs, 0)::numeric / dt.total_runs, 3)
                     ELSE 0 END                AS rate
            FROM days d
            LEFT JOIN daily_affected da ON da.day = d.day
            LEFT JOIN daily_total    dt ON dt.day = d.day
            ORDER BY d.day
            """,
            agent_id, failure_type,
        )

        # 5. Co-occurring failure types (same runs, last 30 days)
        co_rows = await conn.fetch(
            """
            WITH affected AS (
                SELECT DISTINCT run_id
                FROM failure_signals
                WHERE agent_id    = $1
                  AND failure_type = $2
                  AND shadow       = FALSE
                  AND detected_at >= NOW() - INTERVAL '30 days'
            )
            SELECT
                fs.failure_type,
                COUNT(DISTINCT fs.run_id)                                              AS co_count,
                ROUND(COUNT(DISTINCT fs.run_id)::numeric / COUNT(DISTINCT a.run_id), 3) AS co_rate
            FROM affected a
            JOIN failure_signals fs ON fs.run_id = a.run_id
            WHERE fs.agent_id    = $1
              AND fs.failure_type != $2
              AND fs.shadow       = FALSE
            GROUP BY fs.failure_type
            ORDER BY co_rate DESC
            LIMIT 8
            """,
            agent_id, failure_type,
        )

        # 6. Top affected runs (highest confidence, most recent)
        run_rows = await conn.fetch(
            """
            SELECT
                run_id,
                confidence,
                severity,
                step_index,
                detected_at,
                evidence
            FROM failure_signals
            WHERE agent_id    = $1
              AND failure_type = $2
              AND shadow       = FALSE
            ORDER BY confidence DESC, detected_at DESC
            LIMIT 5
            """,
            agent_id, failure_type,
        )

    total_runs    = int(total_row["total"] or 0)
    affected_runs = int(overview_row["affected_runs"] or 0)

    return {
        "failure_type": failure_type,
        "overview": {
            "affected_runs":  affected_runs,
            "total_runs":     total_runs,
            "rate":           round(affected_runs / total_runs, 3) if total_runs else 0.0,
            "avg_confidence": float(overview_row["avg_confidence"] or 0),
            "first_seen":     _ts(overview_row["first_seen"]),
            "last_seen":      _ts(overview_row["last_seen"]),
            "severity_breakdown": {
                "CRITICAL": int(overview_row["critical_count"] or 0),
                "HIGH":     int(overview_row["high_count"] or 0),
                "MEDIUM":   int(overview_row["medium_count"] or 0),
                "LOW":      int(overview_row["low_count"] or 0),
            },
        },
        "step_distribution": {
            "p25":      float(step_row["p25"])      if step_row["p25"]      is not None else None,
            "p50":      float(step_row["p50"])      if step_row["p50"]      is not None else None,
            "p75":      float(step_row["p75"])      if step_row["p75"]      is not None else None,
            "avg_step": float(step_row["avg_step"]) if step_row["avg_step"] is not None else None,
        },
        "evidence_aggregates": [
            {k: v for k, v in dict(r).items() if v is not None}
            for r in evidence_rows
        ],
        "daily_trend": [
            {
                "day":           str(r["day"]),
                "affected_runs": int(r["affected_runs"]),
                "total_runs":    int(r["total_runs"]),
                "rate":          float(r["rate"] or 0),
            }
            for r in trend_rows
        ],
        "co_occurring": [
            {
                "failure_type": r["failure_type"],
                "co_count":     int(r["co_count"]),
                "co_rate":      float(r["co_rate"] or 0),
            }
            for r in co_rows
        ],
        "top_runs": [
            {
                "run_id":      r["run_id"],
                "confidence":  float(r["confidence"]),
                "severity":    r["severity"],
                "step_index":  int(r["step_index"]),
                "detected_at": _ts(r["detected_at"]),
                "evidence":    dict(r["evidence"]),
            }
            for r in run_rows
        ],
    }
