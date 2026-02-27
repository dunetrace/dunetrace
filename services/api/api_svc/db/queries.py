"""
services/api/api_svc/db/queries.py

All read queries for the customer API.
Reads from: events, failure_signals, processed_runs, api_keys.
No writes — this service is read-only.
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
    """
    Returns (rows, total_count).
    Each row: agent_id, last_seen, run_count, signal_count, critical_count, high_count
    """
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
                -- started_at from first event
                (SELECT MIN(e.received_at) FROM events e WHERE e.run_id = pr.run_id) AS started_at,
                -- step_count
                (SELECT MAX(e.step_index) + 1 FROM events e WHERE e.run_id = pr.run_id) AS step_count,
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
    import sys, os
    # Import explainer
    _EXPLAINER = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../services/explainer")
    )
    _SDK = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../packages/sdk-py")
    )
    for p in [_SDK, _EXPLAINER]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from dunetrace.models import FailureSignal, FailureType, Severity
    from app.explainer import explain

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

    started_at = events[0]["timestamp"] if events else None
    completed_at = dict(pr)["processed_at"]
    if hasattr(completed_at, "timestamp"):
        completed_at = completed_at.timestamp()

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
        "step_count":    len(events),
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
) -> tuple[list, int]:
    """List live signals for an agent with optional filters."""
    if not _pool:
        return [], 0

    import json, sys, os

    _EXPLAINER = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../services/explainer")
    )
    _SDK = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../packages/sdk-py")
    )
    for p in [_SDK, _EXPLAINER]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from dunetrace.models import FailureSignal, FailureType, Severity
    from app.explainer import explain

    where = ["agent_id = $1", "shadow = FALSE"]
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
                   step_index, confidence, detected_at, evidence, alerted
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
