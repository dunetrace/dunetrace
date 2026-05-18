from __future__ import annotations

import logging
from typing import Any

try:
    import asyncpg
except ImportError:  # pragma: no cover - allows unit tests without db driver
    asyncpg = None  # type: ignore

from alerts_svc.config import settings

logger = logging.getLogger("dunetrace.alerts.db")

_pool: asyncpg.Pool | None = None  # type: ignore[attr-defined]


async def init_pool() -> None:
    global _pool
    if asyncpg is None:
        raise RuntimeError("asyncpg is not installed")
    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=1,
        max_size=5,
        command_timeout=15,
    )
    logger.info("DB pool ready")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def fetch_unalerted_signals(limit: int = 50) -> list[dict[str, Any]]:
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                failure_type,
                severity,
                run_id,
                agent_id,
                agent_version,
                step_index,
                confidence,
                evidence,
                detected_at
            FROM failure_signals
            WHERE alerted = FALSE
              AND COALESCE(shadow, TRUE) = FALSE
            ORDER BY detected_at ASC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def mark_alerted_batch(signal_ids: list[int]) -> None:
    if not signal_ids or not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE failure_signals
            SET alerted = TRUE
            WHERE id = ANY($1::bigint[])
            """,
            signal_ids,
        )


async def ensure_digest_schema() -> None:
    """Create digest_log table if it doesn't exist."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS digest_log (
                id          BIGSERIAL    PRIMARY KEY,
                digest_type TEXT         NOT NULL DEFAULT 'weekly',
                sent_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
            """
        )


async def was_digest_sent_recently(within_days: int = 6) -> bool:
    """True if a weekly digest was sent within the last `within_days` days."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 1 FROM digest_log
            WHERE digest_type = 'weekly'
              AND sent_at >= NOW() - ($1 || ' days')::INTERVAL
            LIMIT 1
            """,
            str(within_days),
        )
    return row is not None


async def log_digest_sent() -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO digest_log (digest_type) VALUES ('weekly')"
        )


async def fetch_weekly_digest_data() -> dict[str, Any]:
    """Aggregate the last 7 days of signal + run + issue data for the weekly digest."""
    if not _pool:
        return {}
    async with _pool.acquire() as conn:
        # Total runs and signals across all agents in the last 7 days
        totals = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT pr.run_id)              AS total_runs,
                COUNT(DISTINCT pr.agent_id)            AS total_agents,
                COUNT(DISTINCT fs.id)                  AS total_signals
            FROM processed_runs pr
            LEFT JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.shadow = FALSE
            WHERE pr.processed_at >= NOW() - INTERVAL '7 days'
            """
        )

        # Top failure types by affected run count
        top_failures = await conn.fetch(
            """
            SELECT
                fs.failure_type,
                COUNT(DISTINCT fs.run_id)::int  AS affected_runs,
                COUNT(DISTINCT pr.run_id)::int  AS total_runs,
                ROUND(
                    COUNT(DISTINCT fs.run_id)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0), 3
                )                               AS rate
            FROM processed_runs pr
            JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.shadow = FALSE
            WHERE pr.processed_at >= NOW() - INTERVAL '7 days'
            GROUP BY fs.failure_type
            ORDER BY affected_runs DESC
            LIMIT 5
            """
        )

        # Top agents by signal volume with dominant failure type
        top_agents = await conn.fetch(
            """
            WITH agent_signals AS (
                SELECT
                    pr.agent_id,
                    COUNT(DISTINCT fs.id)      AS signal_count,
                    COUNT(DISTINCT pr.run_id)  AS run_count
                FROM processed_runs pr
                JOIN failure_signals fs
                    ON fs.run_id = pr.run_id AND fs.shadow = FALSE
                WHERE pr.processed_at >= NOW() - INTERVAL '7 days'
                GROUP BY pr.agent_id
            ),
            dominant AS (
                SELECT DISTINCT ON (pr.agent_id)
                    pr.agent_id,
                    fs.failure_type
                FROM processed_runs pr
                JOIN failure_signals fs
                    ON fs.run_id = pr.run_id AND fs.shadow = FALSE
                WHERE pr.processed_at >= NOW() - INTERVAL '7 days'
                GROUP BY pr.agent_id, fs.failure_type
                ORDER BY pr.agent_id, COUNT(*) DESC
            )
            SELECT a.agent_id, a.signal_count, a.run_count, d.failure_type AS dominant_failure
            FROM agent_signals a
            JOIN dominant d ON d.agent_id = a.agent_id
            ORDER BY a.signal_count DESC
            LIMIT 5
            """
        )

        # Systemic patterns: failure types at ≥10% of runs per agent in last 7 days
        systemic = await conn.fetch(
            """
            SELECT
                pr.agent_id,
                fs.failure_type,
                COUNT(DISTINCT pr.run_id)::int  AS total_runs,
                COUNT(DISTINCT fs.run_id)::int  AS affected_runs,
                ROUND(
                    COUNT(DISTINCT fs.run_id)::numeric
                    / NULLIF(COUNT(DISTINCT pr.run_id), 0), 3
                )                               AS rate
            FROM processed_runs pr
            JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.shadow = FALSE
            WHERE pr.processed_at >= NOW() - INTERVAL '7 days'
            GROUP BY pr.agent_id, fs.failure_type
            HAVING ROUND(
                COUNT(DISTINCT fs.run_id)::numeric
                / NULLIF(COUNT(DISTINCT pr.run_id), 0), 3
            ) >= 0.10
            ORDER BY rate DESC
            LIMIT 10
            """
        )

        # Issues opened and resolved this week
        issues_opened = await conn.fetchval(
            """
            SELECT COUNT(*) FROM issues
            WHERE first_seen >= NOW() - INTERVAL '7 days'
            """
        ) or 0

        issues_resolved = await conn.fetchval(
            """
            SELECT COUNT(*) FROM issues
            WHERE resolved_at >= NOW() - INTERVAL '7 days'
            """
        ) or 0

    return {
        "total_runs":      int(totals["total_runs"] or 0),
        "total_agents":    int(totals["total_agents"] or 0),
        "total_signals":   int(totals["total_signals"] or 0),
        "top_failures":    [dict(r) for r in top_failures],
        "top_agents":      [dict(r) for r in top_agents],
        "systemic":        [dict(r) for r in systemic],
        "issues_opened":   int(issues_opened),
        "issues_resolved": int(issues_resolved),
    }


async def fetch_signal_rate_context(agent_id: str, failure_type: str) -> dict[str, Any]:
    """Return 7-day rate context for a failure_type on this agent.
    Used to distinguish systemic patterns from one-off alerts in Slack messages.
    Returns: {total_runs, affected_runs, rate, is_systemic}. Empty dict on error."""
    if not _pool:
        return {}
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(DISTINCT pr.run_id)::int AS total_runs,
                    COUNT(DISTINCT fs.run_id)::int AS affected_runs,
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
                    AND fs.failure_type = $2
                WHERE pr.agent_id = $1
                  AND pr.processed_at >= NOW() - INTERVAL '7 days'
                """,
                agent_id,
                failure_type,
            )
        if not row or row["total_runs"] == 0:
            return {}
        rate = float(row["rate"] or 0)
        return {
            "total_runs":    int(row["total_runs"]),
            "affected_runs": int(row["affected_runs"]),
            "rate":          rate,
            "is_systemic":   rate >= 0.10,
        }
    except Exception as exc:
        logger.warning("fetch_signal_rate_context failed: %s", exc)
        return {}


async def fetch_run_tokens(run_ids: list[str]) -> dict[str, dict]:
    """Fetch total prompt+completion tokens and model for a batch of run_ids."""
    if not _pool or not run_ids:
        return {}
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                r.run_id,
                SUM(COALESCE((r.payload->>'prompt_tokens')::integer, 0))    AS prompt_tokens,
                SUM(COALESCE((r.payload->>'completion_tokens')::integer, 0)) AS completion_tokens,
                (SELECT MIN(c.payload->>'model') FROM events c
                 WHERE c.run_id = r.run_id AND c.event_type = 'llm.called'
                   AND c.payload->>'model' IS NOT NULL) AS model
            FROM events r
            WHERE r.run_id = ANY($1::text[])
              AND r.event_type = 'llm.responded'
              AND r.payload->>'prompt_tokens' IS NOT NULL
            GROUP BY r.run_id
            """,
            run_ids,
        )
    return {r["run_id"]: dict(r) for r in rows}
