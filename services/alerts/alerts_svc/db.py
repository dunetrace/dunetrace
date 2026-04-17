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
