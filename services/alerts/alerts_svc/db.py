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
