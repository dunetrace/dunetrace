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
        # See ingest_svc/db/postgres.py::init_pool for why this is required —
        # DATABASE_URL is Supabase's transaction-mode PgBouncer pooler, which
        # is incompatible with asyncpg's default prepared-statement cache.
        statement_cache_size=0,
    )
    logger.info("DB pool ready")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def fetch_unalerted_signals(limit: int = 50) -> list[dict[str, Any]]:
    """Scan unalerted live signals across ALL orgs in one poll — mirrors
    detector_svc's fetch_completed_runs/fetch_stalled_runs, which also scan
    globally rather than per-org. org_id is returned per-row so the worker
    can group and scope downstream policy/dedup checks correctly."""
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
                org_id,
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
    """Create digest_log table if it doesn't exist, and migrate it to be org-scoped.

    Each org gets its own weekly digest send, gated independently — the digest
    aggregates that org's own runs/signals/issues only, so a shared Slack/webhook
    destination never sees another org's data mixed into one message.
    """
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS digest_log (
                id          BIGSERIAL    PRIMARY KEY,
                digest_type TEXT         NOT NULL DEFAULT 'weekly',
                sent_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
            """)
        await conn.execute("ALTER TABLE digest_log ADD COLUMN IF NOT EXISTS org_id TEXT")
        await conn.execute("UPDATE digest_log SET org_id = 'default' WHERE org_id IS NULL")
        await conn.execute("ALTER TABLE digest_log ALTER COLUMN org_id SET NOT NULL")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_digest_log_org_type_sent "
            "ON digest_log (org_id, digest_type, sent_at)"
        )


async def fetch_active_org_ids(within_days: int = 7) -> list[str]:
    """Distinct org_ids with any processed run in the last `within_days` days.
    Orgs with zero activity are skipped entirely — no digest, no digest_log row."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT org_id FROM processed_runs
            WHERE processed_at >= NOW() - ($1 || ' days')::INTERVAL
            """,
            str(within_days),
        )
    return [r["org_id"] for r in rows]


async def was_digest_sent_recently(org_id: str, within_days: int = 6) -> bool:
    """True if org_id's weekly digest was sent within the last `within_days` days."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 1 FROM digest_log
            WHERE org_id = $1
              AND digest_type = 'weekly'
              AND sent_at >= NOW() - ($2 || ' days')::INTERVAL
            LIMIT 1
            """,
            org_id,
            str(within_days),
        )
    return row is not None


async def log_digest_sent(org_id: str) -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO digest_log (digest_type, org_id) VALUES ('weekly', $1)", org_id
        )


async def fetch_weekly_digest_data(org_id: str) -> dict[str, Any]:
    """Aggregate the last 7 days of signal + run + issue data for org_id's weekly digest."""
    if not _pool:
        return {}
    async with _pool.acquire() as conn:
        # Total runs and signals for this org in the last 7 days
        totals = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT pr.run_id)              AS total_runs,
                COUNT(DISTINCT pr.agent_id)            AS total_agents,
                COUNT(DISTINCT fs.id)                  AS total_signals
            FROM processed_runs pr
            LEFT JOIN failure_signals fs
                ON fs.run_id = pr.run_id AND fs.shadow = FALSE
            WHERE pr.org_id = $1
              AND pr.processed_at >= NOW() - INTERVAL '7 days'
            """,
            org_id,
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
            WHERE pr.org_id = $1
              AND pr.processed_at >= NOW() - INTERVAL '7 days'
            GROUP BY fs.failure_type
            ORDER BY affected_runs DESC
            LIMIT 5
            """,
            org_id,
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
                WHERE pr.org_id = $1
                  AND pr.processed_at >= NOW() - INTERVAL '7 days'
                GROUP BY pr.agent_id
            ),
            dominant AS (
                SELECT DISTINCT ON (pr.agent_id)
                    pr.agent_id,
                    fs.failure_type
                FROM processed_runs pr
                JOIN failure_signals fs
                    ON fs.run_id = pr.run_id AND fs.shadow = FALSE
                WHERE pr.org_id = $1
                  AND pr.processed_at >= NOW() - INTERVAL '7 days'
                GROUP BY pr.agent_id, fs.failure_type
                ORDER BY pr.agent_id, COUNT(*) DESC
            )
            SELECT a.agent_id, a.signal_count, a.run_count, d.failure_type AS dominant_failure
            FROM agent_signals a
            JOIN dominant d ON d.agent_id = a.agent_id
            ORDER BY a.signal_count DESC
            LIMIT 5
            """,
            org_id,
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
            WHERE pr.org_id = $1
              AND pr.processed_at >= NOW() - INTERVAL '7 days'
            GROUP BY pr.agent_id, fs.failure_type
            HAVING ROUND(
                COUNT(DISTINCT fs.run_id)::numeric
                / NULLIF(COUNT(DISTINCT pr.run_id), 0), 3
            ) >= 0.10
            ORDER BY rate DESC
            LIMIT 10
            """,
            org_id,
        )

        # Issues opened and resolved this week
        issues_opened = (
            await conn.fetchval(
                """
            SELECT COUNT(*) FROM issues
            WHERE org_id = $1
              AND first_seen >= NOW() - INTERVAL '7 days'
            """,
                org_id,
            )
            or 0
        )

        issues_resolved = (
            await conn.fetchval(
                """
            SELECT COUNT(*) FROM issues
            WHERE org_id = $1
              AND resolved_at >= NOW() - INTERVAL '7 days'
            """,
                org_id,
            )
            or 0
        )

    return {
        "total_runs": int(totals["total_runs"] or 0),
        "total_agents": int(totals["total_agents"] or 0),
        "total_signals": int(totals["total_signals"] or 0),
        "top_failures": [dict(r) for r in top_failures],
        "top_agents": [dict(r) for r in top_agents],
        "systemic": [dict(r) for r in systemic],
        "issues_opened": int(issues_opened),
        "issues_resolved": int(issues_resolved),
    }


async def fetch_signal_rate_context(
    org_id: str, agent_id: str, failure_type: str
) -> dict[str, Any]:
    """Return 7-day rate context for a failure_type on this org's agent.
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
                    AND fs.org_id   = pr.org_id
                    AND fs.agent_id = pr.agent_id
                    AND fs.shadow   = FALSE
                    AND fs.failure_type = $3
                WHERE pr.org_id = $1
                  AND pr.agent_id = $2
                  AND pr.processed_at >= NOW() - INTERVAL '7 days'
                """,
                org_id,
                agent_id,
                failure_type,
            )
        if not row or row["total_runs"] == 0:
            return {}
        rate = float(row["rate"] or 0)
        return {
            "total_runs": int(row["total_runs"]),
            "affected_runs": int(row["affected_runs"]),
            "rate": rate,
            "is_systemic": rate >= 0.10,
        }
    except Exception as exc:
        logger.warning("fetch_signal_rate_context failed: %s", exc)
        return {}


async def fetch_agent_overrides(
    triples: list[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict]:
    """Return agent_detector_overrides rows keyed by (org_id, agent_id, failure_type).
    Gracefully returns {} if the table (or its org_id column) doesn't exist yet —
    agent_detector_overrides is owned by api_svc; until that service's migration
    adds org_id, this degrades to 'no overrides' rather than erroring."""
    if not _pool or not triples:
        return {}
    triple_set = set(triples)
    org_ids = [t[0] for t in triples]
    agent_ids = [t[1] for t in triples]
    ftypes = [t[2] for t in triples]
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT org_id, agent_id, failure_type, fp_count, confidence_floor, silenced
                FROM agent_detector_overrides
                WHERE org_id = ANY($1::text[])
                  AND agent_id = ANY($2::text[])
                  AND failure_type = ANY($3::text[])
                """,
                org_ids,
                agent_ids,
                ftypes,
            )
        return {
            (r["org_id"], r["agent_id"], r["failure_type"]): dict(r)
            for r in rows
            if (r["org_id"], r["agent_id"], r["failure_type"]) in triple_set
        }
    except Exception as exc:
        logger.warning("fetch_agent_overrides failed (table may not exist yet): %s", exc)
        return {}


async def evaluate_alert_policy(
    org_id: str,
    agent_id: str,
    failure_type: str,
    mode: str,
    threshold: int,
    window_runs: int,
) -> tuple[bool, str]:
    """Return (policy_met, human_readable_reason).
    Queries processed_runs and failure_signals to check consecutive / frequency.
    `immediate` always returns (True, "").
    """
    if mode == "immediate":
        return True, ""

    if not _pool:
        return True, ""  # can't check — fail open

    async with _pool.acquire() as conn:
        # Last window_runs completed runs for this org's agent, newest first
        run_rows = await conn.fetch(
            """
            SELECT run_id FROM processed_runs
            WHERE org_id = $1 AND agent_id = $2
            ORDER BY processed_at DESC
            LIMIT $3
            """,
            org_id,
            agent_id,
            window_runs,
        )
        if not run_rows:
            return False, f"no run history yet for {agent_id}"

        run_ids = [r["run_id"] for r in run_rows]

        # Which of those runs had this failure type (any shadow=FALSE signal)?
        flagged_rows = await conn.fetch(
            """
            SELECT DISTINCT run_id FROM failure_signals
            WHERE org_id = $1
              AND agent_id = $2
              AND failure_type = $3
              AND shadow = FALSE
              AND run_id = ANY($4::text[])
            """,
            org_id,
            agent_id,
            failure_type,
            run_ids,
        )
    flagged = {r["run_id"] for r in flagged_rows}

    if mode == "consecutive":
        # The last `threshold` runs must ALL have triggered
        last_n = run_ids[:threshold]
        hit = sum(1 for rid in last_n if rid in flagged)
        met = len(last_n) == threshold and hit == threshold
        if met:
            return True, ""
        return (
            False,
            f"{hit}/{threshold} consecutive runs — waiting for {threshold - hit} more",
        )

    if mode == "frequency":
        hit = sum(1 for rid in run_ids if rid in flagged)
        met = hit >= threshold
        if met:
            return True, ""
        return (
            False,
            f"{hit}/{threshold} of last {len(run_ids)} runs — need {threshold - hit} more",
        )

    # Unknown mode — fail open
    return True, f"unknown mode '{mode}'"


async def ensure_dedup_schema() -> None:
    """Create alert_dedup table if it doesn't exist, and migrate its key to be org-scoped.

    agent_id is not guaranteed unique across orgs, so a dedup window keyed only on
    (agent_id, failure_type) could let one org's alert suppress another org's
    identically-named agent's alert. Widening the key to (org_id, agent_id,
    failure_type) closes that.
    """
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_dedup (
                agent_id         TEXT        NOT NULL,
                failure_type     TEXT        NOT NULL,
                last_alerted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                suppressed_count INTEGER     NOT NULL DEFAULT 0,
                PRIMARY KEY (agent_id, failure_type)
            )
            """)
        await conn.execute("ALTER TABLE alert_dedup ADD COLUMN IF NOT EXISTS org_id TEXT")
        await conn.execute("UPDATE alert_dedup SET org_id = 'default' WHERE org_id IS NULL")
        await conn.execute("ALTER TABLE alert_dedup ALTER COLUMN org_id SET NOT NULL")
        await conn.execute("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'alert_dedup_pkey'
                ) THEN
                    ALTER TABLE alert_dedup DROP CONSTRAINT alert_dedup_pkey;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'alert_dedup_org_agent_failure_type_key'
                ) THEN
                    ALTER TABLE alert_dedup ADD CONSTRAINT alert_dedup_org_agent_failure_type_key
                        PRIMARY KEY (org_id, agent_id, failure_type);
                END IF;
            END $$;
            """)


async def fetch_dedup_states(
    triples: list[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict]:
    """Return dedup records keyed by (org_id, agent_id, failure_type) for the given triples."""
    if not _pool or not triples:
        return {}
    triple_set = set(triples)
    org_ids = [t[0] for t in triples]
    agent_ids = [t[1] for t in triples]
    ftypes = [t[2] for t in triples]
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT org_id, agent_id, failure_type, last_alerted_at, suppressed_count
            FROM alert_dedup
            WHERE org_id = ANY($1::text[])
              AND agent_id = ANY($2::text[])
              AND failure_type = ANY($3::text[])
            """,
            org_ids,
            agent_ids,
            ftypes,
        )
    return {
        (r["org_id"], r["agent_id"], r["failure_type"]): dict(r)
        for r in rows
        if (r["org_id"], r["agent_id"], r["failure_type"]) in triple_set
    }


async def record_alert_sent(org_id: str, agent_id: str, failure_type: str) -> None:
    """Upsert dedup record: reset suppressed_count, stamp now as last_alerted_at."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO alert_dedup (org_id, agent_id, failure_type, last_alerted_at, suppressed_count)
            VALUES ($1, $2, $3, NOW(), 0)
            ON CONFLICT (org_id, agent_id, failure_type)
            DO UPDATE SET last_alerted_at = NOW(), suppressed_count = 0
            """,
            org_id,
            agent_id,
            failure_type,
        )


async def increment_suppressed_count(
    org_id: str, agent_id: str, failure_type: str, count: int
) -> None:
    """Increment suppressed_count for a key that is within its silence window."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE alert_dedup
            SET suppressed_count = suppressed_count + $4
            WHERE org_id = $1 AND agent_id = $2 AND failure_type = $3
            """,
            org_id,
            agent_id,
            failure_type,
            count,
        )


async def fetch_run_tokens(run_ids: list[str]) -> dict[str, dict]:
    """Fetch total prompt+completion tokens and model for a batch of run_ids.
    prompt_tokens may be in llm.called (direct SDK) or llm.responded (LangChain);
    completion_tokens are always in llm.responded. Sum both event types."""
    if not _pool or not run_ids:
        return {}
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                r.run_id,
                -- Prefer prompt_tokens from llm.called (SDK/Hermes pattern).
                -- Fall back to llm.responded when llm.called has none (LangChain pattern).
                -- This mirrors run_builder's "prefer called, fall back to responded" logic
                -- and prevents double-counting when both events carry the field.
                CASE
                    WHEN SUM(CASE WHEN r.event_type = 'llm.called'
                                 THEN COALESCE((r.payload->>'prompt_tokens')::integer, 0)
                                 ELSE 0 END) > 0
                    THEN SUM(CASE WHEN r.event_type = 'llm.called'
                                 THEN COALESCE((r.payload->>'prompt_tokens')::integer, 0)
                                 ELSE 0 END)
                    ELSE SUM(CASE WHEN r.event_type = 'llm.responded'
                                 THEN COALESCE((r.payload->>'prompt_tokens')::integer, 0)
                                 ELSE 0 END)
                END AS prompt_tokens,
                SUM(COALESCE((r.payload->>'completion_tokens')::integer, 0)) AS completion_tokens,
                (SELECT MIN(c.payload->>'model') FROM events c
                 WHERE c.run_id = r.run_id AND c.event_type = 'llm.called'
                   AND c.payload->>'model' IS NOT NULL) AS model
            FROM events r
            WHERE r.run_id = ANY($1::text[])
              AND r.event_type IN ('llm.called', 'llm.responded')
            GROUP BY r.run_id
            """,
            run_ids,
        )
    return {r["run_id"]: dict(r) for r in rows}
