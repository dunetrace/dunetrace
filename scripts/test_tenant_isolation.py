#!/usr/bin/env python3
"""
Cross-tenant isolation smoke test for the A1 multi-tenancy refactor.

Creates two orgs with an IDENTICALLY-NAMED agent in each (the exact scenario that
would leak or pollute data if any org_id filter were missing or wrong), inserts
real rows via the actual application write paths (not raw SQL guesses at schema),
then proves org A cannot see org B's data through the real query functions in
ingest_svc / detector_svc / alerts_svc / api_svc.

Requires a running Postgres reachable at DATABASE_URL (defaults to the local
docker-compose instance). Does NOT go through the HTTP API — the running stack's
AUTH_MODE=dev collapses all API keys to org_id="default", which would defeat the
whole point of this test. Instead this calls the query-layer functions directly,
the same functions the HTTP routers call after resolving org_id via require_org /
_resolve_org_id.

Usage:
    docker compose up -d
    python scripts/test_tenant_isolation.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in [
    "packages/sdk-py",
    "services/explainer",
    "services/ingest",
    "services/detector",
    "services/alerts",
    "services/api",
]:
    p = str(_ROOT / _p)
    if p not in sys.path:
        sys.path.insert(0, p)

import asyncpg

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dunetrace:dunetrace@localhost:5432/dunetrace"
)

_TS = int(time.time())
ORG_A = f"isotest-org-a-{_TS}"
ORG_B = f"isotest-org-b-{_TS}"
AGENT = f"isotest-shared-agent-{_TS}"  # SAME agent_id in both orgs — the whole point
VERSION = "v1"

RUN_A = f"isotest-run-a-{_TS}"
RUN_B = f"isotest-run-b-{_TS}"

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}  {detail}")


async def main() -> None:
    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL, min_size=1, max_size=5, statement_cache_size=0
    )

    # Wire the real pool into every service module we're about to call directly.
    import ingest_svc.db.postgres as ingest_db
    import detector_svc.db as detector_db
    import alerts_svc.db as alerts_db
    import api_svc.db.queries as api_db

    ingest_db._pool = pool
    detector_db._pool = pool
    alerts_db._pool = pool
    api_db._pool = pool

    from ingest_svc.schemas import IngestEvent
    from dunetrace.models import FailureSignal, FailureType, Severity

    print(f"Fixtures: org_a={ORG_A} org_b={ORG_B} agent={AGENT}\n")

    try:
        # ── Fixtures: identical agent in two orgs, via real write paths ──────────
        for org, run_id in [(ORG_A, RUN_A), (ORG_B, RUN_B)]:
            events = [
                IngestEvent(
                    event_type="run.started",
                    run_id=run_id,
                    agent_id=AGENT,
                    agent_version=VERSION,
                    step_index=0,
                    payload={"tools": ["search"]},
                ),
                IngestEvent(
                    event_type="tool.called",
                    run_id=run_id,
                    agent_id=AGENT,
                    agent_version=VERSION,
                    step_index=1,
                    payload={"tool_name": "search", "args": "q"},
                ),
                IngestEvent(
                    event_type="run.completed",
                    run_id=run_id,
                    agent_id=AGENT,
                    agent_version=VERSION,
                    step_index=1,
                    payload={"exit_reason": "completed"},
                ),
            ]
            await ingest_db.insert_events(events, str(uuid.uuid4()), org)
            await detector_db.mark_run_processed(run_id, AGENT, VERSION, "completed", 1, org)

        signal_a = FailureSignal(
            failure_type=FailureType.TOOL_LOOP,
            severity=Severity.HIGH,
            run_id=RUN_A,
            agent_id=AGENT,
            agent_version=VERSION,
            step_index=1,
            confidence=0.9,
            evidence={"tool": "search"},
        )
        signal_b = FailureSignal(
            failure_type=FailureType.TOOL_LOOP,
            severity=Severity.HIGH,
            run_id=RUN_B,
            agent_id=AGENT,
            agent_version=VERSION,
            step_index=1,
            confidence=0.9,
            evidence={"tool": "search"},
        )
        await detector_db.write_signals([signal_a], shadow=False, org_id=ORG_A)
        await detector_db.write_signals([signal_b], shadow=False, org_id=ORG_B)

        await detector_db.upsert_fired_issues(ORG_A, AGENT, ["TOOL_LOOP"])
        await detector_db.upsert_fired_issues(ORG_B, AGENT, ["TOOL_LOOP"])

        await alerts_db.record_alert_sent(ORG_A, AGENT, "TOOL_LOOP")
        # org B deliberately does NOT record an alert — dedup state should differ.

        signal_id_a = await pool.fetchval("SELECT id FROM failure_signals WHERE run_id = $1", RUN_A)
        signal_id_b = await pool.fetchval("SELECT id FROM failure_signals WHERE run_id = $1", RUN_B)

        detector_a = await api_db.create_custom_detector(
            ORG_A, AGENT, "CUSTOM_ISOTEST", "test detector", {"conditions": []}
        )

        print("Fixtures created. Running isolation checks...\n")

        # ── api_svc: get_run_detail ───────────────────────────────────────────────
        own = await api_db.get_run_detail(ORG_A, RUN_A)
        check("get_run_detail: org A sees its own run", own is not None)

        cross = await api_db.get_run_detail(ORG_B, RUN_A)
        check("get_run_detail: org B CANNOT see org A's run", cross is None)

        # ── api_svc: get_signal_by_id ─────────────────────────────────────────────
        own_sig = await api_db.get_signal_by_id(ORG_A, signal_id_a)
        check("get_signal_by_id: org A sees its own signal", own_sig is not None)

        cross_sig = await api_db.get_signal_by_id(ORG_B, signal_id_a)
        check("get_signal_by_id: org B CANNOT see org A's signal", cross_sig is None)

        # ── api_svc: list_agents doesn't merge same-named agent across orgs ──────
        rows_a, total_a = await api_db.list_agents(ORG_A, 0, 50)
        rows_b, total_b = await api_db.list_agents(ORG_B, 0, 50)
        agent_row_a = next((r for r in rows_a if r["agent_id"] == AGENT), None)
        agent_row_b = next((r for r in rows_b if r["agent_id"] == AGENT), None)
        check(
            "list_agents: org A's run_count reflects only its own run",
            agent_row_a is not None and agent_row_a["run_count"] == 1,
            detail=str(agent_row_a),
        )
        check(
            "list_agents: org B's run_count reflects only its own run (not merged with A)",
            agent_row_b is not None and agent_row_b["run_count"] == 1,
            detail=str(agent_row_b),
        )

        # ── api_svc: list_issues isolation ────────────────────────────────────────
        issues_a = await api_db.list_issues(ORG_A, AGENT)
        issues_b = await api_db.list_issues(ORG_B, AGENT)
        check(
            "list_issues: org A sees exactly 1 issue for the shared agent name",
            len(issues_a) == 1,
            detail=str(issues_a),
        )
        check(
            "list_issues: org B sees exactly 1 issue for the shared agent name (its own)",
            len(issues_b) == 1,
            detail=str(issues_b),
        )

        # ── api_svc: custom detector cross-org denial ─────────────────────────────
        own_det = await api_db.get_custom_detector(ORG_A, detector_a["id"])
        check("get_custom_detector: org A sees its own detector", own_det is not None)
        cross_det = await api_db.get_custom_detector(ORG_B, detector_a["id"])
        check("get_custom_detector: org B CANNOT see org A's detector", cross_det is None)

        # ── detector_svc: baseline not polluted by identically-named cross-org agent
        baseline_a = await detector_db.fetch_step_count_baseline(
            ORG_A, AGENT, VERSION, "some-other-run", min_runs=1
        )
        baseline_b = await detector_db.fetch_step_count_baseline(
            ORG_B, AGENT, VERSION, "some-other-run", min_runs=1
        )
        # Each org only has 1 run of its own — baseline should reflect exactly that
        # run's step count (max step_index = 1), not a blended 2-run sample from both orgs.
        check(
            "fetch_step_count_baseline: org A baseline unaffected by org B's identical-name agent",
            baseline_a == 1.0,
            detail=f"baseline_a={baseline_a}",
        )
        check(
            "fetch_step_count_baseline: org B baseline unaffected by org A's identical-name agent",
            baseline_b == 1.0,
            detail=f"baseline_b={baseline_b}",
        )

        # ── alerts_svc: dedup state isolation ─────────────────────────────────────
        dedup_states = await alerts_db.fetch_dedup_states(
            [(ORG_A, AGENT, "TOOL_LOOP"), (ORG_B, AGENT, "TOOL_LOOP")]
        )
        check(
            "fetch_dedup_states: org A has a dedup record (alert was sent)",
            (ORG_A, AGENT, "TOOL_LOOP") in dedup_states,
        )
        check(
            "fetch_dedup_states: org B has NO dedup record (no alert sent there)",
            (ORG_B, AGENT, "TOOL_LOOP") not in dedup_states,
        )

        print(f"\n{_passed} passed, {_failed} failed")

    finally:
        print("\nCleaning up isotest fixtures...")
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM custom_detector_results WHERE agent_id = $1", AGENT)
            await conn.execute("DELETE FROM custom_detectors WHERE agent_id = $1", AGENT)
            await conn.execute("DELETE FROM agent_detector_overrides WHERE agent_id = $1", AGENT)
            await conn.execute("DELETE FROM alert_dedup WHERE agent_id = $1", AGENT)
            await conn.execute("DELETE FROM issues WHERE agent_id = $1", AGENT)
            await conn.execute("DELETE FROM failure_signals WHERE agent_id = $1", AGENT)
            await conn.execute("DELETE FROM processed_runs WHERE agent_id = $1", AGENT)
            await conn.execute("DELETE FROM events WHERE agent_id = $1", AGENT)
        await pool.close()

    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
