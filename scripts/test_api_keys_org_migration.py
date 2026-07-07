#!/usr/bin/env python3
"""
Regression test for the api_keys.customer_id / org_id drift fix.

Background: _MULTI_TENANCY_DDL's customer_id -> org_id rename only fires when
org_id does not already exist on the table. Any install where org_id got created
some other way first (observed on a real dev DB: org_id added, customer_id left
behind with its original per-tenant values, api_keys.agent_id already dropped so
_backfill_org_id()'s own guard makes it a permanent no-op) is left with customer_id
NOT NULL and no default -- create_api_key() only ever writes org_id, so every new
key insert fails with a not-null violation. This script proves the fix: a fresh
install still gets one clean org_id column, and an install caught in the drifted
state gets its real per-tenant customer_id values recovered into org_id (with a
matching organizations row) rather than silently discarded.

Uses a disposable database (not the shared docker-compose one) so it's safe to
run repeatedly and never touches real data.

Requires a running Postgres reachable at DATABASE_URL's host (defaults to the
local docker-compose instance, but connects to its own throwaway database).

Usage:
    docker compose up -d postgres
    python scripts/test_api_keys_org_migration.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in ["packages/schemas-py", "services/ingest"]:
    p = str(_ROOT / _p)
    if p not in sys.path:
        sys.path.insert(0, p)

import asyncpg  # noqa: E402

_BASE_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://dunetrace:dunetrace@localhost:5432/dunetrace"
)
_ADMIN_DSN = _BASE_DSN.rsplit("/", 1)[0] + "/postgres"
_TEST_DB = f"dunetrace_migration_test_{int(time.time())}"
_TEST_DSN = _BASE_DSN.rsplit("/", 1)[0] + "/" + _TEST_DB

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
    admin_conn = await asyncpg.connect(dsn=_ADMIN_DSN)
    try:
        await admin_conn.execute(f'CREATE DATABASE "{_TEST_DB}"')
    finally:
        await admin_conn.close()

    os.environ["DATABASE_URL"] = _TEST_DSN
    from ingest_svc.db import postgres

    try:
        await _run_checks(postgres)
    finally:
        await postgres.close_pool()
        admin_conn = await asyncpg.connect(dsn=_ADMIN_DSN)
        try:
            await admin_conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = '{_TEST_DB}' AND pid <> pg_backend_pid()"
            )
            await admin_conn.execute(f'DROP DATABASE IF EXISTS "{_TEST_DB}"')
        finally:
            await admin_conn.close()

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


async def _run_checks(postgres) -> None:
    await postgres.init_pool()
    await postgres.ensure_schema()

    # ── Scenario 1: fresh install never had customer_id at all ──────────────────
    async with postgres._pool.acquire() as conn:
        cols = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'api_keys'"
            )
        }
    check("fresh install: api_keys has org_id", "org_id" in cols)
    check("fresh install: api_keys has no customer_id", "customer_id" not in cols)

    # ── Scenario 2: simulate the drifted state on this same fresh schema ────────
    # (org_id already exists per scenario 1 -- add customer_id back with a real,
    # distinct tenant value and null out org_id for one row, matching exactly
    # what the dev DB that exposed this bug looked like.)
    async with postgres._pool.acquire() as conn:
        await conn.execute("ALTER TABLE api_keys ALTER COLUMN org_id DROP NOT NULL")
        await conn.execute("ALTER TABLE api_keys ADD COLUMN customer_id TEXT")
        await conn.execute(
            "INSERT INTO api_keys (key, org_id, customer_id) "
            "VALUES ('dt_migrationtest_key', NULL, 'widgets-inc')"
        )

        await postgres.ensure_schema()  # re-run the real migration against the drifted state

        cols_after = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'api_keys'"
            )
        }
        check(
            "drifted install: customer_id dropped after re-migration",
            "customer_id" not in cols_after,
        )

        row = await conn.fetchrow("SELECT org_id FROM api_keys WHERE key = 'dt_migrationtest_key'")
        check(
            "drifted install: org_id recovered from customer_id (not defaulted)",
            row is not None and row["org_id"] == "widgets-inc",
            detail=f"org_id={row['org_id'] if row else None}",
        )

        org_row = await conn.fetchrow("SELECT id FROM organizations WHERE id = 'widgets-inc'")
        check(
            "drifted install: matching organizations row created for the recovered org_id",
            org_row is not None,
        )

        # ── Scenario 3: the original failure this whole thing is regression-testing ──
        await conn.execute(
            "INSERT INTO organizations (id, name) VALUES ('new-org', 'new-org') "
            "ON CONFLICT (id) DO NOTHING"
        )
        try:
            await conn.execute(
                "INSERT INTO api_keys (key, org_id) VALUES ('dt_migrationtest_key2', 'new-org')"
            )
            insert_ok = True
        except Exception as exc:  # noqa: BLE001
            insert_ok = False
            insert_exc = exc
        check(
            "new key insert (org_id only, no customer_id) succeeds post-migration",
            insert_ok,
            detail="" if insert_ok else str(insert_exc),
        )

        # ── Idempotency: running the migration again on an already-clean schema is a no-op ──
        await postgres.ensure_schema()
        cols_idempotent = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'api_keys'"
            )
        }
        check(
            "idempotent: second re-run doesn't reintroduce customer_id",
            "customer_id" not in cols_idempotent,
        )


if __name__ == "__main__":
    asyncio.run(main())
