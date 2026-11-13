#!/usr/bin/env python3
"""
Migrate a plain `events` table to the monthly-range-partitioned form (audit
Finding 13). OPT-IN, OFFLINE migration for large deployments that want the cheap
partition-drop retention path instead of the batched-DELETE fallback that
prune_old_events now runs automatically on unpartitioned tables.

WHAT IT DOES (all inside one transaction where safe):
  1. Verifies `events` exists and is NOT already partitioned (idempotent: exits
     cleanly if already partitioned).
  2. Renames `events` -> `events_premigration`.
  3. Creates the partitioned `events` table (matching ingest_svc's DDL) + indexes.
  4. Creates monthly partitions covering the full historical range of the data,
     plus current+3 future months and an events_default catch-all.
  5. Re-inserts all rows from events_premigration into events (routed to
     partitions) in batches.
  6. Leaves `events_premigration` in place — you drop it manually once you've
     verified row counts match (safety).

REQUIREMENTS / WARNINGS:
  - TAKE A BACKUP FIRST. This rewrites your entire events table.
  - Run with ingest (and anything writing events) STOPPED — this is an offline
    migration; concurrent writes during the rename/re-insert will be lost.
  - TEST ON A COPY of production data first. Re-inserting millions of rows takes
    time and disk (both tables coexist until you drop events_premigration).
  - Set DATABASE_URL in the environment.

USAGE:
  DATABASE_URL=postgres://... python scripts/migrate_events_to_partitioned.py --yes
  (omit --yes for a dry run that only reports what it would do)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date

try:
    import asyncpg
except ImportError:
    print("asyncpg required: pip install asyncpg", file=sys.stderr)
    sys.exit(1)

BATCH = 50_000

PARTITIONED_DDL = """
CREATE TABLE events (
    id             BIGSERIAL        NOT NULL,
    batch_id       TEXT             NOT NULL,
    event_type     TEXT             NOT NULL,
    run_id         TEXT             NOT NULL,
    agent_id       TEXT             NOT NULL,
    agent_version  TEXT             NOT NULL,
    step_index     INTEGER          NOT NULL,
    timestamp      DOUBLE PRECISION NOT NULL,
    payload        JSONB            NOT NULL DEFAULT '{}',
    parent_run_id  TEXT,
    received_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    event_id       TEXT,
    org_id         TEXT,
    trace_id       TEXT,
    conversation_id TEXT,
    PRIMARY KEY (id, received_at)
) PARTITION BY RANGE (received_at);
CREATE INDEX idx_events_event_id ON events(event_id);
CREATE INDEX idx_events_run_id   ON events(run_id);
CREATE INDEX idx_events_agent    ON events(agent_id, received_at DESC);
CREATE INDEX idx_events_type     ON events(event_type);
"""


def _month_bounds(d: date) -> tuple[str, str]:
    start = date(d.year, d.month, 1)
    end = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return start.isoformat(), end.isoformat()


def _months_between(lo: date, hi: date) -> list[date]:
    months, cur = [], date(lo.year, lo.month, 1)
    while cur <= hi:
        months.append(cur)
        cur = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return months


async def main(apply: bool) -> int:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2
    conn = await asyncpg.connect(dsn)
    try:
        relkind = await conn.fetchval("SELECT relkind FROM pg_class WHERE relname='events'")
        if relkind is None:
            print("No `events` table found — nothing to migrate.")
            return 0
        if relkind == "p":
            print("`events` is already partitioned — nothing to do.")
            return 0

        total = await conn.fetchval("SELECT count(*) FROM events")
        rng = await conn.fetchrow(
            "SELECT min(received_at)::date AS lo, max(received_at)::date AS hi FROM events"
        )
        lo = rng["lo"] or date.today()
        hi = rng["hi"] or date.today()
        # extend range to current + 3 future months so live ingest has partitions
        today = date.today()
        far = date(today.year + ((today.month + 3) > 12), ((today.month + 2) % 12) + 1, 1)
        months = _months_between(min(lo, today), max(hi, far))

        print(f"events: plain table, {total} rows, data {lo}..{hi}")
        print(f"would create {len(months)} monthly partitions + events_default")
        if not apply:
            print(
                "\nDRY RUN — re-run with --yes to perform the migration "
                "(BACKUP + stop writers first)."
            )
            return 0

        async with conn.transaction():
            await conn.execute("ALTER TABLE events RENAME TO events_premigration")
            # Renaming a table does NOT rename its indexes — they keep their
            # original names and stay attached to events_premigration. Without
            # this step PARTITIONED_DDL below fails with DuplicateTableError on
            # the first index it recreates (idx_events_event_id), aborting the
            # whole migration. Rename them out of the way rather than dropping
            # them, so events_premigration stays queryable for the row-count
            # verification and any manual rollback.
            await conn.execute(
                """
                DO $$
                DECLARE r record;
                BEGIN
                    FOR r IN
                        SELECT indexname FROM pg_indexes
                        WHERE tablename = 'events_premigration'
                          AND schemaname = current_schema()
                    LOOP
                        EXECUTE format(
                            'ALTER INDEX %I RENAME TO %I',
                            r.indexname, left(r.indexname, 49) || '_premigration'
                        );
                    END LOOP;
                END $$;
                """
            )
            await conn.execute(PARTITIONED_DDL)
            await conn.execute("CREATE TABLE events_default PARTITION OF events DEFAULT")
            for m in months:
                s, e = _month_bounds(m)
                name = f"events_{m.year}{m.month:02d}"
                await conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF events "
                    f"FOR VALUES FROM ('{s}') TO ('{e}')"
                )
            # Re-insert in batches by id from the old table.
            moved = 0
            last_id = 0
            cols = (
                "id,batch_id,event_type,run_id,agent_id,agent_version,step_index,"
                "timestamp,payload,parent_run_id,received_at,event_id,org_id,"
                "trace_id,conversation_id"
            )
            while True:
                n = await conn.execute(
                    f"INSERT INTO events ({cols}) "
                    f"SELECT {cols} FROM events_premigration "
                    f"WHERE id > $1 ORDER BY id LIMIT {BATCH}",
                    last_id,
                )
                cnt = int(n.split()[-1]) if n.startswith("INSERT") else 0
                if cnt == 0:
                    break
                last_id = await conn.fetchval(
                    "SELECT max(id) FROM events_premigration WHERE id <= "
                    "(SELECT max(id) FROM (SELECT id FROM events_premigration WHERE id > $1 "
                    f"ORDER BY id LIMIT {BATCH}) s)",
                    last_id,
                )
                moved += cnt
                print(f"  moved {moved}/{total}")
                if last_id is None:
                    break

        new_total = await conn.fetchval("SELECT count(*) FROM events")
        print(f"\nDone. events (partitioned) now has {new_total} rows (was {total}).")
        print("Verify counts match, then drop the backup: DROP TABLE events_premigration;")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="actually perform the migration")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.yes)))
