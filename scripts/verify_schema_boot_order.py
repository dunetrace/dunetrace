#!/usr/bin/env python3
"""
Boot-order verification for the shared schema, against a real Postgres.

What it proves, on a throwaway server:

  1. BOOT-ORDER MATRIX. For each of the six services in turn, on a FRESH
     database: boot that service's schema-init first, then all the others.
     No exception; schema_version == CURRENT_SCHEMA_VERSION; every table any
     service declares OR queries exists.
  2. CONCURRENT APPLIERS. apply_migrations from N connections at once on a
     fresh database: exactly one application per migration (counted from the
     runner's own "Applying migration" log line, not just the PK).
  3. IDEMPOTENCE. Every init re-run twice more on each matrix database: the
     second and third runs make no schema change (columns, indexes and
     constraints compared before/after).
  4. UPGRADE FROM HEAD. A database pre-created by the committed (HEAD) service
     DDL — `git archive HEAD` into a scratch tree, booted in a subprocess —
     with rows inserted, then brought forward by the working-tree inits:
     every ADD COLUMN of migrations 6+ exists, NOT NULL ones are backfilled,
     no seeded row is lost, and the upgraded database is idempotent too.

The "every table any service queries" list is derived, not hand-written: the
CREATE TABLE names across migrations + services (via check_schema_owners's
parser), plus every FROM/INTO/UPDATE/JOIN <name> in the services' SQL string
literals (walked with `ast`, CTE names excluded). A table nobody declares any
more fails the run.

Safety: the script creates and drops databases, and every service's config.py
loads the repo's .env — which may hold a production DATABASE_URL — into the
environment with setdefault at import. DATABASE_URL is therefore pinned in
os.environ before any service module is imported, and non-loopback hosts are
refused unless --allow-remote is given.

Usage:
    export PATH=<venv>/bin:$PATH          # asyncpg + the services' deps
    python scripts/verify_schema_boot_order.py --start-container
    DATABASE_URL=postgresql://u:p@127.0.0.1:55432/postgres \\
        python scripts/verify_schema_boot_order.py
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent.parent

# The union of every Makefile test target's PYTHONPATH.
SERVICE_PATHS = [
    "packages/schemas-py",
    "packages/sdk-py",
    "services/explainer",
    "services/ingest",
    "services/detector",
    "services/alerts",
    "services/api",
    "services/semantic",
    "services/integrations",
]

SERVICE_PACKAGES = {
    "ingest": "services/ingest/ingest_svc",
    "detector": "services/detector/detector_svc",
    "alerts": "services/alerts/alerts_svc",
    "api": "services/api/api_svc",
    "semantic": "services/semantic/semantic_svc",
    "integrations": "services/integrations/integrations_svc",
}

# docker-compose's own start order; the matrix rotates each service to the front.
ORDER = ["ingest", "detector", "alerts", "api", "semantic", "integrations"]

DEFAULT_PORT = 55432
CONTAINER_NAME = "dunetrace-schema-boot-check"
CONTAINER_IMAGE = "postgres:16-alpine"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _install_paths(root: Path) -> None:
    for rel in reversed(SERVICE_PATHS):
        p = str(root / rel)
        if p not in sys.path:
            sys.path.insert(0, p)


def _dsn_for(admin_dsn: str, dbname: str) -> str:
    parts = urlsplit(admin_dsn)
    return urlunsplit(parts._replace(path=f"/{dbname}"))


# ── Service boots (working tree) ─────────────────────────────────────────────
#
# Each one is exactly what the service's own startup does — init_pool plus the
# ensure_* calls in the order its main/worker makes them — pointed at `dsn`.


async def _boot_ingest(dsn: str) -> None:
    import ingest_svc.db.postgres as m

    m.settings.DATABASE_URL = dsn
    await m.init_pool()
    try:
        await m.ensure_schema()
    finally:
        await m.close_pool()


async def _boot_detector(dsn: str) -> None:
    import detector_svc.db as m

    m.settings.DATABASE_URL = dsn
    await m.init_pool()
    try:
        await m.ensure_detector_schema()
    finally:
        await m.close_pool()


async def _boot_alerts(dsn: str) -> None:
    import alerts_svc.db as m

    m.settings.DATABASE_URL = dsn
    await m.init_pool()
    try:
        await m.ensure_digest_schema()
        await m.ensure_dedup_schema()
        await m.ensure_semantic_signal_column()
        await m.ensure_alert_integrations_schema()
        await m.ensure_alert_claim_columns()
    finally:
        await m.close_pool()


async def _boot_api(dsn: str) -> None:
    import api_svc.db.queries as m

    m.settings.DATABASE_URL = dsn
    await m.init_pool()  # applies migrations and the API's own DDL
    await m.close_pool()


async def _boot_semantic(dsn: str) -> None:
    import semantic_svc.db as m

    m.settings.DATABASE_URL = dsn
    await m.init_pool()
    try:
        await m.ensure_semantic_schema()
    finally:
        await m.close_pool()


async def _boot_integrations(dsn: str) -> None:
    import integrations_svc.db as m

    m.settings.DATABASE_URL = dsn
    await m.init_pool()
    try:
        await m.ensure_integrations_schema()
        await m.ensure_elevenlabs_schema()
    finally:
        await m.close_pool()


BOOTS = {
    "ingest": _boot_ingest,
    "detector": _boot_detector,
    "alerts": _boot_alerts,
    "api": _boot_api,
    "semantic": _boot_semantic,
    "integrations": _boot_integrations,
}


# ── Expected tables ──────────────────────────────────────────────────────────

_SQLISH_RE = re.compile(
    r"^\s*(?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DO|DROP|TRUNCATE)\b", re.M
)
# Uppercase keywords only: the codebase writes SQL keywords in caps and prose
# in lowercase, which is what keeps docstrings ("reads from the table") out.
_REF_RE = re.compile(
    r"(?<!EPOCH )(?<!DISTINCT )\b(?:FROM|INTO|UPDATE|JOIN)\s+(?:ONLY\s+)?"
    r"([a-z_][a-z0-9_]*)\b(?!\s*\()"
)
_CTE_RE = re.compile(r"(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)([a-z_][a-z0-9_]*)\s+AS\s*\(", re.I)
# `) t`, `) ranked` — a derived table's alias.
_SUBQUERY_ALIAS_RE = re.compile(r"\)\s+(?:AS\s+)?([a-z_][a-z0-9_]*)\b")
# EXTRACT(HOUR FROM x) — the FROM is not a table reference.
_EXTRACT_FROM_RE = re.compile(r"(EXTRACT\(\s*\w+\s+)FROM\b", re.I)
_SQL_COMMENT_RE = re.compile(r"--[^\n]*")
_NOT_TABLES = {
    "set",  # ON CONFLICT DO UPDATE SET
    "skip",  # FOR UPDATE SKIP LOCKED
    "nowait",
    "only",
    "lateral",
    "unnest",
    "generate_series",
    "information_schema",
    "pg_catalog",
}


def _literal_strings(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            # f-string: keep the constant parts and mark each hole with
            # non-word characters, so neither a templated name nor an alias
            # right after one (`UPDATE {table} t SET`) reads as a table.
            yield "".join(v.value if isinstance(v, ast.Constant) else "<hole>" for v in node.values)


def referenced_tables(root: Path) -> dict[str, set[str]]:
    """table -> {"service/file.py", ...} for every FROM/INTO/UPDATE/JOIN in SQL
    string literals across the six service packages."""
    refs: dict[str, set[str]] = {}
    for svc, rel in SERVICE_PACKAGES.items():
        # Aliases are collected across the whole package first: a CTE may be
        # a module constant spliced into several queries, so the string that
        # names it is not the string that reads FROM it.
        sql_strings: list[tuple[str, str]] = []
        aliases: set[str] = set()
        for path in sorted((root / rel).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for s in _literal_strings(tree):
                if not _SQLISH_RE.search(s):
                    continue
                s = _EXTRACT_FROM_RE.sub(r"\1", _SQL_COMMENT_RE.sub("", s))
                aliases |= {m.lower() for m in _CTE_RE.findall(s)}
                aliases |= {m.lower() for m in _SUBQUERY_ALIAS_RE.findall(s)}
                sql_strings.append((path.name, s))
        for fname, s in sql_strings:
            for name in _REF_RE.findall(s):
                if (
                    name in aliases
                    or name in _NOT_TABLES
                    or name.startswith(("pg_", "json_", "jsonb_"))
                    or " " in name
                    or name.startswith("events_")  # monthly partitions
                ):
                    continue
                refs.setdefault(name, set()).add(f"{svc}/{fname}")
    return refs


def declared_tables(root: Path) -> set[str]:
    from check_schema_owners import _CREATE_TABLE_RE, _strip_comments, extract_ddl, read_sources

    tables: set[str] = set()
    for src in read_sources(root).values():
        tables |= {t.table for t in extract_ddl(src).tables}
    # Belt and braces: any CREATE TABLE outside the seven scanned files.
    for rel in SERVICE_PACKAGES.values():
        for path in (root / rel).rglob("*.py"):
            text = _strip_comments(path.read_text(encoding="utf-8"))
            tables |= {m.group(1).lower() for m in _CREATE_TABLE_RE.finditer(text)}
    return tables


def added_columns_since(version: int) -> list[tuple[int, str, str]]:
    """(migration, table, column) for every ADD COLUMN in migrations > version."""
    from check_schema_owners import _ADD_COLUMN_RE
    from dunetrace_schemas.migrations import MIGRATIONS

    out = []
    for number, _name, sql in MIGRATIONS:
        if number <= version:
            continue
        for m in _ADD_COLUMN_RE.finditer(sql):
            out.append((number, m.group(1).lower(), m.group(2).lower()))
    return out


# ── DB helpers ───────────────────────────────────────────────────────────────


async def _admin(admin_dsn: str):
    import asyncpg

    return await asyncpg.connect(admin_dsn)


async def fresh_db(admin_dsn: str, name: str) -> str:
    conn = await _admin(admin_dsn)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()
    return _dsn_for(admin_dsn, name)


async def drop_db(admin_dsn: str, name: str) -> None:
    conn = await _admin(admin_dsn)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await conn.close()


async def snapshot(conn) -> dict[str, set[tuple]]:
    cols = await conn.fetch(
        "SELECT table_name, column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_schema = 'public'"
    )
    idx = await conn.fetch(
        "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'"
    )
    cons = await conn.fetch(
        "SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid) "
        "FROM pg_constraint WHERE connamespace = 'public'::regnamespace"
    )
    return {
        "columns": {tuple(r) for r in cols},
        "indexes": {tuple(r) for r in idx},
        "constraints": {tuple(r) for r in cons},
    }


def snapshot_diff(before: dict, after: dict) -> list[str]:
    out = []
    for kind in ("columns", "indexes", "constraints"):
        for row in sorted(after[kind] - before[kind], key=str):
            out.append(f"+ {kind}: {row}")
        for row in sorted(before[kind] - after[kind], key=str):
            out.append(f"- {kind}: {row}")
    return out


async def db_state(dsn: str):
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        version = await conn.fetchval("SELECT COALESCE(MAX(version), 0) FROM schema_version")
        rows = await conn.fetchval("SELECT COUNT(*) FROM schema_version")
        tables = {
            r["tablename"]
            for r in await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        }
        snap = await snapshot(conn)
    finally:
        await conn.close()
    return version, rows, tables, snap


# ── Scenarios ────────────────────────────────────────────────────────────────


class Result:
    def __init__(self, name: str):
        self.name = name
        self.failures: list[str] = []
        self.notes: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.failures

    def fail(self, msg: str) -> None:
        self.failures.append(msg)


async def _boot_all(order: list[str], dsn: str, result: Result, label: str) -> None:
    for svc in order:
        try:
            await BOOTS[svc](dsn)
        except Exception as exc:  # keep going: the later services' errors matter too
            result.fail(f"{label}: {svc} raised {type(exc).__name__}: {exc}")
            result.notes.append(traceback.format_exc())


async def _check_current(dsn: str, expected: set[str], result: Result, label: str):
    from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

    version, rows, tables, snap = await db_state(dsn)
    if version != CURRENT_SCHEMA_VERSION or rows != CURRENT_SCHEMA_VERSION:
        result.fail(
            f"{label}: schema_version max={version} rows={rows}, "
            f"expected {CURRENT_SCHEMA_VERSION}/{CURRENT_SCHEMA_VERSION}"
        )
    missing = sorted(expected - tables)
    if missing:
        result.fail(f"{label}: tables missing after boot: {missing}")
    return snap


async def _check_idempotent(order, dsn, snap1, result: Result, label: str) -> None:
    for rerun in (2, 3):
        await _boot_all(order, dsn, result, f"{label} run {rerun}")
        _v, _r, _t, snap = await db_state(dsn)
        diff = snapshot_diff(snap1, snap)
        if diff:
            result.fail(f"{label}: run {rerun} changed the schema:\n  " + "\n  ".join(diff))


async def run_matrix(admin_dsn: str, expected: set[str]) -> list[Result]:
    results = []
    for i, first in enumerate(ORDER):
        result = Result(f"{first}-first")
        order = [first] + [s for s in ORDER if s != first]
        name = f"boot_{i}_{first}_first"
        dsn = await fresh_db(admin_dsn, name)
        try:
            await _boot_all(order, dsn, result, "boot")
            snap1 = await _check_current(dsn, expected, result, "boot")
            await _check_idempotent(order, dsn, snap1, result, "idempotence")
        finally:
            await drop_db(admin_dsn, name)
        results.append(result)
        print(f"  matrix {result.name:<20} {'OK' if result.ok else 'FAIL'}", flush=True)
    return results


class _ApplyCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.applied: list[int] = []

    def emit(self, record):
        if record.msg.startswith("Applying migration"):
            self.applied.append(record.args[0])


async def run_concurrent(admin_dsn: str, n: int) -> Result:
    import asyncpg

    from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION, apply_migrations

    result = Result(f"concurrent-x{n}")
    name = "boot_concurrent"
    dsn = await fresh_db(admin_dsn, name)
    counter = _ApplyCounter()
    mig_logger = logging.getLogger("dunetrace.migrations")
    saved = (mig_logger.level, mig_logger.propagate)
    mig_logger.setLevel(logging.INFO)
    mig_logger.propagate = False
    mig_logger.addHandler(counter)
    try:
        conns = [await asyncpg.connect(dsn) for _ in range(n)]
        try:
            outcomes = await asyncio.gather(
                *(apply_migrations(c) for c in conns), return_exceptions=True
            )
            for i, outcome in enumerate(outcomes):
                if isinstance(outcome, BaseException):
                    result.fail(f"applier {i} raised {type(outcome).__name__}: {outcome}")
                elif outcome != CURRENT_SCHEMA_VERSION:
                    result.fail(f"applier {i} returned version {outcome}")
            rows = await conns[0].fetch("SELECT version FROM schema_version ORDER BY version")
            versions = [r["version"] for r in rows]
            if versions != list(range(1, CURRENT_SCHEMA_VERSION + 1)):
                result.fail(f"schema_version rows are {versions}")
        finally:
            for c in conns:
                await c.close()
        if sorted(counter.applied) != list(range(1, CURRENT_SCHEMA_VERSION + 1)):
            result.fail(
                f"migrations were applied {len(counter.applied)} times across {n} appliers "
                f"(expected exactly once each): {sorted(counter.applied)}"
            )
        result.notes.append(f"{len(counter.applied)} applications across {n} concurrent appliers")
    finally:
        mig_logger.removeHandler(counter)
        mig_logger.setLevel(saved[0])
        mig_logger.propagate = saved[1]
        await drop_db(admin_dsn, name)
    print(f"  {result.name:<27} {'OK' if result.ok else 'FAIL'}", flush=True)
    return result


# ── Upgrade from HEAD ────────────────────────────────────────────────────────

# Runs inside a subprocess whose PYTHONPATH is a `git archive HEAD` tree, so
# the committed service DDL (and the committed migrations) build the database
# exactly as a deployment on HEAD would have, then seeds rows into every table
# the working tree's migrations go on to ALTER.
_HEAD_CHILD = r"""
import asyncio, json, sys
dsn = sys.argv[1]

SEEDS = [
    ("organizations", {"id": "acme", "name": "Acme"}),
    ("api_keys", {"key": "dt_old_key", "org_id": "acme", "key_hash": "h" * 64,
                  "key_prefix": "dt_old_key", "active": True, "agent_id": "a1",
                  "customer_id": "acme", "company_id": "acme"}),
    ("runs", {"run_id": "r1", "org_id": "acme", "agent_id": "a1", "agent_version": "v1"}),
    ("processed_runs", {"run_id": "r1", "org_id": "acme", "agent_id": "a1",
                        "agent_version": "v1", "trigger": "completed"}),
    ("failure_signals", {"failure_type": "TOOL_LOOP", "severity": "high", "run_id": "r1",
                         "agent_id": "a1", "agent_version": "v1", "step_index": 1,
                         "confidence": 0.9, "evidence": "{}", "org_id": "acme"}),
    ("fixes", {"org_id": "acme", "run_id": "r1", "signal_id": 1, "fix_content": "x",
               "applied_via": "clipboard"}),
    ("policies", {"org_id": "acme", "agent_id": "a1", "name": "p", "condition": "{}",
                  "action": "{}"}),
    ("policy_evaluations", {"org_id": "acme", "policy_id": 1, "policy_name": "p",
                            "agent_id": "a1", "run_id": "r1"}),
    ("custom_detectors", {"org_id": "acme", "agent_id": "a1", "name": "cd",
                          "description": "d", "config_json": "{}"}),
    ("custom_detector_results", {"org_id": "acme", "detector_id": 1, "run_id": "r1",
                                 "agent_id": "a1", "fired": True}),
    ("issues", {"org_id": "acme", "agent_id": "a1", "failure_type": "TOOL_LOOP"}),
    ("org_enabled_packs", {"org_id": "acme", "pack_name": "voice"}),
    ("run_state_metrics", {"run_id": "r1", "org_id": "acme", "agent_id": "a1",
                           "state": "llm", "total_ms": 10, "segment_count": 1}),
    ("org_alert_integrations", {"org_id": "acme", "provider": "slack",
                                "encrypted_credentials": "enc"}),
    ("linear_issue_signals", {"org_id": "acme", "signal_id": 1, "linear_issue_id": "LIN-1"}),
    ("signal_groups", {"org_id": "acme", "agent_id": "a1", "evaluator": "HALLUCINATION",
                       "root_cause_hash": "h", "root_cause_sample": "s"}),
    ("signal_group_members", {"group_id": 1, "signal_id": 1, "run_id": "r1"}),
    ("signal_group_overrides", {"group_id": 1, "fp_count": 1}),
    ("org_semantic_evaluation_usage", {"org_id": "acme", "month": "2026-09", "eval_count": 1}),
    ("semantic_evaluation_log", {"org_id": "acme", "agent_id": "a1",
                                 "evaluator": "HALLUCINATION", "fired": True,
                                 "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0}),
    ("external_evaluation_integrations", {"org_id": "acme", "provider": "langfuse",
                                          "endpoint_url": "http://x",
                                          "encrypted_credentials": "enc"}),
    ("external_evaluation_processed", {"org_id": "acme", "provider": "langfuse",
                                       "external_id": "e1"}),
    ("elevenlabs_integrations", {"org_id": "acme", "encrypted_credentials": "enc"}),
    ("elevenlabs_generations", {"org_id": "acme", "generation_id": "g1",
                                "generated_at": 1.0, "character_count": 5}),
]

async def boot_all():
    import ingest_svc.db.postgres as ingest
    ingest.settings.DATABASE_URL = dsn
    await ingest.init_pool(); await ingest.ensure_schema(); await ingest.close_pool()
    import detector_svc.db as detector
    detector.settings.DATABASE_URL = dsn
    await detector.init_pool(); await detector.ensure_detector_schema(); await detector.close_pool()
    import alerts_svc.db as alerts
    alerts.settings.DATABASE_URL = dsn
    await alerts.init_pool()
    await alerts.ensure_digest_schema(); await alerts.ensure_dedup_schema()
    await alerts.ensure_semantic_signal_column(); await alerts.ensure_alert_integrations_schema()
    await alerts.ensure_alert_claim_columns()
    await alerts.close_pool()
    import api_svc.db.queries as api
    api.settings.DATABASE_URL = dsn
    await api.init_pool(); await api.close_pool()
    import semantic_svc.db as semantic
    semantic.settings.DATABASE_URL = dsn
    await semantic.init_pool(); await semantic.ensure_semantic_schema(); await semantic.close_pool()
    import integrations_svc.db as integ
    integ.settings.DATABASE_URL = dsn
    await integ.init_pool()
    await integ.ensure_integrations_schema(); await integ.ensure_elevenlabs_schema()
    await integ.close_pool()

async def seed():
    import asyncpg
    conn = await asyncpg.connect(dsn)
    counts, failed = {}, False
    for table, values in SEEDS:
        cols = {r["column_name"] for r in await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = $1", table)}
        if not cols:
            print("SEED-SKIP", table, "(not created at HEAD)")
            continue
        use = {k: v for k, v in values.items() if k in cols}
        sql = "INSERT INTO %s (%s) VALUES (%s)" % (
            table, ", ".join(use), ", ".join("$%d" % (i + 1) for i in range(len(use))))
        try:
            await conn.execute(sql, *use.values())
            counts[table] = await conn.fetchval("SELECT COUNT(*) FROM " + table)
        except Exception as exc:
            print("SEED-FAIL", table, type(exc).__name__, exc)
            failed = True
    await conn.close()
    print("SEEDED " + json.dumps(counts))
    return failed

async def main():
    await boot_all()
    from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION
    print("HEAD-VERSION", CURRENT_SCHEMA_VERSION)
    if await seed():
        sys.exit(2)

asyncio.run(main())
"""


def stage_head_tree() -> str:
    tmp = tempfile.mkdtemp(prefix="dunetrace-head-")
    paths = " ".join(SERVICE_PATHS)
    subprocess.run(f"git archive HEAD {paths} | tar -x -C {tmp}", shell=True, cwd=ROOT, check=True)
    return tmp


async def run_upgrade(admin_dsn: str, head_tree: str, expected: set[str], order: list[str]):
    import asyncpg

    result = Result(f"upgrade-from-HEAD ({order[0]}-first)")
    name = f"boot_upgrade_{order[0]}"
    dsn = await fresh_db(admin_dsn, name)
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(str(Path(head_tree) / p) for p in SERVICE_PATHS)
        env["DATABASE_URL"] = dsn
        proc = subprocess.run(
            [sys.executable, "-c", _HEAD_CHILD, dsn],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        head_version, seeded = None, {}
        for line in proc.stdout.splitlines():
            if line.startswith("HEAD-VERSION "):
                head_version = int(line.split()[1])
            elif line.startswith("SEEDED "):
                seeded = json.loads(line[len("SEEDED ") :])
            elif line.startswith("SEED-"):
                result.notes.append(line)
        if proc.returncode != 0 or head_version is None:
            result.fail(
                f"HEAD-tree boot/seed failed (rc={proc.returncode}):\n"
                + proc.stdout[-2000:]
                + proc.stderr[-4000:]
            )
            return result
        result.notes.append(f"HEAD schema version {head_version}; seeded {len(seeded)} tables")

        await _boot_all(order, dsn, result, "upgrade boot")
        snap1 = await _check_current(dsn, expected, result, "upgrade")

        conn = await asyncpg.connect(dsn)
        try:
            for number, table, column in added_columns_since(head_version):
                row = await conn.fetchrow(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2",
                    table,
                    column,
                )
                if row is None:
                    result.fail(f"migration {number}: {table}.{column} missing after upgrade")
                    continue
                if row["is_nullable"] == "NO":
                    nulls = await conn.fetchval(
                        f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
                    )
                    if nulls:
                        result.fail(
                            f"migration {number}: {table}.{column} NOT NULL but {nulls} NULL rows"
                        )
            for table, count in seeded.items():
                now = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                if now != count:
                    result.fail(f"{table}: {count} rows before upgrade, {now} after")
            cons = {
                r["conname"]
                for r in await conn.fetch(
                    "SELECT conname FROM pg_constraint WHERE conrelid = 'issues'::regclass"
                )
            }
            if "issues_org_agent_failure_type_key" not in cons:
                result.fail(f"issues: org-scoped unique key missing, have {sorted(cons)}")
            if "issues_agent_id_failure_type_key" in cons:
                result.fail("issues: pre-tenancy (agent_id, failure_type) key still present")
        finally:
            await conn.close()

        await _check_idempotent(order, dsn, snap1, result, "upgrade idempotence")
    finally:
        await drop_db(admin_dsn, name)
    print(f"  {result.name:<40} {'OK' if result.ok else 'FAIL'}", flush=True)
    return result


# ── Container ────────────────────────────────────────────────────────────────


def start_container(port: int) -> str:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-e",
            "POSTGRES_USER=dt",
            "-e",
            "POSTGRES_PASSWORD=dt",
            "-e",
            "POSTGRES_DB=postgres",
            "-p",
            f"127.0.0.1:{port}:5432",
            CONTAINER_IMAGE,
        ],
        check=True,
        capture_output=True,
    )
    return f"postgresql://dt:dt@127.0.0.1:{port}/postgres"


def stop_container() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


async def wait_ready(dsn: str, timeout: float = 60.0) -> None:
    import asyncpg

    deadline = time.monotonic() + timeout
    while True:
        try:
            conn = await asyncpg.connect(dsn, timeout=2)
            await conn.close()
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.5)


# ── Main ─────────────────────────────────────────────────────────────────────


def _print_result(r: Result) -> None:
    print(f"\n{r.name}: {'OK' if r.ok else 'FAIL'}")
    for n in r.notes:
        if not n.startswith("Traceback"):
            print(f"  note: {n}")
    for f in r.failures:
        print(f"  FAIL: {f}")
    for n in r.notes:
        if n.startswith("Traceback"):
            print("  " + n.replace("\n", "\n  "))


async def _main(args) -> int:
    from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

    admin_dsn = args.database_url
    print(f"target: {admin_dsn.split('@')[-1]}   CURRENT_SCHEMA_VERSION={CURRENT_SCHEMA_VERSION}")
    await wait_ready(admin_dsn)

    declared = declared_tables(ROOT)
    referenced = referenced_tables(ROOT)
    undeclared = {t: sorted(f) for t, f in referenced.items() if t not in declared}
    expected = declared | set(referenced)
    print(f"tables: {len(declared)} declared, {len(referenced)} referenced in SQL")
    static = Result("static: every queried table is declared")
    if undeclared:
        static.fail(f"queried but declared nowhere: {undeclared}")

    results = [static]
    print("boot-order matrix:")
    results += await run_matrix(admin_dsn, expected)
    results.append(await run_concurrent(admin_dsn, args.concurrency))

    if not args.no_upgrade:
        head_tree = stage_head_tree()
        try:
            results.append(await run_upgrade(admin_dsn, head_tree, expected, ORDER))
            results.append(await run_upgrade(admin_dsn, head_tree, expected, ORDER[::-1]))
        finally:
            shutil.rmtree(head_tree, ignore_errors=True)

    for r in results:
        _print_result(r)
    failed = [r for r in results if not r.ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} scenarios passed")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--start-container",
        action="store_true",
        help=f"run a throwaway {CONTAINER_IMAGE} via docker and remove it after",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--no-upgrade", action="store_true", help="skip the upgrade-from-HEAD scenario (needs git)"
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit a non-loopback host (the script creates and drops databases)",
    )
    args = parser.parse_args(argv)

    started = False
    if args.start_container:
        args.database_url = start_container(args.port)
        started = True
    if not args.database_url:
        parser.error("DATABASE_URL is not set; pass --database-url or --start-container")
    host = urlsplit(args.database_url).hostname
    if host not in _LOCAL_HOSTS and not args.allow_remote:
        parser.error(
            f"refusing to create/drop databases on non-local host {host!r} "
            "(pass --allow-remote if you really mean it)"
        )

    # Pin before any service module is imported: their config.py loads the
    # repo .env with setdefault, and that file may name a real database.
    os.environ["DATABASE_URL"] = args.database_url
    os.environ.pop("ENV", None)  # never trip the deployment guard
    logging.basicConfig(level=logging.WARNING)
    _install_paths(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))

    try:
        return asyncio.run(_main(args))
    finally:
        if started and not args.keep_container:
            stop_container()


if __name__ == "__main__":
    sys.exit(main())
