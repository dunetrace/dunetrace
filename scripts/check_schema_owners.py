#!/usr/bin/env python3
"""
Schema-ownership audit: who declares which table, and is it exactly one place?

THE RULE (stated in packages/schemas-py/dunetrace_schemas/migrations.py): a
table touched by more than one service is declared in the migration runner,
not in a service. A single-owner table may stay inline in that service's own
``ensure_schema`` with ``CREATE TABLE IF NOT EXISTS``. Once a table lives in
migrations, the service copy must go — a migration plus one service is two
owners, and two owners is exactly the "whichever service starts first wins"
contract the migration runner exists to replace (``policies.signature`` was
declared by one service, selected by another inside a ``try/except`` that
returned ``[]``, and a cold start in the wrong order silently disabled every
runtime guardrail).

WHAT THIS CHECKS. It scans the six service schema files plus migrations.py as
text (no imports, no database) and extracts:

* every ``CREATE TABLE [IF NOT EXISTS] <name>`` — that file is an *owner* of
  ``<name>``. ``CREATE TABLE ... PARTITION OF`` children are skipped (they are
  storage for a table someone else owns, not a declaration).
* every ``ALTER TABLE <name> ADD COLUMN [IF NOT EXISTS] <col>`` — including
  the ones inside ``DO $$ ... $$`` blocks, multi-line statements, and
  f-strings (the scan is textual, so an f-string is just a string whose
  ``{placeholder}`` is not a table name and is ignored).

It then flags:

* **multi-owner tables** — declared by two or more of the scanned files.
* **stray columns** — an ``ADD COLUMN`` in a file that does not also own the
  table. That is a service growing a table it does not declare, typically a
  migrations-owned one; the column belongs in a new migration.

``tests/test_schema_parity.py`` asserts that duplicated declarations *agree*
column-for-column; this script asserts that they should not exist at all. The
parity test deliberately ignores ``ALTER``; this one does not.

RATCHET. The end state — zero multi-owner tables, zero stray columns — is the
default expectation: with no flags the script exits 1 on any finding. Until the
remaining duplicates are moved into migrations, ``--check`` compares the
current counts against ``scripts/schema_owners_baseline.json`` and fails only
if either count went *up*. ``--baseline`` rewrites that file with the current
counts; the number is meant to only ever go down, and when it reaches zero the
baseline and the default agree.

Usage:
    python scripts/check_schema_owners.py             # strict: any finding fails
    python scripts/check_schema_owners.py --check     # ratchet against baseline
    python scripts/check_schema_owners.py --baseline  # record current counts
    python scripts/check_schema_owners.py --json      # machine-readable report
    python scripts/check_schema_owners.py --max-multi-owner 5

Exit code 0: within budget. 1: a multi-owner table or stray column exceeds the
budget. 2: usage error (unreadable file, malformed baseline).

Stdlib only — it runs in CI before anything is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "scripts" / "schema_owners_baseline.json"

# Owner label -> path relative to the repo root. migrations.py counts as an
# owner: a table it declares must not also be declared by a service.
SCANNED_FILES: dict[str, str] = {
    "migrations": "packages/schemas-py/dunetrace_schemas/migrations.py",
    "ingest": "services/ingest/ingest_svc/db/postgres.py",
    "detector": "services/detector/detector_svc/db.py",
    "api": "services/api/api_svc/db/queries.py",
    "alerts": "services/alerts/alerts_svc/db.py",
    "semantic": "services/semantic/semantic_svc/db.py",
    "integrations": "services/integrations/integrations_svc/db.py",
}

# `CREATE TABLE [IF NOT EXISTS] name` — but not a partition child
# (`CREATE TABLE IF NOT EXISTS events_default PARTITION OF events`). `\w+`
# deliberately does not match an f-string placeholder like `{name}`, so the
# per-month partition loop in ingest is ignored the same way. `\s+` spans
# newlines, so a name on its own line still matches.
#
# The `(?!IF\b)` guard is load-bearing: when the name fails (a `{placeholder}`,
# a partition child, or prose like ``CREATE TABLE IF NOT EXISTS`` in a docstring
# followed by backticks), the regex backtracks out of the optional IF NOT EXISTS
# group and would happily report a table named "if".
_CREATE_TABLE_RE = re.compile(
    r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?!IF\b)(\w+)\b(?!\s+PARTITION\s+OF\b)",
    re.IGNORECASE,
)

# `ALTER TABLE name ADD COLUMN [IF NOT EXISTS] col ...` — `ADD CONSTRAINT`
# does not match (it is not a column), nor does `ALTER COLUMN ... SET NOT NULL`
# (that promotes a column somebody already added; ownership of the column is
# decided where it was added).
_ADD_COLUMN_RE = re.compile(
    r"\bALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\b",
    re.IGNORECASE,
)

# Comments are blanked (not removed) so match positions still map to the
# original line numbers. Both SQL `--` comments and Python `#` comments talk
# about DDL in prose ("CREATE TABLE here — this service never writes either
# table") and would otherwise register phantom tables named "here" or "is".
_SQL_COMMENT_RE = re.compile(r"--[^\n]*")
_PY_COMMENT_RE = re.compile(r"(?m)(?:^|(?<=\s))#[^\n]*")


@dataclass
class TableDecl:
    table: str
    line: int


@dataclass
class ColumnAdd:
    table: str
    column: str
    line: int


@dataclass
class FileDdl:
    tables: list[TableDecl] = field(default_factory=list)
    columns: list[ColumnAdd] = field(default_factory=list)


@dataclass
class StrayColumn:
    owner: str
    table: str
    column: str
    line: int


@dataclass
class Report:
    per_owner: dict[str, FileDdl]
    table_owners: dict[str, list[str]]
    multi_owner: dict[str, list[str]]
    stray_columns: list[StrayColumn]

    @property
    def multi_owner_count(self) -> int:
        return len(self.multi_owner)

    @property
    def stray_column_count(self) -> int:
        return len(self.stray_columns)

    def to_dict(self) -> dict:
        return {
            "owners": {
                owner: {
                    "tables": [asdict(t) for t in ddl.tables],
                    "columns": [asdict(c) for c in ddl.columns],
                }
                for owner, ddl in self.per_owner.items()
            },
            "tables": self.table_owners,
            "multi_owner_tables": self.multi_owner,
            "stray_columns": [asdict(s) for s in self.stray_columns],
            "counts": {
                "multi_owner_tables": self.multi_owner_count,
                "stray_columns": self.stray_column_count,
            },
        }


def _strip_comments(source: str) -> str:
    source = _SQL_COMMENT_RE.sub(lambda m: " " * len(m.group(0)), source)
    return _PY_COMMENT_RE.sub(lambda m: " " * len(m.group(0)), source)


def _line_of(source: str, pos: int) -> int:
    return source.count("\n", 0, pos) + 1


def extract_ddl(source: str) -> FileDdl:
    """Every table declaration and every ADD COLUMN in one file's text."""
    text = _strip_comments(source)
    ddl = FileDdl()
    for m in _CREATE_TABLE_RE.finditer(text):
        ddl.tables.append(TableDecl(table=m.group(1).lower(), line=_line_of(text, m.start())))
    for m in _ADD_COLUMN_RE.finditer(text):
        ddl.columns.append(
            ColumnAdd(
                table=m.group(1).lower(),
                column=m.group(2).lower(),
                line=_line_of(text, m.start()),
            )
        )
    return ddl


def audit(sources: dict[str, str]) -> Report:
    """owner label -> file text. Pure; the only thing main() adds is I/O."""
    per_owner = {owner: extract_ddl(text) for owner, text in sources.items()}

    owners_of: dict[str, set[str]] = defaultdict(set)
    for owner, ddl in per_owner.items():
        for decl in ddl.tables:
            owners_of[decl.table].add(owner)
    table_owners = {t: sorted(o) for t, o in sorted(owners_of.items())}
    multi_owner = {t: o for t, o in table_owners.items() if len(o) > 1}

    stray: list[StrayColumn] = []
    for owner, ddl in per_owner.items():
        owned = {decl.table for decl in ddl.tables}
        for add in ddl.columns:
            if add.table not in owned:
                stray.append(StrayColumn(owner, add.table, add.column, add.line))
    stray.sort(key=lambda s: (s.table, s.column, s.owner, s.line))

    return Report(per_owner, table_owners, multi_owner, stray)


def read_sources(root: Path, files: dict[str, str] | None = None) -> dict[str, str]:
    files = files if files is not None else SCANNED_FILES
    sources: dict[str, str] = {}
    for owner, rel in files.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"{owner}: {path} does not exist")
        sources[owner] = path.read_text(encoding="utf-8")
    return sources


# --- baseline ---------------------------------------------------------------


def baseline_from_report(report: Report) -> dict:
    return {
        "_comment": (
            "Ratchet for scripts/check_schema_owners.py --check. Counts may only "
            "go down; regenerate with --baseline after moving a table into "
            "migrations. The target is 0 and 0."
        ),
        "multi_owner_tables": report.multi_owner_count,
        "stray_columns": report.stray_column_count,
        "tables": sorted(report.multi_owner),
    }


def write_baseline(path: Path, report: Report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline_from_report(report), indent=2) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> dict | None:
    """None when there is no baseline — the caller falls back to the strict
    zero budget, which is the end state anyway."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "multi_owner_tables": int(data["multi_owner_tables"]),
            "stray_columns": int(data["stray_columns"]),
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed baseline {path}: {exc}") from exc


# --- rendering --------------------------------------------------------------


def render(report: Report, files: dict[str, str]) -> str:
    out: list[str] = []
    out.append("Per-service schema declarations")
    out.append("=" * 31)
    for owner, ddl in report.per_owner.items():
        out.append(f"\n{owner}  ({files.get(owner, '?')})")
        if not ddl.tables and not ddl.columns:
            out.append("  (no DDL)")
        for decl in ddl.tables:
            out.append(f"  CREATE TABLE {decl.table}  (line {decl.line})")
        for add in ddl.columns:
            out.append(f"  ADD COLUMN   {add.table}.{add.column}  (line {add.line})")

    out.append("\nOwners per table")
    out.append("=" * 16)
    width = max((len(t) for t in report.table_owners), default=0)
    for table, owners in report.table_owners.items():
        flag = "  <-- MULTI-OWNER" if len(owners) > 1 else ""
        out.append(f"  {table.ljust(width)}  {', '.join(owners)}{flag}")

    out.append("\nStray columns (ADD COLUMN on a table this file does not declare)")
    out.append("=" * 64)
    if not report.stray_columns:
        out.append("  none")
    for s in report.stray_columns:
        out.append(f"  {s.owner}: {s.table}.{s.column}  (line {s.line})")

    out.append("")
    out.append(
        f"multi-owner tables: {report.multi_owner_count}   "
        f"stray columns: {report.stray_column_count}"
    )
    return "\n".join(out)


# --- verdict ----------------------------------------------------------------


def verdict(
    report: Report,
    max_multi_owner: int,
    max_stray: int,
) -> list[str]:
    """Human-readable failures; empty means within budget."""
    problems: list[str] = []
    if report.multi_owner_count > max_multi_owner:
        problems.append(
            f"{report.multi_owner_count} table(s) declared by more than one owner "
            f"(budget {max_multi_owner}): {', '.join(sorted(report.multi_owner))}. "
            "A shared table belongs in packages/schemas-py/dunetrace_schemas/migrations.py "
            "and nowhere else; once it is there, delete the service copy."
        )
    if report.stray_column_count > max_stray:
        rendered = ", ".join(f"{s.owner}:{s.table}.{s.column}" for s in report.stray_columns)
        problems.append(
            f"{report.stray_column_count} stray column(s) (budget {max_stray}): {rendered}. "
            "An ADD COLUMN on a table the file does not declare belongs in a new migration."
        )
    return problems


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to scan")
    parser.add_argument(
        "--max-multi-owner",
        type=int,
        default=0,
        help="tables allowed to have more than one owner (default 0)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=f"write the current counts to {BASELINE_PATH.name} and exit 0",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail only if a count exceeds the baseline file (ratchet)",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="baseline path (default scripts/schema_owners_baseline.json under --root)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quiet", action="store_true", help="only print the verdict")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.root.resolve()
    baseline_path: Path = args.baseline_file or (root / "scripts" / "schema_owners_baseline.json")

    try:
        sources = read_sources(root)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = audit(sources)

    if args.baseline:
        write_baseline(baseline_path, report)
        print(
            f"wrote {baseline_path}: multi_owner_tables={report.multi_owner_count} "
            f"stray_columns={report.stray_column_count}"
        )
        return 0

    max_multi = args.max_multi_owner
    max_stray = 0
    baseline = None
    if args.check:
        try:
            baseline = load_baseline(baseline_path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if baseline is None:
            print(
                f"note: no baseline at {baseline_path}; --check falls back to the strict "
                "zero budget",
                file=sys.stderr,
            )
        else:
            max_multi = max(max_multi, baseline["multi_owner_tables"])
            max_stray = baseline["stray_columns"]

    problems = verdict(report, max_multi, max_stray)

    if args.json:
        payload = report.to_dict()
        payload["baseline"] = baseline
        payload["budget"] = {"multi_owner_tables": max_multi, "stray_columns": max_stray}
        payload["problems"] = problems
        payload["ok"] = not problems
        print(json.dumps(payload, indent=2))
    else:
        if not args.quiet:
            print(render(report, SCANNED_FILES))
            print()
        for p in problems:
            print(f"FAIL: {p}")
        if not problems:
            print(
                f"OK: multi-owner tables {report.multi_owner_count} <= {max_multi}, "
                f"stray columns {report.stray_column_count} <= {max_stray}"
            )
        if baseline is not None and not problems:
            if (
                report.multi_owner_count < baseline["multi_owner_tables"]
                or report.stray_column_count < baseline["stray_columns"]
            ):
                print(
                    "note: counts are below the baseline — lower it with "
                    "`python scripts/check_schema_owners.py --baseline` so the ratchet "
                    "holds the new floor"
                )

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
