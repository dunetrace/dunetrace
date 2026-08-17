"""
Cross-service schema parity.

Every `CREATE TABLE` in this repo is `IF NOT EXISTS`, and 20-odd tables are
declared by more than one service, because the schema is defined inline in each
service that owns a table rather than in a central migration. That means
**whichever service starts first wins**: if two declarations of the same table
drift apart, the schema you get depends on container start order, and the loser's
columns silently don't exist. The failure surfaces far from the cause — an
`UndefinedColumnError` in an unrelated query, only in deployments where the start
order happened to differ.

Nothing else in the suite covers this. `packages/schemas-py/tests/test_sdk_parity.py`
guards the SDK↔schemas enum boundary and `services/ingest/tests/test_sdk_schema_compat.py`
guards the wire format, but neither looks at DDL. This test closes that gap by
parsing every `CREATE TABLE IF NOT EXISTS` out of the service sources and
asserting the duplicated ones agree on column names *and* their type/constraint
text.

Deliberately excluded: `ALTER TABLE` migrations. Divergence there is legitimate
and intentional — `ingest_svc` owns the `org_id NOT NULL` promotion while other
services declare it nullable, which is a documented ownership split rather than
drift. This test only asserts that the `CREATE TABLE` statements themselves
match.

Run: make test-schema-parity
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICES_DIR = REPO_ROOT / "services"

# Captures the body between the outermost parens of a CREATE TABLE IF NOT EXISTS.
# Terminated by a line-leading `)`, then anything up to the statement end.
#
# The `[^;"']*` tail is load-bearing, not slop: `events` is declared
# `... ) PARTITION BY RANGE (received_at);`, so requiring `)` to be followed
# immediately by the terminator makes that match fail, and a failed match doesn't
# stop there — the non-greedy body keeps growing to the *next* `);` in the file
# and swallows whatever table comes after `events`. That silently dropped
# `failure_signals` from the parse until the canary test below caught it.
_CREATE_TABLE_RE = re.compile(
    r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\s*\)[^;\"']*[;\"']",
    re.DOTALL,
)

# Table-level constraints — not columns, so they're skipped when building the
# per-column map. (Constraint parity is not asserted; these are position- and
# name-sensitive in ways that make textual comparison noisy.)
_CONSTRAINT_KEYWORDS = frozenset({"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT", "EXCLUDE"})


def _service_sources() -> list[Path]:
    return sorted(
        p
        for p in SERVICES_DIR.rglob("*.py")
        if "test" not in p.parts and "__pycache__" not in p.parts
    )


def _parse_columns(body: str) -> dict[str, str]:
    """Map column name -> normalized definition text (type + inline constraints)."""
    columns: dict[str, str] = {}
    for raw_line in body.split("\n"):
        line = re.sub(r"--.*$", "", raw_line).strip().rstrip(",").strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0].upper() in _CONSTRAINT_KEYWORDS:
            continue
        # Collapse whitespace and upper-case so formatting differences between
        # two declarations of the same column don't read as drift.
        columns[tokens[0]] = " ".join(tokens[1:]).upper()
    return columns


def _collect_declarations() -> dict[str, dict[Path, dict[str, str]]]:
    """table -> {source file -> {column -> definition}}."""
    declarations: dict[str, dict[Path, dict[str, str]]] = defaultdict(dict)
    for path in _service_sources():
        source = path.read_text()
        for match in _CREATE_TABLE_RE.finditer(source):
            table, body = match.group(1), match.group(2)
            columns = _parse_columns(body)
            if columns:
                declarations[table][path] = columns
    return declarations


DECLARATIONS = _collect_declarations()
DUPLICATED = {t: per for t, per in DECLARATIONS.items() if len(per) > 1}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_parser_finds_the_known_declarations() -> None:
    """Guard against the regex silently matching nothing — which would make
    every parity assertion below vacuously pass."""
    assert len(DECLARATIONS) > 30, (
        f"only parsed {len(DECLARATIONS)} CREATE TABLE statements; the regex has "
        "probably stopped matching the current formatting"
    )
    assert len(DUPLICATED) > 10, (
        f"only {len(DUPLICATED)} tables look duplicated; expected ~20 — parsing may be incomplete"
    )
    # events is partitioned and declared once, by ingest — a canary that we're
    # reading the real files.
    assert "events" in DECLARATIONS
    assert "failure_signals" in DECLARATIONS


@pytest.mark.parametrize("table", sorted(DUPLICATED))
def test_duplicated_table_declares_the_same_columns(table: str) -> None:
    per_file = DUPLICATED[table]
    all_columns = set().union(*(set(c) for c in per_file.values()))

    problems = []
    for path, columns in sorted(per_file.items()):
        missing = all_columns - set(columns)
        if missing:
            problems.append(f"  {_rel(path)} is missing: {', '.join(sorted(missing))}")

    assert not problems, (
        f"`{table}` is declared in {len(per_file)} services with different column "
        f"sets. Because every CREATE TABLE is IF NOT EXISTS, the columns you get "
        f"depend on which service starts first.\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("table", sorted(DUPLICATED))
def test_duplicated_table_declares_the_same_column_types(table: str) -> None:
    per_file = DUPLICATED[table]
    files = sorted(per_file)

    problems = []
    for column in sorted(set().union(*(set(c) for c in per_file.values()))):
        definitions = {f: per_file[f].get(column) for f in files}
        present = {f: d for f, d in definitions.items() if d is not None}
        if len({*present.values()}) > 1:
            rendered = "\n".join(f"      {_rel(f)}: {d!r}" for f, d in present.items())
            problems.append(f"  {column}:\n{rendered}")

    assert not problems, (
        f"`{table}` is declared in {len(per_file)} services with conflicting "
        f"column definitions. The first service to start wins, so this is a "
        f"start-order-dependent bug:\n" + "\n".join(problems)
    )
