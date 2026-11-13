#!/usr/bin/env python3
"""
Tests for check_schema_owners.py itself. A ratchet that silently parses
nothing would pass forever, so the extraction, the ownership rules, the
baseline direction and the real-repo run are each pinned here. Self-contained:
synthetic schema files are written into a temp directory shaped like the
repo (the same relative paths SCANNED_FILES names), so no fixture from the
rest of the repo is needed — except the last test, which runs against the
real tree and the committed baseline.

Run: python -m unittest scripts.test_check_schema_owners
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:  # `python -m unittest scripts.test_check_schema_owners` from the repo root
    from scripts.check_schema_owners import (
        REPO_ROOT,
        SCANNED_FILES,
        audit,
        extract_ddl,
        load_baseline,
        main,
    )
except ImportError:  # `python scripts/test_check_schema_owners.py`
    from check_schema_owners import (  # type: ignore[no-redef]
        REPO_ROOT,
        SCANNED_FILES,
        audit,
        extract_ddl,
        load_baseline,
        main,
    )


def _make_repo(root: Path, contents: dict[str, str]) -> None:
    """Write every scanned file (empty unless given) so read_sources finds it."""
    for owner, rel in SCANNED_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents.get(owner, "# nothing here\n"))


def _run(root: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["--root", str(root), *argv])
    return code, out.getvalue(), err.getvalue()


_WIDGETS = """
_SCHEMA = '''
CREATE TABLE IF NOT EXISTS widgets (
    id      BIGSERIAL PRIMARY KEY,
    org_id  TEXT NOT NULL
);
ALTER TABLE widgets ADD COLUMN IF NOT EXISTS colour TEXT;
'''
"""


class TestExtractDdl(unittest.TestCase):
    def test_bare_sql_string(self):
        ddl = extract_ddl(_WIDGETS)
        self.assertEqual([t.table for t in ddl.tables], ["widgets"])
        self.assertEqual([(c.table, c.column) for c in ddl.columns], [("widgets", "colour")])

    def test_create_without_if_not_exists(self):
        ddl = extract_ddl("CREATE TABLE plain (id INT);")
        self.assertEqual([t.table for t in ddl.tables], ["plain"])

    def test_multiline_alter_and_do_block(self):
        source = """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name = 'gadgets' AND column_name = 'size') THEN
                ALTER TABLE gadgets ADD COLUMN size INTEGER NOT NULL DEFAULT 0;
            END IF;
            ALTER TABLE gadgets
                ADD COLUMN IF NOT EXISTS scopes TEXT[] NOT NULL DEFAULT ARRAY['ingest'];
        END $$;
        """
        ddl = extract_ddl(source)
        self.assertEqual(
            [(c.table, c.column) for c in ddl.columns],
            [("gadgets", "size"), ("gadgets", "scopes")],
        )

    def test_fstring_placeholder_and_partition_children_are_not_tables(self):
        source = """
        await conn.execute("CREATE TABLE IF NOT EXISTS events_default PARTITION OF events DEFAULT")
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS {name} "
            f"PARTITION OF events FOR VALUES FROM ('{start}') TO ('{end}')"
        )
        await conn.execute(f"ALTER TABLE {table} ALTER COLUMN org_id SET NOT NULL")
        """
        ddl = extract_ddl(source)
        self.assertEqual(ddl.tables, [])
        self.assertEqual(ddl.columns, [])

    def test_prose_in_comments_and_docstrings_is_ignored(self):
        source = '''
        """
        every service declared the tables it touched with
        ``CREATE TABLE IF NOT EXISTS`` and grew them with ``ALTER TABLE ... ADD COLUMN IF
        NOT EXISTS``, so "whichever service starts first wins" was the contract.
        """
        # CREATE TABLE here — this service never writes either table).
        x = 1  # ALTER TABLE foo ADD COLUMN bar
        sql = """
        -- CREATE TABLE IF NOT EXISTS is a no-op on an existing table
        CREATE TABLE IF NOT EXISTS real_table (id INT);
        """
        '''
        ddl = extract_ddl(source)
        self.assertEqual([t.table for t in ddl.tables], ["real_table"])
        self.assertEqual(ddl.columns, [])

    def test_add_constraint_and_alter_column_are_not_columns(self):
        source = """
        ALTER TABLE things ADD CONSTRAINT things_pkey PRIMARY KEY (org_id, id);
        ALTER TABLE things ALTER COLUMN org_id SET NOT NULL;
        ALTER TABLE things DROP COLUMN IF EXISTS legacy;
        """
        self.assertEqual(extract_ddl(source).columns, [])

    def test_line_numbers_survive_comment_blanking(self):
        source = "# comment\n-- sql comment\n\nCREATE TABLE IF NOT EXISTS t (id INT);\n"
        self.assertEqual(extract_ddl(source).tables[0].line, 4)


class TestAudit(unittest.TestCase):
    def test_single_owner_is_ok(self):
        report = audit({"api": _WIDGETS, "detector": "CREATE TABLE IF NOT EXISTS runs (id INT);"})
        self.assertEqual(report.multi_owner, {})
        self.assertEqual(report.stray_columns, [])
        self.assertEqual(report.table_owners, {"runs": ["detector"], "widgets": ["api"]})

    def test_two_services_declaring_one_table_is_flagged(self):
        report = audit({"api": _WIDGETS, "detector": _WIDGETS})
        self.assertEqual(report.multi_owner, {"widgets": ["api", "detector"]})
        # The ALTER in each file is on a table that file owns — not stray.
        self.assertEqual(report.stray_columns, [])

    def test_migration_plus_service_is_flagged(self):
        """Once a table lives in migrations the service copy must go: a
        migration plus one service is two owners, not 'the migration wins'."""
        report = audit({"migrations": _WIDGETS, "ingest": _WIDGETS})
        self.assertEqual(report.multi_owner, {"widgets": ["ingest", "migrations"]})

    def test_stray_alter_on_unowned_table_is_flagged(self):
        report = audit(
            {
                "migrations": "CREATE TABLE IF NOT EXISTS failure_signals (id INT);",
                "alerts": (
                    'await conn.execute("ALTER TABLE failure_signals '
                    'ADD COLUMN IF NOT EXISTS alert_claimed_at TIMESTAMPTZ")'
                ),
            }
        )
        self.assertEqual(report.multi_owner, {})
        self.assertEqual(len(report.stray_columns), 1)
        stray = report.stray_columns[0]
        self.assertEqual(
            (stray.owner, stray.table, stray.column),
            ("alerts", "failure_signals", "alert_claimed_at"),
        )

    def test_alter_on_a_table_nobody_declares_is_still_stray(self):
        report = audit({"api": "ALTER TABLE ghosts ADD COLUMN IF NOT EXISTS x INT;"})
        self.assertEqual([s.table for s in report.stray_columns], ["ghosts"])


class TestMainExitCodes(unittest.TestCase):
    def test_clean_repo_exits_zero_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS})
            code, out, _ = _run(root)
        self.assertEqual(code, 0, out)
        self.assertIn("OK:", out)

    def test_multi_owner_exits_one_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS, "detector": _WIDGETS})
            code, out, _ = _run(root)
        self.assertEqual(code, 1)
        self.assertIn("FAIL: 1 table(s) declared by more than one owner", out)

    def test_max_multi_owner_raises_the_budget_but_not_for_strays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS, "detector": _WIDGETS})
            code, _, _ = _run(root, "--max-multi-owner", "1")
            self.assertEqual(code, 0)
            _make_repo(
                root,
                {"api": _WIDGETS, "alerts": "ALTER TABLE widgets ADD COLUMN IF NOT EXISTS z INT;"},
            )
            code, out, _ = _run(root, "--max-multi-owner", "5")
        self.assertEqual(code, 1)
        self.assertIn("stray column", out)

    def test_json_output_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS, "detector": _WIDGETS})
            code, out, _ = _run(root, "--json")
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["counts"], {"multi_owner_tables": 1, "stray_columns": 0})
        self.assertEqual(payload["multi_owner_tables"], {"widgets": ["api", "detector"]})
        self.assertEqual(payload["baseline"], None)

    def test_missing_scanned_file_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = _run(Path(tmp))
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)


class TestBaselineRatchet(unittest.TestCase):
    def test_baseline_records_current_counts_and_check_passes_against_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(
                root,
                {
                    "api": _WIDGETS,
                    "detector": _WIDGETS,
                    "alerts": "ALTER TABLE widgets ADD COLUMN IF NOT EXISTS z INT;",
                },
            )
            code, _, _ = _run(root, "--baseline")
            self.assertEqual(code, 0)
            baseline = load_baseline(root / "scripts" / "schema_owners_baseline.json")
            self.assertEqual(baseline, {"multi_owner_tables": 1, "stray_columns": 1})
            code, out, _ = _run(root, "--check")
        self.assertEqual(code, 0, out)

    def test_check_fails_when_a_count_goes_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS, "detector": _WIDGETS})
            self.assertEqual(_run(root, "--baseline")[0], 0)
            # A third copy does not change the *table* count, so add a new
            # duplicated table and a stray column — both must trip.
            _make_repo(
                root,
                {
                    "api": _WIDGETS + "CREATE TABLE IF NOT EXISTS gizmos (id INT);",
                    "detector": _WIDGETS + "CREATE TABLE IF NOT EXISTS gizmos (id INT);",
                },
            )
            code, out, _ = _run(root, "--check", "--quiet")
            self.assertEqual(code, 1)
            self.assertIn("budget 1", out)
            _make_repo(
                root,
                {
                    "api": _WIDGETS,
                    "detector": _WIDGETS,
                    "alerts": "ALTER TABLE widgets ADD COLUMN IF NOT EXISTS z INT;",
                },
            )
            code, out, _ = _run(root, "--check", "--quiet")
        self.assertEqual(code, 1)
        self.assertIn("stray column(s) (budget 0)", out)

    def test_check_passes_when_a_count_goes_down_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS, "detector": _WIDGETS})
            self.assertEqual(_run(root, "--baseline")[0], 0)
            _make_repo(root, {"api": _WIDGETS})
            code, out, _ = _run(root, "--check")
        self.assertEqual(code, 0)
        self.assertIn("below the baseline", out)

    def test_check_without_a_baseline_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS, "detector": _WIDGETS})
            code, _, err = _run(root, "--check", "--quiet")
        self.assertEqual(code, 1)
        self.assertIn("no baseline", err)

    def test_malformed_baseline_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root, {"api": _WIDGETS})
            (root / "scripts").mkdir()
            (root / "scripts" / "schema_owners_baseline.json").write_text('{"nope": 1}')
            code, _, err = _run(root, "--check")
        self.assertEqual(code, 2)
        self.assertIn("malformed baseline", err)


class TestRealRepo(unittest.TestCase):
    """The committed baseline must match the tree: CI runs --check."""

    def test_scanner_reads_the_real_files(self):
        code, out, _ = _run(REPO_ROOT, "--json")
        payload = json.loads(out)
        # events is partitioned and declared once, by ingest — a canary that
        # the partition-child exclusion did not swallow the parent.
        self.assertEqual(payload["tables"].get("events"), ["ingest"])
        self.assertIn("migrations", payload["tables"].get("failure_signals", []))
        self.assertNotIn("if", payload["tables"])
        self.assertNotIn("events_default", payload["tables"])
        self.assertGreater(len(payload["tables"]), 30)

    def test_real_repo_passes_against_committed_baseline(self):
        code, out, err = _run(REPO_ROOT, "--check", "--quiet")
        self.assertEqual(code, 0, f"{out}\n{err}")


if __name__ == "__main__":
    unittest.main()
