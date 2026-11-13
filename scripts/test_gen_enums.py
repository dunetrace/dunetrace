"""
Tests for the enum generator (scripts/gen_enums.py).

Run: python -m unittest scripts.test_gen_enums   (from repo root)
  or: cd scripts && python -m unittest test_gen_enums

Stdlib only — the docs-consistency CI job that runs this has just the SDK
installed, no pydantic, so ``dunetrace_schemas.enums`` is loaded by file path.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_enums as g  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
_SDK_ENUMS = "packages/sdk-py/dunetrace/_enums.py"
_SCHEMA_ENUMS = "packages/schemas-py/dunetrace_schemas/enums.py"


def _load_by_path(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _members(enum_cls):
    return [(m.name, m.value) for m in enum_cls]


class _TempRepo:
    """A throwaway repo root holding only enum_source.py, so write()/check() can
    be exercised without touching the real generated modules."""

    def __enter__(self) -> pathlib.Path:
        self._dir = tempfile.mkdtemp(prefix="gen_enums_")
        root = pathlib.Path(self._dir)
        src = root / g.SOURCE_REL
        src.parent.mkdir(parents=True)
        shutil.copy(_REPO / g.SOURCE_REL, src)
        return root

    def __exit__(self, *exc):
        shutil.rmtree(self._dir, ignore_errors=True)


class TestGeneratedModulesMatchSource(unittest.TestCase):
    """The modules on disk are exactly what the source produces (what CI's --check runs)."""

    def test_check_passes_on_repo(self):
        out = io.StringIO()
        self.assertEqual(g.check(_REPO, out=out), 0, out.getvalue())

    def test_both_targets_carry_the_header(self):
        for rel in g.TARGETS:
            first_line = (_REPO / rel).read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(first_line, g.HEADER, rel)

    def test_sdk_module_equals_source(self):
        # Imported the normal way: this is the module dunetrace.models re-exports.
        sys.path.insert(0, str(_REPO / "packages" / "sdk-py"))
        try:
            from dunetrace import _enums as sdk_enums
        finally:
            sys.path.pop(0)
        source = g.load_source(_REPO)
        for class_name, members in source.ENUMS:
            expected = [(name, value) for name, value, _ in members]
            self.assertEqual(_members(getattr(sdk_enums, class_name)), expected, class_name)

    def test_schemas_module_equals_source(self):
        schema_enums = _load_by_path(_SCHEMA_ENUMS, "_test_schema_enums")
        source = g.load_source(_REPO)
        for class_name, members in source.ENUMS:
            expected = [(name, value) for name, value, _ in members]
            self.assertEqual(_members(getattr(schema_enums, class_name)), expected, class_name)

    def test_models_reexports_generated_enums(self):
        # `from dunetrace.models import EventType` must be the *same* class as the
        # generated one, not a copy — the whole point of the re-export.
        sys.path.insert(0, str(_REPO / "packages" / "sdk-py"))
        try:
            from dunetrace import _enums as sdk_enums
            from dunetrace import models
        finally:
            sys.path.pop(0)
        self.assertIs(models.EventType, sdk_enums.EventType)
        self.assertIs(models.Severity, sdk_enums.Severity)
        self.assertIs(models.FailureType, sdk_enums.FailureType)

    def test_known_shape(self):
        # Guards against a source edit that silently drops a group: the counts
        # documented in CLAUDE.md / docs, and the one member whose value differs
        # from its name.
        source = g.load_source(_REPO)
        by_name = dict(source.ENUMS)
        self.assertEqual(len(by_name["EventType"]), 24)
        self.assertEqual(len(by_name["Severity"]), 4)
        self.assertEqual(len(by_name["FailureType"]), 39)
        values = {name: value for name, value, _ in by_name["FailureType"]}
        self.assertEqual(values["CONFIDENT_HALLUCINATION"], "CONFIDENT_HALLUCINATION_PROXY")
        self.assertEqual(values["CUSTOM"], "CUSTOM")


class TestGenerator(unittest.TestCase):
    def test_write_is_idempotent(self):
        with _TempRepo() as root:
            first = g.write(root)
            self.assertEqual(sorted(first), sorted(g.TARGETS), "first run writes both")
            snapshot = {rel: (root / rel).read_bytes() for rel in g.TARGETS}
            second = g.write(root)
            self.assertEqual(second, [], "second run changes nothing")
            for rel in g.TARGETS:
                self.assertEqual((root / rel).read_bytes(), snapshot[rel], rel)
            self.assertEqual(g.check(root, out=io.StringIO()), 0)

    def test_output_is_deterministic_across_renders(self):
        self.assertEqual(g.render_all(_REPO), g.render_all(_REPO))

    def test_check_detects_hand_edit(self):
        with _TempRepo() as root:
            g.write(root)
            target = root / _SDK_ENUMS
            text = target.read_text(encoding="utf-8")
            target.write_text(
                text.replace('CUSTOM = "CUSTOM"', 'CUSTOM = "CUSTOM"\n    SNEAKY = "SNEAKY"'),
                encoding="utf-8",
            )
            out = io.StringIO()
            self.assertEqual(g.check(root, out=out), 1)
            report = out.getvalue()
            self.assertIn(_SDK_ENUMS, report)
            self.assertIn("-    SNEAKY", report, "diff shows the hand edit")
            self.assertIn("STALE", report)
            # The untouched target is not reported as stale.
            self.assertNotIn(f"STALE: {_SCHEMA_ENUMS}", report)

    def test_check_detects_missing_target(self):
        with _TempRepo() as root:
            g.write(root)
            (root / _SCHEMA_ENUMS).unlink()
            out = io.StringIO()
            self.assertEqual(g.check(root, out=out), 1)
            self.assertIn(_SCHEMA_ENUMS, out.getvalue())

    def test_check_detects_source_edit(self):
        # The other direction: the source gained a member but nobody regenerated.
        with _TempRepo() as root:
            g.write(root)
            src = root / g.SOURCE_REL
            text = src.read_text(encoding="utf-8")
            src.write_text(
                text.replace(
                    '    ("CUSTOM", "CUSTOM",',
                    '    ("BRAND_NEW", "BRAND_NEW", None),\n    ("CUSTOM", "CUSTOM",',
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            self.assertEqual(g.check(root, out=out), 1)
            self.assertIn('+    BRAND_NEW = "BRAND_NEW"', out.getvalue())

    def test_generated_text_is_importable_and_matches(self):
        text = g.render_module(
            "'''doc'''", [("Colour", [("RED", "red", "warm\nish"), ("BLUE", "blue", None)])]
        )
        ns: dict = {}
        exec(compile(text, "<generated>", "exec"), ns)  # noqa: S102 - our own output
        self.assertEqual(_members(ns["Colour"]), [("RED", "red"), ("BLUE", "blue")])
        self.assertIn('    # warm\n    # ish\n    RED = "red"\n', text)
        self.assertTrue(text.startswith(g.HEADER + "\n"))

    def test_values_are_double_quoted_and_escaped(self):
        # ruff format enforces double quotes; a value containing one must still
        # round-trip rather than produce a syntax error.
        text = g.render_enum("E", [("A", 'say "hi"', None)])
        ns: dict = {}
        exec(compile("from enum import Enum\n" + text, "<generated>", "exec"), ns)  # noqa: S102
        self.assertEqual(ns["E"].A.value, 'say "hi"')

    def test_main_check_exit_codes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g.main(["--check"]), 0)


class TestSourceValidation(unittest.TestCase):
    def test_duplicate_value_rejected(self):
        with self.assertRaises(ValueError):
            g._validate([("E", [("A", "x", None), ("B", "x", None)])])

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            g._validate([("E", [("A", "x", None), ("A", "y", None)])])

    def test_bad_identifier_rejected(self):
        with self.assertRaises(ValueError):
            g._validate([("E", [("not-valid", "x", None)])])

    def test_lowercase_name_rejected(self):
        with self.assertRaises(ValueError):
            g._validate([("E", [("lower", "x", None)])])

    def test_empty_enum_rejected(self):
        with self.assertRaises(ValueError):
            g._validate([("E", [])])


if __name__ == "__main__":
    unittest.main(verbosity=2)
