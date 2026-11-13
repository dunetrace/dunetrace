#!/usr/bin/env python3
"""
Tests for gen_skill_references.py itself.

The orphan sweep is the dangerous part. It deletes files out of another
repository's working tree, and the first version deleted every *.md it had not
just written -- which would have silently removed hand-authored shared
references like verification.md on the next regeneration. That is the headline
case below.

Run: python scripts/test_gen_skill_references.py
  or python -m unittest scripts.test_gen_skill_references
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_skill_references as gsr  # noqa: E402


def run_main(*argv: str) -> tuple[int, str]:
    """Invoke main() with argv, returning (exit code, combined output)."""
    buf = io.StringIO()
    with patch.object(sys, "argv", ["gen_skill_references.py", *argv]):
        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                code = gsr.main()
            except SystemExit as exc:  # --require-all raises rather than returns
                code = 1 if exc.code else 0
                buf.write(str(exc))
    return code, buf.getvalue()


class TestOrphanSweep(unittest.TestCase):
    """It must delete what it owns and nothing else."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_hand_authored_reference_survives(self):
        """The regression. A shared reference is not derived from any guide."""
        run_main("--out", str(self.tmp))
        hand = self.tmp / "verification.md"
        hand.write_text("# Verification\n\nHand written, owned by the skill.\n")

        run_main("--out", str(self.tmp))

        self.assertTrue(hand.exists(), "hand-authored reference was deleted")
        self.assertIn("Hand written", hand.read_text())

    def test_stale_generated_file_is_removed(self):
        """A renamed guide must not leave its old reference behind."""
        run_main("--out", str(self.tmp))
        stale = self.tmp / "zz-removed-framework.md"
        stale.write_text((self.tmp / "crewai.md").read_text())
        self.assertTrue(gsr._is_generated(stale))

        _, out = run_main("--out", str(self.tmp))

        self.assertFalse(stale.exists())
        self.assertIn("zz-removed-framework.md", out)

    def test_check_ignores_hand_authored_files(self):
        """A shared reference must not be reported as drift."""
        run_main("--out", str(self.tmp))
        (self.tmp / "verification.md").write_text("# Verification\n")

        code, out = run_main("--out", str(self.tmp), "--check")

        self.assertEqual(code, 0, out)

    def test_is_generated_discriminates(self):
        gen = self.tmp / "g.md"
        gen.write_text("# x\n\n<!-- GENERATED from docs/integrate-x.md by ... -->\n")
        hand = self.tmp / "h.md"
        hand.write_text("# Verification\n")
        self.assertTrue(gsr._is_generated(gen))
        self.assertFalse(gsr._is_generated(hand))
        self.assertFalse(gsr._is_generated(self.tmp / "missing.md"))


class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_clean_tree_passes(self):
        run_main("--out", str(self.tmp))
        code, out = run_main("--out", str(self.tmp), "--check")
        self.assertEqual(code, 0, out)

    def test_edited_reference_fails(self):
        run_main("--out", str(self.tmp))
        target = self.tmp / "crewai.md"
        target.write_text(target.read_text() + "\nlocal edit\n")
        code, _ = run_main("--out", str(self.tmp), "--check")
        self.assertEqual(code, 1)

    def test_missing_reference_fails(self):
        run_main("--out", str(self.tmp))
        (self.tmp / "crewai.md").unlink()
        code, _ = run_main("--out", str(self.tmp), "--check")
        self.assertEqual(code, 1)


class TestGuideParsing(unittest.TestCase):
    def test_every_guide_has_a_block(self):
        """--require-all is what stops a guide dropping out of the skill."""
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_main("--out", tmp, "--require-all")
            self.assertEqual(code, 0, out)

    def test_require_all_fails_on_a_guide_without_a_block(self):
        guides = sorted(gsr.DOCS.glob("integrate-*.md"))
        self.assertTrue(guides)
        original = guides[0].read_text(encoding="utf-8")
        stripped = gsr.BLOCK_RE.sub("", original)
        self.assertNotEqual(original, stripped, "fixture guide had no block")
        try:
            guides[0].write_text(stripped, encoding="utf-8")
            with tempfile.TemporaryDirectory() as tmp:
                code, out = run_main("--out", tmp, "--require-all")
            self.assertEqual(code, 1)
            self.assertIn(guides[0].name, out)
        finally:
            guides[0].write_text(original, encoding="utf-8")

    def test_generated_references_carry_required_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_main("--out", tmp)
            body = (Path(tmp) / "crewai.md").read_text()
        for needle in (
            "Requires an open `dt.run()`",
            "Target file",
            "Grep signals",
            "GENERATED from docs/",
        ):
            self.assertIn(needle, body)

    def test_run_context_warning_appears_only_when_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_main("--out", tmp)
            needs = (Path(tmp) / "crewai.md").read_text()  # requires_run_context: true
            opens = (Path(tmp) / "langchain.md").read_text()  # opens its own run
        warning = "emits nothing outside an open run"
        self.assertIn(warning, needs)
        self.assertNotIn(warning, opens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
