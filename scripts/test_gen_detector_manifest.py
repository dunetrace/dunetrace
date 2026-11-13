#!/usr/bin/env python3
"""
Tests for gen_detector_manifest.py itself. A coverage report whose counts are
wrong is worse than no coverage report, because it reads as a guarantee. The
audit that keeps the counts honest is the part worth testing: each way a
detector can fall out of the manifest gets a test that fails without it.

Run: python scripts/test_gen_detector_manifest.py
  or python -m unittest scripts.test_gen_detector_manifest
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_detector_manifest as gdm  # noqa: E402

CURATED = json.loads(gdm.CURATED.read_text(encoding="utf-8"))


def _with(mutate) -> dict:
    """A copy of the real curated file with one mutation applied."""
    data = copy.deepcopy(CURATED)
    mutate(data)
    return data


class TestAuditCatchesDrift(unittest.TestCase):
    """Each failure mode must actually fail the build."""

    def _build_with(self, curated: dict):
        """Point the generator at a mutated copy of the curated file on disk.

        Patching Path.read_text is not possible (PosixPath attributes are
        read-only), so swap the module-level path instead.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(curated, fh)
            tmp = Path(fh.name)
        try:
            with patch.object(gdm, "CURATED", tmp):
                return gdm.build()
        finally:
            tmp.unlink(missing_ok=True)

    def test_undescribed_detector_fails(self):
        """A new detector that nobody described must not silently vanish."""
        broken = _with(lambda d: d["detectors"].pop("TOOL_LOOP"))
        with self.assertRaises(SystemExit) as ctx:
            self._build_with(broken)
        self.assertIn("TOOL_LOOP", str(ctx.exception))
        self.assertIn("not described", str(ctx.exception))

    def test_entry_for_nonexistent_detector_fails(self):
        broken = _with(
            lambda d: d["detectors"].update(
                {"NOT_A_DETECTOR": {"requires": ["run"], "enriches": [], "why": "invented"}}
            )
        )
        with self.assertRaises(SystemExit) as ctx:
            self._build_with(broken)
        self.assertIn("NOT_A_DETECTOR", str(ctx.exception))

    def test_unaccounted_field_read_fails(self):
        """The case that matters: a detector reads a field its entry never names."""

        def drop_tool(d):
            d["detectors"]["TOOL_LOOP"]["requires"] = ["run"]
            d["detectors"]["TOOL_LOOP"]["enriches"] = []

        with self.assertRaises(SystemExit) as ctx:
            self._build_with(_with(drop_tool))
        msg = str(ctx.exception)
        self.assertIn("TOOL_LOOP", msg)
        self.assertIn("tool_calls", msg)

    def test_unknown_token_fails(self):
        broken = _with(lambda d: d["detectors"]["TOOL_LOOP"].__setitem__("requires", ["telepathy"]))
        with self.assertRaises(SystemExit) as ctx:
            self._build_with(broken)
        self.assertIn("unknown token", str(ctx.exception))

    def test_implication_satisfies_the_audit(self):
        """llm.tokens must satisfy a detector that reads llm_calls."""

        def tokens_only(d):
            d["detectors"]["EMPTY_LLM_RESPONSE"]["requires"] = ["llm.tokens"]
            d["detectors"]["EMPTY_LLM_RESPONSE"]["enriches"] = []

        self._build_with(_with(tokens_only))  # must not raise


class TestManifestMatchesTheLiveDetectorSet(unittest.TestCase):
    """The repo's own state stays clean, and the counts come from code."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = gdm.build()

    def test_builds_clean(self):
        self.assertTrue(self.manifest["structural"])

    def test_counts_match_the_imported_detector_set(self):
        from detector_svc.db import LIVE_DETECTORS
        from detector_svc.detectors import _DETECTOR_CLASSES

        counts = self.manifest["counts"]
        self.assertEqual(counts["structural"], len(_DETECTOR_CLASSES))
        self.assertEqual(counts["structural_live"], len(LIVE_DETECTORS))
        self.assertEqual(
            counts["structural_live"] + counts["structural_shadow"], counts["structural"]
        )

    def test_voice_pack_count_matches_the_pack(self):
        from dunetrace.packs import PACK_REGISTRY

        pack = PACK_REGISTRY["voice"]
        if isinstance(pack, type):  # registry holds a class in some versions
            pack = pack()
        self.assertEqual(self.manifest["counts"]["voice_pack"], len(pack.detectors))

    def test_every_requires_token_is_declared(self):
        declared = set(self.manifest["tokens"])
        for name, entry in self.manifest["structural"].items():
            for item in entry["requires"]:
                for tok in item if isinstance(item, list) else [item]:
                    self.assertIn(tok, declared, f"{name} requires undeclared {tok!r}")

    def test_every_token_has_an_actionable_hint(self):
        """A gap the report cannot phrase is a gap the user cannot close."""
        for tok in self.manifest["tokens"]:
            self.assertIn(tok, self.manifest["token_hint"], f"{tok!r} has no token_hint")

    def test_yaml_keys_match_detectors_yml(self):
        import yaml

        cfg = yaml.safe_load((gdm.ROOT / "detectors.yml").read_text(encoding="utf-8"))
        shipped = {k for k in cfg["default"] if not k.startswith("_")}
        self.assertEqual({e["yaml_key"] for e in self.manifest["structural"].values()}, shipped)


class TestCoverageEvaluation(unittest.TestCase):
    """The manifest has to produce right answers, not just parse."""

    @classmethod
    def setUpClass(cls):
        cls.m = gdm.build()

    def _can_fire(self, name, have):
        implies = self.m["implies"]
        got = set(have)
        for t in list(got):
            got |= set(implies.get(t, ()))
        for item in self.m["structural"][name]["requires"]:
            alts = item if isinstance(item, list) else [item]
            if not (set(alts) & got):
                return False
        return True

    def test_tool_loop_needs_only_tool_calls(self):
        """The case the curated file exists for: reads llm, does not require it."""
        self.assertTrue(self._can_fire("TOOL_LOOP", {"run", "tool"}))

    def test_rag_detector_needs_retrieval(self):
        self.assertFalse(self._can_fire("RAG_EMPTY_RETRIEVAL", {"run", "llm", "tool"}))
        self.assertTrue(self._can_fire("RAG_EMPTY_RETRIEVAL", {"run", "retrieval"}))

    def test_step_count_inflation_is_baseline_gated(self):
        self.assertFalse(self._can_fire("STEP_COUNT_INFLATION", {"run", "llm", "tool"}))
        self.assertTrue(self._can_fire("STEP_COUNT_INFLATION", {"run", "llm", "tool", "baseline"}))

    def test_unresolved_ambiguity_is_config_gated(self):
        self.assertFalse(self._can_fire("UNRESOLVED_AMBIGUITY", {"run", "tool"}))

    def test_delegation_detectors_need_nested_runs(self):
        for name in ("DELEGATION_LOOP", "HANDOFF_CONTEXT_LOSS"):
            self.assertFalse(self._can_fire(name, {"run", "llm", "tool"}), name)
            self.assertTrue(self._can_fire(name, {"run", "parent_run_id"}), name)

    def test_a_fully_instrumented_agent_reaches_most_detectors(self):
        """Guards against a curation change that quietly strands detectors."""
        full = {
            "run",
            "llm",
            "llm.tokens",
            "tool",
            "tool.output",
            "retrieval",
            "retrieval.content",
            "memory",
            "external",
            "input_text",
            "system_prompt",
            "declared_tools",
            "parent_run_id",
        }
        firing = [n for n in self.m["structural"] if self._can_fire(n, full)]
        # Everything except the two gated on something that is not instrumentation.
        self.assertEqual(
            sorted(set(self.m["structural"]) - set(firing)),
            ["STEP_COUNT_INFLATION", "UNRESOLVED_AMBIGUITY"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
