"""
Tests for Phase 1.4.1's semantic-signal confidence floor: the config loader
(docs/config/semantic-evaluators.yml) and the new gate step in poll_once().
No DB, no real HTTP calls.

Run:
    cd services/alerts
    python -m unittest tests.test_semantic_confidence_floor -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alerts_svc.config import (
    get_semantic_confidence_floor,
    load_semantic_confidence_floors,
)
from alerts_svc.sender import SendResult
import alerts_svc.db as db_module
import alerts_svc.worker as worker_module


class TestEnsureSemanticSignalColumn(unittest.IsolatedAsyncioTestCase):
    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            await db_module.ensure_semantic_signal_column()  # must not raise


class TestLoadSemanticConfidenceFloors(unittest.TestCase):
    def _write(self, contents: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, "w") as f:
            f.write(contents)
        self.addCleanup(os.remove, path)
        return path

    def test_missing_file_returns_empty(self):
        floors = load_semantic_confidence_floors("/nonexistent/semantic-evaluators.yml")
        self.assertEqual(floors, {})

    def test_valid_file_parsed(self):
        path = self._write("HALLUCINATION:\n  alert_confidence_floor: 0.75\n")
        floors = load_semantic_confidence_floors(path)
        self.assertEqual(floors, {"HALLUCINATION": 0.75})

    def test_out_of_range_value_skipped(self):
        path = self._write("HALLUCINATION:\n  alert_confidence_floor: 1.5\n")
        floors = load_semantic_confidence_floors(path)
        self.assertEqual(floors, {})

    def test_non_numeric_value_skipped(self):
        path = self._write("HALLUCINATION:\n  alert_confidence_floor: not_a_number\n")
        floors = load_semantic_confidence_floors(path)
        self.assertEqual(floors, {})

    def test_missing_key_skipped(self):
        path = self._write("HALLUCINATION:\n  some_other_setting: true\n")
        floors = load_semantic_confidence_floors(path)
        self.assertEqual(floors, {})


class TestGetSemanticConfidenceFloor(unittest.TestCase):
    def test_returns_configured_value(self):
        self.assertEqual(
            get_semantic_confidence_floor({"HALLUCINATION": 0.75}, "HALLUCINATION"), 0.75
        )

    def test_falls_back_to_default_for_unlisted_evaluator(self):
        self.assertEqual(get_semantic_confidence_floor({}, "TASK_COMPLETION"), 0.6)

    def test_case_insensitive(self):
        self.assertEqual(
            get_semantic_confidence_floor({"HALLUCINATION": 0.9}, "hallucination"), 0.9
        )


def _semantic_row(id_, confidence, evaluator="HALLUCINATION"):
    return {
        "id": id_,
        "failure_type": evaluator,
        "severity": "HIGH",
        "run_id": f"run-{id_}",
        "agent_id": "agent-1",
        "org_id": "org-1",
        "agent_version": "v1",
        "step_index": 0,
        "confidence": confidence,
        "evidence": {"reasoning": "test"},
        "detected_at": time.time(),
        "source": "semantic",
    }


class TestPollOnceSemanticConfidenceGate(unittest.IsolatedAsyncioTestCase):
    async def _run(self, rows, floors):
        mark_mock = AsyncMock()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=rows)),
            patch("alerts_svc.worker.mark_alerted_batch", mark_mock),
            patch("alerts_svc.worker.load_semantic_confidence_floors", return_value=floors),
            patch(
                "alerts_svc.worker.deliver",
                return_value={"slack": SendResult(True, "slack", 1, 200)},
            ),
        ):
            found, delivered = await worker_module.poll_once()
        return found, delivered, mark_mock

    async def test_below_floor_signal_never_delivered(self):
        rows = [_semantic_row(1, confidence=0.4)]
        found, delivered, mark_mock = await self._run(rows, {"HALLUCINATION": 0.6})

        self.assertEqual(found, 1)
        self.assertEqual(delivered, 0)
        mark_mock.assert_called_once_with([1])  # marked alerted (silently), not delivered

    async def test_above_floor_signal_delivered(self):
        rows = [_semantic_row(1, confidence=0.9)]
        found, delivered, mark_mock = await self._run(rows, {"HALLUCINATION": 0.6})

        self.assertEqual(found, 1)
        self.assertEqual(delivered, 1)

    async def test_exactly_at_floor_is_not_suppressed(self):
        # Gate uses strict "<", so confidence == floor still alerts.
        rows = [_semantic_row(1, confidence=0.6)]
        found, delivered, _mark_mock = await self._run(rows, {"HALLUCINATION": 0.6})
        self.assertEqual(delivered, 1)

    async def test_structural_signal_unaffected_by_semantic_gate(self):
        row = _semantic_row(1, confidence=0.1)
        row["source"] = "structural"
        row["failure_type"] = "TOOL_LOOP"
        found, delivered, _mark_mock = await self._run([row], {"HALLUCINATION": 0.6})
        # No FP override, no policy pending — should deliver despite low confidence,
        # since the semantic floor only applies to source == "semantic".
        self.assertEqual(delivered, 1)

    async def test_missing_source_key_treated_as_structural(self):
        # Rows from before the source column existed (or a stale fixture) —
        # .get("source") is None, must not accidentally match "semantic".
        row = _semantic_row(1, confidence=0.1)
        del row["source"]
        row["failure_type"] = "TOOL_LOOP"
        found, delivered, _mark_mock = await self._run([row], {"HALLUCINATION": 0.6})
        self.assertEqual(delivered, 1)

    async def test_per_evaluator_floor_override_respected(self):
        rows = [_semantic_row(1, confidence=0.75, evaluator="TASK_COMPLETION")]
        found, delivered, _mark_mock = await self._run(rows, {"TASK_COMPLETION": 0.8})
        self.assertEqual(delivered, 0)

    async def test_default_floor_used_when_evaluator_not_in_config(self):
        rows = [_semantic_row(1, confidence=0.65, evaluator="TASK_COMPLETION")]
        found, delivered, _mark_mock = await self._run(rows, {})  # no config entries at all
        self.assertEqual(delivered, 1)  # 0.65 > default 0.6


if __name__ == "__main__":
    unittest.main(verbosity=2)
