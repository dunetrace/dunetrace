"""
services/detector/tests/test_worker.py

Tests for the detector worker — run_builder and process_run logic.
No DB required — all async DB calls are mocked.

Run:
    cd services/detector
    pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from detector_svc.run_builder import build_run_state
from dunetrace.models import FailureType
import detector_svc.worker  # must be imported before patch() can resolve "detector_svc.worker.*"


# ── Event factories ────────────────────────────────────────────────────────────

def evt(event_type: str, step_index: int = 1, payload: dict = None, **kw) -> dict:
    return {
        "event_type":    event_type,
        "run_id":        kw.get("run_id", "run-test-1"),
        "agent_id":      kw.get("agent_id", "agent-test"),
        "agent_version": kw.get("agent_version", "abc12345"),
        "step_index":    step_index,
        "timestamp":     time.time(),
        "payload":       payload or {},
        "parent_run_id": kw.get("parent_run_id"),
    }


def tool_evt(tool_name: str, step: int) -> dict:
    return evt("tool.called", step, {"tool_name": tool_name, "args_hash": "aa"})


def retrieval_evt(index: str, count: int, score: float = None, step: int = 1) -> dict:
    return evt("retrieval.responded", step, {
        "index_name": index, "result_count": count, "top_score": score,
    })


def run_started(tools: list = None, step: int = 0) -> dict:
    return evt("run.started", step, {
        "input_hash": "abc123",
        "model": "gpt-4o",
        "tools": ["web_search", "calculator"] if tools is None else tools,
    })


def run_completed(step: int = 10) -> dict:
    return evt("run.completed", step, {"exit_reason": "final_answer", "total_steps": step})


def llm_evt(step: int) -> dict:
    return evt("llm.called", step, {"model": "gpt-4o"})


def llm_responded_evt(step: int, prompt_tokens: int = 500) -> dict:
    return evt("llm.responded", step, {"prompt_tokens": prompt_tokens, "finish_reason": "stop"})


# ── RunBuilder tests ───────────────────────────────────────────────────────────

class TestRunBuilder(unittest.TestCase):

    def test_raises_on_empty_events(self):
        with self.assertRaises(ValueError):
            build_run_state([])

    def test_basic_identity_fields(self):
        state = build_run_state([run_started()])
        self.assertEqual(state.run_id,       "run-test-1")
        self.assertEqual(state.agent_id,     "agent-test")
        self.assertEqual(state.agent_version,"abc12345")

    def test_extracts_available_tools_from_run_started(self):
        state = build_run_state([run_started(tools=["web_search", "calculator"])])
        self.assertEqual(state.available_tools, ["web_search", "calculator"])

    def test_extracts_exit_reason_from_run_completed(self):
        state = build_run_state([run_started(), run_completed()])
        self.assertEqual(state.exit_reason, "final_answer")

    def test_exit_reason_error_from_run_errored(self):
        state = build_run_state([
            run_started(),
            evt("run.errored", 5, {"error_type": "RuntimeError"}),
        ])
        self.assertEqual(state.exit_reason, "error")

    def test_tool_calls_extracted(self):
        events = [
            run_started(),
            tool_evt("web_search", 1),
            tool_evt("calculator", 2),
            tool_evt("web_search", 3),
        ]
        state = build_run_state(events)
        self.assertEqual(len(state.tool_calls), 3)
        self.assertEqual(state.tool_calls[0].tool_name, "web_search")
        self.assertEqual(state.tool_calls[1].tool_name, "calculator")
        self.assertEqual(state.tool_calls[2].tool_name, "web_search")

    def test_tool_calls_preserve_step_order(self):
        events = [
            run_started(),
            tool_evt("b", 3),
            tool_evt("a", 1),
            tool_evt("c", 2),
        ]
        # Events arrive sorted by step_index from DB
        events_sorted = sorted(events, key=lambda e: e["step_index"])
        state = build_run_state(events_sorted)
        names = [c.tool_name for c in state.tool_calls]
        self.assertEqual(names, ["a", "c", "b"])

    def test_retrievals_extracted(self):
        events = [
            run_started(),
            retrieval_evt("docs", count=5, score=0.87, step=1),
            run_completed(),
        ]
        state = build_run_state(events)
        self.assertEqual(len(state.retrievals), 1)
        self.assertEqual(state.retrievals[0].result_count, 5)
        self.assertAlmostEqual(state.retrievals[0].top_score, 0.87)

    def test_empty_retrieval_extracted(self):
        events = [
            run_started(),
            retrieval_evt("docs", count=0, score=None, step=1),
            run_completed(),
        ]
        state = build_run_state(events)
        self.assertEqual(state.retrievals[0].result_count, 0)
        self.assertIsNone(state.retrievals[0].top_score)

    def test_current_step_is_max_step_index(self):
        events = [
            run_started(step=0),
            llm_evt(step=1),
            tool_evt("web_search", step=2),
            llm_evt(step=3),
            run_completed(step=4),
        ]
        state = build_run_state(events)
        self.assertEqual(state.current_step, 4)

    def test_events_list_populated(self):
        events = [run_started(), llm_evt(1), tool_evt("web_search", 2), run_completed(3)]
        state = build_run_state(events)
        self.assertEqual(len(state.events), 4)

    def test_unknown_event_type_skipped_gracefully(self):
        events = [run_started(), evt("future.unknown.type", 1), run_completed(2)]
        state = build_run_state(events)  # should not raise
        self.assertEqual(state.run_id, "run-test-1")

    def test_missing_payload_handled(self):
        raw = {
            "event_type": "tool.called", "run_id": "r1",
            "agent_id": "a1", "agent_version": "v1",
            "step_index": 1, "timestamp": time.time(),
            "payload": None, "parent_run_id": None,
        }
        state = build_run_state([raw])
        self.assertEqual(len(state.tool_calls), 1)
        self.assertEqual(state.tool_calls[0].tool_name, "unknown")


# ── Detector integration via RunBuilder ───────────────────────────────────────

class TestDetectorIntegrationViaRunBuilder(unittest.TestCase):
    """
    Verifies that RunBuilder produces RunState that correctly
    triggers detectors. This is the key integration test.
    """

    def _run(self, events):
        from dunetrace.detectors import run_detectors
        state = build_run_state(events)
        return run_detectors(state)

    def test_tool_loop_detected(self):
        events = [
            run_started(tools=["web_search"]),
            tool_evt("web_search", 1),
            tool_evt("web_search", 2),
            tool_evt("web_search", 3),
            tool_evt("web_search", 4),
            tool_evt("web_search", 5),
            run_completed(6),
        ]
        signals = self._run(events)
        types = [s.failure_type for s in signals]
        self.assertIn(FailureType.TOOL_LOOP, types)

    def test_tool_thrashing_detected(self):
        events = [
            run_started(tools=["a", "b"]),
            tool_evt("a", 1), tool_evt("b", 2),
            tool_evt("a", 3), tool_evt("b", 4),
            tool_evt("a", 5), tool_evt("b", 6),
            run_completed(7),
        ]
        signals = self._run(events)
        types = [s.failure_type for s in signals]
        self.assertIn(FailureType.TOOL_THRASHING, types)

    def test_tool_avoidance_detected(self):
        events = [
            run_started(tools=["web_search"]),
            llm_evt(1), llm_responded_evt(1),
            llm_evt(2), llm_responded_evt(2),
            run_completed(3),
        ]
        signals = self._run(events)
        types = [s.failure_type for s in signals]
        self.assertIn(FailureType.TOOL_AVOIDANCE, types)

    def test_rag_empty_retrieval_detected(self):
        events = [
            run_started(tools=["search"]),
            retrieval_evt("docs", count=0, score=None, step=1),
            run_completed(2),
        ]
        signals = self._run(events)
        types = [s.failure_type for s in signals]
        self.assertIn(FailureType.RAG_EMPTY_RETRIEVAL, types)

    def test_no_signal_on_healthy_run(self):
        events = [
            run_started(tools=["web_search"]),
            tool_evt("web_search", 1),
            tool_evt("web_search", 2),   # same tool twice — below threshold
            tool_evt("calculator", 3),
            run_completed(4),
        ]
        signals = self._run(events)
        # No structural failures in this run
        self.assertEqual(signals, [])

    def test_no_signal_when_no_tools_available(self):
        """TOOL_AVOIDANCE should not fire if agent has no tools."""
        events = [
            run_started(tools=[]),
            llm_evt(1),
            run_completed(2),
        ]
        signals = self._run(events)
        types = [s.failure_type for s in signals]
        self.assertNotIn(FailureType.TOOL_AVOIDANCE, types)

    def test_multiple_signals_on_same_run(self):
        """TOOL_LOOP can fire alongside TOOL_AVOIDANCE on a bad run."""
        # No tools in run_started, but still has tool calls = loop
        events = [
            run_started(tools=["web_search"]),
            tool_evt("web_search", 1),
            tool_evt("web_search", 2),
            tool_evt("web_search", 3),
            tool_evt("web_search", 4),
            tool_evt("web_search", 5),
            run_completed(6),
        ]
        signals = self._run(events)
        self.assertGreaterEqual(len(signals), 1)


# ── Process run (async, mocked DB) ────────────────────────────────────────────

class TestProcessRun(unittest.IsolatedAsyncioTestCase):

    async def test_process_run_writes_signals_for_looping_run(self):
        events = [
            run_started(tools=["web_search"]),
            tool_evt("web_search", 1),
            tool_evt("web_search", 2),
            tool_evt("web_search", 3),
            tool_evt("web_search", 4),
            tool_evt("web_search", 5),
            run_completed(6),
        ]

        written_signals = []
        written_shadow  = []

        async def mock_write(signals, shadow):
            written_signals.extend(signals)
            written_shadow.append(shadow)
            return len(signals)

        with patch("detector_svc.worker.fetch_run_events",   AsyncMock(return_value=events)), \
             patch("detector_svc.worker.write_signals",       mock_write), \
             patch("detector_svc.worker.mark_run_processed",  AsyncMock()):
            from detector_svc.worker import process_run
            count = await process_run("run-test-1", "agent-test", "abc1", "completed")

        self.assertGreater(count, 0)
        self.assertIn(
            FailureType.TOOL_LOOP,
            [s.failure_type for s in written_signals]
        )

    async def test_process_run_shadow_mode_by_default(self):
        """All signals should be shadow=True since LIVE_DETECTORS is empty."""
        events = [
            run_started(tools=["web_search"]),
            *[tool_evt("web_search", i) for i in range(1, 6)],
            run_completed(6),
        ]

        captured_shadow = []

        async def mock_write(signals, shadow):
            captured_shadow.append(shadow)
            return len(signals)

        with patch("detector_svc.worker.fetch_run_events",  AsyncMock(return_value=events)), \
             patch("detector_svc.worker.write_signals",      mock_write), \
             patch("detector_svc.worker.mark_run_processed", AsyncMock()), \
             patch("detector_svc.worker.LIVE_DETECTORS",     set()):  # empty = all shadow
            from detector_svc.worker import process_run
            await process_run("run-1", "agent-1", "v1", "completed")

        self.assertTrue(all(s is True for s in captured_shadow),
                        "All signals should be shadow when LIVE_DETECTORS is empty")

    async def test_process_run_marks_processed_even_with_no_signals(self):
        events = [run_started(tools=["web_search"]), tool_evt("web_search", 1), run_completed(2)]
        mark_mock = AsyncMock()

        with patch("detector_svc.worker.fetch_run_events",  AsyncMock(return_value=events)), \
             patch("detector_svc.worker.write_signals",      AsyncMock(return_value=0)), \
             patch("detector_svc.worker.mark_run_processed", mark_mock):
            from detector_svc.worker import process_run
            await process_run("run-1", "agent-1", "v1", "completed")

        mark_mock.assert_called_once()

    async def test_process_run_handles_empty_events_gracefully(self):
        mark_mock = AsyncMock()

        with patch("detector_svc.worker.fetch_run_events",  AsyncMock(return_value=[])), \
             patch("detector_svc.worker.mark_run_processed", mark_mock):
            from detector_svc.worker import process_run
            count = await process_run("run-empty", "a", "v", "completed")

        self.assertEqual(count, 0)
        mark_mock.assert_called_once()

    async def test_poll_once_processes_completed_and_stalled(self):
        completed = [{"run_id": "r1", "agent_id": "a1", "agent_version": "v1",
                      "trigger": "completed"}]
        stalled   = [{"run_id": "r2", "agent_id": "a1", "agent_version": "v1",
                      "trigger": "stalled"}]

        healthy_events = [
            run_started(tools=["web_search"]),
            tool_evt("web_search", 1),
            run_completed(2),
        ]

        with patch("detector_svc.worker.fetch_completed_runs",
                   AsyncMock(return_value=completed)), \
             patch("detector_svc.worker.fetch_stalled_runs",
                   AsyncMock(return_value=stalled)), \
             patch("detector_svc.worker.fetch_run_events",
                   AsyncMock(return_value=healthy_events)), \
             patch("detector_svc.worker.write_signals",      AsyncMock(return_value=0)), \
             patch("detector_svc.worker.mark_run_processed", AsyncMock()):
            from detector_svc.worker import poll_once
            runs, signals = await poll_once()

        self.assertEqual(runs, 2)

    async def test_poll_once_returns_zero_when_no_work(self):
        with patch("detector_svc.worker.fetch_completed_runs", AsyncMock(return_value=[])), \
             patch("detector_svc.worker.fetch_stalled_runs",   AsyncMock(return_value=[])):
            from detector_svc.worker import poll_once
            runs, signals = await poll_once()

        self.assertEqual(runs, 0)
        self.assertEqual(signals, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
