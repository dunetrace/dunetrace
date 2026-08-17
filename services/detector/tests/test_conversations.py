"""
Tests for conversation modeling (Phase 3.1) — the runs/conversations
registry in detector_svc.db and its wiring into detector_svc.worker's
process_run(). All DB calls are mocked — no real Postgres needed.

Run:
    cd services/detector
    python -m pytest tests/test_conversations.py -v
"""

from __future__ import annotations

import sys
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/detector"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detector_svc.db as db_module

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_pool(fetchrow_return=None):
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))
    pool._conn = conn
    return pool


class _AsyncCtx:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *_):
        pass


# ── upsert_run_and_conversation ─────────────────────────────────────────────────


class TestUpsertRunAndConversation(unittest.IsolatedAsyncioTestCase):
    async def test_noop_when_pool_is_none(self):
        with patch.object(db_module, "_pool", None):
            # Should not raise
            await db_module.upsert_run_and_conversation(
                "run-1", "org-1", "agent-1", "v1", time.time(), "conv_123"
            )

    async def test_no_conversation_id_skips_conversations_upsert_but_still_registers_run(self):
        pool = _make_pool()
        with patch.object(db_module, "_pool", pool):
            await db_module.upsert_run_and_conversation(
                "run-1", "org-1", "agent-1", "v1", time.time(), None
            )
        # Only one query executed — the runs insert. No conversations fetchrow.
        pool._conn.fetchrow.assert_not_called()
        pool._conn.execute.assert_called_once()
        call_args = pool._conn.execute.call_args
        self.assertIn("INSERT INTO runs", call_args[0][0])
        # conversation_id positional arg (5th bound param) is None
        self.assertIsNone(call_args[0][5])

    async def test_conversation_id_present_upserts_conversation_then_links_run(self):
        pool = _make_pool(fetchrow_return={"id": 42})
        with patch.object(db_module, "_pool", pool):
            await db_module.upsert_run_and_conversation(
                "run-1", "org-1", "agent-1", "v1", time.time(), "conv_123"
            )
        pool._conn.fetchrow.assert_called_once()
        conv_call = pool._conn.fetchrow.call_args
        self.assertIn("INSERT INTO conversations", conv_call[0][0])
        self.assertEqual(conv_call[0][1:], ("org-1", "agent-1", "conv_123"))

        pool._conn.execute.assert_called_once()
        run_call = pool._conn.execute.call_args
        self.assertIn("INSERT INTO runs", run_call[0][0])
        self.assertEqual(run_call[0][5], 42)  # conversation_id FK resolved from fetchrow

    async def test_run_id_and_agent_fields_bound_correctly(self):
        pool = _make_pool(fetchrow_return={"id": 7})
        ts = time.time()
        with patch.object(db_module, "_pool", pool):
            await db_module.upsert_run_and_conversation(
                "run-xyz", "org-2", "agent-2", "v2", ts, "conv_abc"
            )
        run_call = pool._conn.execute.call_args
        args = run_call[0]
        self.assertEqual(args[1], "run-xyz")
        self.assertEqual(args[2], "org-2")
        self.assertEqual(args[3], "agent-2")
        self.assertEqual(args[4], "v2")
        self.assertEqual(args[6], ts)


# ── process_run() wiring ────────────────────────────────────────────────────────


def _evt(event_type: str, step_index: int = 1, payload: dict = None, **kw) -> dict:
    return {
        "event_type": event_type,
        "run_id": kw.get("run_id", "run-test-1"),
        "agent_id": kw.get("agent_id", "agent-test"),
        "agent_version": kw.get("agent_version", "abc12345"),
        "step_index": step_index,
        "timestamp": kw.get("timestamp", time.time()),
        "payload": payload or {},
        "parent_run_id": kw.get("parent_run_id"),
        "conversation_id": kw.get("conversation_id"),
    }


def _run_started(step: int = 0, conversation_id=None, timestamp=None) -> dict:
    return _evt(
        "run.started",
        step,
        {"input_text": "hi", "model": "gpt-4o", "tools": []},
        conversation_id=conversation_id,
        timestamp=timestamp or time.time(),
    )


def _run_completed(step: int = 2, conversation_id=None) -> dict:
    return _evt(
        "run.completed",
        step,
        {"exit_reason": "final_answer", "total_steps": step},
        conversation_id=conversation_id,
    )


class TestProcessRunConversationWiring(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_id_extracted_and_passed_through(self):
        events = [
            _run_started(conversation_id="conv_123"),
            _run_completed(conversation_id="conv_123"),
        ]
        with (
            patch("detector_svc.worker.fetch_run_events", AsyncMock(return_value=events)),
            patch("detector_svc.worker.write_signals", AsyncMock(return_value=0)),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()),
            patch("detector_svc.worker.upsert_run_and_conversation", AsyncMock()) as upsert_mock,
        ):
            from detector_svc.worker import process_run

            await process_run("run-test-1", "agent-test", "abc1", "completed", "org-1")

        upsert_mock.assert_called_once()
        args = upsert_mock.call_args[0]
        self.assertEqual(args[0], "run-test-1")
        self.assertEqual(args[1], "org-1")
        self.assertEqual(args[2], "agent-test")
        self.assertEqual(args[3], "abc1")
        self.assertEqual(args[5], "conv_123")

    async def test_backward_compat_no_conversation_id_still_registers_run_with_none(self):
        """Old SDK calls / single-turn agents omit conversation_id entirely —
        process_run() must still call upsert_run_and_conversation (to
        register the run itself), just with conversation_id=None."""
        events = [_run_started(), _run_completed()]
        with (
            patch("detector_svc.worker.fetch_run_events", AsyncMock(return_value=events)),
            patch("detector_svc.worker.write_signals", AsyncMock(return_value=0)),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()),
            patch("detector_svc.worker.upsert_run_and_conversation", AsyncMock()) as upsert_mock,
        ):
            from detector_svc.worker import process_run

            await process_run("run-test-1", "agent-test", "abc1", "completed", "org-1")

        upsert_mock.assert_called_once()
        self.assertIsNone(upsert_mock.call_args[0][5])

    async def test_conversation_registry_failure_does_not_block_processing(self):
        """A bug in conversation registry updates must never prevent
        mark_run_processed from being called — same isolation as issue
        tracking's own try/except."""
        events = [_run_started(), _run_completed()]
        with (
            patch("detector_svc.worker.fetch_run_events", AsyncMock(return_value=events)),
            patch("detector_svc.worker.write_signals", AsyncMock(return_value=0)),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()) as mark_mock,
            patch(
                "detector_svc.worker.upsert_run_and_conversation",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            from detector_svc.worker import process_run

            count = await process_run("run-test-1", "agent-test", "abc1", "completed", "org-1")

        mark_mock.assert_called_once()
        self.assertEqual(count, 0)

    async def test_started_at_falls_back_to_first_event_timestamp_when_run_started_missing(self):
        """Defensive fallback — run.started missing from the event list
        (data loss / instrumentation gap) must not crash the registry
        update; started_at falls back to the earliest event's timestamp."""
        fallback_ts = 1_700_000_000.0
        events = [
            _evt("tool.called", 1, {"tool_name": "search"}, timestamp=fallback_ts),
            _run_completed(step=2),
        ]
        with (
            patch("detector_svc.worker.fetch_run_events", AsyncMock(return_value=events)),
            patch("detector_svc.worker.write_signals", AsyncMock(return_value=0)),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()),
            patch("detector_svc.worker.upsert_run_and_conversation", AsyncMock()) as upsert_mock,
        ):
            from detector_svc.worker import process_run

            await process_run("run-test-1", "agent-test", "abc1", "completed", "org-1")

        upsert_mock.assert_called_once()
        self.assertEqual(upsert_mock.call_args[0][4], fallback_ts)


if __name__ == "__main__":
    unittest.main()
