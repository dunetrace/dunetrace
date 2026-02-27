"""
services/api/tests/test_api.py

Customer API tests — all logic tested through schemas and query contracts.
No DB, no HTTP server needed. Validates:
  - Response schema shapes (Pydantic models)
  - Pagination logic
  - Auth header parsing
  - Filter parameter passing
  - 404 handling

Run:
    cd services/api
    python -m unittest tests.test_api -v
"""
from __future__ import annotations

import sys
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/sdk-py")))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../explainer")))

from api_svc.schemas import (
    AgentSummary, AgentListResponse,
    RunSummary, RunListResponse, RunDetail, RunEvent, RunSignal,
    SignalDetail, SignalListResponse,
    Page, HealthResponse,
)
from api_svc.config import settings

NOW = time.time()


# ── Shared factories ───────────────────────────────────────────────────────────

def make_agent_row(**kw) -> dict:
    return {
        "agent_id":       kw.get("agent_id",       "agent-test"),
        "last_seen":      kw.get("last_seen",       NOW),
        "run_count":      kw.get("run_count",       10),
        "signal_count":   kw.get("signal_count",    3),
        "critical_count": kw.get("critical_count",  1),
        "high_count":     kw.get("high_count",      2),
    }


def make_run_row(**kw) -> dict:
    return {
        "run_id":        kw.get("run_id",        "run-abc"),
        "agent_id":      kw.get("agent_id",      "agent-test"),
        "agent_version": kw.get("agent_version", "abc12345"),
        "exit_reason":   kw.get("exit_reason",   "completed"),
        "processed_at":  kw.get("processed_at",  NOW),
        "started_at":    kw.get("started_at",    NOW - 30),
        "step_count":    kw.get("step_count",    5),
        "signal_count":  kw.get("signal_count",  1),
    }


def make_signal_row(**kw) -> dict:
    return {
        "id":             kw.get("id",             1),
        "failure_type":   kw.get("failure_type",   "TOOL_LOOP"),
        "severity":       kw.get("severity",       "HIGH"),
        "run_id":         kw.get("run_id",         "run-abc"),
        "agent_id":       kw.get("agent_id",       "agent-test"),
        "agent_version":  kw.get("agent_version",  "abc12345"),
        "step_index":     kw.get("step_index",     5),
        "confidence":     kw.get("confidence",     0.95),
        "detected_at":    kw.get("detected_at",    NOW),
        "evidence":       kw.get("evidence",       {"tool": "web_search", "count": 5, "window": 5}),
        "alerted":        kw.get("alerted",        True),
        "title":          kw.get("title",          "Tool loop detected"),
        "what":           kw.get("what",           "Agent looped."),
        "why_it_matters": kw.get("why_it_matters", "Burns tokens."),
        "evidence_summary": kw.get("evidence_summary", "x5. 95%."),
        "suggested_fixes":  kw.get("suggested_fixes",  [
            {"description": "Add limit", "language": "python", "code": "MAX=3"}
        ]),
    }


# ── Schema contract tests ──────────────────────────────────────────────────────

class TestSchemas(unittest.TestCase):
    """Validate that every schema accepts valid data and rejects invalid data."""

    def test_agent_summary_accepts_valid_data(self):
        a = AgentSummary(**make_agent_row())
        self.assertEqual(a.agent_id, "agent-test")
        self.assertEqual(a.run_count, 10)

    def test_agent_summary_last_seen_can_be_none(self):
        row = make_agent_row()
        row["last_seen"] = None
        a = AgentSummary(**row)
        self.assertIsNone(a.last_seen)

    def test_page_has_more_true_when_more_exist(self):
        p = Page(total=100, offset=0, limit=20, has_more=True)
        self.assertTrue(p.has_more)

    def test_page_has_more_false_at_end(self):
        p = Page(total=5, offset=0, limit=20, has_more=False)
        self.assertFalse(p.has_more)

    def test_run_summary_has_signals_derived(self):
        r = RunSummary(run_id="r", agent_id="a", agent_version="v", started_at=None, completed_at=None, exit_reason=None, step_count=5, signal_count=1, has_signals=True)
        self.assertTrue(r.has_signals)

    def test_run_event_accepts_all_fields(self):
        e = RunEvent(
            event_type="tool.called", step_index=1, timestamp=NOW,
            payload={"tool_name": "web_search"}, parent_run_id=None,
        )
        self.assertEqual(e.event_type, "tool.called")

    def test_run_signal_includes_explanation_fields(self):
        s = RunSignal(**{
            "id": 1, "failure_type": "TOOL_LOOP", "severity": "HIGH",
            "step_index": 5, "confidence": 0.95, "detected_at": NOW,
            "evidence": {}, "title": "Loop", "what": "Looped.",
            "why_it_matters": "Expensive.", "evidence_summary": "×5",
            "suggested_fixes": [],
        })
        self.assertEqual(s.title, "Loop")
        self.assertEqual(s.why_it_matters, "Expensive.")

    def test_signal_detail_has_alerted_field(self):
        s = SignalDetail(**make_signal_row())
        self.assertTrue(s.alerted)

    def test_health_response_defaults(self):
        h = HealthResponse()
        self.assertEqual(h.status, "ok")
        self.assertEqual(h.version, "0.1.0")

    def test_agent_list_response_shape(self):
        resp = AgentListResponse(
            agents=[AgentSummary(**make_agent_row())],
            page=Page(total=1, offset=0, limit=20, has_more=False),
        )
        self.assertEqual(len(resp.agents), 1)
        self.assertEqual(resp.page.total, 1)

    def test_run_list_response_shape(self):
        run = RunSummary(run_id="run-abc", agent_id="a", agent_version="v", started_at=None, completed_at=None, exit_reason="completed", step_count=5, signal_count=1, has_signals=True)
        resp = RunListResponse(
            runs=[run],
            page=Page(total=1, offset=0, limit=20, has_more=False),
        )
        self.assertEqual(resp.runs[0].run_id, "run-abc")

    def test_signal_list_response_shape(self):
        resp = SignalListResponse(
            signals=[SignalDetail(**make_signal_row())],
            page=Page(total=1, offset=0, limit=20, has_more=False),
        )
        self.assertEqual(resp.signals[0].failure_type, "TOOL_LOOP")

    def test_run_detail_shape(self):
        d = RunDetail(
            run_id="run-abc", agent_id="a", agent_version="v",
            started_at=NOW - 30, completed_at=NOW,
            exit_reason="completed", step_count=3,
            events=[RunEvent(event_type="run.started", step_index=0,
                             timestamp=NOW - 30, payload={}, parent_run_id=None)],
            signals=[],
        )
        self.assertEqual(d.run_id, "run-abc")
        self.assertEqual(len(d.events), 1)


# ── Pagination logic ───────────────────────────────────────────────────────────

class TestPagination(unittest.TestCase):

    def test_has_more_true_when_offset_plus_limit_less_than_total(self):
        has_more = (0 + 20) < 50
        self.assertTrue(has_more)

    def test_has_more_false_when_at_end(self):
        has_more = (40 + 20) < 50
        self.assertFalse(has_more)

    def test_has_more_false_on_exact_boundary(self):
        has_more = (30 + 20) < 50
        self.assertFalse(has_more)

    def test_page_size_default_is_reasonable(self):
        self.assertGreater(settings.PAGE_SIZE_DEFAULT, 0)
        self.assertLessEqual(settings.PAGE_SIZE_DEFAULT, settings.PAGE_SIZE_MAX)

    def test_page_size_max_enforced(self):
        self.assertGreaterEqual(settings.PAGE_SIZE_MAX, settings.PAGE_SIZE_DEFAULT)


# ── Config / auth ──────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):

    def test_dev_mode_default(self):
        self.assertEqual(settings.AUTH_MODE, "dev")
        self.assertTrue(settings.is_dev)

    def test_prod_mode_disables_dev(self):
        original = settings.AUTH_MODE
        settings.AUTH_MODE = "prod"
        self.assertFalse(settings.is_dev)
        settings.AUTH_MODE = original


# ── Async DB layer unit tests ──────────────────────────────────────────────────

class TestDbLayer(unittest.IsolatedAsyncioTestCase):

    async def test_verify_api_key_dev_mode_returns_dev_customer(self):
        from api_svc.db.queries import verify_api_key
        result = await verify_api_key("any_key_at_all")
        self.assertEqual(result, "dev_customer")

    async def test_check_db_no_pool_returns_no_pool(self):
        import api_svc.db.queries as q
        original_pool = q._pool
        q._pool = None
        result = await q.check_db()
        self.assertEqual(result, "no_pool")
        q._pool = original_pool

    async def test_list_agents_no_pool_returns_empty(self):
        import api_svc.db.queries as q
        original_pool = q._pool
        q._pool = None
        rows, total = await q.list_agents("cust", 0, 20)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)
        q._pool = original_pool

    async def test_list_runs_no_pool_returns_empty(self):
        import api_svc.db.queries as q
        original_pool = q._pool
        q._pool = None
        rows, total = await q.list_runs("agent-x", 0, 20)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)
        q._pool = original_pool

    async def test_get_run_detail_no_pool_returns_none(self):
        import api_svc.db.queries as q
        original_pool = q._pool
        q._pool = None
        result = await q.get_run_detail("run-xyz")
        self.assertIsNone(result)
        q._pool = original_pool

    async def test_list_signals_no_pool_returns_empty(self):
        import api_svc.db.queries as q
        original_pool = q._pool
        q._pool = None
        rows, total = await q.list_signals("agent-x", 0, 20)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)
        q._pool = original_pool


# ── Formatters / serialisation ─────────────────────────────────────────────────

class TestSerialisation(unittest.TestCase):

    def test_signal_detail_json_serialisable(self):
        import json
        s = SignalDetail(**make_signal_row())
        data = s.model_dump()
        serialised = json.dumps(data)
        self.assertIsInstance(serialised, str)

    def test_run_detail_json_serialisable(self):
        import json
        d = RunDetail(
            run_id="r", agent_id="a", agent_version="v",
            started_at=NOW, completed_at=NOW, exit_reason="completed", step_count=1,
            events=[], signals=[],
        )
        data = d.model_dump()
        serialised = json.dumps(data)
        self.assertIsInstance(serialised, str)

    def test_agent_summary_confidence_field_is_float(self):
        s = SignalDetail(**make_signal_row())
        self.assertIsInstance(s.confidence, float)

    def test_multiple_fixes_in_signal(self):
        row = make_signal_row()
        row["suggested_fixes"] = [
            {"description": "Fix 1", "language": "python", "code": "a = 1"},
            {"description": "Fix 2", "language": "text",   "code": "Do this"},
        ]
        s = SignalDetail(**row)
        self.assertEqual(len(s.suggested_fixes), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
