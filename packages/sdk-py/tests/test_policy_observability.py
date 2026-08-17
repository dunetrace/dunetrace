"""
Policy evaluation observability (Phase 5) — SDK side.

Covers the rate limiter, the PolicyEvaluationRecord / reason builder, structured
DEBUG logging, and end-to-end record shipping (a policy.evaluated event with the
right payload) when reporting is enabled — and silence when it isn't.

Run: python -m unittest tests.test_policy_observability -v
"""

from __future__ import annotations

import logging
import unittest

from dunetrace import Dunetrace
from dunetrace.models import EventType
from dunetrace.policies import eval_logger
from dunetrace.policies.observability import (
    EvaluationRateLimiter,
    PolicyEvaluationRecord,
    build_reason,
)


class _CapturingExporter:
    """Records every event the client emits (synchronous, in-process)."""

    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)

    def policy_evals(self):
        return [e for e in self.events if e.event_type == EventType.POLICY_EVALUATED]


def _client(reporting=False, exporter=None):
    c = Dunetrace(
        api_key="k",
        policy_evaluation_reporting=reporting,
        exporters=[exporter] if exporter else None,
    )
    c._ship = lambda b: None
    return c


REFUND_POLICY = dict(
    name="refund-guard",
    condition={
        "trigger": "before_tool_call",
        "operator": "eq",
        "value": "refund",
        "match": {"args.amount": {"gt": 10000}},
    },
    action={"type": "require_approval"},
)


class TestRateLimiter(unittest.TestCase):
    def test_admits_up_to_limit(self):
        rl = EvaluationRateLimiter(limit_per_minute=5, sample_rate=0.0)
        admits = [rl.admit("p", now=100.0)[0] for _ in range(5)]
        self.assertEqual(admits, [True] * 5)

    def test_drops_beyond_limit_without_sampling(self):
        rl = EvaluationRateLimiter(limit_per_minute=3, sample_rate=0.0)
        for _ in range(3):
            rl.admit("p", now=100.0)
        admit, sampled = rl.admit("p", now=100.0)
        self.assertFalse(admit)
        self.assertTrue(sampled)

    def test_samples_deterministically_beyond_limit(self):
        rl = EvaluationRateLimiter(limit_per_minute=2, sample_rate=0.5)  # 1-in-2 beyond
        for _ in range(2):
            rl.admit("p", now=100.0)
        # beyond-limit calls: 1st dropped, 2nd sampled-in, 3rd dropped, 4th in...
        outcomes = [rl.admit("p", now=100.0) for _ in range(4)]
        self.assertEqual([o[0] for o in outcomes], [False, True, False, True])
        self.assertTrue(all(o[1] for o in outcomes))  # all flagged sampled

    def test_window_resets_after_60s(self):
        rl = EvaluationRateLimiter(limit_per_minute=1, sample_rate=0.0)
        self.assertTrue(rl.admit("p", now=100.0)[0])
        self.assertFalse(rl.admit("p", now=130.0)[0])  # same window
        self.assertTrue(rl.admit("p", now=161.0)[0])  # new window

    def test_per_policy_isolation(self):
        rl = EvaluationRateLimiter(limit_per_minute=1, sample_rate=0.0)
        self.assertTrue(rl.admit("a", now=100.0)[0])
        self.assertTrue(rl.admit("b", now=100.0)[0])  # different key, own budget


class TestRecordAndReason(unittest.TestCase):
    def test_to_dict_shape(self):
        r = PolicyEvaluationRecord(
            policy_name="p",
            policy_id=7,
            agent_id="a",
            run_id="r",
            trigger="before_tool_call",
            trigger_matched=True,
            fired=False,
            conditions=[{"field_path": "args.amount", "result": False}],
            reason="x",
            sampled=False,
            ts=1.0,
        )
        d = r.to_dict()
        self.assertEqual(d["policy_id"], 7)
        self.assertEqual(d["fired"], False)
        self.assertEqual(d["conditions"][0]["field_path"], "args.amount")

    def test_reason_fired(self):
        self.assertIn("fired", build_reason("t", True, True, []))

    def test_reason_trigger_not_matched(self):
        self.assertIn("trigger", build_reason("cost_usd", False, False, []))

    def test_reason_points_at_failing_condition(self):
        conds = [
            {
                "field_path": "args.amount",
                "operator": "gt",
                "expected": 10000,
                "actual": 500,
                "result": False,
            }
        ]
        reason = build_reason("before_tool_call", True, False, conds)
        self.assertIn("args.amount", reason)
        self.assertIn("500", reason)


class TestStructuredLogging(unittest.TestCase):
    def setUp(self):
        self._records = []

        class _Cap(logging.Handler):
            def emit(_self, rec):
                self._records.append(rec)

        self._handler = _Cap()
        eval_logger.addHandler(self._handler)
        self._prev_level = eval_logger.level
        eval_logger.setLevel(logging.DEBUG)

    def tearDown(self):
        eval_logger.removeHandler(self._handler)
        eval_logger.setLevel(self._prev_level)

    def test_evaluation_logged_with_structured_extra(self):
        c = _client(reporting=False)  # logging alone, no shipping
        c.add_policy(**REFUND_POLICY)
        with c.run("billing") as run:
            run.request_approval = lambda *a, **k: None
            run.tool_called("refund", {"amount": 500})  # below threshold
        c.shutdown(timeout=2)
        evals = [getattr(r, "policy_evaluation", None) for r in self._records]
        evals = [e for e in evals if e]
        self.assertTrue(evals)
        rec = evals[-1]
        self.assertEqual(rec["policy_name"], "refund-guard")
        self.assertFalse(rec["fired"])
        self.assertIn("args.amount", rec["conditions"][0]["field_path"])


class TestShipping(unittest.TestCase):
    def test_reporting_enabled_ships_policy_evaluated_event(self):
        exp = _CapturingExporter()
        c = _client(reporting=True, exporter=exp)
        c.add_policy(**REFUND_POLICY)
        with c.run("billing") as run:
            run.request_approval = lambda *a, **k: None
            run.tool_called("refund", {"amount": 25000})  # above threshold → fires
        c.shutdown(timeout=2)
        evals = exp.policy_evals()
        self.assertTrue(evals)
        payload = evals[-1].payload
        self.assertEqual(payload["policy_name"], "refund-guard")
        self.assertTrue(payload["fired"])
        self.assertEqual(payload["conditions"][0]["actual"], 25000)

    def test_reporting_disabled_ships_nothing(self):
        exp = _CapturingExporter()
        # reporting off AND eval logger not at DEBUG → observer inactive
        eval_logger.setLevel(logging.WARNING)
        c = _client(reporting=False, exporter=exp)
        c.add_policy(**REFUND_POLICY)
        with c.run("billing") as run:
            run.request_approval = lambda *a, **k: None
            run.tool_called("refund", {"amount": 25000})
        c.shutdown(timeout=2)
        self.assertEqual(exp.policy_evals(), [])

    def test_policy_evaluated_not_in_run_state_events(self):
        # Shipping must not pollute the run's own event/trace list.
        exp = _CapturingExporter()
        c = _client(reporting=True, exporter=exp)
        c.add_policy(**REFUND_POLICY)
        with c.run("billing") as run:
            run.request_approval = lambda *a, **k: None
            run.tool_called("refund", {"amount": 25000})
            state_event_types = [e.event_type for e in run.state.events]
        c.shutdown(timeout=2)
        self.assertNotIn(EventType.POLICY_EVALUATED, state_event_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
