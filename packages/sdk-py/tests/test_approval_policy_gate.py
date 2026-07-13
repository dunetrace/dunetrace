"""
Tests for the require_approval policy gate (Capability 2, Phase 2.5): a
before_tool_call policy makes run.tool_called() (and the @dt.tool decorators)
block on approval before the tool runs. request_approval/arequest_approval are
patched here — the flow itself is covered by test_approval_flow.py; these tests
verify the *wiring*: which tool triggers the gate, per-call semantics, and
denial propagation.

Run: python -m unittest tests.test_approval_policy_gate -v
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from dunetrace.client import DunetraceClient
from dunetrace.policies import ApprovalDenied, Policy, PolicyEngine


def _make_client() -> DunetraceClient:
    c = DunetraceClient(api_key="dt_test", api_url="http://localhost:8002", debug=False)
    c._ship = lambda batch: None
    return c


def _approve_policy(value="wire_money", operator="eq", agent_id="*"):
    return dict(
        name="approve-guarded",
        agent_id=agent_id,
        condition={"trigger": "before_tool_call", "operator": operator, "value": value},
        action={"type": "require_approval", "params": {"timeout_s": 120}},
    )


# ── Engine-level matching ─────────────────────────────────────────────────────


class TestFindApprovalPolicy(unittest.TestCase):
    def _engine(self, *policies):
        e = PolicyEngine()
        for p in policies:
            e.add(Policy.from_dict(p))
        return e

    def test_matches_guarded_tool(self):
        e = self._engine(_approve_policy("wire_money"))
        self.assertIsNotNone(e.find_approval_policy("agent", "wire_money"))

    def test_does_not_match_other_tool(self):
        e = self._engine(_approve_policy("wire_money"))
        self.assertIsNone(e.find_approval_policy("agent", "search"))

    def test_respects_agent_scoping(self):
        e = self._engine(_approve_policy(agent_id="billing-agent"))
        self.assertIsNone(e.find_approval_policy("other-agent", "wire_money"))
        self.assertIsNotNone(e.find_approval_policy("billing-agent", "wire_money"))

    def test_ignores_non_approval_before_tool_call_is_impossible(self):
        # A before_tool_call trigger with a non-require_approval action still
        # isn't returned (find_approval_policy filters on action type too).
        p = _approve_policy()
        p["action"] = {"type": "log"}
        e = self._engine(p)
        self.assertIsNone(e.find_approval_policy("agent", "wire_money"))

    def test_empty_engine_returns_none(self):
        self.assertIsNone(PolicyEngine().find_approval_policy("agent", "wire_money"))


# ── Sync gate via tool_called ─────────────────────────────────────────────────


class TestSyncGate(unittest.TestCase):
    def test_guarded_tool_triggers_request_approval(self):
        c = _make_client()
        c.add_policy(**_approve_policy("wire_money"))
        with c.run("agent") as run:
            run.request_approval = MagicMock()
            run.tool_called("wire_money", {"amt": 500})
            run.request_approval.assert_called_once()
            self.assertEqual(run.request_approval.call_args.args[0], "wire_money")
            # timeout_s from the policy params threads through
            self.assertEqual(run.request_approval.call_args.kwargs["timeout_s"], 120)
        c.shutdown(timeout=2)

    def test_unguarded_tool_not_gated(self):
        c = _make_client()
        c.add_policy(**_approve_policy("wire_money"))
        with c.run("agent") as run:
            run.request_approval = MagicMock()
            run.tool_called("search", {"q": "hi"})
            run.request_approval.assert_not_called()
        c.shutdown(timeout=2)

    def test_no_policy_means_no_gate(self):
        c = _make_client()
        with c.run("agent") as run:
            run.request_approval = MagicMock()
            run.tool_called("wire_money")
            run.request_approval.assert_not_called()
        c.shutdown(timeout=2)

    def test_denial_propagates_out_of_tool_called(self):
        c = _make_client()
        c.add_policy(**_approve_policy("wire_money"))
        with c.run("agent") as run:
            run.request_approval = MagicMock(side_effect=ApprovalDenied("wire_money", "denied", 1))
            with self.assertRaises(ApprovalDenied):
                run.tool_called("wire_money")
        c.shutdown(timeout=2)

    def test_gated_per_call_not_once_per_run(self):
        c = _make_client()
        c.add_policy(**_approve_policy("wire_money"))
        with c.run("agent") as run:
            run.request_approval = MagicMock()
            run.tool_called("wire_money")
            run.tool_called("wire_money")
            self.assertEqual(run.request_approval.call_count, 2)
        c.shutdown(timeout=2)


# ── Decorator gates ───────────────────────────────────────────────────────────


class TestSyncDecoratorGate(unittest.TestCase):
    def test_denied_tool_does_not_execute(self):
        c = _make_client()
        c.add_policy(**_approve_policy("charge_card"))
        ran = {"v": False}

        @c.tool("charge_card")
        def charge_card(amount):
            ran["v"] = True
            return "charged"

        with c.run("agent") as run:
            run.request_approval = MagicMock(side_effect=ApprovalDenied("charge_card", "denied", 1))
            with self.assertRaises(ApprovalDenied):
                charge_card(100)
        c.shutdown(timeout=2)
        self.assertFalse(ran["v"])  # tool body never executed


class TestAsyncDecoratorGate(unittest.TestCase):
    def test_async_denied_tool_does_not_execute(self):
        c = _make_client()
        c.add_policy(**_approve_policy("charge_card"))
        ran = {"v": False}

        @c.tool("charge_card")
        async def charge_card(amount):
            ran["v"] = True
            return "charged"

        async def go():
            with c.run("agent") as run:
                run.arequest_approval = AsyncMock(
                    side_effect=ApprovalDenied("charge_card", "denied", 1)
                )
                await charge_card(100)

        with self.assertRaises(ApprovalDenied):
            asyncio.run(go())
        c.shutdown(timeout=2)
        self.assertFalse(ran["v"])

    def test_async_granted_tool_runs(self):
        c = _make_client()
        c.add_policy(**_approve_policy("charge_card"))
        ran = {"v": False}

        @c.tool("charge_card")
        async def charge_card(amount):
            ran["v"] = True
            return "charged"

        async def go():
            with c.run("agent") as run:
                run.arequest_approval = AsyncMock(return_value=None)  # granted
                return await charge_card(100)

        result = asyncio.run(go())
        c.shutdown(timeout=2)
        self.assertTrue(ran["v"])
        self.assertEqual(result, "charged")


if __name__ == "__main__":
    unittest.main(verbosity=2)
