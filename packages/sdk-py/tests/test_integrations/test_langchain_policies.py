"""
Tests proving DunetraceCallbackHandler actually enforces policies registered
via dt.add_policy() — the LangChain/LangGraph integration used to silently
skip policy evaluation entirely (events were emitted with self._client._emit()
directly, bypassing RunContext._check_policies()). These tests would fail
against the old implementation.

Requires the real `langchain` extra (langchain-core + langgraph) — unlike
test_langchain.py, this suite is not runnable against a bare stub since it
exercises real LangGraph tool-calling control flow. Skipped entirely (not
failed) when that extra isn't installed — install with
`pip install -e "packages/sdk-py[dev,langchain]"` to run it.
"""

from __future__ import annotations

import unittest

from dunetrace.integrations.langchain import DunetraceCallbackHandler
from dunetrace.models import EventType
from dunetrace.policies import Policy, PolicyEngine, PolicyViolation

try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    _HAS_LANGCHAIN_EXTRA = True
except ImportError:
    _HAS_LANGCHAIN_EXTRA = False


if _HAS_LANGCHAIN_EXTRA:

    class _ToolCallingFakeModel(FakeMessagesListChatModel):
        """FakeMessagesListChatModel.bind_tools() raises NotImplementedError by
        default; create_react_agent requires it. We script exact tool_calls on
        each scripted AIMessage instead of relying on real tool-schema binding."""

        def bind_tools(self, tools, **kwargs):
            return self


class _FakeClient:
    """Minimal stand-in for Dunetrace with a *real* PolicyEngine — a bare
    MagicMock()'s auto-mocked _policy_engine has len() == 0, which makes
    RunContext._check_policies() a no-op and would hide this entire class of
    bug."""

    def __init__(self) -> None:
        self.emitted: list = []
        self._policy_engine = PolicyEngine()
        self._ingest_url = None
        self._api_key = None

    def _emit(self, event) -> None:
        self.emitted.append(event)

    def flush(self) -> None:
        pass


def _make_handler(policies=None, tools=None, model="fake"):
    client = _FakeClient()
    for p in policies or []:
        client._policy_engine.add(p)
    handler = DunetraceCallbackHandler(
        client, agent_id="test-agent", model=model, tools=tools or ["counting_tool"]
    )
    return handler, client


def _stop_policy(value: int = 3) -> Policy:
    return Policy(
        name="tool-call-cap",
        condition={"trigger": "tool_call_count", "operator": "gt", "value": value},
        action={"type": "stop"},
    )


# ── End-to-end: real LangGraph agent, real tool-calling control flow ───────────


@unittest.skipUnless(_HAS_LANGCHAIN_EXTRA, "requires the langchain/langgraph extra")
class TestPolicyStopsRealAgentRun(unittest.TestCase):
    def setUp(self):
        self.call_count = 0

        @tool
        def counting_tool(x: str) -> str:
            """A tool that counts how many times it actually runs."""
            self.call_count += 1
            return f"result {self.call_count}"

        self.counting_tool = counting_tool

    def _build_agent(self, n_tool_calls: int = 5):
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "counting_tool", "args": {"x": str(i)}, "id": str(i)}],
            )
            for i in range(n_tool_calls)
        ]
        responses.append(AIMessage(content="done"))
        model = _ToolCallingFakeModel(responses=responses)
        return create_react_agent(model, [self.counting_tool])

    def test_stop_policy_raises_policy_violation_out_of_agent_invoke(self):
        agent = self._build_agent()
        handler, client = _make_handler(policies=[_stop_policy(value=3)])

        with self.assertRaises(PolicyViolation):
            agent.invoke({"messages": [("human", "go")]}, config={"callbacks": [handler]})

    def test_stop_policy_prevents_the_overlimit_tool_call_from_running(self):
        """The core bug: previously the policy never fired at all, so every
        scripted tool call would run to completion regardless of the cap."""
        agent = self._build_agent()
        handler, client = _make_handler(policies=[_stop_policy(value=3)])

        with self.assertRaises(PolicyViolation):
            agent.invoke({"messages": [("human", "go")]}, config={"callbacks": [handler]})

        self.assertEqual(self.call_count, 3)

    def test_no_stop_policy_lets_all_tool_calls_run(self):
        """Regression: without a stop policy, behavior is unchanged."""
        agent = self._build_agent(n_tool_calls=3)
        handler, client = _make_handler(policies=[])

        agent.invoke({"messages": [("human", "go")]}, config={"callbacks": [handler]})

        self.assertEqual(self.call_count, 3)

    def test_higher_threshold_does_not_block_a_run_under_the_cap(self):
        agent = self._build_agent(n_tool_calls=2)
        handler, client = _make_handler(policies=[_stop_policy(value=5)])

        agent.invoke({"messages": [("human", "go")]}, config={"callbacks": [handler]})

        self.assertEqual(self.call_count, 2)

    def test_policy_violation_surfaces_as_run_errored_with_policy_name(self):
        agent = self._build_agent()
        handler, client = _make_handler(policies=[_stop_policy(value=3)])

        with self.assertRaises(PolicyViolation):
            agent.invoke({"messages": [("human", "go")]}, config={"callbacks": [handler]})

        errored = next(e for e in client.emitted if e.event_type == EventType.RUN_ERRORED)
        self.assertEqual(errored.payload["exit_reason"], "policy_violation")
        self.assertEqual(errored.payload["policy_name"], "tool-call-cap")

    def test_no_run_completed_event_when_policy_stops_the_run(self):
        agent = self._build_agent()
        handler, client = _make_handler(policies=[_stop_policy(value=3)])

        with self.assertRaises(PolicyViolation):
            agent.invoke({"messages": [("human", "go")]}, config={"callbacks": [handler]})

        self.assertNotIn(EventType.RUN_COMPLETED, [e.event_type for e in client.emitted])

    def test_policy_triggering_call_is_still_traced(self):
        """The call that trips the policy is emitted before the run stops —
        so the dashboard shows exactly which call tripped it, not a gap."""
        agent = self._build_agent()
        handler, client = _make_handler(policies=[_stop_policy(value=3)])

        with self.assertRaises(PolicyViolation):
            agent.invoke({"messages": [("human", "go")]}, config={"callbacks": [handler]})

        tool_called_events = [e for e in client.emitted if e.event_type == EventType.TOOL_CALLED]
        self.assertEqual(len(tool_called_events), 4)  # 3 that ran + the one that tripped the cap


# ── Direct handler calls: log-only policies, tool_called() path in isolation ──


class TestNonStopPolicies(unittest.TestCase):
    def test_log_policy_does_not_raise(self):
        handler, client = _make_handler(
            policies=[
                Policy(
                    name="just-log",
                    condition={"trigger": "tool_call_count", "operator": "gt", "value": 0},
                    action={"type": "log"},
                )
            ]
        )
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start({"name": "search"}, "q", run_id="lc-t", parent_run_id="lc-1")
        self.assertIn(EventType.POLICY_TRIGGERED, [e.event_type for e in client.emitted])

    def test_tool_call_count_metric_reflects_the_call_that_just_happened(self):
        """Regression for the old dead _RunCtx.tool_call_count counter: the
        metric the policy engine sees must come from the real RunContext."""
        triggered_at = []

        class _RecordingEngine(PolicyEngine):
            def evaluate(self, agent_id, metrics, triggered):
                triggered_at.append(metrics["tool_call_count"])
                return super().evaluate(agent_id, metrics, triggered)

        client = _FakeClient()
        client._policy_engine = _RecordingEngine()
        client._policy_engine.add(
            Policy(
                name="never-fires",
                condition={"trigger": "tool_call_count", "operator": "gt", "value": 999},
                action={"type": "stop"},
            )
        )
        handler = DunetraceCallbackHandler(client, agent_id="a", model="fake", tools=["search"])
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start({"name": "search"}, "q1", run_id="t1", parent_run_id="lc-1")
        handler.on_tool_start({"name": "search"}, "q2", run_id="t2", parent_run_id="lc-1")
        self.assertEqual(triggered_at, [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
