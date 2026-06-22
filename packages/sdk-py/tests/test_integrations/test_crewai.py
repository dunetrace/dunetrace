"""
Tests for DunetraceCrewCallback (CrewAI 1.x integration).

Stubs out crewai so no crewai package is required.
Run:
    cd packages/sdk-py
    python -m unittest tests.test_integrations.test_crewai -v
"""

from __future__ import annotations

import sys
import threading
import types
import time
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub crewai hooks so we can import without the real package
# ---------------------------------------------------------------------------
def _stub_crewai() -> None:
    if "crewai" in sys.modules:
        return

    crewai = types.ModuleType("crewai")
    crewai_hooks = types.ModuleType("crewai.hooks")
    crewai_hooks_llm = types.ModuleType("crewai.hooks.llm_hooks")
    crewai_hooks_tool = types.ModuleType("crewai.hooks.tool_hooks")

    _before_llm_hooks: list = []
    _after_llm_hooks: list = []
    _before_tool_hooks: list = []
    _after_tool_hooks: list = []

    class LLMCallHookContext:
        def __init__(self, llm=None, messages=None, response=""):
            self.llm = llm
            self.messages = messages or []
            self.response = response

    class ToolCallHookContext:
        def __init__(self, tool_name="tool", tool_input=None, tool_result=None, error=None):
            self.tool_name = tool_name
            self.tool_input = tool_input or {}
            self.tool_result = tool_result
            self.error = error

    crewai_hooks.get_before_llm_call_hooks = lambda: list(_before_llm_hooks)
    crewai_hooks.get_after_llm_call_hooks = lambda: list(_after_llm_hooks)
    crewai_hooks.get_before_tool_call_hooks = lambda: list(_before_tool_hooks)
    crewai_hooks.get_after_tool_call_hooks = lambda: list(_after_tool_hooks)

    def register_before_llm(fn):
        _before_llm_hooks.append(fn)

    def register_after_llm(fn):
        _after_llm_hooks.append(fn)

    def register_before_tool(fn):
        _before_tool_hooks.append(fn)

    def register_after_tool(fn):
        _after_tool_hooks.append(fn)

    def unregister_before_llm(fn):
        _before_llm_hooks.remove(fn)

    def unregister_after_llm(fn):
        _after_llm_hooks.remove(fn)

    def unregister_before_tool(fn):
        _before_tool_hooks.remove(fn)

    def unregister_after_tool(fn):
        _after_tool_hooks.remove(fn)

    crewai_hooks.register_before_llm_call_hook = register_before_llm
    crewai_hooks.register_after_llm_call_hook = register_after_llm
    crewai_hooks.register_before_tool_call_hook = register_before_tool
    crewai_hooks.register_after_tool_call_hook = register_after_tool
    crewai_hooks.unregister_before_llm_call_hook = unregister_before_llm
    crewai_hooks.unregister_after_llm_call_hook = unregister_after_llm
    crewai_hooks.unregister_before_tool_call_hook = unregister_before_tool
    crewai_hooks.unregister_after_tool_call_hook = unregister_after_tool

    crewai_hooks_llm.LLMCallHookContext = LLMCallHookContext
    crewai_hooks_tool.ToolCallHookContext = ToolCallHookContext

    crewai.hooks = crewai_hooks
    sys.modules["crewai"] = crewai
    sys.modules["crewai.hooks"] = crewai_hooks
    sys.modules["crewai.hooks.llm_hooks"] = crewai_hooks_llm
    sys.modules["crewai.hooks.tool_hooks"] = crewai_hooks_tool


_stub_crewai()

from crewai.hooks.llm_hooks import LLMCallHookContext  # noqa: E402
from crewai.hooks.tool_hooks import ToolCallHookContext  # noqa: E402

from dunetrace import Dunetrace  # noqa: E402
from dunetrace.integrations.crewai import DunetraceCrewCallback  # noqa: E402
from dunetrace.models import EventType  # noqa: E402


def _make_dt() -> tuple[Dunetrace, list]:
    from dunetrace.policies import PolicyEngine

    dt = Dunetrace.__new__(Dunetrace)
    dt._ingest_url = "http://localhost:8001/v1/ingest"
    dt._api_key = ""
    dt._otel_exporter = None
    dt._policy_engine = PolicyEngine()
    emitted: list = []
    dt._emit = lambda event: emitted.append(event)
    dt.flush = lambda: None
    return dt, emitted


class TestCrewCallbackLLMHooks(unittest.TestCase):
    def setUp(self):
        self.dt, self.emitted = _make_dt()
        self.cb = DunetraceCrewCallback(self.dt, agent_id="test-crew", model="gpt-4o")

    def test_before_llm_emits_llm_called(self):
        llm_mock = MagicMock()
        llm_mock.model = "gpt-4o"
        ctx = LLMCallHookContext(llm=llm_mock, messages=[{"role": "user", "content": "hello"}])

        with self.dt.run("test-crew", user_input="hello"):
            self.cb._before_llm(ctx)

        called = [e for e in self.emitted if e.event_type == EventType.LLM_CALLED]
        self.assertEqual(len(called), 1)

    def test_after_llm_emits_llm_responded(self):
        ctx = LLMCallHookContext(response="The capital is Paris.")

        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_llm(ctx)
            self.cb._after_llm(ctx)

        responded = [e for e in self.emitted if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)

    def test_llm_responded_has_no_completion_tokens(self):
        """CrewAI hooks don't expose token counts — llm.responded should not have them."""
        ctx = LLMCallHookContext(response="answer")
        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_llm(ctx)
            self.cb._after_llm(ctx)

        responded = [e for e in self.emitted if e.event_type == EventType.LLM_RESPONDED][0]
        # completion_tokens=0 from run_context default is fine; prompt_tokens must not appear
        self.assertNotIn("prompt_tokens", responded.payload)

    def test_no_run_context_no_emit(self):
        ctx = LLMCallHookContext(response="answer")
        self.cb._before_llm(ctx)
        self.cb._after_llm(ctx)
        self.assertEqual(self.emitted, [])

    def test_llm_hooks_not_emitted_outside_run(self):
        ctx = LLMCallHookContext(messages=[{"content": "q"}])
        # No active run
        self.cb._before_llm(ctx)
        self.assertEqual(len(self.emitted), 0)

    def test_model_name_from_llm_attribute(self):
        llm_mock = MagicMock()
        llm_mock.model = "claude-3-5-sonnet"
        ctx = LLMCallHookContext(llm=llm_mock, messages=[])

        with self.dt.run("test-crew", user_input="q") as run:
            self.cb._before_llm(ctx)

        called = [e for e in self.emitted if e.event_type == EventType.LLM_CALLED][0]
        self.assertEqual(called.payload.get("model"), "claude-3-5-sonnet")

    def test_after_llm_when_before_not_called_no_crash(self):
        """after_llm without a prior before_llm must not crash (timer miss)."""
        ctx = LLMCallHookContext(response="late")
        with self.dt.run("test-crew", user_input="q"):
            self.cb._after_llm(ctx)  # no _before_llm → t0 is None → latency ~0


class TestCrewCallbackToolHooks(unittest.TestCase):
    def setUp(self):
        self.dt, self.emitted = _make_dt()
        self.cb = DunetraceCrewCallback(self.dt, agent_id="test-crew")

    def test_before_tool_emits_tool_called(self):
        ctx = ToolCallHookContext(tool_name="web_search", tool_input={"query": "LLMs"})
        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_tool(ctx)

        called = [e for e in self.emitted if e.event_type == EventType.TOOL_CALLED]
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0].payload.get("tool_name"), "web_search")

    def test_after_tool_success_emits_tool_responded(self):
        ctx = ToolCallHookContext(
            tool_name="web_search",
            tool_input={"query": "q"},
            tool_result="Some result text",
        )
        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_tool(ctx)
            self.cb._after_tool(ctx)

        responded = [e for e in self.emitted if e.event_type == EventType.TOOL_RESPONDED]
        self.assertEqual(len(responded), 1)
        self.assertTrue(responded[0].payload.get("success"))

    def test_after_tool_failure_result_none_marks_failure(self):
        """Regression: tool_result=None must not be silently treated as success."""
        before_ctx = ToolCallHookContext(tool_name="fetch", tool_input={})
        after_ctx = ToolCallHookContext(
            tool_name="fetch",
            tool_result=None,  # tool raised, no result
        )
        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_tool(before_ctx)
            self.cb._after_tool(after_ctx)

        responded = [e for e in self.emitted if e.event_type == EventType.TOOL_RESPONDED]
        self.assertFalse(responded[0].payload.get("success"))

    def test_after_tool_explicit_error_marks_failure(self):
        """When ctx.error is set, the tool must be reported as failed."""
        ctx = ToolCallHookContext(
            tool_name="read_file",
            tool_result="partial content",
            error=RuntimeError("permission denied"),
        )
        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_tool(ctx)
            self.cb._after_tool(ctx)

        responded = [e for e in self.emitted if e.event_type == EventType.TOOL_RESPONDED]
        self.assertFalse(responded[0].payload.get("success"))

    def test_after_tool_empty_string_result_is_success(self):
        """Empty string is a valid (if unusual) successful result."""
        before_ctx = ToolCallHookContext(tool_name="ping")
        after_ctx = ToolCallHookContext(tool_name="ping", tool_result="")
        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_tool(before_ctx)
            self.cb._after_tool(after_ctx)

        responded = [e for e in self.emitted if e.event_type == EventType.TOOL_RESPONDED]
        self.assertTrue(responded[0].payload.get("success"))

    def test_output_length_from_result(self):
        ctx = ToolCallHookContext(tool_name="t", tool_result="hello world")
        with self.dt.run("test-crew", user_input="q"):
            self.cb._before_tool(ctx)
            self.cb._after_tool(ctx)

        responded = [e for e in self.emitted if e.event_type == EventType.TOOL_RESPONDED][0]
        self.assertEqual(responded.payload.get("output_length"), 11)

    def test_no_run_context_no_emit(self):
        ctx = ToolCallHookContext(tool_name="t", tool_result="x")
        self.cb._before_tool(ctx)
        self.cb._after_tool(ctx)
        self.assertEqual(self.emitted, [])

    def test_concurrent_different_tools_isolated(self):
        """Two tools active concurrently in separate threads must not share timers."""
        results = {}

        def _call_tool(name):
            with self.dt.run("test-crew", user_input=name):
                before = ToolCallHookContext(tool_name=name, tool_input={})
                self.cb._before_tool(before)
                time.sleep(0.01)
                after = ToolCallHookContext(tool_name=name, tool_result="ok")
                self.cb._after_tool(after)

        t1 = threading.Thread(target=_call_tool, args=("tool_a",))
        t2 = threading.Thread(target=_call_tool, args=("tool_b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both tools must have responded — no timer collisions
        responded = [e for e in self.emitted if e.event_type == EventType.TOOL_RESPONDED]
        self.assertEqual(len(responded), 2)
        tool_names = {r.payload.get("tool_name") for r in responded}
        self.assertEqual(tool_names, {"tool_a", "tool_b"})


class TestCrewCallbackInstallUninstall(unittest.TestCase):
    def test_install_registers_hooks(self):
        import crewai.hooks as ch

        dt, _ = _make_dt()
        cb = DunetraceCrewCallback(dt, agent_id="crew")
        before = len(ch.get_before_llm_call_hooks())
        cb.install()
        self.assertGreater(len(ch.get_before_llm_call_hooks()), before)
        cb.uninstall()

    def test_install_is_idempotent(self):
        import crewai.hooks as ch

        dt, _ = _make_dt()
        cb = DunetraceCrewCallback(dt, agent_id="crew")
        cb.install()
        count_after_first = len(ch.get_before_llm_call_hooks())
        cb.install()  # second install
        self.assertEqual(len(ch.get_before_llm_call_hooks()), count_after_first)
        cb.uninstall()

    def test_uninstall_removes_hooks(self):
        import crewai.hooks as ch

        dt, _ = _make_dt()
        cb = DunetraceCrewCallback(dt, agent_id="crew")
        cb.install()
        before = len(ch.get_before_llm_call_hooks())
        cb.uninstall()
        self.assertLess(len(ch.get_before_llm_call_hooks()), before)

    def test_uninstall_without_install_no_crash(self):
        dt, _ = _make_dt()
        cb = DunetraceCrewCallback(dt, agent_id="crew")
        cb.uninstall()  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
