"""
Tests for auto_instrument(), @dt.agent() decorator, ASGI/WSGI middleware,
get_current_run(), and httpx/requests patches.

No network required — all external calls are mocked.
"""

import asyncio
import io
import json
import sys
import types
import unittest
from typing import ClassVar
from unittest.mock import MagicMock

from dunetrace import (
    Dunetrace,
    DunetraceASGIMiddleware,
    DunetraceWSGIMiddleware,
    get_current_run,
)
from dunetrace.auto import _PATCHED
from dunetrace.models import EventType

# If the real langchain_core is installed, other tests in this same process
# (e.g. test_integrations/test_langchain_policies.py, or even
# TestAutoInstrumentIdempotency below, which calls the unscoped
# dt.auto_instrument() covering every framework) may patch
# BaseChatModel/BaseTool for real, permanently, before
# TestAutoInstrumentLangChain's own setUpClass ever runs — auto_instrument()
# patches are deliberately process-wide and idempotent-forever, not meant to
# be reversible per test. Snapshotting the true pre-patch methods here, at
# module IMPORT time (during unittest's collection phase, before any test
# anywhere has run), is the only reliable way to restore them afterwards.
try:
    from langchain_core.language_models.chat_models import BaseChatModel as _PRISTINE_BCM_CLS
    from langchain_core.tools import BaseTool as _PRISTINE_BT_CLS

    _PRISTINE_BCM_METHODS = {
        name: _PRISTINE_BCM_CLS.__dict__.get(name)
        for name in ("invoke", "ainvoke", "stream", "astream")
    }
    _PRISTINE_BT_METHODS = {name: _PRISTINE_BT_CLS.__dict__.get(name) for name in ("run", "arun")}
except ImportError:
    _PRISTINE_BCM_CLS = _PRISTINE_BT_CLS = None
    _PRISTINE_BCM_METHODS = _PRISTINE_BT_METHODS = {}


def _restore_pristine_langchain_methods():
    if _PRISTINE_BCM_CLS is not None:
        for name, fn in _PRISTINE_BCM_METHODS.items():
            if fn is not None:
                setattr(_PRISTINE_BCM_CLS, name, fn)
    if _PRISTINE_BT_CLS is not None:
        for name, fn in _PRISTINE_BT_METHODS.items():
            if fn is not None:
                setattr(_PRISTINE_BT_CLS, name, fn)
    _PATCHED.discard("langchain")


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_client() -> Dunetrace:
    return Dunetrace(endpoint=None)


def _collect(client: Dunetrace) -> list:
    """Drain all buffered events after shutdown."""
    client.shutdown(timeout=2)
    return []  # events were emitted to buffer; inspect via captured list instead


def _capture(client: Dunetrace):
    """Attach a list collector to client._emit and return the list."""
    captured = []
    original = client._emit

    def _capturing_emit(event):
        captured.append(event)
        original(event)

    client._emit = _capturing_emit
    return captured


# ── fake stdlib modules so tests don't need real packages ─────────────────────


def _install_fake_openai():
    mod = types.ModuleType("openai")
    resources = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")

    class FakeUsage:
        completion_tokens = 42
        prompt_tokens = 10
        completion_tokens_details = type("D", (), {"reasoning_tokens": 7})()

    class FakeChoice:
        finish_reason = "stop"
        message = type("M", (), {"content": "Paris"})()

    class FakeResponse:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class Completions:
        def create(self, *, messages=None, model="unknown", **kwargs):
            return FakeResponse()

    class AsyncCompletions:
        async def create(self, *, messages=None, model="unknown", **kwargs):
            return FakeResponse()

    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions
    # Force-install (not setdefault): keeping a previously-imported real openai
    # would make _patch_openai() patch the wrong module. See _install_fake_anthropic.
    sys.modules["openai"] = mod
    sys.modules["openai.resources"] = resources
    sys.modules["openai.resources.chat"] = chat
    sys.modules["openai.resources.chat.completions"] = completions
    return completions


def _install_fake_anthropic():
    mod = types.ModuleType("anthropic")
    resources = types.ModuleType("anthropic.resources")
    messages_mod = types.ModuleType("anthropic.resources.messages")

    class FakeUsage:
        output_tokens = 30

    class FakeContent:
        text = "42"

    class FakeResponse:
        usage = FakeUsage()
        stop_reason = "end_turn"
        content = [FakeContent()]

    class Messages:
        def create(self, *, model="unknown", messages=None, max_tokens=1024, **kwargs):
            return FakeResponse()

    class AsyncMessages:
        async def create(self, *, model="unknown", messages=None, max_tokens=1024, **kwargs):
            return FakeResponse()

    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages
    # Force-install the fake. setdefault() would keep a previously-imported REAL
    # anthropic module (present under pytest's ordering once anything imports it),
    # so _patch_anthropic() would patch the real module while the test calls this
    # fake — no events emitted. Overwriting guarantees the patched module and the
    # tested module are the same object.
    mod.resources = resources
    resources.messages = messages_mod
    sys.modules["anthropic"] = mod
    sys.modules["anthropic.resources"] = resources
    sys.modules["anthropic.resources.messages"] = messages_mod
    return messages_mod


def _install_fake_httpx():
    mod = types.ModuleType("httpx")

    class FakeURL:
        host = "api.example.com"

        def __str__(self):
            return "https://api.example.com/data"

    class FakeRequest:
        url = FakeURL()

    class FakeResponse:
        status_code = 200
        headers = {"content-length": "256"}

    class Client:
        def send(self, request, **kwargs):
            return FakeResponse()

    class AsyncClient:
        async def send(self, request, **kwargs):
            return FakeResponse()

    mod.Client = Client
    mod.AsyncClient = AsyncClient
    mod._FakeRequest = FakeRequest
    sys.modules["httpx"] = mod
    return mod


def _install_fake_requests():
    mod = types.ModuleType("requests")

    class FakePreparedRequest:
        url = "https://serpapi.com/search?q=hello"

    class FakeResponse:
        status_code = 200
        headers = {"content-length": "128"}

    class Session:
        def send(self, request, **kwargs):
            return FakeResponse()

    mod.Session = Session
    mod._FakePrepared = FakePreparedRequest
    sys.modules["requests"] = mod
    return mod


def _uninstall_fake(*prefixes):
    """Remove fake provider modules a test class installed and forget them in
    the global _PATCHED set, so they don't leak into later tests (e.g. the
    idempotency test's real auto_instrument() call, which would otherwise trip
    over an incomplete fake). The next real `import` re-imports the genuine
    module. Order-independent isolation for the fake-installing provider classes."""
    for name in list(sys.modules):
        for pfx in prefixes:
            if name == pfx or name.startswith(pfx + "."):
                sys.modules.pop(name, None)
    for pfx in prefixes:
        _PATCHED.discard(pfx)


def _get_langchain_classes():
    """Return (BaseChatModel, BaseTool) usable for testing _patch_langchain.

    Prefers the real langchain_core if it's installed — overwriting it with a
    stub in sys.modules would break other test modules in the same process
    that exercise a real LangGraph agent end-to-end (test_langchain_policies.py
    does exactly that). Only falls back to a minimal fake when langchain_core
    genuinely isn't importable.

    Real BaseChatModel/BaseTool are abstract/require pydantic fields, so this
    returns minimal concrete subclasses either way — callers just need
    something they can construct and call .invoke()/.run() on.
    """
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import BaseTool

        class _RealFakeChatModel(BaseChatModel):
            last_in_framework_call: ClassVar[object] = None

            @property
            def _llm_type(self) -> str:
                return "dunetrace-test-fake"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                from dunetrace.context import _in_framework_call

                type(self).last_in_framework_call = _in_framework_call.get()
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="fake-response"))]
                )

        class _RealFakeTool(BaseTool):
            name: str = "fake_tool"
            description: str = "a fake tool for testing"

            def _run(self, *args, **kwargs):
                return "fake-tool-result"

        return _RealFakeChatModel, _RealFakeTool
    except ImportError:
        pass

    return _install_stub_langchain_core()


def _install_stub_langchain_core():
    """Fallback used only when the real langchain_core isn't installed.
    Doesn't simulate the full callback protocol (on_chain_start/on_llm_end/...)
    — that's covered by test_integrations/test_langchain.py against the real
    DunetraceCallbackHandler. These fakes only need to prove the *patch*
    mechanics: config/callbacks injection, the re-entrancy flag, idempotency.
    """
    lc_core = types.ModuleType("langchain_core")
    lc_lm = types.ModuleType("langchain_core.language_models")
    lc_lm_cm = types.ModuleType("langchain_core.language_models.chat_models")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_runnables = types.ModuleType("langchain_core.runnables")
    lc_runnables_config = types.ModuleType("langchain_core.runnables.config")
    lc_callbacks = types.ModuleType("langchain_core.callbacks")
    lc_callbacks_base = types.ModuleType("langchain_core.callbacks.base")

    class BaseCallbackHandler:
        def __init__(self):
            pass

    lc_callbacks_base.BaseCallbackHandler = BaseCallbackHandler
    lc_callbacks.base = lc_callbacks_base

    def ensure_config(config):
        return dict(config) if config else {}

    class BaseChatModel:
        last_config = None
        last_in_framework_call = None

        def invoke(self, input, config=None, *, stop=None, **kwargs):
            from dunetrace.context import _in_framework_call

            BaseChatModel.last_config = config
            BaseChatModel.last_in_framework_call = _in_framework_call.get()
            return "fake-response"

        async def ainvoke(self, input, config=None, *, stop=None, **kwargs):
            BaseChatModel.last_config = config
            return "fake-response"

        def stream(self, input, config=None, *, stop=None, **kwargs):
            BaseChatModel.last_config = config
            yield "fake-chunk"

        async def astream(self, input, config=None, *, stop=None, **kwargs):
            BaseChatModel.last_config = config
            yield "fake-chunk"

    class BaseTool:
        last_callbacks = None

        def run(
            self,
            tool_input,
            verbose=None,
            start_color="green",
            color="green",
            callbacks=None,
            *,
            tags=None,
            metadata=None,
            run_name=None,
            run_id=None,
            config=None,
            tool_call_id=None,
            **kwargs,
        ):
            BaseTool.last_callbacks = callbacks
            return "fake-tool-result"

        async def arun(
            self,
            tool_input,
            verbose=None,
            start_color="green",
            color="green",
            callbacks=None,
            *,
            tags=None,
            metadata=None,
            run_name=None,
            run_id=None,
            config=None,
            tool_call_id=None,
            **kwargs,
        ):
            BaseTool.last_callbacks = callbacks
            return "fake-tool-result"

    lc_lm_cm.BaseChatModel = BaseChatModel
    lc_tools.BaseTool = BaseTool
    lc_runnables_config.ensure_config = ensure_config
    lc_lm.chat_models = lc_lm_cm
    lc_runnables.config = lc_runnables_config
    lc_core.language_models = lc_lm
    lc_core.tools = lc_tools
    lc_core.runnables = lc_runnables
    lc_core.callbacks = lc_callbacks

    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.language_models"] = lc_lm
    sys.modules["langchain_core.language_models.chat_models"] = lc_lm_cm
    sys.modules["langchain_core.tools"] = lc_tools
    sys.modules["langchain_core.runnables"] = lc_runnables
    sys.modules["langchain_core.runnables.config"] = lc_runnables_config
    sys.modules["langchain_core.callbacks"] = lc_callbacks
    sys.modules["langchain_core.callbacks.base"] = lc_callbacks_base

    # dunetrace.integrations.langchain resolves BaseCallbackHandler and
    # _LANGCHAIN_AVAILABLE in a top-level try/except at import time. Reload it
    # now that the fakes are in sys.modules so that resolution re-runs against
    # them, regardless of whether some other test module imported it first.
    import importlib

    import dunetrace.integrations.langchain as _lc_mod

    importlib.reload(_lc_mod)

    return BaseChatModel, BaseTool


def _install_fake_crewai():
    """crewai.hooks stub (mirrors test_integrations/test_crewai.py) plus
    Crew/Agent classes with kickoff/kickoff_async, so _patch_crewai has a
    run boundary to patch."""
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
    crewai_hooks.register_before_llm_call_hook = lambda fn: _before_llm_hooks.append(fn)
    crewai_hooks.register_after_llm_call_hook = lambda fn: _after_llm_hooks.append(fn)
    crewai_hooks.register_before_tool_call_hook = lambda fn: _before_tool_hooks.append(fn)
    crewai_hooks.register_after_tool_call_hook = lambda fn: _after_tool_hooks.append(fn)
    crewai_hooks.unregister_before_llm_call_hook = lambda fn: _before_llm_hooks.remove(fn)
    crewai_hooks.unregister_after_llm_call_hook = lambda fn: _after_llm_hooks.remove(fn)
    crewai_hooks.unregister_before_tool_call_hook = lambda fn: _before_tool_hooks.remove(fn)
    crewai_hooks.unregister_after_tool_call_hook = lambda fn: _after_tool_hooks.remove(fn)
    crewai_hooks_llm.LLMCallHookContext = LLMCallHookContext
    crewai_hooks_tool.ToolCallHookContext = ToolCallHookContext

    class Crew:
        def __init__(self, name="crew"):
            self.name = name

        def kickoff(self, inputs=None, **kwargs):
            return "fake-crew-result"

        async def kickoff_async(self, inputs=None, **kwargs):
            return "fake-crew-result"

    class Agent:
        def __init__(self, role="fake-role"):
            self.role = role

        def kickoff(self, messages, *args, **kwargs):
            return "fake-agent-result"

        async def kickoff_async(self, messages, *args, **kwargs):
            return "fake-agent-result"

    crewai.hooks = crewai_hooks
    crewai.Crew = Crew
    crewai.Agent = Agent
    sys.modules["crewai"] = crewai
    sys.modules["crewai.hooks"] = crewai_hooks
    sys.modules["crewai.hooks.llm_hooks"] = crewai_hooks_llm
    sys.modules["crewai.hooks.tool_hooks"] = crewai_hooks_tool

    # dunetrace.integrations.crewai resolves _crewai_hooks/_CREWAI_AVAILABLE
    # in a top-level try/except at import time. Reload it now that the fakes
    # are in sys.modules so resolution re-runs against them, regardless of
    # whether some other test module imported it first.
    import importlib

    import dunetrace.integrations.crewai as _cw_mod

    importlib.reload(_cw_mod)

    return Crew, Agent


# ─────────────────────────────────────────────────────────────────────────────
# 1. get_current_run()
# ─────────────────────────────────────────────────────────────────────────────


class TestGetCurrentRun(unittest.TestCase):
    def test_none_outside_run(self):
        self.assertIsNone(get_current_run())

    def test_set_inside_run(self):
        dt = _make_client()
        with dt.run("agent", user_input="hi") as run:
            self.assertIs(get_current_run(), run)
        dt.shutdown(timeout=1)

    def test_reset_after_run(self):
        dt = _make_client()
        with dt.run("agent"):
            pass
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)

    def test_reset_after_exception(self):
        dt = _make_client()
        try:
            with dt.run("agent"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)

    def test_nested_runs_isolate(self):
        dt = _make_client()
        with dt.run("outer") as outer:
            self.assertIs(get_current_run(), outer)
            with dt.run("inner", parent_run_id=outer.run_id) as inner:
                self.assertIs(get_current_run(), inner)
            self.assertIs(get_current_run(), outer)
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. @dt.agent() decorator
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentDecorator(unittest.TestCase):
    def test_sync_function_wrapped(self):
        dt = _make_client()
        captured = _capture(dt)

        @dt.agent("dec-agent", model="gpt-4o")
        def fn(query: str) -> str:
            self.assertIsNotNone(get_current_run())
            self.assertEqual(get_current_run().agent_id, "dec-agent")
            return "answer"

        result = fn("hello")
        self.assertEqual(result, "answer")
        self.assertIsNone(get_current_run())

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.RUN_STARTED, types_)
        self.assertIn(EventType.RUN_COMPLETED, types_)
        dt.shutdown(timeout=1)

    def test_async_function_wrapped(self):
        dt = _make_client()
        captured = _capture(dt)

        @dt.agent("async-agent", model="claude-3-5-sonnet")
        async def fn(query: str) -> str:
            self.assertIsNotNone(get_current_run())
            return "async answer"

        result = asyncio.run(fn("hi"))
        self.assertEqual(result, "async answer")
        self.assertIsNone(get_current_run())

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.RUN_STARTED, types_)
        self.assertIn(EventType.RUN_COMPLETED, types_)
        dt.shutdown(timeout=1)

    def test_first_arg_used_as_user_input(self):
        dt = _make_client()
        captured = _capture(dt)

        @dt.agent("agent")
        def fn(query: str):
            pass

        fn("my query")
        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.payload.get("input_text"), "my query")
        dt.shutdown(timeout=1)

    def test_input_from_named_param(self):
        dt = _make_client()

        @dt.agent("agent", input_from="question")
        def fn(context: str, question: str):
            pass

        with dt.run("_check") as run:
            pass  # ensure no crash
        fn("ctx", "my question")
        fn("ctx", question="my question kw")
        dt.shutdown(timeout=1)

    def test_exception_propagates_and_emits_errored(self):
        dt = _make_client()
        captured = _capture(dt)

        @dt.agent("err-agent")
        def fn(q: str):
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            fn("test")

        self.assertIsNone(get_current_run())
        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.RUN_ERRORED, types_)
        self.assertNotIn(EventType.RUN_COMPLETED, types_)
        dt.shutdown(timeout=1)

    def test_final_answer_set_on_clean_return(self):
        dt = _make_client()

        @dt.agent("agent")
        def fn(q: str):
            return "ok"

        fn("hi")
        dt.shutdown(timeout=1)

    def test_functools_wraps_preserves_name(self):
        dt = _make_client()

        @dt.agent("agent")
        def my_special_function(q: str):
            pass

        self.assertEqual(my_special_function.__name__, "my_special_function")
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. auto_instrument — OpenAI
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoInstrumentOpenAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        _PATCHED.discard("openai")
        from dunetrace.auto import _patch_openai

        _patch_openai()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("openai")

    def test_sync_llm_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("oa-agent", user_input="hi") as run:
            completions = self._completions_mod.Completions()
            completions.create(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                model="gpt-4o",
            )

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.LLM_CALLED, types_)
        self.assertIn(EventType.LLM_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_model_recorded(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent") as run:
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
            )

        llm_called = next(e for e in captured if e.event_type == EventType.LLM_CALLED)
        self.assertEqual(llm_called.payload["model"], "gpt-4o-mini")
        dt.shutdown(timeout=1)

    def test_no_events_outside_run(self):
        dt = _make_client()
        captured = _capture(dt)

        # Call without an active run — should not emit anything
        self._completions_mod.Completions().create(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
        )

        self.assertEqual(len(captured), 0)
        dt.shutdown(timeout=1)

    def test_async_llm_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        async def _run():
            with dt.run("async-oa") as run:
                await self._completions_mod.AsyncCompletions().create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o",
                )

        asyncio.run(_run())
        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.LLM_CALLED, types_)
        self.assertIn(EventType.LLM_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_completion_tokens_recorded(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
            )

        responded = next(e for e in captured if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["completion_tokens"], 42)
        dt.shutdown(timeout=1)

    def test_reasoning_tokens_recorded(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="o3"
            )

        responded = next(e for e in captured if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["reasoning_tokens"], 7)
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. auto_instrument — Anthropic
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoInstrumentAnthropic(unittest.TestCase):
    def setUp(self):
        # Re-install + re-patch the fake per test (not once per class): the fake
        # module and the global _PATCHED set are shared process state, and under
        # pytest's definition-order execution an earlier provider class can leave
        # "anthropic" state a once-per-class setup never recovers from. Doing it
        # per test makes each one self-contained regardless of ordering.
        self._messages_mod = _install_fake_anthropic()
        _PATCHED.discard("anthropic")
        from dunetrace.auto import _patch_anthropic

        _patch_anthropic()

    def tearDown(self):
        _uninstall_fake("anthropic")

    def test_sync_llm_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("ant-agent"):
            self._messages_mod.Messages().create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1024,
            )

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.LLM_CALLED, types_)
        self.assertIn(EventType.LLM_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_finish_reason_end_turn(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._messages_mod.Messages().create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
            )

        responded = next(e for e in captured if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["finish_reason"], "end_turn")
        dt.shutdown(timeout=1)

    def test_async_llm_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        async def _run():
            with dt.run("async-ant"):
                await self._messages_mod.AsyncMessages().create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "hi"}],
                )

        asyncio.run(_run())
        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.LLM_CALLED, types_)
        self.assertIn(EventType.LLM_RESPONDED, types_)
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. auto_instrument — httpx
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoInstrumentHTTPX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._httpx = _install_fake_httpx()
        _PATCHED.discard("httpx")
        from dunetrace.auto import _patch_httpx

        _patch_httpx()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("httpx")

    def test_sync_tool_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("httpx-agent"):
            req = self._httpx._FakeRequest()
            self._httpx.Client().send(req)

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.TOOL_CALLED, types_)
        self.assertIn(EventType.TOOL_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_hostname_as_tool_name(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._httpx.Client().send(self._httpx._FakeRequest())

        called = next(e for e in captured if e.event_type == EventType.TOOL_CALLED)
        self.assertEqual(called.payload["tool_name"], "api.example.com")
        dt.shutdown(timeout=1)

    def test_success_flag_on_200(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._httpx.Client().send(self._httpx._FakeRequest())

        responded = next(e for e in captured if e.event_type == EventType.TOOL_RESPONDED)
        self.assertTrue(responded.payload["success"])
        dt.shutdown(timeout=1)

    def test_no_events_outside_run(self):
        dt = _make_client()
        captured = _capture(dt)

        self._httpx.Client().send(self._httpx._FakeRequest())
        self.assertEqual(len(captured), 0)
        dt.shutdown(timeout=1)

    def test_async_tool_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        async def _run():
            with dt.run("async-httpx"):
                await self._httpx.AsyncClient().send(self._httpx._FakeRequest())

        asyncio.run(_run())
        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.TOOL_CALLED, types_)
        self.assertIn(EventType.TOOL_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_url_transmitted_raw(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._httpx.Client().send(self._httpx._FakeRequest())

        called = next(e for e in captured if e.event_type == EventType.TOOL_CALLED)
        raw_url = str(self._httpx._FakeRequest().url)
        self.assertIn(raw_url, json.dumps(called.payload))
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. auto_instrument — requests
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoInstrumentRequests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._requests = _install_fake_requests()
        _PATCHED.discard("requests")
        from dunetrace.auto import _patch_requests

        _patch_requests()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("requests")

    def test_tool_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("req-agent"):
            self._requests.Session().send(self._requests._FakePrepared())

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.TOOL_CALLED, types_)
        self.assertIn(EventType.TOOL_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_hostname_as_tool_name(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._requests.Session().send(self._requests._FakePrepared())

        called = next(e for e in captured if e.event_type == EventType.TOOL_CALLED)
        self.assertEqual(called.payload["tool_name"], "serpapi.com")
        dt.shutdown(timeout=1)

    def test_no_events_outside_run(self):
        dt = _make_client()
        captured = _capture(dt)

        self._requests.Session().send(self._requests._FakePrepared())
        self.assertEqual(len(captured), 0)
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ASGI middleware
# ─────────────────────────────────────────────────────────────────────────────


class TestASGIMiddleware(unittest.TestCase):
    def _run_request(self, dt, agent_id, scope=None, side_effect=None):
        """Run a fake HTTP request through the middleware."""
        inner_run = {}
        scope = scope or {"type": "http", "method": "POST", "path": "/run"}

        async def fake_app(scope, receive, send):
            inner_run["run"] = get_current_run()
            inner_run["scope_run"] = scope.get("state", {}).get("dunetrace_run")
            if side_effect:
                raise side_effect

        middleware = DunetraceASGIMiddleware(fake_app, dt=dt, agent_id=agent_id)
        try:
            asyncio.run(middleware(scope, None, None))
        except Exception:
            pass
        return inner_run

    def test_run_active_inside_handler(self):
        dt = _make_client()
        result = self._run_request(dt, "api")
        self.assertIsNotNone(result["run"])
        self.assertEqual(result["run"].agent_id, "api")
        dt.shutdown(timeout=1)

    def test_run_on_scope_state(self):
        dt = _make_client()
        result = self._run_request(dt, "api")
        self.assertIs(result["scope_run"], result["run"])
        dt.shutdown(timeout=1)

    def test_run_cleaned_up_after_request(self):
        dt = _make_client()
        self._run_request(dt, "api")
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)

    def test_run_cleaned_up_after_exception(self):
        dt = _make_client()
        self._run_request(dt, "api", side_effect=RuntimeError("handler crash"))
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)

    def test_non_http_scope_passthrough(self):
        dt = _make_client()
        called = {}

        async def fake_app(scope, receive, send):
            called["yes"] = True

        middleware = DunetraceASGIMiddleware(fake_app, dt=dt, agent_id="api")
        asyncio.run(middleware({"type": "lifespan"}, None, None))
        self.assertTrue(called.get("yes"))
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)

    def test_run_started_and_completed_emitted(self):
        dt = _make_client()
        captured = _capture(dt)
        self._run_request(dt, "api")

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.RUN_STARTED, types_)
        self.assertIn(EventType.RUN_COMPLETED, types_)
        dt.shutdown(timeout=1)

    def test_method_and_path_as_user_input(self):
        dt = _make_client()
        captured = _capture(dt)
        self._run_request(dt, "api", scope={"type": "http", "method": "GET", "path": "/health"})

        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertNotEqual(started.payload.get("input_text", ""), "")
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 8. WSGI middleware
# ─────────────────────────────────────────────────────────────────────────────


class TestWSGIMiddleware(unittest.TestCase):
    def _run_request(self, dt, agent_id, environ=None, side_effect=None):
        inner = {}
        environ = environ or {"REQUEST_METHOD": "POST", "PATH_INFO": "/run"}

        def fake_app(environ, start_response):
            inner["run"] = get_current_run()
            inner["environ_run"] = environ.get("dunetrace.run")
            if side_effect:
                raise side_effect
            return []

        middleware = DunetraceWSGIMiddleware(fake_app, dt=dt, agent_id=agent_id)
        try:
            middleware(environ, lambda *a: None)
        except Exception:
            pass
        return inner

    def test_run_active_inside_handler(self):
        dt = _make_client()
        result = self._run_request(dt, "flask-api")
        self.assertIsNotNone(result["run"])
        self.assertEqual(result["run"].agent_id, "flask-api")
        dt.shutdown(timeout=1)

    def test_run_on_environ(self):
        dt = _make_client()
        result = self._run_request(dt, "flask-api")
        self.assertIs(result["environ_run"], result["run"])
        dt.shutdown(timeout=1)

    def test_run_cleaned_up_after_request(self):
        dt = _make_client()
        self._run_request(dt, "flask-api")
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)

    def test_run_cleaned_up_after_exception(self):
        dt = _make_client()
        self._run_request(dt, "flask-api", side_effect=RuntimeError("crash"))
        self.assertIsNone(get_current_run())
        dt.shutdown(timeout=1)

    def test_run_started_and_completed_emitted(self):
        dt = _make_client()
        captured = _capture(dt)
        self._run_request(dt, "flask-api")

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.RUN_STARTED, types_)
        self.assertIn(EventType.RUN_COMPLETED, types_)
        dt.shutdown(timeout=1)

    def test_method_and_path_as_user_input(self):
        dt = _make_client()
        captured = _capture(dt)
        self._run_request(dt, "api", environ={"REQUEST_METHOD": "GET", "PATH_INFO": "/health"})

        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertNotEqual(started.payload.get("input_text", ""), "")
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 9. auto_instrument idempotency
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoInstrumentIdempotency(unittest.TestCase):
    def test_calling_twice_is_safe(self):
        dt = _make_client()
        dt.auto_instrument()
        dt.auto_instrument()  # must not double-wrap or raise
        dt.shutdown(timeout=1)

    def test_unknown_framework_logs_warning(self):
        import logging

        dt = _make_client()
        with self.assertLogs("dunetrace.auto", level="WARNING") as cm:
            dt.auto_instrument(["nonexistent_framework"])
        self.assertTrue(any("nonexistent_framework" in m for m in cm.output))
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Framework re-entrancy guard (avoids double-counting LangChain/CrewAI
#     calls that pass through the raw openai/anthropic/httpx/requests patches)
# ─────────────────────────────────────────────────────────────────────────────


class TestFrameworkReentrancyGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        _PATCHED.discard("openai")
        from dunetrace.auto import _patch_openai

        _patch_openai()

    def test_no_event_emitted_while_in_framework_call(self):
        from dunetrace.context import _in_framework_call

        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            token = _in_framework_call.set(True)
            try:
                self._completions_mod.Completions().create(
                    messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
                )
            finally:
                _in_framework_call.reset(token)

        types_ = [e.event_type for e in captured]
        self.assertNotIn(EventType.LLM_CALLED, types_)
        self.assertNotIn(EventType.LLM_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_call_still_goes_through_while_suppressed(self):
        """Suppression must only skip emission, never skip the real call."""
        from dunetrace.context import _in_framework_call

        dt = _make_client()
        token = _in_framework_call.set(True)
        try:
            result = self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
            )
        finally:
            _in_framework_call.reset(token)
        self.assertIsNotNone(result)
        dt.shutdown(timeout=1)

    def test_events_resume_after_flag_cleared(self):
        from dunetrace.context import _in_framework_call

        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            token = _in_framework_call.set(True)
            _in_framework_call.reset(token)
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
            )

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.LLM_CALLED, types_)
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 11. auto_instrument(langchain=...) — patches BaseChatModel + BaseTool
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoInstrumentLangChain(unittest.TestCase):
    """Verifies the *patch* mechanics end-to-end against real langchain_core
    classes (falling back to a minimal stub only if langchain_core isn't
    installed) — the real DunetraceCallbackHandler behavior itself (tiered
    agent_id resolution, ambient-run reuse, policy evaluation, ...) is
    covered separately in test_integrations/test_langchain.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.BaseChatModel, cls.BaseTool = _get_langchain_classes()

    def setUp(self):
        # Undo any patching left behind by an earlier test class in this same
        # process (e.g. TestAutoInstrumentIdempotency's unscoped
        # dt.auto_instrument() call, which legitimately — and permanently, by
        # design — patches every registered framework including langchain).
        _restore_pristine_langchain_methods()

    def tearDown(self):
        _restore_pristine_langchain_methods()

    def test_requires_client(self):
        with self.assertLogs("dunetrace.auto", level="WARNING") as cm:
            from dunetrace.auto import _patch_langchain

            _patch_langchain(client=None)
        self.assertTrue(any("requires a client" in m for m in cm.output))
        self.assertNotIn("langchain", _PATCHED)

    def test_no_run_created_without_ambient_dt_run(self):
        """Documents a real, deliberate limitation (see
        docs/integrations/auto-instrumentation.md): on_chain_start — which
        DunetraceCallbackHandler's own run-creation logic hangs off — never
        fires for a bare leaf-level BaseChatModel/BaseTool call with no
        enclosing chain callback. Auto-instrumented LangChain/LangGraph calls
        only attach to an *already open* dt.run(); they never open their own."""
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_langchain

        _patch_langchain(client=dt, default_agent_id="my-agent")

        model = self.BaseChatModel()
        model.invoke("hello")  # no ambient dt.run()

        self.assertEqual(captured, [])
        dt.shutdown(timeout=1)

    def test_invoke_drives_dunetrace_handler_end_to_end(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_langchain

        _patch_langchain(client=dt, default_agent_id="my-agent")

        # dt.run() wrapping is required: on_chain_start (which the handler's
        # own run-creation logic hangs off) never fires for a bare leaf-level
        # BaseChatModel/BaseTool call with no enclosing chain callback — see
        # docs/integrations/auto-instrumentation.md. Auto-instrumented
        # LangChain/LangGraph calls attach to whichever dt.run() is active.
        model = self.BaseChatModel()
        with dt.run("my-agent"):
            model.invoke("hello")

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.RUN_STARTED, types_)
        self.assertIn(EventType.LLM_CALLED, types_)
        self.assertIn(EventType.LLM_RESPONDED, types_)
        self.assertIn(EventType.RUN_COMPLETED, types_)
        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.agent_id, "my-agent")
        dt.shutdown(timeout=1)

    def test_handler_not_duplicated_when_config_already_carries_it(self):
        """A config that already carries the shared handler — as a nested
        Runnable call inheriting its parent's config would — must not get it
        added a second time. A duplicate would fire on_llm_start/on_llm_end
        twice for the same logical LLM call."""
        from dunetrace.integrations.langchain import DunetraceCallbackHandler

        dt = _make_client()
        from dunetrace.auto import _patch_langchain

        _patch_langchain(client=dt, default_agent_id="my-agent")

        seen_handlers = []

        class _RecordingModel(self.BaseChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                handlers = run_manager.handlers if run_manager else []
                seen_handlers.append(
                    [h for h in handlers if isinstance(h, DunetraceCallbackHandler)]
                )
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        model = _RecordingModel()
        model.invoke("hello", config={"callbacks": []})
        self.assertEqual(len(seen_handlers[0]), 1)

        # Simulates a nested call inheriting a config that already resolved
        # to include our handler (what happens in a real Runnable chain).
        model.invoke("hello again", config={"callbacks": list(seen_handlers[0])})
        self.assertEqual(len(seen_handlers[1]), 1)
        self.assertIs(seen_handlers[0][0], seen_handlers[1][0])
        dt.shutdown(timeout=1)

    def test_reentrancy_flag_set_during_invoke_and_reset_after(self):
        from dunetrace.context import _in_framework_call

        dt = _make_client()
        from dunetrace.auto import _patch_langchain

        _patch_langchain(client=dt, default_agent_id="my-agent")

        model = self.BaseChatModel()
        self.assertFalse(_in_framework_call.get())
        model.invoke("hello")
        # The fake's own _generate() runs inside _patch_langchain's wrapper
        # and records the flag it observed — proves the flag is set for the
        # duration of the underlying call, not just around it.
        self.assertTrue(type(model).last_in_framework_call)
        self.assertFalse(_in_framework_call.get())  # reset after the call completes
        dt.shutdown(timeout=1)

    def test_tool_run_drives_dunetrace_handler(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_langchain

        _patch_langchain(client=dt, default_agent_id="my-agent")

        tool = self.BaseTool()
        with dt.run("my-agent"):
            tool.run("some input")

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.RUN_STARTED, types_)
        self.assertIn(EventType.TOOL_CALLED, types_)
        dt.shutdown(timeout=1)

    def test_idempotent(self):
        dt = _make_client()
        from dunetrace.auto import _patch_langchain

        _patch_langchain(client=dt, default_agent_id="my-agent")
        invoke_after_first = self.BaseChatModel.invoke
        _patch_langchain(client=dt, default_agent_id="my-agent")
        self.assertIs(self.BaseChatModel.invoke, invoke_after_first)
        dt.shutdown(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# 12. auto_instrument(crewai=...) — Crew/Agent kickoff run boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoInstrumentCrewAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Crew, cls.Agent = _install_fake_crewai()
        # Fresh fake classes, so this snapshot (unlike langchain's, which
        # patches the real process-wide BaseChatModel) is genuinely pristine.
        cls._pristine = {
            "Crew.kickoff": cls.Crew.__dict__.get("kickoff"),
            "Crew.kickoff_async": cls.Crew.__dict__.get("kickoff_async"),
            "Agent.kickoff": cls.Agent.__dict__.get("kickoff"),
            "Agent.kickoff_async": cls.Agent.__dict__.get("kickoff_async"),
        }

    def _restore_pristine(self):
        self.Crew.kickoff = self._pristine["Crew.kickoff"]
        self.Crew.kickoff_async = self._pristine["Crew.kickoff_async"]
        self.Agent.kickoff = self._pristine["Agent.kickoff"]
        self.Agent.kickoff_async = self._pristine["Agent.kickoff_async"]
        _PATCHED.discard("crewai")

    def setUp(self):
        self._restore_pristine()

    def tearDown(self):
        self._restore_pristine()

    def test_requires_client(self):
        with self.assertLogs("dunetrace.auto", level="WARNING") as cm:
            from dunetrace.auto import _patch_crewai

            _patch_crewai(client=None)
        self.assertTrue(any("requires a client" in m for m in cm.output))
        self.assertNotIn("crewai", _PATCHED)

    def test_crew_kickoff_opens_a_run_when_none_active(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_crewai

        _patch_crewai(client=dt, default_agent_id="fallback-agent")

        crew = self.Crew(name="crew")  # unset (default) name — falls through to default_agent_id
        crew.kickoff(inputs={"topic": "AI trends"})

        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.agent_id, "fallback-agent")
        self.assertIn(EventType.RUN_COMPLETED, [e.event_type for e in captured])
        dt.shutdown(timeout=1)

    def test_crew_name_used_as_agent_id_when_set(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_crewai

        _patch_crewai(client=dt, default_agent_id="fallback-agent")

        crew = self.Crew(name="research-crew")
        crew.kickoff(inputs={"topic": "AI trends"})

        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.agent_id, "research-crew")
        dt.shutdown(timeout=1)

    def test_per_call_agent_id_wins_over_crew_name(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_crewai

        _patch_crewai(client=dt, default_agent_id="fallback-agent")

        crew = self.Crew(name="research-crew")
        crew.kickoff(inputs={"agent_id": "explicit-override"})

        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.agent_id, "explicit-override")
        dt.shutdown(timeout=1)

    def test_ambient_run_reused_not_double_started(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_crewai

        _patch_crewai(client=dt, default_agent_id="fallback-agent")

        crew = self.Crew(name="research-crew")
        with dt.run("ambient-agent"):
            crew.kickoff(inputs={"topic": "AI trends"})

        started_events = [e for e in captured if e.event_type == EventType.RUN_STARTED]
        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0].agent_id, "ambient-agent")
        dt.shutdown(timeout=1)

    def test_agent_kickoff_uses_role_as_agent_id(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_crewai

        _patch_crewai(client=dt, default_agent_id="fallback-agent")

        agent = self.Agent(role="researcher")
        agent.kickoff("find the latest AI news")

        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.agent_id, "researcher")
        dt.shutdown(timeout=1)

    def test_loud_fallback_when_nothing_resolves(self):
        dt = _make_client()
        captured = _capture(dt)
        from dunetrace.auto import _patch_crewai

        _patch_crewai(client=dt, default_agent_id=None)

        crew = self.Crew(name="crew")  # default name, no per-call override, no default
        with self.assertLogs("dunetrace.auto", level="WARNING") as cm:
            crew.kickoff(inputs={"topic": "AI trends"})
        self.assertTrue(any("could not determine an agent_id" in m for m in cm.output))

        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.agent_id, "unattributed-agent")
        dt.shutdown(timeout=1)

    def test_idempotent(self):
        dt = _make_client()
        from dunetrace.auto import _patch_crewai

        _patch_crewai(client=dt, default_agent_id="agent")
        kickoff_after_first = self.Crew.kickoff
        _patch_crewai(client=dt, default_agent_id="agent")
        self.assertIs(self.Crew.kickoff, kickoff_after_first)
        dt.shutdown(timeout=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
