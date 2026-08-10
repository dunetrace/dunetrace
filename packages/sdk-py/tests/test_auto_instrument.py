"""
Tests for auto_instrument(), @dt.agent() decorator, ASGI/WSGI middleware,
get_current_run(), and httpx/requests patches.

No network required — all external calls are mocked.
"""

import asyncio
import io
import json
import sys
import time
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
from dunetrace.policies import PolicyViolation

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


# ── fake streams ──────────────────────────────────────────────────────────────
#
# Every provider stream is both an iterator and a context manager, and the
# proxy has to work through either. These two mirror that shape. Verified
# against the real classes: openai.Stream, anthropic MessageStreamManager /
# MessageStream, and mistralai EventStream / EventStreamAsync.


class _FakeSyncStream:
    """A real iterator, like the classes it stands in for: openai.Stream
    (_streaming.py) and mistralai's EventStream both define __iter__ returning
    self plus __next__. Modelling them as mere iterables previously hid the fact
    that the proxy was not an iterator either."""

    def __init__(self, chunks):
        self._chunks = iter(list(chunks))
        self.entered = False
        self.exited = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = iter(list(chunks))
        self.exited = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False


class _FakeStreamManager:
    """Anthropic's MessageStreamManager supports only the context-manager
    protocol, and __enter__ hands back a different object (a MessageStream).
    The proxy has to rebind onto it, which this exercises."""

    def __init__(self, events, final_message=None):
        self._stream = _FakeSyncStream(events)
        if final_message is not None:
            self._stream.get_final_message = lambda: final_message
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self._stream

    def __exit__(self, *exc):
        return False


class _FakeAsyncStreamManager:
    """Async twin of _FakeStreamManager.

    get_final_message is an *async* function here, matching the real
    anthropic.lib.streaming.AsyncMessageStream (verified against 0.116.0, where
    MessageStream.get_final_message is `def` and AsyncMessageStream's is
    `async def`). Stubbing it as a plain lambda is what previously let an
    un-awaited-coroutine bug pass this suite.
    """

    def __init__(self, events, final_message=None):
        self._stream = _FakeAsyncStream(events)
        if final_message is not None:

            async def _get_final_message():
                return final_message

            self._stream.get_final_message = _get_final_message

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *exc):
        return False


def _mistral_stream_chunks():
    """Shape confirmed against a live La Plateforme stream: text arrives on
    choices[0].delta.content and usage lands on the final chunk only."""

    def event(text=None, finish=None, usage=None):
        choice = types.SimpleNamespace(
            delta=types.SimpleNamespace(content=text), finish_reason=finish
        )
        return types.SimpleNamespace(data=types.SimpleNamespace(choices=[choice], usage=usage))

    final_usage = types.SimpleNamespace(prompt_tokens=20, completion_tokens=8, total_tokens=28)
    return [event("1, "), event("2, "), event("3", finish="stop", usage=final_usage)]


def _openai_stream_chunks(include_usage=False):
    def chunk(text=None, finish=None, usage=None, choices=None):
        if choices is None:
            choices = [
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=text), finish_reason=finish
                )
            ]
        return types.SimpleNamespace(choices=choices, usage=usage)

    out = [chunk("Hello"), chunk(" world"), chunk(None, finish="stop")]
    if include_usage:
        # The usage-only chunk include_usage appends carries an empty choices
        # list, which is exactly why we don't inject the option ourselves.
        out.append(
            chunk(usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3), choices=[])
        )
    return out


def _anthropic_stream_events():
    return [
        types.SimpleNamespace(
            type="message_start",
            message=types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=12)),
        ),
        types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(text="4")),
        types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(text="2")),
        types.SimpleNamespace(
            type="message_delta",
            delta=types.SimpleNamespace(stop_reason="end_turn"),
            usage=types.SimpleNamespace(output_tokens=30),
        ),
    ]


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
            if kwargs.get("stream"):
                return _FakeSyncStream(_openai_stream_chunks(bool(kwargs.get("stream_options"))))
            return FakeResponse()

    class AsyncCompletions:
        async def create(self, *, messages=None, model="unknown", **kwargs):
            if kwargs.get("stream"):
                return _FakeAsyncStream(_openai_stream_chunks(bool(kwargs.get("stream_options"))))
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
        input_tokens = 12

    class FakeContent:
        text = "42"

    class FakeResponse:
        usage = FakeUsage()
        stop_reason = "end_turn"
        content = [FakeContent()]

    class Messages:
        def create(self, *, model="unknown", messages=None, max_tokens=1024, **kwargs):
            if kwargs.get("stream"):
                return _FakeSyncStream(_anthropic_stream_events())
            return FakeResponse()

        def stream(self, *, model="unknown", messages=None, max_tokens=1024, **kwargs):
            return _FakeStreamManager(_anthropic_stream_events(), FakeResponse())

    class AsyncMessages:
        async def create(self, *, model="unknown", messages=None, max_tokens=1024, **kwargs):
            if kwargs.get("stream"):
                return _FakeAsyncStream(_anthropic_stream_events())
            return FakeResponse()

        def stream(self, *, model="unknown", messages=None, max_tokens=1024, **kwargs):
            # Not a coroutine in the real SDK either — it returns an async
            # context manager directly.
            return _FakeAsyncStreamManager(_anthropic_stream_events(), FakeResponse())

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


def _install_fake_mistral():
    """Fake the mistralai v2 module tree.

    v2 dropped the top-level __init__.py and moved everything under
    mistralai.client, so the fake mirrors that layout. _patch_mistral() imports
    mistralai.client.chat, and patching a module the test doesn't call would
    emit nothing, so all three levels are force-installed for the same reason
    _install_fake_anthropic documents.
    """
    mod = types.ModuleType("mistralai")
    client_mod = types.ModuleType("mistralai.client")
    chat_mod = types.ModuleType("mistralai.client.chat")

    class FakeUsage:
        prompt_tokens = 11
        completion_tokens = 22
        total_tokens = 33

    class FakeMessage:
        content = "Paris"

    class FakeChoice:
        finish_reason = "stop"
        message = FakeMessage()

    class FakeResponse:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class Chat:
        def complete(self, *, model="unknown", messages=None, **kwargs):
            if chat_mod._raise is not None:
                raise chat_mod._raise
            return chat_mod._response()

        async def complete_async(self, *, model="unknown", messages=None, **kwargs):
            if chat_mod._raise is not None:
                raise chat_mod._raise
            return chat_mod._response()

        def stream(self, *, model="unknown", messages=None, **kwargs):
            if chat_mod._raise is not None:
                raise chat_mod._raise
            return _FakeSyncStream(_mistral_stream_chunks())

        async def stream_async(self, *, model="unknown", messages=None, **kwargs):
            # Coroutine that resolves to the stream, matching the real SDK.
            return _FakeAsyncStream(_mistral_stream_chunks())

        def parse(self, *, model="unknown", messages=None, **kwargs):
            # Mirrors the real SDK, where parse() delegates to complete().
            # Keeping that shape here is what makes the double-count test real.
            return self.complete(model=model, messages=messages, **kwargs)

    class Embeddings:
        def create(self, *, model="unknown", inputs=None, **kwargs):
            # An embedding response has no choices, only usage.
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(prompt_tokens=17, completion_tokens=0)
            )

        async def create_async(self, *, model="unknown", inputs=None, **kwargs):
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(prompt_tokens=17, completion_tokens=0)
            )

    class Fim:
        def complete(self, *, model="unknown", prompt=None, **kwargs):
            choice = types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(content="return 42"),
            )
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=4),
                choices=[choice],
            )

        async def complete_async(self, *, model="unknown", prompt=None, **kwargs):
            return self.complete(model=model, prompt=prompt, **kwargs)

        def stream(self, *, model="unknown", prompt=None, **kwargs):
            return _FakeSyncStream(_mistral_stream_chunks())

        async def stream_async(self, *, model="unknown", prompt=None, **kwargs):
            return _FakeAsyncStream(_mistral_stream_chunks())

    chat_mod.Chat = Chat
    # Tests swap these to drive the error path and the alternate content shapes
    # without standing up a second fake module tree.
    chat_mod._response = FakeResponse
    chat_mod._default_response = FakeResponse
    chat_mod._raise = None
    chat_mod._FakeUsage = FakeUsage

    emb_mod = types.ModuleType("mistralai.client.embeddings")
    emb_mod.Embeddings = Embeddings
    fim_mod = types.ModuleType("mistralai.client.fim")
    fim_mod.Fim = Fim

    # MistralAzure and MistralGCP carry their own Chat/Fim classes in separate
    # modules; mistralai.azure.client.chat.Chat is NOT mistralai.client.chat.Chat
    # (verified against 2.9.1). Independent classes built from a copy of the
    # unpatched __dict__, deliberately NOT subclasses: a subclass would inherit
    # whatever the core class was patched with and get wrapped a second time,
    # which is not how the real package behaves.
    def _independent_copy(cls):
        return type(cls.__name__, (), dict(cls.__dict__))

    azure_chat_mod = types.ModuleType("mistralai.azure.client.chat")
    azure_chat_mod.Chat = _independent_copy(Chat)
    gcp_chat_mod = types.ModuleType("mistralai.gcp.client.chat")
    gcp_chat_mod.Chat = _independent_copy(Chat)
    gcp_fim_mod = types.ModuleType("mistralai.gcp.client.fim")
    gcp_fim_mod.Fim = _independent_copy(Fim)

    mod.client = client_mod
    client_mod.chat = chat_mod
    client_mod.embeddings = emb_mod
    client_mod.fim = fim_mod
    sys.modules["mistralai"] = mod
    sys.modules["mistralai.client"] = client_mod
    sys.modules["mistralai.client.chat"] = chat_mod
    sys.modules["mistralai.client.embeddings"] = emb_mod
    sys.modules["mistralai.client.fim"] = fim_mod
    for _name, _module in (
        ("mistralai.azure", types.ModuleType("mistralai.azure")),
        ("mistralai.azure.client", types.ModuleType("mistralai.azure.client")),
        ("mistralai.azure.client.chat", azure_chat_mod),
        ("mistralai.gcp", types.ModuleType("mistralai.gcp")),
        ("mistralai.gcp.client", types.ModuleType("mistralai.gcp.client")),
        ("mistralai.gcp.client.chat", gcp_chat_mod),
        ("mistralai.gcp.client.fim", gcp_fim_mod),
    ):
        sys.modules[_name] = _module
    chat_mod._azure_chat = azure_chat_mod.Chat
    chat_mod._gcp_chat = gcp_chat_mod.Chat
    chat_mod._gcp_fim = gcp_fim_mod.Fim
    return chat_mod


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


class TestAutoInstrumentMistral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._chat_mod = _install_fake_mistral()
        _PATCHED.discard("mistral")
        from dunetrace.auto import _patch_mistral

        _patch_mistral()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("mistralai", "mistral")

    def tearDown(self):
        self._chat_mod._raise = None
        self._chat_mod._response = self._chat_mod._default_response

    def test_sync_llm_events_emitted(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("mistral-agent", user_input="hi"):
            self._chat_mod.Chat().complete(
                messages=[{"role": "user", "content": "Capital of France?"}],
                model="mistral-large-latest",
            )

        types_ = [e.event_type for e in captured]
        self.assertIn(EventType.LLM_CALLED, types_)
        self.assertIn(EventType.LLM_RESPONDED, types_)
        dt.shutdown(timeout=1)

    def test_model_and_provider_recorded(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._chat_mod.Chat().complete(
                messages=[{"role": "user", "content": "hi"}],
                model="mistral-small-latest",
            )

        called = next(e for e in captured if e.event_type == EventType.LLM_CALLED)
        self.assertEqual(called.payload["model"], "mistral-small-latest")
        self.assertEqual(called.payload["provider"], "mistral")
        dt.shutdown(timeout=1)

    def test_real_prompt_tokens_backfilled_from_usage(self):
        """llm_called sends a chars//4 estimate; llm_responded overrides it with
        the exact count Mistral returns."""
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent") as run:
            self._chat_mod.Chat().complete(
                messages=[{"role": "user", "content": "x" * 400}],
                model="mistral-large-latest",
            )
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.prompt_tokens, 11)
        self.assertEqual(lc.completion_tokens, 22)
        responded = next(e for e in captured if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["prompt_tokens"], 11)
        dt.shutdown(timeout=1)

    def test_provider_on_reconstructed_llm_call(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._chat_mod.Chat().complete(
                messages=[{"role": "user", "content": "hi"}],
                model="mistral-large-latest",
            )
            self.assertEqual(run.state.llm_calls[-1].provider, "mistral")
        dt.shutdown(timeout=1)

    def test_output_text_and_finish_reason(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._chat_mod.Chat().complete(
                messages=[{"role": "user", "content": "hi"}],
                model="mistral-large-latest",
            )
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.output_text, "Paris")
        self.assertEqual(lc.output_length, 5)
        self.assertEqual(lc.finish_reason, "stop")
        dt.shutdown(timeout=1)

    def test_list_shaped_content_is_joined(self):
        """Mistral assistant content is a string or a list of chunks. Both have
        to produce text, not a crash."""

        class Chunk:
            def __init__(self, text):
                self.text = text

        class ListMessage:
            content = [Chunk("Par"), Chunk("is")]

        class ListChoice:
            finish_reason = "stop"
            message = ListMessage()

        class ListResponse:
            usage = self._chat_mod._FakeUsage()
            choices = [ListChoice()]

        self._chat_mod._response = ListResponse
        dt = _make_client()

        with dt.run("agent") as run:
            self._chat_mod.Chat().complete(
                messages=[{"role": "user", "content": "hi"}],
                model="mistral-large-latest",
            )
            self.assertEqual(run.state.llm_calls[-1].output_text, "Paris")
        dt.shutdown(timeout=1)

    def test_tool_call_reply_with_null_content_does_not_crash(self):
        class NullMessage:
            content = None

        class NullChoice:
            finish_reason = "tool_calls"
            message = NullMessage()

        class NullResponse:
            usage = self._chat_mod._FakeUsage()
            choices = [NullChoice()]

        self._chat_mod._response = NullResponse
        dt = _make_client()

        with dt.run("agent") as run:
            self._chat_mod.Chat().complete(
                messages=[{"role": "user", "content": "hi"}],
                model="mistral-large-latest",
            )
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.output_length, 0)
        self.assertEqual(lc.finish_reason, "tool_calls")
        dt.shutdown(timeout=1)

    def test_no_events_outside_run(self):
        dt = _make_client()
        captured = _capture(dt)

        self._chat_mod.Chat().complete(
            messages=[{"role": "user", "content": "hi"}],
            model="mistral-large-latest",
        )

        self.assertEqual(len(captured), 0)
        dt.shutdown(timeout=1)

    def test_error_is_reraised_and_recorded(self):
        self._chat_mod._raise = RuntimeError("mistral is down")
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            with self.assertRaises(RuntimeError):
                self._chat_mod.Chat().complete(
                    messages=[{"role": "user", "content": "hi"}],
                    model="mistral-large-latest",
                )

        responded = next(e for e in captured if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["finish_reason"], "error")
        dt.shutdown(timeout=1)

    def test_idempotent(self):
        from dunetrace.auto import _patch_mistral

        patched_once = self._chat_mod.Chat.complete
        _patch_mistral()
        self.assertIs(self._chat_mod.Chat.complete, patched_once)

    def test_parse_does_not_double_count(self):
        """Chat.parse delegates to Chat.complete in the real SDK, so patching
        both would emit two llm.called events for one API call."""
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._chat_mod.Chat().parse(
                messages=[{"role": "user", "content": "hi"}],
                model="mistral-large-latest",
            )

        called = [e for e in captured if e.event_type == EventType.LLM_CALLED]
        responded = [e for e in captured if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(called), 1)
        self.assertEqual(len(responded), 1)
        dt.shutdown(timeout=1)


class TestMistralStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._chat_mod = _install_fake_mistral()
        _PATCHED.discard("mistral")
        from dunetrace.auto import _patch_mistral

        _patch_mistral()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("mistralai", "mistral")

    def test_context_manager_stream_totals(self):
        dt = _make_client()

        with dt.run("agent") as run:
            with self._chat_mod.Chat().stream(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": "count"}],
            ) as stream:
                chunks = list(stream)
            lc = run.state.llm_calls[-1]

        self.assertEqual(len(chunks), 3)
        self.assertEqual(lc.prompt_tokens, 20)
        self.assertEqual(lc.completion_tokens, 8)
        self.assertEqual(lc.output_text, "1, 2, 3")
        self.assertEqual(lc.finish_reason, "stop")
        dt.shutdown(timeout=1)

    def test_plain_iteration_without_context_manager(self):
        dt = _make_client()

        with dt.run("agent") as run:
            stream = self._chat_mod.Chat().stream(
                model="mistral-small-latest", messages=[{"role": "user", "content": "c"}]
            )
            list(stream)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.completion_tokens, 8)
        dt.shutdown(timeout=1)

    def test_emits_exactly_once_when_both_paths_run(self):
        """Iterating to exhaustion and exiting the context manager both reach
        the emit, so the once-guard has to hold."""
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            with self._chat_mod.Chat().stream(
                model="mistral-small-latest", messages=[{"role": "user", "content": "c"}]
            ) as stream:
                list(stream)

        responded = [e for e in captured if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)
        dt.shutdown(timeout=1)

    def test_early_break_still_emits(self):
        """A caller who stops reading has still paid for the tokens consumed."""
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            stream = self._chat_mod.Chat().stream(
                model="mistral-small-latest", messages=[{"role": "user", "content": "c"}]
            )
            for _ in stream:
                break

        responded = [e for e in captured if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)
        dt.shutdown(timeout=1)

    def test_async_stream_totals(self):
        dt = _make_client()

        async def _run():
            with dt.run("agent") as run:
                stream = await self._chat_mod.Chat().stream_async(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": "c"}],
                )
                async for _ in stream:
                    pass
                return run.state.llm_calls[-1]

        lc = asyncio.run(_run())
        self.assertEqual(lc.prompt_tokens, 20)
        self.assertEqual(lc.completion_tokens, 8)
        self.assertEqual(lc.output_text, "1, 2, 3")
        dt.shutdown(timeout=1)

    def test_async_context_manager_stream(self):
        dt = _make_client()

        async def _run():
            with dt.run("agent") as run:
                stream = await self._chat_mod.Chat().stream_async(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": "c"}],
                )
                async with stream as s:
                    async for _ in s:
                        pass
                return run.state.llm_calls[-1]

        lc = asyncio.run(_run())
        self.assertEqual(lc.completion_tokens, 8)
        dt.shutdown(timeout=1)

    def test_chunks_are_not_retained(self):
        """The brief's no-memory-leak requirement. The proxy keeps ints and text
        fragments, never chunk objects, so a long stream costs no more than the
        equivalent non-streaming response."""
        import gc
        import time as _time
        import weakref

        from dunetrace.auto import _StreamProxy, _mistral_stream_collector

        class WeakChunk:
            # SimpleNamespace can't be weak-referenced, and the point of this
            # test is to hold weakrefs to the chunk objects themselves.
            def __init__(self, data):
                self.data = data

        dt = _make_client()
        chunks = [WeakChunk(c.data) for c in _mistral_stream_chunks()]
        refs = [weakref.ref(c) for c in chunks]

        # Driving the proxy directly rather than repatching Chat.stream, which
        # would leak a class-level mutation into every later test.
        with dt.run("agent") as run:
            # The patcher emits llm_called before handing back the proxy, and
            # llm_responded backfills the most recent LlmCall, so the preamble
            # has to happen here too.
            run.llm_called("mistral-small-latest", prompt_tokens=0, provider="mistral")
            proxy = _StreamProxy(
                _FakeSyncStream(chunks), run, _time.monotonic(), _mistral_stream_collector
            )
            with proxy as stream:
                list(stream)
            lc = run.state.llm_calls[-1] if run.state.llm_calls else None

        del chunks, proxy, stream
        gc.collect()
        self.assertTrue(
            all(r() is None for r in refs),
            "stream proxy retained chunk objects after the stream closed",
        )
        # The measurement still happened; only the chunk objects were released.
        self.assertIsNotNone(lc)
        self.assertEqual(lc.completion_tokens, 8)
        dt.shutdown(timeout=1)

    def test_stream_outside_run_is_untouched(self):
        dt = _make_client()
        captured = _capture(dt)

        stream = self._chat_mod.Chat().stream(
            model="mistral-small-latest", messages=[{"role": "user", "content": "c"}]
        )
        list(stream)

        self.assertEqual(len(captured), 0)
        dt.shutdown(timeout=1)


class TestMistralAsyncEmbeddingsAndFim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._chat_mod = _install_fake_mistral()
        _PATCHED.discard("mistral")
        from dunetrace.auto import _patch_mistral

        _patch_mistral()
        cls._emb = sys.modules["mistralai.client.embeddings"]
        cls._fim = sys.modules["mistralai.client.fim"]

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("mistralai", "mistral")

    def test_complete_async(self):
        dt = _make_client()

        async def _run():
            with dt.run("agent") as run:
                await self._chat_mod.Chat().complete_async(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": "hi"}],
                )
                return run.state.llm_calls[-1]

        lc = asyncio.run(_run())
        self.assertEqual(lc.provider, "mistral")
        self.assertEqual(lc.prompt_tokens, 11)
        self.assertEqual(lc.completion_tokens, 22)
        dt.shutdown(timeout=1)

    def test_embeddings_captured_with_input_tokens_only(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._emb.Embeddings().create(model="mistral-embed", inputs=["a", "b"])
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.model, "mistral-embed")
        self.assertEqual(lc.provider, "mistral")
        self.assertEqual(lc.prompt_tokens, 17)
        self.assertIsNone(lc.completion_tokens)
        self.assertEqual(lc.output_length, 0)
        dt.shutdown(timeout=1)

    def test_embeddings_cost_uses_the_input_only_rate(self):
        from dunetrace.policies import compute_run_cost

        dt = _make_client()
        with dt.run("agent") as run:
            self._emb.Embeddings().create(model="mistral-embed", inputs=["a"])
            lc = run.state.llm_calls[-1]

        self.assertAlmostEqual(compute_run_cost([lc]), 17 * 0.10e-6)
        dt.shutdown(timeout=1)

    def test_embeddings_async(self):
        dt = _make_client()

        async def _run():
            with dt.run("agent") as run:
                await self._emb.Embeddings().create_async(model="mistral-embed", inputs=["a"])
                return run.state.llm_calls[-1]

        self.assertEqual(asyncio.run(_run()).prompt_tokens, 17)
        dt.shutdown(timeout=1)

    def test_fim_completion_captured(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._fim.Fim().complete(model="codestral-latest", prompt="def f():")
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.model, "codestral-latest")
        self.assertEqual(lc.provider, "mistral")
        self.assertEqual(lc.prompt_tokens, 5)
        self.assertEqual(lc.completion_tokens, 4)
        self.assertEqual(lc.output_text, "return 42")
        dt.shutdown(timeout=1)

    def test_fim_streaming_captured(self):
        dt = _make_client()

        with dt.run("agent") as run:
            with self._fim.Fim().stream(model="codestral-latest", prompt="x") as s:
                list(s)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.completion_tokens, 8)
        dt.shutdown(timeout=1)


class TestOpenAIStreamingCapture(unittest.TestCase):
    """Before this, a streamed OpenAI call emitted completion_tokens=0,
    finish_reason "stop" and empty output, because a Stream object has no
    .usage."""

    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        _PATCHED.discard("openai")
        from dunetrace.auto import _patch_openai

        _patch_openai()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("openai")

    def test_stream_without_include_usage_estimates_rather_than_zero(self):
        dt = _make_client()

        with dt.run("agent") as run:
            stream = self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                stream=True,
            )
            list(stream)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.output_text, "Hello world")
        self.assertEqual(lc.finish_reason, "stop")
        # "Hello world" is 11 chars, so the 4-chars-per-token estimate is 2.
        self.assertEqual(lc.completion_tokens, 2)
        self.assertNotEqual(lc.completion_tokens, 0)
        dt.shutdown(timeout=1)

    def test_stream_with_include_usage_uses_real_totals(self):
        dt = _make_client()

        with dt.run("agent") as run:
            stream = self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                stream=True,
                stream_options={"include_usage": True},
            )
            list(stream)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.prompt_tokens, 7)
        self.assertEqual(lc.completion_tokens, 3)
        dt.shutdown(timeout=1)

    def test_async_stream(self):
        dt = _make_client()

        async def _run():
            with dt.run("agent") as run:
                stream = await self._completions_mod.AsyncCompletions().create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o",
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for _ in stream:
                    pass
                return run.state.llm_calls[-1]

        lc = asyncio.run(_run())
        self.assertEqual(lc.completion_tokens, 3)
        self.assertEqual(lc.output_text, "Hello world")
        dt.shutdown(timeout=1)

    def test_non_streaming_path_is_unchanged(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
            )
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.completion_tokens, 42)
        self.assertEqual(lc.reasoning_tokens, 7)
        dt.shutdown(timeout=1)


class TestAnthropicStreamingCapture(unittest.TestCase):
    """messages.stream() emitted nothing at all before this, since it calls
    self._post directly and never routes through the patched create()."""

    @classmethod
    def setUpClass(cls):
        cls._messages_mod = _install_fake_anthropic()
        _PATCHED.discard("anthropic")
        from dunetrace.auto import _patch_anthropic

        _patch_anthropic()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("anthropic")

    def test_create_with_stream_true(self):
        dt = _make_client()

        with dt.run("agent") as run:
            stream = self._messages_mod.Messages().create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            list(stream)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.prompt_tokens, 12)
        self.assertEqual(lc.completion_tokens, 30)
        self.assertEqual(lc.output_text, "42")
        self.assertEqual(lc.finish_reason, "end_turn")
        dt.shutdown(timeout=1)

    def test_messages_stream_manager(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent") as run:
            with self._messages_mod.Messages().stream(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
            ) as stream:
                list(stream)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.prompt_tokens, 12)
        self.assertEqual(lc.completion_tokens, 30)
        self.assertEqual(lc.output_text, "42")
        responded = [e for e in captured if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)
        dt.shutdown(timeout=1)

    def test_finalizer_recovers_totals_when_proxy_was_never_iterated(self):
        """A caller consuming the stream through an SDK helper leaves the
        collector empty, so the finalizer reads get_final_message() instead."""
        dt = _make_client()

        with dt.run("agent") as run:
            with self._messages_mod.Messages().stream(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass  # never iterated
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.completion_tokens, 30)
        self.assertEqual(lc.prompt_tokens, 12)
        dt.shutdown(timeout=1)

    def test_async_messages_stream_manager(self):
        dt = _make_client()

        async def _run():
            with dt.run("agent") as run:
                async with self._messages_mod.AsyncMessages().stream(
                    model="claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "hi"}],
                ) as stream:
                    async for _ in stream:
                        pass
                return run.state.llm_calls[-1]

        lc = asyncio.run(_run())
        self.assertEqual(lc.completion_tokens, 30)
        dt.shutdown(timeout=1)

    def test_async_finalizer_recovers_totals_when_proxy_was_never_iterated(self):
        """The async counterpart of the sync finalizer test, and the shape the
        documented idiom actually takes: `async with ... as s: async for text in
        s.text_stream:` reaches text_stream on the inner manager through
        __getattr__, so the proxy's collector never sees an event and everything
        has to come from get_final_message().

        Regression: that getter is `async def` on AsyncMessageStream but plain
        `def` on MessageStream. Calling it synchronously produced an un-awaited
        coroutine, so the run recorded completion_tokens=None with
        output_length=0 and finish_reason="stop" — which is exactly the firing
        condition of EmptyLlmResponseDetector.
        """
        dt = _make_client()

        async def _run():
            with dt.run("agent") as run:
                async with self._messages_mod.AsyncMessages().stream(
                    model="claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "hi"}],
                ):
                    pass  # never iterated
                return run.state.llm_calls[-1]

        lc = asyncio.run(_run())
        self.assertEqual(lc.completion_tokens, 30)
        self.assertEqual(lc.prompt_tokens, 12)
        # Not the EMPTY_LLM_RESPONSE shape (finish_reason "stop" + zero length).
        self.assertNotEqual(lc.output_length, 0)
        dt.shutdown(timeout=1)

    def test_non_streaming_path_is_unchanged(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._messages_mod.Messages().create(
                model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}]
            )
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.completion_tokens, 30)
        self.assertEqual(lc.finish_reason, "end_turn")
        dt.shutdown(timeout=1)


class TestStreamBackfillsItsOwnCall(unittest.TestCase):
    """A stream's response lands whenever the caller drains it, which may be
    after other LLM calls have started. The response must back-fill the call the
    stream belongs to, not whichever call happens to be last at drain time."""

    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        _PATCHED.discard("openai")
        from dunetrace.auto import _patch_openai

        _patch_openai()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("openai")

    def test_call_interleaved_between_open_and_drain_is_not_clobbered(self):
        dt = _make_client()
        completions = self._completions_mod.Completions()

        with dt.run("agent") as run:
            stream = completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                stream=True,
                stream_options={"include_usage": True},
            )
            # A whole non-streamed call completes while the stream is still open.
            completions.create(
                messages=[{"role": "user", "content": "summarise"}],
                model="gpt-4o-mini",
            )
            list(stream)
            calls = list(run.state.llm_calls)

        self.assertEqual(len(calls), 2)

        streamed, direct = calls[0], calls[1]
        self.assertEqual(streamed.model, "gpt-4o")
        self.assertEqual(streamed.completion_tokens, 3)
        self.assertEqual(streamed.prompt_tokens, 7)
        self.assertEqual(streamed.output_text, "Hello world")

        # The interleaved call keeps its own totals — previously the stream's
        # values were written over the top of them.
        self.assertEqual(direct.model, "gpt-4o-mini")
        self.assertEqual(direct.completion_tokens, 42)
        self.assertEqual(direct.prompt_tokens, 10)
        self.assertEqual(direct.output_text, "Paris")
        dt.shutdown(timeout=1)

    def test_undrained_stream_leaves_other_calls_alone(self):
        """A stream abandoned without being drained must not retroactively
        rewrite a later call when the proxy is finally collected."""
        dt = _make_client()
        completions = self._completions_mod.Completions()

        with dt.run("agent") as run:
            stream = completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                stream=True,
            )
            iterator = iter(stream)
            next(iterator)
            del iterator  # closes the generator -> _emit fires on the error path
            completions.create(
                messages=[{"role": "user", "content": "again"}],
                model="gpt-4o-mini",
            )
            calls = list(run.state.llm_calls)

        self.assertEqual(calls[1].model, "gpt-4o-mini")
        self.assertEqual(calls[1].completion_tokens, 42)
        self.assertEqual(calls[1].output_text, "Paris")
        dt.shutdown(timeout=1)


class TestStreamProxyIteratorProtocol(unittest.TestCase):
    """The wrapped streams are all first-class iterators (openai.Stream,
    mistralai EventStream, anthropic MessageStream), so the proxy has to be one
    too — instrumentation must not change what a caller can do with the object."""

    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        _PATCHED.discard("openai")
        from dunetrace.auto import _patch_openai

        _patch_openai()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("openai")

    def _stream(self, run_unused=None):
        return self._completions_mod.Completions().create(
            messages=[{"role": "user", "content": "hi"}], model="gpt-4o", stream=True
        )

    def test_next_on_the_proxy_works(self):
        """Peeking the first chunk is ordinary usage and used to raise
        TypeError: '_StreamProxy' object is not an iterator."""
        dt = _make_client()
        with dt.run("agent"):
            stream = self._stream()
            first = next(stream)
        self.assertEqual(first.choices[0].delta.content, "Hello")
        dt.shutdown(timeout=1)

    def test_iter_is_idempotent(self):
        dt = _make_client()
        with dt.run("agent"):
            stream = self._stream()
            self.assertIs(iter(stream), iter(stream))
            self.assertIs(iter(stream), stream)
        dt.shutdown(timeout=1)

    def test_partial_consume_then_resume_records_the_whole_stream(self):
        """Previously the first partial pass latched the emit, and every chunk
        read afterwards reached the caller but never the run."""
        import itertools

        dt = _make_client()
        with dt.run("agent") as run:
            stream = self._stream()
            head = list(itertools.islice(stream, 1))
            rest = list(stream)
            lc = run.state.llm_calls[-1]

        self.assertEqual(len(head), 1)
        self.assertEqual(len(rest), 2)
        # "Hello" + " world" — nothing dropped between the two passes.
        self.assertEqual(lc.output_text, "Hello world")
        dt.shutdown(timeout=1)

    def test_async_proxy_is_an_async_iterator(self):
        dt = _make_client()

        async def _run():
            with dt.run("agent"):
                stream = await self._completions_mod.AsyncCompletions().create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o",
                    stream=True,
                )
                self.assertIs(stream.__aiter__(), stream)
                return await stream.__anext__()

        first = asyncio.run(_run())
        self.assertEqual(first.choices[0].delta.content, "Hello")
        dt.shutdown(timeout=1)


class TestStreamFailureIsRecordedAsError(unittest.TestCase):
    """A stream that dies mid-flight must not look like a clean completion —
    EmptyLlmResponseDetector fires on finish_reason=="stop" with zero output."""

    def _proxy(self, run, chunks_then_raise):
        from dunetrace.auto import _StreamProxy, _openai_stream_collector

        class _Exploding:
            def __init__(self):
                self._it = iter(chunks_then_raise)

            def __iter__(self):
                return self

            def __next__(self):
                item = next(self._it)
                if isinstance(item, Exception):
                    raise item
                return item

        run.llm_called("gpt-4o", prompt_tokens=100, provider="openai")
        return _StreamProxy(_Exploding(), run, time.monotonic(), _openai_stream_collector)

    def test_mid_stream_error_sets_error_finish_reason(self):
        dt = _make_client()

        class ProviderError(RuntimeError):
            pass

        with dt.run("agent") as run:
            proxy = self._proxy(
                run, [_openai_stream_chunks()[0], ProviderError("connection reset")]
            )
            with self.assertRaises(ProviderError):
                list(proxy)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.finish_reason, "error")
        dt.shutdown(timeout=1)

    def test_error_text_is_reported_on_the_event(self):
        dt = _make_client()
        captured = _capture(dt)

        class ProviderError(RuntimeError):
            pass

        with dt.run("agent") as run:
            proxy = self._proxy(run, [ProviderError("connection reset")])
            with self.assertRaises(ProviderError):
                list(proxy)

        responded = [e for e in captured if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)
        self.assertEqual(responded[0].payload["finish_reason"], "error")
        self.assertIn("connection reset", responded[0].payload["error"])
        dt.shutdown(timeout=1)

    def test_clean_stream_is_still_reported_as_stop(self):
        dt = _make_client()
        with dt.run("agent") as run:
            proxy = self._proxy(run, list(_openai_stream_chunks()))
            list(proxy)
            lc = run.state.llm_calls[-1]
        self.assertEqual(lc.finish_reason, "stop")
        self.assertIsNone(lc.error if hasattr(lc, "error") else None)
        dt.shutdown(timeout=1)


class TestOpenAIStreamedToolCallsAreBilled(unittest.TestCase):
    """A streamed tool-calling turn carries no content — the whole output is on
    delta.tool_calls — so the text-length fallback had nothing to measure and
    the step reported zero output tokens."""

    @staticmethod
    def _tool_call_chunks():
        def chunk(*, name=None, arguments=None, finish=None):
            fn = types.SimpleNamespace(name=name, arguments=arguments)
            delta = types.SimpleNamespace(
                content=None, tool_calls=[types.SimpleNamespace(function=fn)]
            )
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=delta, finish_reason=finish)],
                usage=None,
            )

        return [
            chunk(name="get_weather", arguments=""),
            chunk(arguments='{"location": "Paris", "unit": "celsius"}'),
            chunk(finish="tool_calls"),
        ]

    def test_tool_call_only_stream_reports_nonzero_completion_tokens(self):
        from dunetrace.auto import _StreamProxy, _openai_stream_collector

        dt = _make_client()
        with dt.run("agent") as run:
            run.llm_called("gpt-4o", prompt_tokens=100, provider="openai")
            proxy = _StreamProxy(
                _FakeSyncStream(self._tool_call_chunks()),
                run,
                time.monotonic(),
                _openai_stream_collector,
            )
            list(proxy)
            lc = run.state.llm_calls[-1]

        # "get_weather" (11) + the 40-char JSON argument blob = 51 chars, so the
        # same chars//4 heuristic the text path uses gives 12. The point is that
        # it is not zero — the exact heuristic is shared with the text path.
        self.assertEqual(lc.completion_tokens, 12)
        self.assertEqual(lc.finish_reason, "tool_calls")
        dt.shutdown(timeout=1)

    def test_real_usage_still_wins_over_the_estimate(self):
        from dunetrace.auto import _StreamProxy, _openai_stream_collector

        chunks = self._tool_call_chunks()
        chunks.append(
            types.SimpleNamespace(
                choices=[],
                usage=types.SimpleNamespace(prompt_tokens=90, completion_tokens=60),
            )
        )
        dt = _make_client()
        with dt.run("agent") as run:
            run.llm_called("gpt-4o", prompt_tokens=100, provider="openai")
            proxy = _StreamProxy(
                _FakeSyncStream(chunks), run, time.monotonic(), _openai_stream_collector
            )
            list(proxy)
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.completion_tokens, 60)
        self.assertEqual(lc.prompt_tokens, 90)
        dt.shutdown(timeout=1)


class TestStreamedCallsEnforcePolicies(unittest.TestCase):
    """A `stop` policy is runtime prevention, not telemetry. It has to survive
    the streaming path the same way it survives a non-streamed call."""

    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        _PATCHED.discard("openai")
        from dunetrace.auto import _patch_openai

        _patch_openai()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("openai")

    @staticmethod
    def _client_with_stop_policy() -> Dunetrace:
        dt = Dunetrace(endpoint=None)
        dt.add_policy(
            "halt-on-spend",
            {"trigger": "cost_usd", "operator": "gt", "value": 0.0},
            {"type": "stop"},
        )
        return dt

    def test_stop_policy_raises_when_stream_is_drained_cleanly(self):
        """Regression: the emit was made on the error path unconditionally, so
        the violation was logged and swallowed. The policy was already marked
        fired, so it never got another chance — a stop policy silently stopped
        working for every streamed call."""
        dt = self._client_with_stop_policy()

        with self.assertRaises(PolicyViolation):
            with dt.run("agent"):
                stream = self._completions_mod.Completions().create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o",
                    stream=True,
                    stream_options={"include_usage": True},
                )
                list(stream)
        dt.shutdown(timeout=1)

    def test_host_exception_still_wins_over_the_policy(self):
        """The other half of the contract: when the caller's own iteration blows
        up, our violation must not displace their exception."""

        class ProviderError(RuntimeError):
            pass

        class _ExplodingStream:
            def __iter__(self):
                yield _openai_stream_chunks(True)[0]
                raise ProviderError("connection reset")

        dt = self._client_with_stop_policy()
        from dunetrace.auto import _StreamProxy, _openai_stream_collector

        with self.assertRaises(ProviderError):
            with dt.run("agent") as run:
                run.llm_called("gpt-4o", prompt_tokens=100, provider="openai")
                proxy = _StreamProxy(
                    _ExplodingStream(), run, time.monotonic(), _openai_stream_collector
                )
                list(proxy)
        dt.shutdown(timeout=1)


class TestRealPromptTokensAcrossProviders(unittest.TestCase):
    """All three providers return an exact prompt token count. llm_called sends
    a chars//4 estimate first; llm_responded overrides it with the real number
    so cost_usd, and the cost_usd policy trigger, are exact rather than
    approximate."""

    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        cls._messages_mod = _install_fake_anthropic()
        _PATCHED.discard("openai")
        _PATCHED.discard("anthropic")
        from dunetrace.auto import _patch_anthropic, _patch_openai

        _patch_openai()
        _patch_anthropic()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("openai", "anthropic")

    def test_openai_uses_usage_prompt_tokens(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "x" * 4000}],
                model="gpt-4o",
            )
            lc = run.state.llm_calls[-1]

        # 4000 chars would estimate ~1000 tokens. The fake reports 10.
        self.assertEqual(lc.prompt_tokens, 10)
        dt.shutdown(timeout=1)

    def test_anthropic_uses_usage_input_tokens(self):
        dt = _make_client()

        with dt.run("agent") as run:
            self._messages_mod.Messages().create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "x" * 4000}],
            )
            lc = run.state.llm_calls[-1]

        self.assertEqual(lc.prompt_tokens, 12)
        dt.shutdown(timeout=1)

    def test_missing_usage_keeps_the_estimate_rather_than_zeroing(self):
        """llm_responded ignores a falsy prompt_tokens, so a response without a
        usage block must not wipe out the estimate llm_called recorded."""

        class NoUsageResponse:
            usage = None
            choices = self._completions_mod.Completions().create(messages=[], model="x").choices

        dt = _make_client()
        original = self._completions_mod.Completions.create
        try:
            self._completions_mod.Completions.create = lambda self_, **kw: NoUsageResponse()
            _PATCHED.discard("openai")
            from dunetrace.auto import _patch_openai

            _patch_openai()

            with dt.run("agent") as run:
                self._completions_mod.Completions().create(
                    messages=[{"role": "user", "content": "x" * 400}],
                    model="gpt-4o",
                )
                lc = run.state.llm_calls[-1]

            self.assertEqual(lc.prompt_tokens, 100)
        finally:
            self._completions_mod.Completions.create = original
            _PATCHED.discard("openai")
            dt.shutdown(timeout=1)


class TestMistralNotInstalled(unittest.TestCase):
    def test_patch_is_a_noop_without_mistralai(self):
        """Backward compat: an SDK user who has never heard of Mistral sees no
        error, no warning, and no state change."""
        _uninstall_fake("mistralai", "mistral")
        saved = {k: v for k, v in sys.modules.items() if k.startswith("mistralai")}
        for k in list(sys.modules):
            if k == "mistralai" or k.startswith("mistralai."):
                del sys.modules[k]
        sys.modules["mistralai"] = None  # force ImportError on import

        from dunetrace.auto import _patch_mistral

        try:
            _patch_mistral()
            self.assertNotIn("mistral", _PATCHED)
        finally:
            del sys.modules["mistralai"]
            sys.modules.update(saved)
            _PATCHED.discard("mistral")

    def test_auto_instrument_all_frameworks_survives_missing_mistralai(self):
        dt = _make_client()
        saved = {k: v for k, v in sys.modules.items() if k.startswith("mistralai")}
        for k in list(sys.modules):
            if k == "mistralai" or k.startswith("mistralai."):
                del sys.modules[k]
        sys.modules["mistralai"] = None

        try:
            dt.auto_instrument(["mistral"])
        finally:
            del sys.modules["mistralai"]
            sys.modules.update(saved)
            _PATCHED.discard("mistral")
            dt.shutdown(timeout=1)


def _install_fake_botocore():
    """Fake botocore's one interception point.

    Shapes verified against real botocore 1.43.67 (see the module comment on
    _patch_botocore): `BaseClient._make_api_call(self, operation_name,
    api_params)`, `client.meta.service_model.service_name`, the four
    bedrock-runtime operation names, and each operation's response keys. The
    real client was also driven end-to-end during development; this fake exists
    so the suite doesn't need boto3 installed.
    """
    mod = types.ModuleType("botocore")
    client_mod = types.ModuleType("botocore.client")

    class BaseClient:
        def __init__(self, service_name="bedrock-runtime"):
            self.meta = types.SimpleNamespace(
                service_model=types.SimpleNamespace(service_name=service_name)
            )

        def _make_api_call(self, operation_name, api_params):
            if client_mod._raise is not None:
                raise client_mod._raise
            return client_mod._response

    client_mod.BaseClient = BaseClient
    client_mod._response = {}
    # Tests set this to drive the provider-error path.
    client_mod._raise = None
    mod.client = client_mod
    sys.modules["botocore"] = mod
    sys.modules["botocore.client"] = client_mod
    return client_mod


class TestAutoInstrumentBedrock(unittest.TestCase):
    """Bedrock goes through boto3, and botocore rides on urllib3 — so neither
    the vendor-SDK patches nor the httpx/requests patches ever saw it."""

    @classmethod
    def setUpClass(cls):
        cls._botocore = _install_fake_botocore()
        _PATCHED.discard("botocore")
        from dunetrace.auto import _patch_botocore

        _patch_botocore()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("botocore")

    def _call(self, operation, params, response, service_name="bedrock-runtime"):
        self._botocore._response = response
        client = self._botocore.BaseClient(service_name)
        dt = _make_client()
        with dt.run("agent") as run:
            out = client._make_api_call(operation, params)
            calls = list(run.state.llm_calls)
        dt.shutdown(timeout=1)
        return out, calls

    def test_converse_records_tokens_text_and_stop_reason(self):
        _, calls = self._call(
            "Converse",
            {"modelId": "mistral.mistral-large-2407-v1:0", "messages": [{"role": "user"}]},
            {
                "output": {"message": {"content": [{"text": "Bonjour"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 42, "outputTokens": 7},
            },
        )
        lc = calls[-1]
        self.assertEqual(lc.model, "mistral.mistral-large-2407-v1:0")
        self.assertEqual(lc.provider, "bedrock")
        self.assertEqual(lc.prompt_tokens, 42)
        self.assertEqual(lc.completion_tokens, 7)
        self.assertEqual(lc.finish_reason, "end_turn")
        self.assertEqual(lc.output_text, "Bonjour")

    def test_invoke_model_reads_headers_and_never_consumes_the_body(self):
        """The payload is a streaming body in a model-specific format. Reading
        it would hand the caller an empty stream, so tokens come from headers."""

        class Body:
            def __init__(self):
                self.read_called = False

            def read(self):
                self.read_called = True
                return b"{}"

        body = Body()
        _, calls = self._call(
            "InvokeModel",
            {"modelId": "mistral.mistral-7b-instruct-v0:2", "body": '{"prompt": "hi"}'},
            {
                "body": body,
                "ResponseMetadata": {
                    "HTTPHeaders": {
                        "x-amzn-bedrock-input-token-count": "120",
                        "x-amzn-bedrock-output-token-count": "34",
                    }
                },
            },
        )
        lc = calls[-1]
        self.assertFalse(body.read_called)
        self.assertEqual(lc.prompt_tokens, 120)
        self.assertEqual(lc.completion_tokens, 34)
        # Not "stop" — that plus output_length 0 is EMPTY_LLM_RESPONSE's
        # firing condition, and an unread body is not an empty response.
        self.assertNotEqual(lc.finish_reason, "stop")

    def test_converse_stream_is_wrapped_and_passes_every_event_through(self):
        events = [
            {"contentBlockDelta": {"delta": {"text": "Bon"}}},
            {"contentBlockDelta": {"delta": {"text": "jour"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 5}}},
        ]
        self._botocore._response = {"stream": iter(events)}
        client = self._botocore.BaseClient()
        dt = _make_client()
        with dt.run("agent") as run:
            resp = client._make_api_call("ConverseStream", {"modelId": "m", "messages": []})
            seen = list(resp["stream"])
            lc = run.state.llm_calls[-1]

        self.assertEqual(len(seen), 4)
        self.assertEqual(lc.output_text, "Bonjour")
        self.assertEqual(lc.prompt_tokens, 11)
        self.assertEqual(lc.completion_tokens, 5)
        self.assertEqual(lc.finish_reason, "end_turn")
        dt.shutdown(timeout=1)

    def test_invoke_stream_reads_bedrock_invocation_metrics(self):
        """Chunk payloads are model-specific JSON, but the
        amazon-bedrock-invocationMetrics object on the final chunk is not."""
        chunks = [
            {"chunk": {"bytes": json.dumps({"outputs": [{"text": "a"}]}).encode()}},
            {
                "chunk": {
                    "bytes": json.dumps(
                        {
                            "amazon-bedrock-invocationMetrics": {
                                "inputTokenCount": 88,
                                "outputTokenCount": 19,
                            }
                        }
                    ).encode()
                }
            },
        ]
        self._botocore._response = {"body": iter(chunks)}
        client = self._botocore.BaseClient()
        dt = _make_client()
        with dt.run("agent") as run:
            resp = client._make_api_call(
                "InvokeModelWithResponseStream", {"modelId": "m", "body": "{}"}
            )
            seen = list(resp["body"])
            lc = run.state.llm_calls[-1]

        self.assertEqual(len(seen), 2)
        self.assertEqual(lc.prompt_tokens, 88)
        self.assertEqual(lc.completion_tokens, 19)
        dt.shutdown(timeout=1)

    def test_non_bedrock_services_are_untouched(self):
        """Every boto3 call in the process reaches this wrapper — S3, SQS and
        the rest must pass straight through."""
        _, calls = self._call("ListBuckets", {}, {"Buckets": []}, service_name="s3")
        self.assertEqual(calls, [])

    def test_non_llm_bedrock_operations_are_untouched(self):
        _, calls = self._call("ApplyGuardrail", {}, {"action": "NONE"})
        self.assertEqual(calls, [])

    def test_provider_error_is_recorded_as_an_error(self):
        """A throttled or failed Bedrock call must not look like a clean one."""
        client = self._botocore.BaseClient()
        self._botocore._raise = RuntimeError("ThrottlingException")
        dt = _make_client()
        try:
            with dt.run("agent") as run:
                with self.assertRaises(RuntimeError):
                    client._make_api_call("Converse", {"modelId": "m", "messages": []})
                lc = run.state.llm_calls[-1]
        finally:
            self._botocore._raise = None
        self.assertEqual(lc.finish_reason, "error")
        dt.shutdown(timeout=1)


class TestLlmCallsAreNotDoubleCountedAsHttp(unittest.TestCase):
    """Every vendor SDK here rides on httpx. With both patchers on, one LLM call
    used to be recorded twice — once as llm.called, once as tool.called named
    after the hostname — inflating tool_call_count, which is both a policy
    trigger and what TOOL_LOOP counts."""

    @classmethod
    def setUpClass(cls):
        cls._completions_mod = _install_fake_openai()
        cls._httpx_mod = _install_fake_httpx()
        # The provider SDK issues its request through httpx, exactly as the real
        # openai/anthropic/mistralai clients do.
        _orig_create = cls._completions_mod.Completions.create

        def _create_via_httpx(self, *, messages=None, model="unknown", **kwargs):
            cls._httpx_mod.Client().send(cls._httpx_mod._FakeRequest())
            return _orig_create(self, messages=messages, model=model, **kwargs)

        cls._completions_mod.Completions.create = _create_via_httpx

        _PATCHED.discard("openai")
        _PATCHED.discard("httpx")
        from dunetrace.auto import _patch_httpx, _patch_openai

        _patch_openai()
        _patch_httpx()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("openai", "httpx")

    def test_llm_call_does_not_also_register_as_a_tool_call(self):
        dt = _make_client()
        with dt.run("agent") as run:
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
            )
            tool_calls = list(run.state.tool_calls)
            llm_calls = list(run.state.llm_calls)

        self.assertEqual(len(llm_calls), 1)
        self.assertEqual(tool_calls, [])
        dt.shutdown(timeout=1)

    def test_genuine_tool_http_in_the_same_run_is_still_recorded(self):
        """Suppression must be scoped to the provider call, not the whole run."""
        dt = _make_client()
        with dt.run("agent") as run:
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
            )
            self._httpx_mod.Client().send(self._httpx_mod._FakeRequest())
            tool_calls = list(run.state.tool_calls)

        self.assertEqual(len(tool_calls), 1)
        dt.shutdown(timeout=1)

    def test_suppression_flag_is_reset_after_the_call(self):
        from dunetrace.context import _http_suppressed

        dt = _make_client()
        with dt.run("agent"):
            self._completions_mod.Completions().create(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
            )
            self.assertFalse(_http_suppressed.get())
        dt.shutdown(timeout=1)


class TestMistralHyperscalerClients(unittest.TestCase):
    """MistralAzure and MistralGCP have their own Chat/Fim classes in separate
    modules. Patching only mistralai.client.* left every hyperscaler-hosted call
    uninstrumented while _patch_mistral still reported success — and made
    _mistral_deployment's azure/gcp branches unreachable."""

    @classmethod
    def setUpClass(cls):
        cls._chat_mod = _install_fake_mistral()
        _PATCHED.discard("mistral")
        from dunetrace.auto import _patch_mistral

        _patch_mistral()

    @classmethod
    def tearDownClass(cls):
        _uninstall_fake("mistralai", "mistral")

    def test_hyperscaler_chat_classes_are_distinct_from_the_core_one(self):
        self.assertIsNot(self._chat_mod._azure_chat, self._chat_mod.Chat)
        self.assertIsNot(self._chat_mod._gcp_chat, self._chat_mod.Chat)

    def test_azure_chat_complete_is_instrumented(self):
        dt = _make_client()
        with dt.run("agent") as run:
            self._chat_mod._azure_chat().complete(
                model="mistral-large-latest", messages=[{"role": "user", "content": "hi"}]
            )
            lc = run.state.llm_calls[-1]
        self.assertEqual(lc.model, "mistral-large-latest")
        self.assertEqual(lc.provider, "mistral")
        dt.shutdown(timeout=1)

    def test_gcp_chat_stream_is_instrumented(self):
        dt = _make_client()
        with dt.run("agent") as run:
            stream = self._chat_mod._gcp_chat().stream(
                model="mistral-large-latest", messages=[{"role": "user", "content": "hi"}]
            )
            list(stream)
            lc = run.state.llm_calls[-1]
        self.assertEqual(lc.provider, "mistral")
        self.assertIsNotNone(lc.completion_tokens)
        dt.shutdown(timeout=1)

    def test_gcp_fim_is_instrumented(self):
        dt = _make_client()
        with dt.run("agent") as run:
            self._chat_mod._gcp_fim().complete(model="codestral-latest", prompt="def f(")
            lc = run.state.llm_calls[-1]
        self.assertEqual(lc.model, "codestral-latest")
        dt.shutdown(timeout=1)

    def test_each_call_is_recorded_once(self):
        """The hyperscaler classes must not be subclasses of the core one — that
        would wrap an already-wrapped method and emit twice."""
        dt = _make_client()
        captured = _capture(dt)
        with dt.run("agent"):
            self._chat_mod._azure_chat().complete(
                model="mistral-large-latest", messages=[{"role": "user", "content": "hi"}]
            )
        called = [e for e in captured if e.event_type == EventType.LLM_CALLED]
        responded = [e for e in captured if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(called), 1)
        self.assertEqual(len(responded), 1)
        dt.shutdown(timeout=1)


class TestMistralDeploymentDetection(unittest.TestCase):
    """_mistral_deployment classifies without touching the network. Class
    identity has to beat the URL, because MistralAzure built with no explicit
    server_url resolves to https://api.mistral.ai."""

    @staticmethod
    def _stub(module, url):
        cls = type("Chat", (), {})
        cls.__module__ = module
        obj = cls()
        obj.sdk_configuration = types.SimpleNamespace(get_server_details=lambda: (url, {}))
        return obj

    def test_direct(self):
        from dunetrace.auto import _mistral_deployment

        stub = self._stub("mistralai.client.chat", "https://api.mistral.ai")
        self.assertEqual(_mistral_deployment(stub), "direct")

    def test_azure_by_class_identity_beats_fallback_url(self):
        from dunetrace.auto import _mistral_deployment

        stub = self._stub("mistralai.azure.client.chat", "https://api.mistral.ai")
        self.assertEqual(_mistral_deployment(stub), "azure")

    def test_azure_by_hostname(self):
        from dunetrace.auto import _mistral_deployment

        stub = self._stub("mistralai.client.chat", "https://foo.inference.ai.azure.com")
        self.assertEqual(_mistral_deployment(stub), "azure")

    def test_gcp_by_class_identity(self):
        from dunetrace.auto import _mistral_deployment

        stub = self._stub("mistralai.gcp.client.chat", "https://api.mistral.ai")
        self.assertEqual(_mistral_deployment(stub), "gcp")

    def test_self_hosted(self):
        from dunetrace.auto import _mistral_deployment

        stub = self._stub("mistralai.client.chat", "http://localhost:8000")
        self.assertEqual(_mistral_deployment(stub), "self_hosted")

    def test_per_call_server_url_wins_over_client_default(self):
        from dunetrace.auto import _mistral_deployment

        stub = self._stub("mistralai.client.chat", "https://api.mistral.ai")
        self.assertEqual(_mistral_deployment(stub, "http://vllm.internal:8000"), "self_hosted")

    def test_unreadable_config_is_unknown_not_a_crash(self):
        from dunetrace.auto import _mistral_deployment

        cls = type("Chat", (), {})
        cls.__module__ = "mistralai.client.chat"
        self.assertEqual(_mistral_deployment(cls()), "unknown")


class TestMistralSdkPricing(unittest.TestCase):
    """The SDK price table feeds compute_run_cost, which feeds the cost_usd
    policy trigger. A missing model silently prices at _DEFAULT_PRICE."""

    def _cost(self, model, prompt, completion):
        from dunetrace.models import LlmCall
        from dunetrace.policies import compute_run_cost

        lc = LlmCall(
            model=model,
            prompt_tokens=prompt,
            finish_reason="stop",
            latency_ms=1,
            step_index=0,
            timestamp=0.0,
            completion_tokens=completion,
        )
        return compute_run_cost([lc])

    def test_current_mistral_rates(self):
        # USD for 1M input + 1M output, verified against
        # https://mistral.ai/pricing/api on 2026-08-08.
        cases = [
            ("mistral-large-latest", 2.00),
            ("mistral-large-2512", 2.00),
            ("mistral-medium-latest", 9.00),
            ("mistral-medium-3-5", 9.00),
            ("mistral-small-latest", 0.75),
            ("ministral-3b-latest", 0.20),
            ("ministral-8b-2512", 0.30),
            ("ministral-14b-latest", 0.40),
            ("codestral-2508", 1.20),
        ]
        for model, expected in cases:
            with self.subTest(model=model):
                self.assertAlmostEqual(self._cost(model, 1_000_000, 1_000_000), expected)

    def test_embeddings_bill_input_only(self):
        self.assertAlmostEqual(self._cost("mistral-embed", 1_000_000, 0), 0.10)
        self.assertAlmostEqual(self._cost("codestral-embed", 1_000_000, 0), 0.15)

    def test_codestral_embed_does_not_take_the_chat_rate(self):
        """_price_for walks the table in insertion order with a substring
        fallback, so codestral-embed has to be listed before codestral."""
        self.assertAlmostEqual(self._cost("codestral-embed", 1_000_000, 0), 0.15)

    def test_ministral_does_not_collide_with_mistral_small(self):
        self.assertAlmostEqual(self._cost("ministral-8b-latest", 1_000_000, 0), 0.15)

    def test_mistral_no_longer_falls_back_to_default_price(self):
        from dunetrace.policies import _DEFAULT_PRICE

        default = 1_000_000 * _DEFAULT_PRICE["input"]
        self.assertNotAlmostEqual(self._cost("mistral-large-latest", 1_000_000, 0), default)


if __name__ == "__main__":
    unittest.main(verbosity=2)
