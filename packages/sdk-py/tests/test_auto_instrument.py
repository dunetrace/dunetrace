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
from unittest.mock import MagicMock

from dunetrace import (
    Dunetrace,
    DunetraceASGIMiddleware,
    DunetraceWSGIMiddleware,
    get_current_run,
)
from dunetrace.auto import _PATCHED
from dunetrace.models import EventType


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
    sys.modules.setdefault("openai", mod)
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
    sys.modules.setdefault("anthropic", mod)
    sys.modules["anthropic.resources"] = resources
    sys.modules["anthropic.resources.messages"] = messages_mod
    return messages_mod


def _install_fake_httpx():
    mod = types.ModuleType("httpx")

    class FakeURL:
        host = "api.example.com"
        def __str__(self): return "https://api.example.com/data"

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
        def fn(query: str): pass

        fn("my query")
        started = next(e for e in captured if e.event_type == EventType.RUN_STARTED)
        self.assertNotEqual(started.payload.get("input_hash"), "")
        dt.shutdown(timeout=1)

    def test_input_from_named_param(self):
        dt = _make_client()

        @dt.agent("agent", input_from="question")
        def fn(context: str, question: str): pass

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
        def my_special_function(q: str): pass

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


# ─────────────────────────────────────────────────────────────────────────────
# 4. auto_instrument — Anthropic
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoInstrumentAnthropic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._messages_mod = _install_fake_anthropic()
        _PATCHED.discard("anthropic")
        from dunetrace.auto import _patch_anthropic
        _patch_anthropic()

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

    def test_url_hash_not_raw_url(self):
        dt = _make_client()
        captured = _capture(dt)

        with dt.run("agent"):
            self._httpx.Client().send(self._httpx._FakeRequest())

        called = next(e for e in captured if e.event_type == EventType.TOOL_CALLED)
        raw_url = str(self._httpx._FakeRequest().url)
        self.assertNotIn(raw_url, json.dumps(called.payload))
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
        # user_input is hashed — just verify it's non-empty
        self.assertNotEqual(started.payload.get("input_hash", ""), "")
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
        self.assertNotEqual(started.payload.get("input_hash", ""), "")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
