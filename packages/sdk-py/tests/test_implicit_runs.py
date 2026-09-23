"""
Runs that start and end without dt.run(): the DSN, implicit runs and how they
close (idle, explicit run, process exit, crash), the crash hooks, and the
framework entry points that open exact runs.

No network: every client is built with endpoint=None and events are read off
the client's _emit.
"""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

from dunetrace import Dunetrace, get_current_run
from dunetrace import implicit as impl
from dunetrace.auto import _PATCHED, _patch_fastapi, _patch_flask, _patch_langgraph_entry
from dunetrace.context import _current_run
from dunetrace.models import EventType

# Snapshot LangGraph's pristine entry points at IMPORT time (pytest collects
# every module before running any test). Another module's unscoped
# dt.auto_instrument() patches Pregel for the whole process with its own
# client, so each test here restores these first and re-patches for its own.
try:
    from langgraph.pregel import Pregel as _PRISTINE_PREGEL

    _PRISTINE_PREGEL_METHODS = {
        name: _PRISTINE_PREGEL.__dict__.get(name)
        for name in ("invoke", "ainvoke", "stream", "astream")
    }
except ImportError:
    _PRISTINE_PREGEL = None
    _PRISTINE_PREGEL_METHODS = {}


def _restore_pristine_langgraph():
    if _PRISTINE_PREGEL is not None:
        for name, fn in _PRISTINE_PREGEL_METHODS.items():
            if fn is not None:
                setattr(_PRISTINE_PREGEL, name, fn)
    _PATCHED.discard("langgraph")


def _client(**kw) -> Dunetrace:
    return Dunetrace(endpoint=None, **kw)


def _capture(client: Dunetrace) -> list:
    captured: list = []
    original = client._emit

    def _emit(event):
        captured.append(event)
        original(event)

    client._emit = _emit
    return captured


def _by_type(events, event_type: EventType):
    return [e for e in events if e.event_type == event_type]


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        impl._reset_for_tests()
        self._token = _current_run.set(None)
        self._env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("DUNETRACE_")}

    def tearDown(self) -> None:
        _current_run.reset(self._token)
        for k in list(os.environ):
            if k.startswith("DUNETRACE_"):
                del os.environ[k]
        os.environ.update(self._env)
        impl._reset_for_tests()


# ── DSN ───────────────────────────────────────────────────────────────────────


class TestDsn(_Base):
    def test_parse_splits_endpoint_and_key(self):
        self.assertEqual(
            impl.parse_dsn("https://dt_live_abc@ingest.example.com"),
            ("https://ingest.example.com", "dt_live_abc"),
        )
        self.assertEqual(
            impl.parse_dsn("http://k%40ey@localhost:8001/base/"),
            ("http://localhost:8001/base", "k@ey"),
        )

    def test_parse_rejects_missing_key_or_bad_scheme(self):
        for bad in ("https://ingest.example.com", "ftp://k@host", "not a url", "https://@host"):
            with self.assertRaises(ValueError):
                impl.parse_dsn(bad)

    def test_host_only_for_logs(self):
        self.assertEqual(impl.dsn_host("https://secret@ingest.example.com/x"), "ingest.example.com")

    def test_client_reads_dsn_from_env(self):
        os.environ["DUNETRACE_DSN"] = "https://dt_live_env@ingest.example.com"
        c = Dunetrace()
        try:
            self.assertEqual(c._ingest_url, "https://ingest.example.com/v1/ingest")
            self.assertEqual(c._api_key, "dt_live_env")
        finally:
            c.shutdown(timeout=1)

    def test_explicit_settings_win_over_dsn(self):
        c = Dunetrace(
            dsn="https://dt_live_dsn@ingest.example.com",
            endpoint="http://localhost:9001",
            api_key="k2",
        )
        try:
            self.assertEqual(c._ingest_url, "http://localhost:9001/v1/ingest")
            self.assertEqual(c._api_key, "k2")
        finally:
            c.shutdown(timeout=1)

    def test_malformed_dsn_is_ignored_with_defaults_kept(self):
        c = Dunetrace(dsn="https://ingest.example.com")  # no key
        try:
            self.assertEqual(c._ingest_url, "http://localhost:8001/v1/ingest")
            self.assertEqual(c._api_key, "")
        finally:
            c.shutdown(timeout=1)


# ── Implicit runs ─────────────────────────────────────────────────────────────


class TestImplicitRuns(_Base):
    def test_first_patched_call_opens_a_marked_run_and_later_calls_attach(self):
        c = _client()
        events = _capture(c)
        try:
            run = c._resolve_run("openai.chat.completions.create")
            self.assertIsNotNone(run)
            self.assertTrue(run.implicit)
            self.assertIs(get_current_run(), run)
            started = _by_type(events, EventType.RUN_STARTED)
            self.assertEqual(len(started), 1)
            self.assertTrue(started[0].payload["implicit"])
            self.assertEqual(started[0].payload["opened_by"], "openai.chat.completions.create")
            # The next call in the same context attaches instead of opening another.
            self.assertIs(c._resolve_run("anthropic.messages.create"), run)
            self.assertEqual(len(_by_type(events, EventType.RUN_STARTED)), 1)
        finally:
            c.shutdown(timeout=1)

    def test_agent_id_falls_back_to_init_then_env_then_script_name(self):
        c = _client()
        events = _capture(c)
        try:
            c._resolve_run("x")
            self.assertEqual(events[0].agent_id, os.path.basename(sys.argv[0]) or "implicit")
            c._implicit.close_all("idle")
            os.environ["DUNETRACE_AGENT_ID"] = "from-env"
            _current_run.set(None)
            c._resolve_run("x")
            self.assertEqual(events[-1].agent_id, "from-env")
            c._implicit.close_all("idle")
            c._default_agent_id = "from-init"
            _current_run.set(None)
            c._resolve_run("x")
            self.assertEqual(events[-1].agent_id, "from-init")
        finally:
            c.shutdown(timeout=1)

    def test_idle_closes_the_run_with_exit_reason_idle(self):
        c = _client(implicit_run_idle_s=5)
        events = _capture(c)
        try:
            run = c._resolve_run("x")
            # run.started is emitted by the client, not stored on the run's own
            # state, so with no events yet idleness counts from the open.
            opened = c._implicit._open[run.run_id].opened_at
            self.assertEqual(c._implicit.sweep(now=opened + 4), 0)
            self.assertEqual(c._implicit.sweep(now=opened + 5), 1)
            done = _by_type(events, EventType.RUN_COMPLETED)
            self.assertEqual(len(done), 1)
            self.assertEqual(done[0].payload["exit_reason"], "idle")
            self.assertTrue(run._implicit_closed)
            # The stale context variable reads as no run: a new call opens a new one.
            again = c._resolve_run("x")
            self.assertIsNot(again, run)
            self.assertEqual(len(_by_type(events, EventType.RUN_STARTED)), 2)
        finally:
            c.shutdown(timeout=1)

    def test_idle_is_measured_from_the_last_event_not_the_open(self):
        c = _client(implicit_run_idle_s=5)
        try:
            run = c._resolve_run("x")
            t0 = c._implicit._open[run.run_id].opened_at
            run.llm_called("gpt-4o", prompt_tokens=1)
            run.state.events[-1].timestamp = t0 + 4  # activity just before the deadline
            self.assertEqual(c._implicit.sweep(now=t0 + 6), 0)
            self.assertEqual(c._implicit.sweep(now=t0 + 9), 1)
        finally:
            c.shutdown(timeout=1)

    def test_explicit_run_closes_the_implicit_one_and_does_not_nest_under_it(self):
        c = _client()
        events = _capture(c)
        try:
            implicit_run = c._resolve_run("x")
            with c.run("real-agent") as run:
                self.assertFalse(run.implicit)
                self.assertIsNone(events[-1].parent_run_id)
            done = _by_type(events, EventType.RUN_COMPLETED)
            self.assertEqual([e.run_id for e in done], [implicit_run.run_id, run.run_id])
            self.assertEqual(done[0].payload["exit_reason"], "explicit_run_opened")
        finally:
            c.shutdown(timeout=1)

    def test_shutdown_closes_open_implicit_runs_with_process_exit(self):
        c = _client()
        events = _capture(c)
        run = c._resolve_run("x")
        c.shutdown(timeout=1)
        done = _by_type(events, EventType.RUN_COMPLETED)
        self.assertEqual([e.run_id for e in done], [run.run_id])
        self.assertEqual(done[0].payload["exit_reason"], "process_exit")
        self.assertEqual(c._implicit.open_count(), 0)

    def test_disabled_by_argument_or_env(self):
        c = _client(implicit_runs=False)
        try:
            self.assertIsNone(c._resolve_run("x"))
        finally:
            c.shutdown(timeout=1)
        os.environ["DUNETRACE_IMPLICIT_RUNS"] = "0"
        c = _client()
        try:
            self.assertIsNone(c._resolve_run("x"))
        finally:
            c.shutdown(timeout=1)

    def test_bare_resolve_without_a_client_stays_attach_only(self):
        self.assertIsNone(impl.resolve_run(None, "x"))
        c = _client()
        try:
            with c.run("a") as run:
                self.assertIs(impl.resolve_run(None, "x"), run)
        finally:
            c.shutdown(timeout=1)

    def test_closing_from_another_thread_is_safe(self):
        """The idle sweep runs on its own thread; closing there must emit the
        terminal event and leave the opening context usable."""
        c = _client()
        events = _capture(c)
        try:
            run = c._resolve_run("x")
            t = threading.Thread(target=lambda: c._implicit.close(run.run_id, "idle"))
            t.start()
            t.join(2)
            self.assertEqual(len(_by_type(events, EventType.RUN_COMPLETED)), 1)
            self.assertIsNotNone(c._resolve_run("x"))
        finally:
            c.shutdown(timeout=1)


# ── Crash hooks ───────────────────────────────────────────────────────────────


class TestCrashHooks(_Base):
    def test_unhandled_exception_errors_the_implicit_run(self):
        c = _client()
        events = _capture(c)
        impl.install_crash_hooks(c)
        try:
            run = c._resolve_run("x")
            exc = RuntimeError("boom")
            self.assertEqual(impl.record_crash(exc, "sys.excepthook"), 1)
            errored = _by_type(events, EventType.RUN_ERRORED)
            self.assertEqual([e.run_id for e in errored], [run.run_id])
            self.assertEqual(errored[0].payload["error_type"], "RuntimeError")
            self.assertEqual(errored[0].payload["error"], "boom")
            # Recorded once: the same exception reaching another hook is ignored.
            self.assertEqual(impl.record_crash(exc, "threading.excepthook"), 0)
        finally:
            c.shutdown(timeout=1)

    def test_crash_with_no_run_open_records_a_one_event_run(self):
        c = _client()
        events = _capture(c)
        impl.install_crash_hooks(c)
        try:
            self.assertEqual(impl.record_crash(ValueError("bad"), "sys.excepthook"), 1)
            self.assertEqual(
                [e.event_type for e in events], [EventType.RUN_STARTED, EventType.RUN_ERRORED]
            )
            self.assertEqual(events[0].payload["opened_by"], "sys.excepthook")
            self.assertTrue(events[0].payload["implicit"])
        finally:
            c.shutdown(timeout=1)

    def test_exception_already_recorded_by_dt_run_is_not_recorded_twice(self):
        c = _client()
        events = _capture(c)
        impl.install_crash_hooks(c)
        try:
            with self.assertRaises(RuntimeError):
                with c.run("a"):
                    raise RuntimeError("inside")
            # dt.run() emitted run.errored and marked the exception.
            self.assertEqual(len(_by_type(events, EventType.RUN_ERRORED)), 1)
            exc = None
            try:
                with c.run("b"):
                    raise RuntimeError("again")
            except RuntimeError as e:
                exc = e
            self.assertEqual(impl.record_crash(exc, "sys.excepthook"), 0)
            self.assertEqual(len(_by_type(events, EventType.RUN_ERRORED)), 2)
        finally:
            c.shutdown(timeout=1)

    def test_hooks_chain_to_the_previous_ones_and_install_once(self):
        seen = []
        previous = sys.excepthook
        sys.excepthook = lambda t, v, tb: seen.append(v)
        try:
            c = _client()
            impl.install_crash_hooks(c)
            hook = sys.excepthook
            impl.install_crash_hooks(c)
            self.assertIs(sys.excepthook, hook, "installed once per process")
            exc = RuntimeError("top")
            sys.excepthook(RuntimeError, exc, None)
            self.assertEqual(seen, [exc])
            c.shutdown(timeout=1)
        finally:
            sys.excepthook = previous

    def test_init_installs_hooks_only_when_implicit_runs_are_enabled(self):
        previous = sys.excepthook
        try:
            c = _client(implicit_runs=False)
            c.init(frameworks=[])
            self.assertIs(sys.excepthook, previous)
            c.shutdown(timeout=1)
            c = _client()
            c.init(frameworks=[])
            self.assertIsNot(sys.excepthook, previous)
            c.shutdown(timeout=1)
        finally:
            sys.excepthook = previous


# ── Framework entry points ────────────────────────────────────────────────────


class TestLangGraphEntryPoint(_Base):
    @classmethod
    def setUpClass(cls):
        if _PRISTINE_PREGEL is None:
            raise unittest.SkipTest("langgraph not installed")

    def setUp(self):
        super().setUp()
        _restore_pristine_langgraph()

    def tearDown(self):
        super().tearDown()
        _restore_pristine_langgraph()

    def _graph(self, name=None):
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            n: int

        g = StateGraph(S)
        g.add_node("bump", lambda s: {"n": s["n"] + 1})
        g.add_edge(START, "bump")
        g.add_edge("bump", END)
        return g.compile(name=name) if name else g.compile()

    def test_invoke_opens_an_exact_run_and_closes_it_on_return(self):
        c = _client()
        events = _capture(c)
        try:
            c.init(agent_id="graph-agent", frameworks=["langgraph"])
            out = self._graph().invoke({"n": 1})
            self.assertEqual(out["n"], 2)
            types_ = [e.event_type for e in events]
            self.assertEqual(types_[0], EventType.RUN_STARTED)
            self.assertEqual(types_[-1], EventType.RUN_COMPLETED)
            self.assertEqual(events[0].agent_id, "graph-agent")
            self.assertEqual(events[0].payload["opened_by"], "langgraph.invoke")
            self.assertNotIn(
                "implicit", events[0].payload, "an entry-point run is exact, not guessed"
            )
            self.assertEqual(events[-1].payload["exit_reason"], "final_answer")
            self.assertIsNone(get_current_run())
        finally:
            c.shutdown(timeout=1)

    def test_a_named_graph_uses_its_name_and_a_nested_invoke_attaches(self):
        c = _client()
        events = _capture(c)
        try:
            c.init(frameworks=["langgraph"])
            outer = self._graph(name="planner")
            inner = self._graph()
            with c.run("declared") as run:
                inner.invoke({"n": 0})  # a run is active: attaches, opens nothing
                self.assertIs(get_current_run(), run)
            outer.invoke({"n": 0})
            started = _by_type(events, EventType.RUN_STARTED)
            self.assertEqual([e.agent_id for e in started], ["declared", "planner"])
        finally:
            c.shutdown(timeout=1)

    def test_stream_closes_the_run_when_the_stream_is_exhausted(self):
        c = _client()
        events = _capture(c)
        try:
            c.init(frameworks=["langgraph"])
            chunks = list(self._graph().stream({"n": 1}))
            self.assertTrue(chunks)
            self.assertEqual(events[-1].event_type, EventType.RUN_COMPLETED)
        finally:
            c.shutdown(timeout=1)


class _FakeModule(types.ModuleType):
    pass


class TestWebFrameworkMiddleware(_Base):
    """FastAPI and Flask are not test dependencies, so the constructor patches
    are exercised against minimal stand-ins registered in sys.modules."""

    def tearDown(self):
        super().tearDown()
        for name in ("fastapi", "flask"):
            sys.modules.pop(name, None)
            _PATCHED.discard(name)

    def test_fastapi_apps_get_the_asgi_middleware(self):
        from dunetrace.middleware import DunetraceASGIMiddleware

        mod = _FakeModule("fastapi")

        class FastAPI:
            def __init__(self, title="API"):
                self.title = title
                self.added = []

            def add_middleware(self, cls, **options):
                self.added.append((cls, options))

        mod.FastAPI = FastAPI
        sys.modules["fastapi"] = mod
        c = _client()
        try:
            _patch_fastapi(client=c, default_agent_id="")
            app = FastAPI(title="billing-api")
            self.assertEqual(len(app.added), 1)
            cls, options = app.added[0]
            self.assertIs(cls, DunetraceASGIMiddleware)
            self.assertIs(options["dt"], c)
            self.assertEqual(options["agent_id"], "billing-api")
        finally:
            c.shutdown(timeout=1)

    def test_flask_apps_get_the_wsgi_middleware(self):
        from dunetrace.middleware import DunetraceWSGIMiddleware

        mod = _FakeModule("flask")

        class Flask:
            def __init__(self, name):
                self.name = name
                self.wsgi_app = lambda environ, start_response: [b"ok"]

        mod.Flask = Flask
        sys.modules["flask"] = mod
        c = _client()
        try:
            _patch_flask(client=c, default_agent_id="")
            app = Flask("shop")
            self.assertIsInstance(app.wsgi_app, DunetraceWSGIMiddleware)
        finally:
            c.shutdown(timeout=1)

    def test_no_client_means_no_patch(self):
        mod = _FakeModule("fastapi")
        calls = []

        class FastAPI:
            def __init__(self):
                calls.append(1)

            def add_middleware(self, *a, **k):
                raise AssertionError("must not be called")

        mod.FastAPI = FastAPI
        sys.modules["fastapi"] = mod
        _patch_fastapi(client=None)
        FastAPI()
        self.assertEqual(calls, [1])
        self.assertNotIn("fastapi", _PATCHED)


if __name__ == "__main__":
    unittest.main()
