"""At-exit flush: a short-lived process must still ship its events.

The drain thread is a daemon, so the interpreter kills it during teardown. Without
an atexit hook a script that instruments a run and then exits ships *nothing* —
the events sit in the ring buffer until the process dies. These tests run real
subprocesses against a real HTTP receiver, because that is the only way to
observe interpreter shutdown behaviour honestly; an in-process test can assert
the hook is registered but not that it actually flushes on the way out.
"""

from __future__ import annotations

import atexit
import gc
import http.server
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest

from dunetrace.client import Dunetrace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Sink(http.server.BaseHTTPRequestHandler):
    events: list = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
            type(self).events.extend(body.get("events", []))
        except Exception:
            pass
        self.send_response(202)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"policies":[]}')

    def log_message(self, *args):
        pass


_SCRIPT = textwrap.dedent(
    """
    import sys, logging
    sys.path.insert(0, {root!r})
    logging.disable(logging.CRITICAL)
    from dunetrace import Dunetrace
    dt = Dunetrace(endpoint="http://127.0.0.1:{port}", api_key="k")
    with dt.run("agent", user_input="hi") as run:
        run.llm_called("gpt-4o", 10)
        run.llm_responded(completion_tokens=3)
        run.final_answer()
    {tail}
    """
)


class TestAtExitFlush(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Sink.events = []

    def _run_script(self, tail: str = "", env_extra: dict | None = None) -> list:
        src = _SCRIPT.format(root=_REPO_ROOT, port=self.port, tail=tail)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            path = fh.name
        env = dict(os.environ)
        env["PYTHONPATH"] = _REPO_ROOT
        if env_extra:
            env.update(env_extra)
        try:
            proc = subprocess.run([sys.executable, path], env=env, capture_output=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode()[:800])
        finally:
            os.unlink(path)
        return _Sink.events

    def test_short_lived_process_flushes_without_explicit_shutdown(self):
        events = self._run_script()
        types = [e.get("event_type") for e in events]
        self.assertIn("run.started", types)
        self.assertIn("run.completed", types)

    def test_explicit_shutdown_still_flushes_exactly_once(self):
        events = self._run_script(tail="dt.shutdown()")
        types = [e.get("event_type") for e in events]
        self.assertIn("run.completed", types)
        # The at-exit hook must have been unregistered — no duplicate delivery.
        self.assertEqual(types.count("run.completed"), 1, types)

    def test_atexit_flush_can_be_disabled(self):
        events = self._run_script(env_extra={"DUNETRACE_ATEXIT_TIMEOUT": "0"})
        self.assertEqual(events, [])


class TestAtExitRegistration(unittest.TestCase):
    """In-process checks that don't need a subprocess."""

    def test_shutdown_unregisters_the_hook_and_is_idempotent(self):
        dt = Dunetrace()
        hook = dt._atexit_hook
        self.assertIsNotNone(hook)
        dt.shutdown(timeout=1)
        self.assertIsNone(dt._atexit_hook)
        dt.shutdown(timeout=1)  # must not raise

    def test_hook_does_not_keep_the_client_alive(self):
        """A strong ref in the atexit registry would pin every client forever."""
        import weakref

        dt = Dunetrace()
        ref = weakref.ref(dt)
        hook = dt._atexit_hook
        self.assertIsNotNone(hook)
        # The closure must not hold the client strongly.
        self.assertNotIn(dt, getattr(hook, "__closure__", ()) or ())
        cells = [c.cell_contents for c in (hook.__closure__ or ())]
        self.assertTrue(
            any(isinstance(c, weakref.ref) for c in cells),
            "atexit hook should capture the client via weakref",
        )
        dt.shutdown(timeout=1)
        del dt, hook, cells
        gc.collect()
        # Now that the drain thread also holds only a weakref, nothing pins the
        # client and it must actually be collected.
        self.assertIsNone(ref(), "client was not collected — something holds it strongly")

    def test_disabled_when_timeout_is_zero(self):
        prev = os.environ.get("DUNETRACE_ATEXIT_TIMEOUT")
        os.environ["DUNETRACE_ATEXIT_TIMEOUT"] = "0"
        try:
            dt = Dunetrace()
            self.assertIsNone(dt._atexit_hook)
            dt.shutdown(timeout=1)
        finally:
            if prev is None:
                os.environ.pop("DUNETRACE_ATEXIT_TIMEOUT", None)
            else:
                os.environ["DUNETRACE_ATEXIT_TIMEOUT"] = prev

    def test_bad_timeout_env_falls_back_to_default(self):
        prev = os.environ.get("DUNETRACE_ATEXIT_TIMEOUT")
        os.environ["DUNETRACE_ATEXIT_TIMEOUT"] = "not-a-number"
        try:
            dt = Dunetrace()
            self.assertIsNotNone(dt._atexit_hook)
            dt.shutdown(timeout=1)
        finally:
            if prev is None:
                os.environ.pop("DUNETRACE_ATEXIT_TIMEOUT", None)
            else:
                os.environ["DUNETRACE_ATEXIT_TIMEOUT"] = prev


if __name__ == "__main__":
    unittest.main()
