"""Client lifecycle and credential handling.

Two properties that are easy to regress and expensive to notice:

  * a client the caller merely drops must not leak its drain thread — anything
    building clients per tenant / per request / per test would otherwise
    accumulate one OS thread each for the life of the process;
  * the API key must never appear in a URL. Query strings are written verbatim
    to web-server and proxy access logs, so a key sent that way is disclosed on
    every request, to every hop, permanently.
"""

from __future__ import annotations

import gc
import threading
import time
import unittest
import urllib.request
import weakref
from unittest.mock import patch

from dunetrace.client import Dunetrace


def _drain_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "dunetrace-drain")


class TestDrainThreadLifecycle(unittest.TestCase):
    def test_dropped_client_does_not_leak_its_drain_thread(self):
        before = _drain_threads()

        def make_and_drop():
            clients = [Dunetrace(endpoint="http://127.0.0.1:9") for _ in range(10)]
            for c in clients:
                with c.run("agent") as run:
                    run.final_answer()
            del c  # the loop variable outlives the loop and would pin one client
            clients.clear()

        make_and_drop()
        gc.collect()

        # The threads park on a flush-interval wait, so give them a beat to
        # notice the referent is gone and return.
        for _ in range(40):
            if _drain_threads() <= before:
                break
            time.sleep(0.1)

        self.assertLessEqual(
            _drain_threads(),
            before,
            "drain threads survived their clients — the thread is holding a "
            "strong reference to the client again (see _drain_loop's docstring)",
        )

    def test_dropped_client_is_garbage_collectable(self):
        client = Dunetrace(endpoint="http://127.0.0.1:9")
        ref = weakref.ref(client)
        del client
        gc.collect()
        self.assertIsNone(ref(), "client was not collected — something still holds it strongly")

    def test_explicit_shutdown_still_stops_the_thread(self):
        before = _drain_threads()
        client = Dunetrace(endpoint="http://127.0.0.1:9")
        self.assertGreater(_drain_threads(), before)
        client.shutdown(timeout=2)
        self.assertLessEqual(_drain_threads(), before)

    def test_events_still_ship_after_the_weakref_refactor(self):
        """The loop re-acquires the client each iteration — make sure it still
        actually drains rather than silently shipping nothing."""
        shipped = []
        client = Dunetrace(endpoint="http://127.0.0.1:9")
        client._ship = lambda batch: (shipped.extend(batch), True)[1]
        with client.run("agent") as run:
            run.llm_called("gpt-4o", 5)
            run.final_answer()
        client.shutdown(timeout=2)
        self.assertTrue(shipped, "no events reached _ship")
        self.assertIn("run.completed", [e.event_type.value for e in shipped])


class TestApiKeyNeverInUrl(unittest.TestCase):
    def _capture_policy_request(self) -> urllib.request.Request:
        captured = {}

        class _Resp:
            def read(self):
                return b'{"policies": []}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _fake_urlopen(req, timeout=None):
            captured["req"] = req
            return _Resp()

        client = Dunetrace(endpoint="http://127.0.0.1:9", api_key="dt_live_secret")
        try:
            with patch("urllib.request.urlopen", _fake_urlopen):
                client._fetch_policies("agent-1")
        finally:
            client.shutdown(timeout=1)
        self.assertIn("req", captured, "policy fetch never issued a request")
        return captured["req"]

    def test_key_is_not_in_the_query_string(self):
        req = self._capture_policy_request()
        self.assertNotIn("dt_live_secret", req.full_url)
        self.assertNotIn("api_key", req.full_url)

    def test_key_is_sent_as_a_bearer_header(self):
        req = self._capture_policy_request()
        # urllib title-cases header keys it stores.
        headers = {k.lower(): v for k, v in req.header_items()}
        self.assertEqual(headers.get("authorization"), "Bearer dt_live_secret")

    def test_agent_id_is_still_in_the_query_string(self):
        req = self._capture_policy_request()
        self.assertIn("agent_id=agent-1", req.full_url)


if __name__ == "__main__":
    unittest.main()
