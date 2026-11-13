"""
Remote policy fetch resilience (hardening item 1.4).

A failed remote fetch used to be marked as fetched *before* the request went
out, so it inherited the 60s success TTL and logged only at DEBUG: an agent
could run unguarded for a minute with nothing visible anywhere. Now:

  * mark_fetched() is called only on success; begin_fetch()/end_fetch() are
    the stampede guard instead.
  * a failure is retried on an exponential backoff (2s, 4s, 8s, cap 15s),
    never after the TTL.
  * the first failure in a streak logs WARNING, later ones DEBUG, and the
    success that ends a streak INFO ("recovered after N failures").
  * the last loaded bundle stays enforced through failures (fail-open).
  * policy.evaluated carries policy_bundle_stale / policy_bundle_age_s.
  * an opt-in on-disk cache (policy_cache_path) primes a fresh process through
    PolicyEngine.load(), so a tampered cache cannot inject an enforcing policy.

Fully offline: urllib.request.urlopen is patched throughout.

Run: python -m pytest tests/test_policy_fetch_resilience.py -v
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from dunetrace import Dunetrace
from dunetrace.models import EventType
from dunetrace.policies import PolicyEngine, eval_logger

# Never contacted — urlopen is patched in every test that reaches it. The host
# is asserted on in log messages.
ENDPOINT = "http://policy-host.invalid:9"
HOST = "policy-host.invalid:9"


def _client(**kw) -> Dunetrace:
    kw.setdefault("endpoint", ENDPOINT)
    kw.setdefault("api_key", "dt_live_test")
    c = Dunetrace(**kw)
    c._ship = lambda b: None  # keep the drain thread off the network
    return c


def _response(body):
    """A urlopen() context manager yielding a response whose read() is body."""
    resp = MagicMock()
    resp.read.return_value = body if isinstance(body, bytes) else json.dumps(body).encode()
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


def _remote(policy_id: int, action: str = "log", name: str | None = None, signature=None):
    p = {
        "id": policy_id,
        "name": name or f"remote-{policy_id}",
        "agent_id": "*",
        "condition": {"trigger": "tool_call_count", "operator": "gt", "value": 3},
        "action": {"type": action},
        "enabled": True,
        "priority": 100,
    }
    if signature is not None:
        p["signature"] = signature
    return p


def _fetch_failure_records(records):
    return [r for r in records if "remote policy fetch failed" in r.getMessage()]


# ── Engine bookkeeping ─────────────────────────────────────────────────────────


class TestEngineBackoff(unittest.TestCase):
    def test_failure_never_starts_the_success_ttl(self):
        engine = PolicyEngine()
        engine.mark_fetch_failed("a")
        self.assertNotIn("a", engine._fetch_times)
        self.assertEqual(engine.fetch_failure_streak("a"), 1)

    def test_needs_fetch_again_after_backoff_not_after_ttl(self):
        engine = PolicyEngine()
        with patch("dunetrace.policies.time.monotonic") as mono:
            mono.return_value = 1000.0
            engine.mark_fetch_failed("a")
            self.assertFalse(engine.needs_fetch("a"), "no immediate retry")
            mono.return_value = 1001.9
            self.assertFalse(engine.needs_fetch("a"))
            mono.return_value = 1002.0  # first backoff is 2s
            self.assertTrue(engine.needs_fetch("a"))
            # Nowhere near the 60s TTL — that is the point.
            self.assertLess(mono.return_value - 1000.0, PolicyEngine._FETCH_TTL / 10)

    def test_backoff_doubles_and_caps(self):
        self.assertEqual(
            [PolicyEngine.fetch_backoff(n) for n in range(1, 7)],
            [2.0, 4.0, 8.0, 15.0, 15.0, 15.0],
        )
        engine = PolicyEngine()
        with patch("dunetrace.policies.time.monotonic") as mono:
            now = 0.0
            for expected in (2.0, 4.0, 8.0, 15.0, 15.0):
                mono.return_value = now
                engine.mark_fetch_failed("a")
                mono.return_value = now + expected - 0.01
                self.assertFalse(engine.needs_fetch("a"), f"retried early at backoff {expected}")
                mono.return_value = now + expected
                self.assertTrue(engine.needs_fetch("a"), f"not retried at backoff {expected}")
                now = now + expected + 1.0

    def test_backoff_cap_is_well_below_the_ttl(self):
        self.assertLessEqual(PolicyEngine._FETCH_BACKOFF_MAX, PolicyEngine._FETCH_TTL / 4)

    def test_success_clears_the_streak_and_starts_the_ttl(self):
        engine = PolicyEngine()
        engine.mark_fetch_failed("a")
        engine.mark_fetch_failed("a")
        self.assertEqual(engine.fetch_failure_streak("a"), 2)
        engine.mark_fetched("a")
        self.assertEqual(engine.fetch_failure_streak("a"), 0)
        self.assertFalse(engine.needs_fetch("a"))
        self.assertIn("a", engine._fetch_times)

    def test_failure_after_a_success_retries_on_backoff(self):
        """A failed *refresh* must not wait out the TTL either."""
        engine = PolicyEngine()
        with patch("dunetrace.policies.time.monotonic") as mono:
            mono.return_value = 100.0
            engine.mark_fetched("a")
            mono.return_value = 170.0  # TTL expired → refresh attempted → fails
            engine.mark_fetch_failed("a")
            self.assertFalse(engine.needs_fetch("a"))
            mono.return_value = 172.0
            self.assertTrue(engine.needs_fetch("a"))

    def test_streak_is_per_agent(self):
        engine = PolicyEngine()
        engine.mark_fetch_failed("a")
        self.assertEqual(engine.fetch_failure_streak("b"), 0)
        self.assertTrue(engine.needs_fetch("b"))
        self.assertFalse(engine.needs_fetch("a"))

    def test_in_flight_guard(self):
        engine = PolicyEngine()
        self.assertTrue(engine.begin_fetch("a"))
        self.assertFalse(engine.begin_fetch("a"), "second claim must be refused")
        self.assertFalse(engine.needs_fetch("a"), "not due while on the wire")
        self.assertTrue(engine.begin_fetch("b"), "guard is per agent")
        engine.end_fetch("a")
        self.assertTrue(engine.needs_fetch("a"))
        self.assertTrue(engine.begin_fetch("a"))

    def test_bundle_status(self):
        engine = PolicyEngine()
        self.assertEqual(engine.bundle_status("a", remote_configured=False), (False, None))
        self.assertEqual(engine.bundle_status("a", remote_configured=True), (True, None))
        engine.mark_fetched("a")
        stale, age = engine.bundle_status("a", remote_configured=True)
        self.assertFalse(stale)
        self.assertGreaterEqual(age, 0.0)
        self.assertLess(age, 5.0)
        engine._fetch_times["a"] = time.monotonic() - (PolicyEngine._FETCH_TTL + 1)
        stale, age = engine.bundle_status("a", remote_configured=True)
        self.assertTrue(stale)
        self.assertGreater(age, PolicyEngine._FETCH_TTL)


# ── Client fetch path ──────────────────────────────────────────────────────────


class TestClientFetchFailure(unittest.TestCase):
    def setUp(self):
        self.c = _client()
        self.engine = self.c._policy_engine

    def tearDown(self):
        self.c.shutdown(timeout=1)

    def test_failure_does_not_mark_fetched_and_retries_after_backoff(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with patch("dunetrace.policies.time.monotonic") as mono:
                mono.return_value = 100.0
                self.c._fetch_policies("agent-1")
                self.assertNotIn("agent-1", self.engine._fetch_times)
                self.assertEqual(self.engine.fetch_failure_streak("agent-1"), 1)
                self.assertFalse(self.engine.needs_fetch("agent-1"))
                mono.return_value = 102.0
                self.assertTrue(self.engine.needs_fetch("agent-1"))
        self.assertEqual(self.engine._fetch_in_flight, set(), "guard released after failure")

    def test_malformed_body_is_a_failure(self):
        with patch("urllib.request.urlopen", return_value=_response(b"<html>oops</html>")):
            self.c._fetch_policies("agent-1")
        self.assertEqual(self.engine.fetch_failure_streak("agent-1"), 1)
        self.assertNotIn("agent-1", self.engine._fetch_times)

    def test_non_object_body_is_a_failure(self):
        with patch("urllib.request.urlopen", return_value=_response([1, 2, 3])):
            self.c._fetch_policies("agent-1")
        self.assertEqual(self.engine.fetch_failure_streak("agent-1"), 1)

    def test_first_failure_warns_then_debug_then_info_on_recovery(self):
        err = urllib.error.HTTPError(ENDPOINT, 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertLogs("dunetrace", level="DEBUG") as first:
                self.c._fetch_policies("agent-1")
            recs = _fetch_failure_records(first.records)
            self.assertEqual([r.levelno for r in recs], [logging.WARNING])
            msg = recs[0].getMessage()
            self.assertIn("'agent-1'", msg)
            self.assertIn(HOST, msg)
            self.assertIn("HTTPError", msg)
            self.assertIn("503", msg)
            self.assertIn("retrying in 2s", msg)

            # Same streak, backoff elapsed → DEBUG only.
            self.engine._fetch_failures["agent-1"] = (time.monotonic() - 100, 1)
            with self.assertLogs("dunetrace", level="DEBUG") as second:
                self.c._fetch_policies("agent-1")
            recs = _fetch_failure_records(second.records)
            self.assertEqual([r.levelno for r in recs], [logging.DEBUG])
            self.assertIn("failure 2 in a row", recs[0].getMessage())
            self.assertIn("retrying in 4s", recs[0].getMessage())

        # Recovery → INFO naming the streak length, streak cleared, TTL started.
        self.engine._fetch_failures["agent-1"] = (time.monotonic() - 100, 2)
        with patch("urllib.request.urlopen", return_value=_response({"policies": []})):
            with self.assertLogs("dunetrace", level="INFO") as third:
                self.c._fetch_policies("agent-1")
        recovered = [r for r in third.records if "recovered after 2 failure" in r.getMessage()]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].levelno, logging.INFO)
        self.assertIn(HOST, recovered[0].getMessage())
        self.assertEqual(self.engine.fetch_failure_streak("agent-1"), 0)
        self.assertIn("agent-1", self.engine._fetch_times)
        self.assertFalse(self.engine.needs_fetch("agent-1"))

    def test_success_without_a_streak_logs_no_recovery(self):
        with patch("urllib.request.urlopen", return_value=_response({"policies": []})):
            with self.assertLogs("dunetrace", level="DEBUG") as logs:
                self.c._fetch_policies("agent-1")
        self.assertFalse(any("recovered" in r.getMessage() for r in logs.records))
        self.assertFalse(any(r.levelno > logging.DEBUG for r in logs.records))

    def test_in_flight_guard_blocks_a_second_concurrent_fetch(self):
        release = threading.Event()
        calls: list = []

        def slow_urlopen(req, timeout=None):
            calls.append(req.full_url)
            release.wait(2)
            return _response({"policies": []})

        with patch("urllib.request.urlopen", slow_urlopen):
            t1 = threading.Thread(target=self.c._fetch_policies, args=("agent-1",))
            t1.start()
            for _ in range(400):  # wait for the first request to be on the wire
                if calls:
                    break
                time.sleep(0.005)
            self.assertEqual(len(calls), 1)
            self.assertFalse(self.engine.needs_fetch("agent-1"), "in flight → not due")
            self.c._fetch_policies("agent-1")  # must return without a second request
            self.assertEqual(len(calls), 1)
            release.set()
            t1.join(2)
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.engine.needs_fetch("agent-1"), "success TTL now in effect")
        self.assertEqual(self.engine._fetch_in_flight, set())

    def test_last_bundle_retained_after_failed_refresh(self):
        with patch("urllib.request.urlopen", return_value=_response({"policies": [_remote(1)]})):
            self.c._fetch_policies("agent-1")
        self.assertEqual(len(self.engine), 1)
        self.engine._fetch_times["agent-1"] = time.monotonic() - (PolicyEngine._FETCH_TTL + 1)
        self.assertTrue(self.engine.needs_fetch("agent-1"))
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                self.c._fetch_policies("agent-1")
        self.assertEqual(len(self.engine), 1, "fail-open: last bundle still enforced")
        self.assertEqual(self.engine.fetch_failure_streak("agent-1"), 1)
        self.assertIn("still enforcing the last loaded bundle", logs.output[0])

    def test_failure_with_no_bundle_says_so(self):
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                self.c._fetch_policies("agent-1")
        self.assertIn("no remote policies are loaded", logs.output[0])

    def test_run_start_does_not_spawn_a_fetch_during_backoff(self):
        self.engine.mark_fetch_failed("billing")  # backoff just started
        self.c._detector_config_store.mark_fetched("billing")  # config fresh: nothing else due
        with patch.object(self.c, "_fetch_policies") as fetch:
            with self.c.run("billing"):
                pass
            time.sleep(0.05)  # the shared fetch thread, had one started, would have run
        fetch.assert_not_called()


# ── Staleness on the wire ──────────────────────────────────────────────────────


class _CapturingExporter:
    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)


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


class TestStalenessOnTheWire(unittest.TestCase):
    def _reporting_client(self, **kw) -> tuple:
        exp = _CapturingExporter()
        c = _client(policy_evaluation_reporting=True, exporters=[exp], **kw)
        # Keep the background prefetch (policies + detector config, one
        # thread) off the wire; the engine state under test is set directly.
        c._fetch_remote_config = lambda *a, **k: None  # type: ignore[method-assign]
        return c, exp

    def _evaluate(self, c, exp) -> dict:
        c.add_policy(**REFUND_POLICY)
        try:
            with c.run("billing") as run:
                run.request_approval = lambda *a, **k: None
                run.tool_called("refund", {"amount": 25000})
        finally:
            c.shutdown(timeout=1)
        evals = [e for e in exp.events if e.event_type == EventType.POLICY_EVALUATED]
        self.assertTrue(evals, "reporting enabled but nothing shipped")
        return evals[-1].payload

    def test_never_fetched_is_stale_with_null_age(self):
        c, exp = self._reporting_client()
        payload = self._evaluate(c, exp)
        self.assertTrue(payload["policy_bundle_stale"])
        self.assertIsNone(payload["policy_bundle_age_s"])

    def test_fresh_bundle_is_not_stale_and_carries_age(self):
        c, exp = self._reporting_client()
        c._policy_engine.mark_fetched("billing")
        payload = self._evaluate(c, exp)
        self.assertFalse(payload["policy_bundle_stale"])
        self.assertIsInstance(payload["policy_bundle_age_s"], float)
        self.assertGreaterEqual(payload["policy_bundle_age_s"], 0.0)
        self.assertLess(payload["policy_bundle_age_s"], 5.0)

    def test_overdue_bundle_is_stale(self):
        c, exp = self._reporting_client()
        c._policy_engine._fetch_times["billing"] = time.monotonic() - (PolicyEngine._FETCH_TTL + 10)
        payload = self._evaluate(c, exp)
        self.assertTrue(payload["policy_bundle_stale"])
        self.assertGreater(payload["policy_bundle_age_s"], PolicyEngine._FETCH_TTL)

    def test_local_only_client_is_never_stale(self):
        with patch.dict(os.environ, {"DUNETRACE_API_KEY": ""}):
            c, exp = self._reporting_client(api_key="")
        payload = self._evaluate(c, exp)
        self.assertFalse(payload["policy_bundle_stale"])
        self.assertIsNone(payload["policy_bundle_age_s"])

    def test_default_wire_format_is_untouched(self):
        """Only the opt-in policy.evaluated payload grew. With reporting off,
        no policy.evaluated is shipped and no run event carries the fields."""
        eval_logger.setLevel(logging.WARNING)
        exp = _CapturingExporter()
        c = _client(policy_evaluation_reporting=False, exporters=[exp])
        c._fetch_remote_config = lambda *a, **k: None  # type: ignore[method-assign]
        c.add_policy(**REFUND_POLICY)
        try:
            with c.run("billing") as run:
                run.request_approval = lambda *a, **k: None
                run.tool_called("refund", {"amount": 25000})
        finally:
            c.shutdown(timeout=1)
        self.assertTrue(exp.events)
        for e in exp.events:
            self.assertNotEqual(e.event_type, EventType.POLICY_EVALUATED)
            self.assertNotIn("policy_bundle_stale", e.payload)
            self.assertNotIn("policy_bundle_age_s", e.payload)


# ── Optional on-disk cache ─────────────────────────────────────────────────────


class TestPolicyCache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="dt-policy-cache-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_off_by_default(self):
        with patch.dict(os.environ, {"DUNETRACE_POLICY_CACHE_PATH": ""}):
            c = _client()
        try:
            self.assertEqual(c._policy_cache_path, "")
            with patch(
                "urllib.request.urlopen", return_value=_response({"policies": [_remote(1)]})
            ):
                c._fetch_policies("agent-1")
        finally:
            c.shutdown(timeout=1)
        self.assertEqual(os.listdir(self.dir), [])

    def test_env_var_configures_path(self):
        with patch.dict(os.environ, {"DUNETRACE_POLICY_CACHE_PATH": self.dir}):
            c = _client()
        c.shutdown(timeout=1)
        self.assertEqual(c._policy_cache_path, self.dir)

    def test_round_trip_primes_a_fresh_process_and_downgrades_unsigned_enforcing(self):
        body = {
            "policies": [
                _remote(1, "log", name="remote-log"),
                _remote(2, "stop", name="remote-stop"),
            ]
        }
        c1 = _client(policy_cache_path=self.dir)
        try:
            with patch("urllib.request.urlopen", return_value=_response(body)):
                c1._fetch_policies("agent-1")
        finally:
            c1.shutdown(timeout=1)
        path = c1._policy_cache_file("agent-1")
        self.assertTrue(os.path.exists(path))
        self.assertEqual([n for n in os.listdir(self.dir) if ".tmp-" in n], [], "atomic write")
        with open(path, encoding="utf-8") as fh:
            cached = json.load(fh)
        self.assertEqual(cached["agent_id"], "agent-1")
        # The raw server JSON is what is cached; the downgrade is applied at load.
        self.assertEqual(cached["policies"][1]["action"]["type"], "stop")

        # Fresh process, server down: primed from cache through load().
        c2 = _client(policy_cache_path=self.dir)
        engine = c2._policy_engine
        try:
            with patch("urllib.request.urlopen", side_effect=OSError("server down")):
                with self.assertLogs("dunetrace", level="INFO") as logs:
                    c2._fetch_policies("agent-1")
        finally:
            c2.shutdown(timeout=1)
        by_name = {p.name: p for p in engine._policies}
        self.assertIn("remote-log", by_name)
        self.assertIn("remote-stop", by_name)
        self.assertEqual(
            by_name["remote-stop"].action["type"],
            "log",
            "an unsigned enforcing policy from the cache is downgraded exactly like a live one",
        )
        self.assertTrue(any("primed 2 cached" in r.getMessage() for r in logs.records))
        # Stale until a real fetch succeeds, and retried on the backoff.
        self.assertEqual(engine.bundle_status("agent-1", True), (True, None))
        self.assertNotIn("agent-1", engine._fetch_times)
        self.assertEqual(engine.fetch_failure_streak("agent-1"), 1)

    def test_tampered_cache_cannot_inject_an_enforcing_policy_when_a_secret_is_set(self):
        c = _client(policy_cache_path=self.dir, policy_secret="shared-secret")
        with open(c._policy_cache_file("agent-1"), "w", encoding="utf-8") as fh:
            json.dump(
                {"agent_id": "agent-1", "policies": [_remote(9, "stop", signature="deadbeef")]},
                fh,
            )
        try:
            with patch("urllib.request.urlopen", side_effect=OSError("down")):
                with self.assertLogs("dunetrace", level="WARNING") as logs:
                    c._fetch_policies("agent-1")
        finally:
            c.shutdown(timeout=1)
        self.assertEqual(len(c._policy_engine), 0)
        self.assertTrue(any("failed signature verification" in line for line in logs.output))

    def test_cache_for_a_different_agent_is_ignored(self):
        c = _client(policy_cache_path=self.dir)
        with open(c._policy_cache_file("agent-1"), "w", encoding="utf-8") as fh:
            json.dump({"agent_id": "agent-2", "policies": [_remote(1)]}, fh)
        try:
            with patch("urllib.request.urlopen", side_effect=OSError("down")):
                with self.assertLogs("dunetrace", level="WARNING") as logs:
                    c._fetch_policies("agent-1")
        finally:
            c.shutdown(timeout=1)
        self.assertEqual(len(c._policy_engine), 0)
        self.assertTrue(any("not a bundle for" in line for line in logs.output))

    def test_corrupt_cache_is_logged_not_fatal(self):
        c = _client(policy_cache_path=self.dir)
        with open(c._policy_cache_file("agent-1"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        try:
            with patch(
                "urllib.request.urlopen", return_value=_response({"policies": [_remote(1)]})
            ):
                with self.assertLogs("dunetrace", level="WARNING") as logs:
                    c._fetch_policies("agent-1")
        finally:
            c.shutdown(timeout=1)
        self.assertTrue(any("could not load policy cache" in line for line in logs.output))
        self.assertEqual(len(c._policy_engine), 1, "the live fetch still went through")

    def test_cache_is_read_once_and_never_over_a_live_bundle(self):
        c = _client(policy_cache_path=self.dir)
        engine = c._policy_engine
        try:
            engine.load([_remote(1, name="live")], agent_id="agent-1")
            with open(c._policy_cache_file("agent-1"), "w", encoding="utf-8") as fh:
                json.dump({"agent_id": "agent-1", "policies": [_remote(2, name="cached")]}, fh)
            with patch("urllib.request.urlopen", side_effect=OSError("down")):
                c._fetch_policies("agent-1")
                engine._fetch_failures["agent-1"] = (time.monotonic() - 100, 1)
                c._fetch_policies("agent-1")
        finally:
            c.shutdown(timeout=1)
        self.assertEqual([p.name for p in engine._policies], ["live"])
        self.assertEqual(c._policy_cache_tried, {"agent-1"})

    def test_unwritable_cache_warns_once_then_debug(self):
        c = _client(policy_cache_path=os.path.join(self.dir, "not-a-dir"))
        try:
            with open(c._policy_cache_path, "w", encoding="utf-8") as fh:
                fh.write("a file where a directory should be")
            engine = c._policy_engine
            with patch("urllib.request.urlopen", return_value=_response({"policies": []})):
                with self.assertLogs("dunetrace", level="DEBUG") as first:
                    c._fetch_policies("agent-1")
                engine._fetch_times["agent-1"] = time.monotonic() - (PolicyEngine._FETCH_TTL + 1)
                with self.assertLogs("dunetrace", level="DEBUG") as second:
                    c._fetch_policies("agent-1")
        finally:
            c.shutdown(timeout=1)
        w1 = [r for r in first.records if "could not write policy cache" in r.getMessage()]
        w2 = [r for r in second.records if "could not write policy cache" in r.getMessage()]
        self.assertEqual([r.levelno for r in w1], [logging.WARNING])
        self.assertEqual([r.levelno for r in w2], [logging.DEBUG])
        self.assertIn(
            "agent-1", engine._fetch_times, "a cache write failure is not a fetch failure"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
