"""
Server-authoritative detector config in the SDK's in-path pass (hardening
item 2.2-sdk).

The in-path signal pass (run_detectors(context="policy"), driven by a
trigger="signal" policy) used to run the TIER1_DETECTORS module singletons with
class defaults, so a threshold tuned in detectors.yml never reached it. Now the
client pulls GET /v1/detector-config alongside /v1/policies — same TTL, backoff,
in-flight guard and staleness verdict, via the shared RemoteFetchState — and
run_context runs the per-agent list built from it:

  * build_detectors(): in-path caps preserved and only ever LOWERED by the
    server; unknown kwargs skipped with one WARNING; non-in-path detectors
    never built; packs appended; None → plain TIER1_DETECTORS.
  * DetectorConfigStore: per-agent (config, detectors, baselines, fetched_at),
    generation counter, stale verdict.
  * client._fetch_detector_config(): success / failure / backoff / logging,
    one background thread for both endpoints.
  * run_context: store.detectors_for(agent) drives the signal pass, baselines
    are seeded onto RunState, the cached subset follows the store generation,
    policy.evaluated carries detector_config_stale / detector_config_age_s.

Fully offline: urllib.request.urlopen is patched wherever a fetch is reached.

Run: python -m pytest tests/test_detector_config.py -v
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from dunetrace import Dunetrace
from dunetrace.detector_config import (
    BASELINE_FIELDS,
    BUDGET_CAPS,
    DetectorConfigStore,
    apply_baselines,
    build_detectors,
)
from dunetrace.detectors import (
    DETECTOR_KEYS,
    TIER1_DETECTORS,
    ContextBloatDetector,
    DelegationLoopDetector,
    HandoffContextLossDetector,
    PromptInjectionDetector,
    ScattershotToolUseDetector,
    ToolLoopDetector,
    UngroundedDestinationDetector,
    UnresolvedAmbiguityDetector,
)
from dunetrace.models import EventType, RunState, Severity
from dunetrace.policies import PolicyEngine
from dunetrace.remote_fetch import RemoteFetchState

ENDPOINT = "http://config-host.invalid:9"
HOST = "config-host.invalid:9"


def _config(detectors=None, packs=None, baselines=None, schema_version=1) -> dict:
    return {
        "schema_version": schema_version,
        "agent_id": "agent-1",
        "agent_version": "v1",
        "generated_at": time.time(),
        "ttl_s": 60,
        "detectors": detectors or {},
        "packs": packs or [],
        "baselines": baselines,
    }


def _by_name(detectors) -> dict:
    return {d.name: d for d in detectors}


def _inpath(cls):
    return next(d for d in TIER1_DETECTORS if type(d) is cls)


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


class _CapturingExporter:
    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)


# ── One map, one bookkeeping implementation ────────────────────────────────────


class TestSharedPieces(unittest.TestCase):
    def test_detector_keys_cover_every_detector_class_once(self):
        self.assertEqual(len(DETECTOR_KEYS), 34)
        classes = list(DETECTOR_KEYS.values())
        self.assertEqual(len(set(classes)), 34, "a class is mapped under two keys")
        for d in TIER1_DETECTORS:
            self.assertIn(type(d), classes, f"{type(d).__name__} has no yaml key")
        for cls in (PromptInjectionDetector, HandoffContextLossDetector, DelegationLoopDetector):
            self.assertIn(cls, classes)
            self.assertNotIn(cls, {type(d) for d in TIER1_DETECTORS})

    def test_store_and_engine_share_the_fetch_bookkeeping(self):
        self.assertTrue(issubclass(PolicyEngine, RemoteFetchState))
        self.assertTrue(issubclass(DetectorConfigStore, RemoteFetchState))
        for attr in ("_FETCH_TTL", "_FETCH_BACKOFF_BASE", "_FETCH_BACKOFF_MAX"):
            self.assertEqual(getattr(PolicyEngine, attr), getattr(DetectorConfigStore, attr))
        for method in ("needs_fetch", "begin_fetch", "end_fetch", "mark_fetched"):
            self.assertIs(getattr(PolicyEngine, method), getattr(RemoteFetchState, method))
            self.assertIs(getattr(DetectorConfigStore, method), getattr(RemoteFetchState, method))
        self.assertIs(PolicyEngine.bundle_status, DetectorConfigStore.bundle_status)


# ── build_detectors ────────────────────────────────────────────────────────────


class TestBuildDetectors(unittest.TestCase):
    def test_none_config_is_the_plain_tier1_list(self):
        self.assertIs(build_detectors(None), TIER1_DETECTORS)

    def test_non_object_config_falls_back_to_tier1(self):
        with self.assertLogs("dunetrace.detector_config", level="WARNING"):
            self.assertIs(build_detectors([1, 2]), TIER1_DETECTORS)  # type: ignore[arg-type]

    def test_empty_config_rebuilds_every_inpath_detector_fresh(self):
        built = build_detectors(_config())
        self.assertEqual([d.name for d in built], [d.name for d in TIER1_DETECTORS])
        for fresh, singleton in zip(built, TIER1_DETECTORS):
            self.assertIsNot(fresh, singleton, "per-agent instances, not the shared singletons")
            self.assertIs(type(fresh), type(singleton))

    def test_server_threshold_applies(self):
        built = _by_name(build_detectors(_config({"tool_loop": {"THRESHOLD": 2, "WINDOW": 4}})))
        self.assertEqual(built["TOOL_LOOP"].THRESHOLD, 2)
        self.assertEqual(built["TOOL_LOOP"].WINDOW, 4)
        self.assertEqual(ToolLoopDetector.THRESHOLD, 3, "class default untouched")
        self.assertEqual(_inpath(ToolLoopDetector).THRESHOLD, 3, "singleton untouched")

    def test_non_inpath_detectors_are_never_built(self):
        cfg = _config(
            {
                "prompt_injection_signal": {"SEVERITY": "LOW"},
                "handoff_context_loss": {"SIZE_DROP_THRESHOLD": 0.1},
                "delegation_loop": {"MIN_LOOP_RUNS": 1},
            }
        )
        types = {type(d) for d in build_detectors(cfg)}
        for cls in (PromptInjectionDetector, HandoffContextLossDetector, DelegationLoopDetector):
            self.assertNotIn(cls, types)

    def test_inpath_caps_survive_a_server_override_of_other_params(self):
        cfg = _config({"ungrounded_destination": {"CASE_SENSITIVE": True}})
        ud = _by_name(build_detectors(cfg))["UNGROUNDED_DESTINATION"]
        singleton = _inpath(UngroundedDestinationDetector)
        self.assertTrue(ud.CASE_SENSITIVE)
        for cap in ("MAX_SCAN_NS", "MAX_SURFACE_CHARS", "MAX_ARGS_CHARS", "TOOL_NAME_SCOPE"):
            self.assertEqual(getattr(ud, cap), getattr(singleton, cap), cap)
        self.assertLess(ud.MAX_SCAN_NS, UngroundedDestinationDetector.MAX_SCAN_NS)

    def test_server_may_lower_a_cap(self):
        cfg = _config(
            {
                "ungrounded_destination": {"MAX_ARGS_CHARS": 1_000, "MAX_SCAN_NS": 500_000},
                "unresolved_ambiguity": {"MAX_OUTPUT_CHARS": 5_000},
                "tool_loop": {"MAX_COST_NS": 250_000},
            }
        )
        built = _by_name(build_detectors(cfg))
        self.assertEqual(built["UNGROUNDED_DESTINATION"].MAX_ARGS_CHARS, 1_000)
        self.assertEqual(built["UNGROUNDED_DESTINATION"].MAX_SCAN_NS, 500_000)
        self.assertEqual(built["UNRESOLVED_AMBIGUITY"].MAX_OUTPUT_CHARS, 5_000)
        self.assertEqual(built["TOOL_LOOP"].MAX_COST_NS, 250_000)

    def test_server_may_not_raise_a_cap(self):
        """The in-path budget is the customer's request latency; detectors.yml
        tunes the worker's own instance and may not spend more of ours."""
        singleton = _inpath(UngroundedDestinationDetector)
        cfg = _config(
            {
                "ungrounded_destination": {
                    "MAX_SCAN_NS": UngroundedDestinationDetector.MAX_SCAN_NS,  # class default, 50x
                    "MAX_SURFACE_CHARS": singleton.MAX_SURFACE_CHARS * 10,
                    "MAX_ARGS_CHARS": singleton.MAX_ARGS_CHARS,  # equal: no-op
                    "MAX_CANDIDATES_PER_RUN": 10_000,
                    "MAX_DEPTH": 100,
                    "MAX_NODES": 10**9,
                },
                "unresolved_ambiguity": {"MAX_OUTPUT_CHARS": 10**8, "MAX_SCAN_NS": 10**10},
                "tool_loop": {"MAX_COST_NS": 10**9},
            }
        )
        with self.assertNoLogs("dunetrace.detector_config", level="WARNING"):
            built = _by_name(build_detectors(cfg))
        ud = built["UNGROUNDED_DESTINATION"]
        for cap in ("MAX_SCAN_NS", "MAX_SURFACE_CHARS", "MAX_ARGS_CHARS"):
            self.assertEqual(getattr(ud, cap), getattr(singleton, cap), cap)
        for cap in ("MAX_CANDIDATES_PER_RUN", "MAX_DEPTH", "MAX_NODES"):
            self.assertEqual(getattr(ud, cap), getattr(UngroundedDestinationDetector, cap), cap)
        ua = built["UNRESOLVED_AMBIGUITY"]
        self.assertEqual(ua.MAX_OUTPUT_CHARS, _inpath(UnresolvedAmbiguityDetector).MAX_OUTPUT_CHARS)
        self.assertEqual(ua.MAX_SCAN_NS, _inpath(UnresolvedAmbiguityDetector).MAX_SCAN_NS)
        self.assertEqual(built["TOOL_LOOP"].MAX_COST_NS, ToolLoopDetector.MAX_COST_NS)

    def test_every_budget_cap_is_a_real_attribute_somewhere(self):
        known = set()
        for cls in DETECTOR_KEYS.values():
            for klass in cls.__mro__:
                known.update(k for k in vars(klass) if k.isupper())
        self.assertTrue(BUDGET_CAPS <= known, BUDGET_CAPS - known)

    def test_non_numeric_cap_is_skipped_with_a_warning(self):
        cfg = _config({"ungrounded_destination": {"MAX_SCAN_NS": "fast"}})
        with self.assertLogs("dunetrace.detector_config", level="WARNING") as logs:
            ud = _by_name(build_detectors(cfg))["UNGROUNDED_DESTINATION"]
        self.assertEqual(ud.MAX_SCAN_NS, _inpath(UngroundedDestinationDetector).MAX_SCAN_NS)
        self.assertIn("non-numeric MAX_SCAN_NS", logs.output[0])

    def test_unknown_kwarg_warns_once_per_detector_and_key(self):
        cfg = _config({"tool_loop": {"THRESHOLD": 2, "THREHOLD": 9}})
        with self.assertLogs("dunetrace.detector_config", level="DEBUG") as logs:
            first = _by_name(build_detectors(cfg))
            build_detectors(cfg)
            build_detectors(_config({"tool_loop": {"THREHOLD": 9}}))
        warnings = [r for r in logs.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1, [r.getMessage() for r in warnings])
        self.assertIn("'tool_loop'", warnings[0].getMessage())
        self.assertIn("'THREHOLD'", warnings[0].getMessage())
        self.assertEqual(first["TOOL_LOOP"].THRESHOLD, 2, "the known key still applied")
        # A different (detector, key) pair warns on its own.
        with self.assertLogs("dunetrace.detector_config", level="WARNING") as logs2:
            build_detectors(_config({"retry_storm": {"THREHOLD": 9}}))
        self.assertEqual(len(logs2.output), 1)

    def test_unknown_detector_key_is_debug_only(self):
        with self.assertNoLogs("dunetrace.detector_config", level="INFO"):
            built = build_detectors(_config({"not_a_detector": {"THRESHOLD": 1}}))
        self.assertEqual(len(built), len(TIER1_DETECTORS))

    def test_severity_string_becomes_the_enum(self):
        built = _by_name(build_detectors(_config({"context_bloat": {"SEVERITY": "HIGH"}})))
        self.assertIs(built["CONTEXT_BLOAT"].SEVERITY, Severity.HIGH)
        self.assertIs(ContextBloatDetector.SEVERITY, Severity.MEDIUM)

    def test_invalid_severity_is_skipped_with_a_warning(self):
        with self.assertLogs("dunetrace.detector_config", level="WARNING"):
            built = _by_name(build_detectors(_config({"context_bloat": {"SEVERITY": "LOUD"}})))
        self.assertIs(built["CONTEXT_BLOAT"].SEVERITY, Severity.MEDIUM)

    def test_constructor_rejection_falls_back_to_the_inpath_instance(self):
        # Scattershot validates MIN_DISTINCT_TOOLS is a positive int in __init__.
        cfg = _config({"scattershot_tool_use": {"MIN_DISTINCT_TOOLS": 6.0}})
        with self.assertLogs("dunetrace.detector_config", level="WARNING") as logs:
            built = _by_name(build_detectors(cfg))
        self.assertIs(built["SCATTERSHOT_TOOL_USE"], _inpath(ScattershotToolUseDetector))
        self.assertIn("could not be applied", logs.output[0])
        self.assertEqual(len(built), len(TIER1_DETECTORS), "no detector lost")

    def test_pack_detectors_are_appended_with_class_defaults(self):
        from dunetrace.packs import PACK_REGISTRY

        voice = PACK_REGISTRY["voice"]
        built = build_detectors(_config(packs=["voice", "no-such-pack"]))
        self.assertEqual(len(built), len(TIER1_DETECTORS) + len(voice.detectors))
        tail = built[len(TIER1_DETECTORS) :]
        self.assertEqual([type(d) for d in tail], list(voice.detectors))
        self.assertTrue(all(d.pack == "voice" for d in tail))
        self.assertEqual(len(build_detectors(_config(packs=[]))), len(TIER1_DETECTORS))


# ── Baselines ──────────────────────────────────────────────────────────────────


class TestBaselines(unittest.TestCase):
    def test_every_mapped_field_exists_on_runstate(self):
        state = RunState(run_id="r", agent_id="a", agent_version="v")
        for attr in BASELINE_FIELDS.values():
            self.assertTrue(hasattr(state, attr), attr)
            self.assertIsNone(getattr(state, attr))

    def test_apply_fills_only_none_fields_and_skips_nulls(self):
        state = RunState(run_id="r", agent_id="a", agent_version="v", baseline_p75_steps=7.0)
        apply_baselines(
            state,
            {
                "p75_steps": 40,
                "p75_token_growth": 2.5,
                "p75_latency_tool_ms": None,
                "p75_total_tokens": "lots",
                "p75_unknown": 1.0,
            },
        )
        self.assertEqual(state.baseline_p75_steps, 7.0, "caller-set value kept")
        self.assertEqual(state.baseline_p75_token_growth, 2.5)
        self.assertIsInstance(state.baseline_p75_token_growth, float)
        self.assertIsNone(state.baseline_p75_latency_tool)
        self.assertIsNone(state.baseline_p75_total_tokens)

    def test_apply_tolerates_no_baselines(self):
        state = RunState(run_id="r", agent_id="a", agent_version="v")
        apply_baselines(state, None)
        apply_baselines(state, "nope")  # type: ignore[arg-type]
        self.assertIsNone(state.baseline_p75_steps)


# ── DetectorConfigStore ────────────────────────────────────────────────────────


class TestDetectorConfigStore(unittest.TestCase):
    def test_defaults_before_any_load(self):
        store = DetectorConfigStore()
        self.assertIs(store.detectors_for("a"), TIER1_DETECTORS)
        self.assertIsNone(store.baselines_for("a"))
        self.assertFalse(store.has_config("a"))
        self.assertEqual(store.generation, 0)
        self.assertTrue(store.is_stale("a"))
        self.assertFalse(store.is_stale("a", remote_configured=False))
        self.assertTrue(store.needs_fetch("a"))

    def test_load_is_per_agent_and_bumps_generation(self):
        store = DetectorConfigStore()
        entry = store.load(
            "a", _config({"tool_loop": {"THRESHOLD": 2}}, baselines={"p75_steps": 9})
        )
        self.assertEqual(store.generation, 1)
        self.assertIs(store.entry_for("a"), entry)
        self.assertEqual(_by_name(store.detectors_for("a"))["TOOL_LOOP"].THRESHOLD, 2)
        self.assertEqual(store.baselines_for("a"), {"p75_steps": 9})
        self.assertIs(store.detectors_for("b"), TIER1_DETECTORS, "other agents untouched")
        self.assertLessEqual(time.monotonic() - entry.fetched_at, 5.0)
        store.load("a", _config())
        self.assertEqual(store.generation, 2)
        self.assertEqual(_by_name(store.detectors_for("a"))["TOOL_LOOP"].THRESHOLD, 3)

    def test_load_rejects_a_non_object(self):
        with self.assertRaises(ValueError):
            DetectorConfigStore().load("a", [1])  # type: ignore[arg-type]

    def test_stale_follows_the_shared_ttl(self):
        store = DetectorConfigStore()
        store.load("a", _config())
        self.assertTrue(store.is_stale("a"), "loaded but not marked fetched → stale")
        store.mark_fetched("a")
        stale, age = store.bundle_status("a", True)
        self.assertFalse(stale)
        self.assertGreaterEqual(age, 0.0)
        store._fetch_times["a"] = time.monotonic() - (RemoteFetchState._FETCH_TTL + 1)
        self.assertTrue(store.is_stale("a"))
        self.assertTrue(store.needs_fetch("a"))

    def test_unknown_schema_version_warns_once_and_still_applies(self):
        store = DetectorConfigStore()
        with self.assertLogs("dunetrace.detector_config", level="WARNING") as logs:
            store.load("a", _config({"tool_loop": {"THRESHOLD": 2}}, schema_version=2))
            store.load("a", _config(schema_version=2))
        self.assertEqual(len([m for m in logs.output if "schema_version" in m]), 1)
        self.assertEqual(store.generation, 2)


# ── Client fetch ───────────────────────────────────────────────────────────────


def _config_fetch_records(records):
    return [r for r in records if "detector config fetch failed" in r.getMessage()]


class TestClientFetch(unittest.TestCase):
    def setUp(self):
        self.c = _client()
        self.store = self.c._detector_config_store

    def tearDown(self):
        self.c.shutdown(timeout=1)

    def test_success_loads_the_store_and_starts_the_ttl(self):
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _response(_config({"tool_loop": {"THRESHOLD": 2}}))
            with self.assertNoLogs("dunetrace", level="INFO"):
                self.c._fetch_detector_config("agent-1", "v1")
            req = urlopen.call_args.args[0]
        self.assertEqual(
            req.full_url, f"{ENDPOINT}/v1/detector-config?agent_id=agent-1&agent_version=v1"
        )
        self.assertEqual(req.get_header("Authorization"), "Bearer dt_live_test")
        self.assertNotIn("api_key", req.full_url)
        self.assertEqual(_by_name(self.store.detectors_for("agent-1"))["TOOL_LOOP"].THRESHOLD, 2)
        self.assertIn("agent-1", self.store._fetch_times)
        self.assertFalse(self.store.needs_fetch("agent-1"))
        self.assertEqual(self.store._fetch_in_flight, set())

    def test_agent_version_is_optional_and_url_encoded(self):
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _response(_config())
            self.c._fetch_detector_config("team/agent")
            self.assertEqual(
                urlopen.call_args.args[0].full_url,
                f"{ENDPOINT}/v1/detector-config?agent_id=team%2Fagent",
            )

    def test_failure_does_not_mark_fetched_and_retries_after_backoff(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with patch("dunetrace.remote_fetch.time.monotonic") as mono:
                mono.return_value = 100.0
                self.c._fetch_detector_config("agent-1")
                self.assertNotIn("agent-1", self.store._fetch_times)
                self.assertEqual(self.store.fetch_failure_streak("agent-1"), 1)
                self.assertFalse(self.store.needs_fetch("agent-1"))
                mono.return_value = 102.0
                self.assertTrue(self.store.needs_fetch("agent-1"))
        self.assertIs(self.store.detectors_for("agent-1"), TIER1_DETECTORS, "fail-open")
        self.assertEqual(self.store._fetch_in_flight, set(), "guard released after failure")

    def test_non_object_body_is_a_failure(self):
        with patch("urllib.request.urlopen", return_value=_response([1, 2, 3])):
            with self.assertLogs("dunetrace", level="WARNING"):
                self.c._fetch_detector_config("agent-1")
        self.assertEqual(self.store.fetch_failure_streak("agent-1"), 1)
        self.assertFalse(self.store.has_config("agent-1"))

    def test_first_failure_warns_then_debug_then_info_on_recovery(self):
        err = urllib.error.HTTPError(ENDPOINT, 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertLogs("dunetrace", level="DEBUG") as first:
                self.c._fetch_detector_config("agent-1")
            recs = _config_fetch_records(first.records)
            self.assertEqual([r.levelno for r in recs], [logging.WARNING])
            msg = recs[0].getMessage()
            self.assertIn("'agent-1'", msg)
            self.assertIn(HOST, msg)
            self.assertIn("HTTPError", msg)
            self.assertIn("retrying in 2s", msg)
            self.assertIn("class defaults", msg)

            self.store._fetch_failures["agent-1"] = (time.monotonic() - 100, 1)
            with self.assertLogs("dunetrace", level="DEBUG") as second:
                self.c._fetch_detector_config("agent-1")
            recs = _config_fetch_records(second.records)
            self.assertEqual([r.levelno for r in recs], [logging.DEBUG])
            self.assertIn("failure 2 in a row", recs[0].getMessage())

        self.store._fetch_failures["agent-1"] = (time.monotonic() - 100, 2)
        with patch("urllib.request.urlopen", return_value=_response(_config())):
            with self.assertLogs("dunetrace", level="INFO") as third:
                self.c._fetch_detector_config("agent-1")
        recovered = [r for r in third.records if "recovered after 2 failure" in r.getMessage()]
        self.assertEqual(len(recovered), 1)
        self.assertIn("detector config", recovered[0].getMessage())
        self.assertEqual(self.store.fetch_failure_streak("agent-1"), 0)
        self.assertFalse(self.store.needs_fetch("agent-1"))

    def test_last_config_retained_after_failed_refresh(self):
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _response(_config({"tool_loop": {"THRESHOLD": 2}}))
            self.c._fetch_detector_config("agent-1")
        self.store._fetch_times["agent-1"] = time.monotonic() - (RemoteFetchState._FETCH_TTL + 1)
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            with self.assertLogs("dunetrace", level="WARNING") as logs:
                self.c._fetch_detector_config("agent-1")
        self.assertEqual(_by_name(self.store.detectors_for("agent-1"))["TOOL_LOOP"].THRESHOLD, 2)
        self.assertIn("still using the last configuration received", logs.output[0])
        self.assertTrue(self.store.is_stale("agent-1"))

    def test_in_flight_guard_blocks_a_second_concurrent_fetch(self):
        release = threading.Event()
        calls: list = []

        def slow_urlopen(req, timeout=None):
            calls.append(req.full_url)
            release.wait(2)
            return _response(_config())

        with patch("urllib.request.urlopen", slow_urlopen):
            t1 = threading.Thread(target=self.c._fetch_detector_config, args=("agent-1",))
            t1.start()
            for _ in range(400):
                if calls:
                    break
                time.sleep(0.005)
            self.assertEqual(len(calls), 1)
            self.assertFalse(self.store.needs_fetch("agent-1"), "in flight → not due")
            self.c._fetch_detector_config("agent-1")
            self.assertEqual(len(calls), 1)
            release.set()
            t1.join(2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.store._fetch_in_flight, set())

    def test_one_thread_pulls_both_endpoints(self):
        seen: list = []

        def urlopen(req, timeout=None):
            seen.append(req.full_url)
            if "/v1/policies" in req.full_url:
                return _response({"policies": []})
            return _response(_config({"tool_loop": {"THRESHOLD": 2}}))

        with patch("urllib.request.urlopen", urlopen):
            self.c._fetch_remote_config("agent-1", "v1")
        self.assertEqual(
            seen,
            [
                f"{ENDPOINT}/v1/policies?agent_id=agent-1",
                f"{ENDPOINT}/v1/detector-config?agent_id=agent-1&agent_version=v1",
            ],
        )
        self.assertIn("agent-1", self.c._policy_engine._fetch_times)
        self.assertIn("agent-1", self.store._fetch_times)

    def test_each_endpoint_keeps_its_own_backoff(self):
        """A policy fetch in backoff must not stop a due detector-config fetch,
        and vice versa — the bookkeeping is shared code, not shared state."""
        self.c._policy_engine.mark_fetch_failed("agent-1")
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _response(_config())
            self.c._fetch_remote_config("agent-1")
            self.assertEqual(urlopen.call_count, 1)
            self.assertIn("/v1/detector-config", urlopen.call_args.args[0].full_url)

    def test_run_start_spawns_the_shared_fetch_when_only_the_store_is_due(self):
        self.c._policy_engine.mark_fetched("billing")  # policies fresh, config never fetched
        with patch.object(self.c, "_fetch_remote_config") as fetch:
            with self.c.run("billing"):
                pass
            for _ in range(200):
                if fetch.called:
                    break
                time.sleep(0.005)
        fetch.assert_called_once()
        agent_id, version = fetch.call_args.args
        self.assertEqual(agent_id, "billing")
        self.assertTrue(version)

    def test_run_start_spawns_nothing_when_both_are_fresh(self):
        self.c._policy_engine.mark_fetched("billing")
        self.store.mark_fetched("billing")
        with patch.object(self.c, "_fetch_remote_config") as fetch:
            with self.c.run("billing"):
                pass
        fetch.assert_not_called()

    def test_local_client_never_fetches(self):
        c = _client(api_key="")
        try:
            with patch("urllib.request.urlopen") as urlopen:
                c._fetch_detector_config("agent-1")
                c._fetch_remote_config("agent-1")
            urlopen.assert_not_called()
        finally:
            c.shutdown(timeout=1)


# ── The in-path pass ───────────────────────────────────────────────────────────


def _loop_run(run) -> None:
    """Five tool calls with 'search' twice — a TOOL_LOOP at THRESHOLD=2, not 3."""
    for name in ("search", "fetch", "parse", "search", "summarise"):
        run.tool_called(name, {"q": name})
        run.tool_responded(name, success=True)


class TestInPathPass(unittest.TestCase):
    def _client_with_signal_policy(self) -> tuple:
        exp = _CapturingExporter()
        c = _client(exporters=[exp])
        c._fetch_remote_config = lambda *a, **k: None  # type: ignore[method-assign]
        c.add_policy(
            "halt-on-loop",
            {"trigger": "signal", "operator": "contains", "value": "TOOL_LOOP"},
            {"type": "log"},
        )
        return c, exp

    def _triggered(self, exp) -> list:
        return [e for e in exp.events if e.event_type == EventType.POLICY_TRIGGERED]

    def test_defaults_before_any_config(self):
        c, exp = self._client_with_signal_policy()
        try:
            with c.run("agent") as run:
                _loop_run(run)
                self.assertEqual([d.name for d in run._policy_detectors], ["TOOL_LOOP"])
                self.assertIs(run._policy_detectors[0], _inpath(ToolLoopDetector))
        finally:
            c.shutdown(timeout=1)
        self.assertEqual(self._triggered(exp), [], "class default THRESHOLD=3 does not fire")

    def test_server_threshold_drives_the_signal_pass(self):
        c, exp = self._client_with_signal_policy()
        c._detector_config_store.load("agent", _config({"tool_loop": {"THRESHOLD": 2}}))
        try:
            with c.run("agent") as run:
                _loop_run(run)
                self.assertEqual(run._policy_detectors[0].THRESHOLD, 2)
        finally:
            c.shutdown(timeout=1)
        fired = self._triggered(exp)
        # A `log` policy is not one-shot: it fires on every evaluation while
        # the signal is present (both events of the looping step here).
        self.assertGreaterEqual(len(fired), 1)
        self.assertEqual({e.payload["policy_name"] for e in fired}, {"halt-on-loop"})
        self.assertIn("TOOL_LOOP", fired[0].payload["value"])

    def test_other_agents_config_does_not_apply(self):
        c, exp = self._client_with_signal_policy()
        c._detector_config_store.load("other-agent", _config({"tool_loop": {"THRESHOLD": 2}}))
        try:
            with c.run("agent") as run:
                _loop_run(run)
        finally:
            c.shutdown(timeout=1)
        self.assertEqual(self._triggered(exp), [])

    def test_subset_regenerates_when_the_store_generation_changes_mid_run(self):
        c, exp = self._client_with_signal_policy()
        store = c._detector_config_store
        try:
            with c.run("agent") as run:
                run.tool_called("search", {})
                run.tool_responded("search", success=True)
                first_key = run._policy_detectors_key
                self.assertEqual(first_key[1], 0)
                store.load("agent", _config({"tool_loop": {"THRESHOLD": 2}}))
                for name in ("fetch", "parse", "search", "summarise"):
                    run.tool_called(name, {})
                    run.tool_responded(name, success=True)
                self.assertEqual(run._policy_detectors_key[1], 1)
                self.assertEqual(run._policy_detectors[0].THRESHOLD, 2)
        finally:
            c.shutdown(timeout=1)
        self.assertGreaterEqual(len(self._triggered(exp)), 1)

    def test_per_step_cache_still_holds_within_a_step(self):
        c, exp = self._client_with_signal_policy()
        try:
            with c.run("agent") as run:
                with patch("dunetrace.detectors.run_detectors", return_value=[]) as rd:
                    run.tool_called("search", {})
                    run.tool_responded("search", success=True)
                    run.llm_called("m", 10)
                    run.llm_responded(completion_tokens=5, output="out")
            calls = rd.call_count
        finally:
            c.shutdown(timeout=1)
        self.assertGreaterEqual(calls, 1)
        self.assertLessEqual(calls, 2, "one detector run per step, not per event")
        self.assertEqual(rd.call_args.kwargs["context"], "policy")
        self.assertEqual([d.name for d in rd.call_args.kwargs["detectors"]], ["TOOL_LOOP"])

    def test_baselines_are_seeded_onto_the_run_state(self):
        c, exp = self._client_with_signal_policy()
        c._detector_config_store.load(
            "agent", _config(baselines={"p75_steps": 12, "p75_token_growth": 1.7})
        )
        try:
            with c.run("agent") as run:
                self.assertIsNone(run.state.baseline_p75_steps, "not before the pass runs")
                run.tool_called("search", {})
                run.tool_responded("search", success=True)
                self.assertEqual(run.state.baseline_p75_steps, 12.0)
                self.assertEqual(run.state.baseline_p75_token_growth, 1.7)
                self.assertIsNone(run.state.baseline_p75_duration_s)
        finally:
            c.shutdown(timeout=1)

    def test_pack_detectors_join_the_full_battery(self):
        exp = _CapturingExporter()
        c = _client(exporters=[exp])
        c._fetch_remote_config = lambda *a, **k: None  # type: ignore[method-assign]
        # An expression condition can't be resolved to failure types → full battery.
        c.add_policy(
            "any-signal",
            {"trigger": "signal", "operator": "contains", "value": {"any": ["X"]}},
            {"type": "log"},
        )
        c._detector_config_store.load("agent", _config(packs=["voice"]))
        try:
            with c.run("agent") as run:
                run.tool_called("search", {})
                run.tool_responded("search", success=True)
                names = [d.name for d in run._policy_detectors]
        finally:
            c.shutdown(timeout=1)
        from dunetrace.packs import PACK_REGISTRY

        self.assertEqual(len(names), len(TIER1_DETECTORS) + len(PACK_REGISTRY["voice"].detectors))

    def test_mock_client_without_a_store_runs_tier1(self):
        from dunetrace.run_context import RunContext

        client = MagicMock()
        client._policy_engine = PolicyEngine()
        ctx = RunContext(client, "agent", "v", [], "hi")
        self.assertIsNone(ctx._detector_config_store())


# ── policy.evaluated ───────────────────────────────────────────────────────────


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
        self.assertTrue(payload["detector_config_stale"])
        self.assertIsNone(payload["detector_config_age_s"])
        self.assertIn("policy_bundle_stale", payload, "1.4 fields still present")

    def test_fresh_config_is_not_stale_and_carries_age(self):
        c, exp = self._reporting_client()
        c._detector_config_store.mark_fetched("billing")
        payload = self._evaluate(c, exp)
        self.assertFalse(payload["detector_config_stale"])
        self.assertIsInstance(payload["detector_config_age_s"], float)
        self.assertGreaterEqual(payload["detector_config_age_s"], 0.0)
        self.assertLess(payload["detector_config_age_s"], 5.0)

    def test_overdue_config_is_stale(self):
        c, exp = self._reporting_client()
        c._detector_config_store._fetch_times["billing"] = time.monotonic() - (
            RemoteFetchState._FETCH_TTL + 10
        )
        payload = self._evaluate(c, exp)
        self.assertTrue(payload["detector_config_stale"])
        self.assertGreater(payload["detector_config_age_s"], RemoteFetchState._FETCH_TTL)

    def test_the_two_verdicts_are_independent(self):
        c, exp = self._reporting_client()
        c._policy_engine.mark_fetched("billing")
        payload = self._evaluate(c, exp)
        self.assertFalse(payload["policy_bundle_stale"])
        self.assertTrue(payload["detector_config_stale"])

    def test_local_only_client_is_never_stale(self):
        with patch.dict(os.environ, {"DUNETRACE_API_KEY": ""}):
            c, exp = self._reporting_client(api_key="")
        payload = self._evaluate(c, exp)
        self.assertFalse(payload["detector_config_stale"])
        self.assertIsNone(payload["detector_config_age_s"])


if __name__ == "__main__":
    unittest.main()
