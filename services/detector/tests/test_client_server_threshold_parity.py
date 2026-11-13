"""
Golden parity: the SDK's in-path detector list and the detector worker's list
agree on the same detectors.yml (hardening item 2.2-sdk).

Two consumers read detectors.yml: the worker (detector_svc.detectors, via
detector_svc.config_loader) instantiates its battery from it directly; the
ingest service (dunetrace_schemas.detector_config) serves the same merged
kwargs to the SDK at GET /v1/detector-config, and the SDK
(dunetrace.detector_config.build_detectors) instantiates its in-path list from
that payload. This suite is the only place with all three on its PYTHONPATH,
so it is where the parity is pinned:

  * one class map — detector_svc's _DETECTOR_CLASSES IS the SDK's DETECTOR_KEYS,
    and both equal the parser's BUILTIN_DETECTOR_KEYS;
  * the same yaml, through both parsers, yields instances whose tunables agree;
  * both lists produce identical failure-type sets over golden RunStates — a
    tool loop that only fires under the agent-category override, a clean run,
    a context-bloat run, and a baseline-driven run;
  * the in-path caps survive on the client instances and a server attempt to
    RAISE one is ignored, while a LOWERED cap applies on both sides;
  * the baselines mapping the SDK carries matches the response keys ingest
    serves.

Run: PYTHONPATH=packages/schemas-py:packages/sdk-py:services/detector \
       python -m pytest services/detector/tests/test_client_server_threshold_parity.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from dunetrace.detector_config import BASELINE_FIELDS, apply_baselines, build_detectors
from dunetrace.detectors import (
    DETECTOR_KEYS,
    TIER1_DETECTORS,
    ToolThrashingDetector,
    UngroundedDestinationDetector,
    run_detectors,
)
from dunetrace.models import LlmCall, RunState, ToolCall
from dunetrace_schemas.baselines import BASELINE_RESPONSE_KEYS
from dunetrace_schemas.detector_config import (
    BUILTIN_DETECTOR_KEYS,
    effective_detector_kwargs,
    json_safe_kwargs,
)
from dunetrace_schemas.detector_config import load_detector_kwargs as parse_for_ingest

import detector_svc.detectors as det_module
from detector_svc.config_loader import load_detector_kwargs as parse_for_worker

GOLDEN_AGENT = "golden-agent"
OTHER_AGENT = "other-agent"  # no section of its own → `default`

# A default override, an agent-category override, a LOWERED cap (applies on
# both sides), and two attempts to RAISE an in-path budget (server-side only).
FIXTURE_YAML = """
default:
  tool_loop:
    threshold: 3
    window: 5
  context_bloat:
    growth_factor: 2.0
    min_last_tokens: 500
  ungrounded_destination:
    max_scan_ns: 50000000    # the class default: 50x the in-path cap
    max_args_chars: 2000     # below the in-path cap of 4000
  tool_thrashing:
    max_cost_ns: 5000000     # cross-cutting budget, 5x the in-path 1ms

golden-agent:
  tool_loop:
    threshold: 2
  context_bloat:
    growth_factor: 1.5
"""

_INPATH_UD = next(d for d in TIER1_DETECTORS if type(d) is UngroundedDestinationDetector)


def _tool(name: str, step: int) -> ToolCall:
    return ToolCall(
        tool_name=name, args="{}", step_index=step, timestamp=1_000.0 + step, success=True
    )


def _llm(tokens: int, step: int) -> LlmCall:
    return LlmCall(
        model="m",
        prompt_tokens=tokens,
        finish_reason="stop",
        latency_ms=50,
        step_index=step,
        timestamp=1_000.0 + step,
        output_length=40,
        completion_tokens=20,
    )


def _state(agent_id: str, tool_calls, llm_calls) -> RunState:
    return RunState(
        run_id="golden",
        agent_id=agent_id,
        agent_version="v1",
        available_tools=["search", "fetch", "parse", "summarise"],
        tool_calls=list(tool_calls),
        llm_calls=list(llm_calls),
        current_step=max([c.step_index for c in tool_calls] + [c.step_index for c in llm_calls]),
        exit_reason="final_answer",
    )


def loop_state(agent_id: str) -> RunState:
    """'search' twice inside a five-call window: a TOOL_LOOP at THRESHOLD=2, not 3."""
    names = ["search", "fetch", "parse", "search", "summarise"]
    return _state(agent_id, [_tool(n, i + 1) for i, n in enumerate(names)], [_llm(900, 1)])


def clean_state(agent_id: str) -> RunState:
    return _state(
        agent_id,
        [_tool("search", 1), _tool("fetch", 3)],
        [_llm(1000, 2), _llm(1100, 4), _llm(1200, 5)],
    )


def bloat_state(agent_id: str) -> RunState:
    """Prompt tokens 2000 → 3600 (growth 1.8): fires at GROWTH_FACTOR 1.5, not 2.0 or 3.0."""
    return _state(
        agent_id,
        [_tool("search", 1), _tool("fetch", 3)],
        [_llm(2000, 2), _llm(3000, 4), _llm(3600, 5)],
    )


def _types(signals) -> set:
    return {s.failure_type.value for s in signals}


class ParityCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp()
        cls.path = os.path.join(cls._dir, "detectors.yml")
        with open(cls.path, "w") as fh:
            fh.write(FIXTURE_YAML)
        cls.worker_cfg = parse_for_worker(
            cls.path, known_detectors=set(det_module._DETECTOR_CLASSES)
        )
        cls.ingest_cfg = parse_for_ingest(cls.path, known_detectors=set(BUILTIN_DETECTOR_KEYS))

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.path)
        os.rmdir(cls._dir)

    def server_list(self, agent_id: str):
        with patch.object(det_module, "_CONFIG", self.worker_cfg):
            with patch("detector_svc.packs.get_enabled_packs", AsyncMock(return_value=set())):
                return asyncio.run(det_module.get_detectors(agent_id, "org-1"))

    def payload(self, agent_id: str, baselines=None) -> dict:
        """What GET /v1/detector-config returns for agent_id, after a JSON round trip."""
        body = {
            "schema_version": 1,
            "agent_id": agent_id,
            "agent_version": "v1",
            "generated_at": 0.0,
            "ttl_s": 60,
            "detectors": json_safe_kwargs(effective_detector_kwargs(self.ingest_cfg, agent_id)),
            "packs": [],
            "baselines": baselines,
        }
        return json.loads(json.dumps(body))

    def client_list(self, agent_id: str):
        return build_detectors(self.payload(agent_id))


class TestOneMap(ParityCase):
    def test_worker_map_is_the_sdk_map(self):
        self.assertIs(det_module._DETECTOR_CLASSES, DETECTOR_KEYS)
        self.assertEqual(set(DETECTOR_KEYS), set(BUILTIN_DETECTOR_KEYS))
        self.assertEqual(len(DETECTOR_KEYS), 34)

    def test_both_parsers_see_the_same_overrides(self):
        for category in ("default", GOLDEN_AGENT):
            worker = {d: set(k) for d, k in self.worker_cfg[category].items()}
            ingest = {d: set(k) for d, k in self.ingest_cfg[category].items()}
            self.assertEqual(worker, ingest, category)

    def test_baseline_keys_match_what_ingest_serves(self):
        self.assertEqual(set(BASELINE_FIELDS), set(BASELINE_RESPONSE_KEYS.values()))
        state = RunState(run_id="r", agent_id="a", agent_version="v")
        for attr in BASELINE_FIELDS.values():
            self.assertTrue(hasattr(state, attr), attr)


class TestTunableParity(ParityCase):
    def _pairs(self, agent_id: str):
        server = {d.name: d for d in self.server_list(agent_id)}
        client = {d.name: d for d in self.client_list(agent_id)}
        self.assertTrue(set(client) <= set(server))
        return server, client

    def test_client_list_is_the_inpath_subset_in_server_order(self):
        server, client = self._pairs(GOLDEN_AGENT)
        self.assertEqual(
            [d.name for d in self.client_list(GOLDEN_AGENT)],
            [d.name for d in self.server_list(GOLDEN_AGENT) if d.name in client],
        )
        for name in ("PROMPT_INJECTION_SIGNAL", "HANDOFF_CONTEXT_LOSS", "DELEGATION_LOOP"):
            self.assertIn(name, server)
            self.assertNotIn(name, client)

    def test_thresholds_agree_for_the_category_and_for_default(self):
        for agent_id, threshold, growth in ((GOLDEN_AGENT, 2, 1.5), (OTHER_AGENT, 3, 2.0)):
            server, client = self._pairs(agent_id)
            self.assertEqual(server["TOOL_LOOP"].THRESHOLD, threshold, agent_id)
            self.assertEqual(client["TOOL_LOOP"].THRESHOLD, threshold, agent_id)
            self.assertEqual(server["TOOL_LOOP"].WINDOW, client["TOOL_LOOP"].WINDOW)
            self.assertEqual(server["CONTEXT_BLOAT"].GROWTH_FACTOR, growth, agent_id)
            self.assertEqual(client["CONTEXT_BLOAT"].GROWTH_FACTOR, growth, agent_id)
            self.assertEqual(server["CONTEXT_BLOAT"].MIN_LAST_TOKENS, 500)
            self.assertEqual(client["CONTEXT_BLOAT"].MIN_LAST_TOKENS, 500)

    def test_every_non_cap_tunable_agrees(self):
        """Beyond the fixture's overrides: every UPPERCASE attribute that is
        not an in-path cap reads the same on the two instances."""
        from dunetrace.detector_config import BUDGET_CAPS

        server, client = self._pairs(GOLDEN_AGENT)
        for name, c in client.items():
            s = server[name]
            self.assertIs(type(s), type(c), name)
            for klass in type(c).__mro__:
                for attr in vars(klass):
                    if attr.isupper() and attr not in BUDGET_CAPS and attr != "TOOL_NAME_SCOPE":
                        self.assertEqual(getattr(s, attr), getattr(c, attr), f"{name}.{attr}")

    def test_inpath_caps_survive_and_a_raise_is_ignored(self):
        server, client = self._pairs(GOLDEN_AGENT)
        ud_s, ud_c = server["UNGROUNDED_DESTINATION"], client["UNGROUNDED_DESTINATION"]
        # The server honours the yaml on its own instance ...
        self.assertEqual(ud_s.MAX_SCAN_NS, 50_000_000)
        # ... the client keeps its in-path cap, because the raise is ignored.
        self.assertEqual(ud_c.MAX_SCAN_NS, _INPATH_UD.MAX_SCAN_NS)
        self.assertEqual(ud_c.MAX_SCAN_NS, 1_000_000)
        self.assertEqual(ud_c.MAX_SURFACE_CHARS, _INPATH_UD.MAX_SURFACE_CHARS)
        self.assertEqual(ud_c.TOOL_NAME_SCOPE, _INPATH_UD.TOOL_NAME_SCOPE)
        self.assertIsNotNone(ud_c.TOOL_NAME_SCOPE)
        self.assertIsNone(ud_s.TOOL_NAME_SCOPE, "server instance is unscoped, as before")
        # The cross-cutting budget too.
        self.assertEqual(server["TOOL_THRASHING"].MAX_COST_NS, 5_000_000)
        self.assertEqual(client["TOOL_THRASHING"].MAX_COST_NS, ToolThrashingDetector.MAX_COST_NS)

    def test_a_lowered_cap_applies_on_both_sides(self):
        server, client = self._pairs(GOLDEN_AGENT)
        self.assertEqual(server["UNGROUNDED_DESTINATION"].MAX_ARGS_CHARS, 2_000)
        self.assertEqual(client["UNGROUNDED_DESTINATION"].MAX_ARGS_CHARS, 2_000)
        self.assertLess(2_000, _INPATH_UD.MAX_ARGS_CHARS)

    def test_singletons_and_class_defaults_are_untouched(self):
        self._pairs(GOLDEN_AGENT)
        self.assertEqual(next(d for d in TIER1_DETECTORS if d.name == "TOOL_LOOP").THRESHOLD, 3)
        self.assertEqual(UngroundedDestinationDetector.MAX_ARGS_CHARS, 1_000_000)


class TestGoldenRuns(ParityCase):
    def assert_parity(self, agent_id: str, state_factory, must_fire=(), must_not_fire=()):
        server = _types(run_detectors(state_factory(agent_id), self.server_list(agent_id)))
        client = _types(
            run_detectors(state_factory(agent_id), self.client_list(agent_id), "policy")
        )
        self.assertEqual(server, client, f"{agent_id}: server={server} client={client}")
        for t in must_fire:
            self.assertIn(t, client, agent_id)
        for t in must_not_fire:
            self.assertNotIn(t, client, agent_id)
        return client

    def test_tool_loop_fires_only_under_the_category_override(self):
        self.assert_parity(GOLDEN_AGENT, loop_state, must_fire=["TOOL_LOOP"])
        self.assert_parity(OTHER_AGENT, loop_state, must_not_fire=["TOOL_LOOP"])
        # And the pre-existing client behaviour (class defaults) agrees with `default`.
        self.assertNotIn("TOOL_LOOP", _types(run_detectors(loop_state(OTHER_AGENT))))

    def test_clean_run_is_clean_on_both_sides(self):
        for agent_id in (GOLDEN_AGENT, OTHER_AGENT):
            self.assertEqual(self.assert_parity(agent_id, clean_state), set())

    def test_context_bloat_fires_only_under_the_category_override(self):
        self.assert_parity(GOLDEN_AGENT, bloat_state, must_fire=["CONTEXT_BLOAT"])
        self.assert_parity(OTHER_AGENT, bloat_state, must_not_fire=["CONTEXT_BLOAT"])
        self.assertNotIn("CONTEXT_BLOAT", _types(run_detectors(bloat_state(OTHER_AGENT))))

    def test_baseline_driven_threshold_agrees(self):
        """With a P75 token-growth baseline the static GROWTH_FACTOR is bypassed
        on both sides: the worker sets state.baseline_* directly, the SDK
        copies the payload's baselines onto the state before the pass."""
        baselines = {key: None for key in BASELINE_RESPONSE_KEYS.values()}
        baselines["p75_token_growth"] = 0.8  # threshold 0.8 × INFLATION_FACTOR 2.0 = 1.6 < 1.8

        server_state = bloat_state(OTHER_AGENT)
        server_state.baseline_p75_token_growth = 0.8
        server = _types(run_detectors(server_state, self.server_list(OTHER_AGENT)))

        client_state = bloat_state(OTHER_AGENT)
        payload = self.payload(OTHER_AGENT, baselines=baselines)
        apply_baselines(client_state, payload["baselines"])
        self.assertEqual(client_state.baseline_p75_token_growth, 0.8)
        client = _types(run_detectors(client_state, build_detectors(payload), "policy"))

        self.assertEqual(server, client)
        self.assertIn("CONTEXT_BLOAT", client)
        # Without the baseline `default`'s 2.0 does not fire on growth 1.8.
        self.assert_parity(OTHER_AGENT, bloat_state, must_not_fire=["CONTEXT_BLOAT"])


if __name__ == "__main__":
    unittest.main()
