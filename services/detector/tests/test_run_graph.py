"""
Delegation run-graph construction (Capability 2, Phase 2.2).

Walks a run's parent_run_id chain into an ordered ancestor chain and derives the
agent-delegation graph from it. Pure traversal — the lineage fetch is injected,
so no database is needed.

Run: python -m unittest tests.test_run_graph -v
"""

from __future__ import annotations

import asyncio
import unittest

from dunetrace import Dunetrace
from dunetrace.detectors import DelegationLoopDetector
from dunetrace.models import EventType

from detector_svc.run_graph import (
    RunNode,
    agent_sequence,
    build_agent_delegation_edges,
    build_ancestor_chain,
    find_cycle,
)


def _fake_fetcher(runs: dict):
    """runs: run_id -> (agent_id, parent_run_id). Returns an async fetcher."""

    async def fetch(run_id):
        if run_id not in runs:
            return None
        agent_id, parent = runs[run_id]
        return {
            "run_id": run_id,
            "agent_id": agent_id,
            "agent_version": "v1",
            "parent_run_id": parent,
        }

    return fetch


def _node(run_id, agent_id, parent):
    return RunNode(run_id=run_id, agent_id=agent_id, agent_version="v1", parent_run_id=parent)


class TestBuildAncestorChain(unittest.TestCase):
    def _chain(self, start, runs, **kw):
        return asyncio.run(build_ancestor_chain(start, _fake_fetcher(runs), **kw))

    def test_single_run_no_parent(self):
        chain = self._chain(_node("r1", "A", None), {})
        self.assertEqual([n.run_id for n in chain], ["r1"])

    def test_walks_to_root(self):
        runs = {
            "r2": ("B", "r1"),
            "r1": ("A", None),
        }
        chain = self._chain(_node("r3", "A", "r2"), runs)
        self.assertEqual([n.run_id for n in chain], ["r3", "r2", "r1"])
        self.assertEqual([n.agent_id for n in chain], ["A", "B", "A"])

    def test_stops_on_missing_ancestor(self):
        # r2's parent id points at a run not yet ingested.
        runs = {"r2": ("B", "r-not-ingested")}
        chain = self._chain(_node("r3", "A", "r2"), runs)
        self.assertEqual([n.run_id for n in chain], ["r3", "r2"])

    def test_respects_max_depth(self):
        runs = {f"r{i}": (f"A{i}", f"r{i - 1}") for i in range(1, 50)}
        runs["r0"] = ("A0", None)
        chain = self._chain(_node("r50", "A50", "r49"), runs, max_depth=5)
        self.assertEqual(len(chain), 6)  # start + 5 hops

    def test_defends_against_runid_cycle(self):
        # Corrupt data: r1 -> r2 -> r1. Must not loop forever.
        runs = {"r1": ("A", "r2"), "r2": ("B", "r1")}
        chain = self._chain(_node("r1", "A", "r2"), runs)
        self.assertEqual([n.run_id for n in chain], ["r1", "r2"])  # stops at repeat


class TestAgentDelegationEdges(unittest.TestCase):
    def test_edges_point_parent_to_child(self):
        # chain leaf-first: A(r3) <- B(r2) <- A(r1). Parent delegates to child.
        chain = [_node("r3", "A", "r2"), _node("r2", "B", "r1"), _node("r1", "A", None)]
        edges = build_agent_delegation_edges(chain)
        self.assertEqual(edges, {"B": {"A"}, "A": {"B"}})  # mutual delegation -> cycle

    def test_no_edges_for_single_node(self):
        self.assertEqual(build_agent_delegation_edges([_node("r1", "A", None)]), {})

    def test_agent_sequence_is_root_first_and_keeps_repetition(self):
        chain = [
            _node("r4", "B", "r3"),
            _node("r3", "A", "r2"),
            _node("r2", "B", "r1"),
            _node("r1", "A", None),
        ]
        self.assertEqual(agent_sequence(chain), ["A", "B", "A", "B"])


class TestFindCycle(unittest.TestCase):
    def test_no_cycle_in_linear_hierarchy(self):
        edges = {"A": {"B"}, "B": {"C"}, "C": {"D"}}
        self.assertIsNone(find_cycle(edges))

    def test_no_cycle_in_fan_out(self):
        edges = {"A": {"B", "C", "D"}}
        self.assertIsNone(find_cycle(edges))

    def test_detects_two_agent_mutual_cycle(self):
        edges = {"A": {"B"}, "B": {"A"}}
        cycle = find_cycle(edges)
        self.assertIsNotNone(cycle)
        self.assertEqual(cycle[0], cycle[-1])  # closes on itself
        self.assertEqual(set(cycle), {"A", "B"})

    def test_detects_three_agent_cycle(self):
        edges = {"A": {"B"}, "B": {"C"}, "C": {"A"}}
        cycle = find_cycle(edges)
        self.assertIsNotNone(cycle)
        self.assertEqual(set(cycle), {"A", "B", "C"})

    def test_empty_graph_has_no_cycle(self):
        self.assertIsNone(find_cycle({}))


class TestDelegationLoopEndToEnd(unittest.TestCase):
    """Full substrate: real SDK nested runs (auto-threaded parent_run_id) ->
    run.started events -> ancestor-chain walk -> DFS -> DELEGATION_LOOP fires.
    The DB lineage fetch is faked from the emitted events (offline)."""

    def test_nested_sdk_runs_form_a_detected_loop(self):
        class _Cap:
            def __init__(self):
                self.events = []

            def handle(self, e):
                self.events.append(e)

        exp = _Cap()
        c = Dunetrace(api_key="k", exporters=[exp])
        c._ship = lambda b: None

        # Agents A and B delegate back and forth: A->B->A->B->A (5 nested runs).
        # parent_run_id is auto-threaded by the SDK (Phase 2.1), no manual ids.
        with c.run("A"):
            with c.run("B"):
                with c.run("A"):
                    with c.run("B"):
                        with c.run("A") as leaf:
                            leaf_id = leaf.run_id
                            leaf_parent = leaf._parent_run_id
        c.shutdown(timeout=1)

        # Build a lineage fetcher from the emitted run.started events (stands in
        # for db.fetch_run_lineage).
        started = {e.run_id: e for e in exp.events if e.event_type == EventType.RUN_STARTED}

        async def fake_fetch(run_id):
            e = started.get(run_id)
            if e is None:
                return None
            return {
                "run_id": e.run_id,
                "agent_id": e.agent_id,
                "agent_version": e.agent_version,
                "parent_run_id": e.parent_run_id,
            }

        # The worker's start node (as process_run would construct it for the leaf).
        start = RunNode(leaf_id, "A", "v", leaf_parent)
        chain = asyncio.run(build_ancestor_chain(start, fake_fetch))
        self.assertEqual(agent_sequence(chain), ["A", "B", "A", "B", "A"])

        cycle = find_cycle(build_agent_delegation_edges(chain))
        sig = DelegationLoopDetector().evaluate_delegation_cycle(
            cycle, agent_sequence(chain), leaf_id, "A", "v"
        )
        self.assertIsNotNone(sig, "DELEGATION_LOOP did not fire end-to-end")
        self.assertEqual(sig.evidence["loop_run_count"], 5)
        self.assertEqual(sig.evidence["cycle_agents"], ["A", "B"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
