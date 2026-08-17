"""
Automatic parent_run_id threading (Capability 2, Phase 2.1).

A run opened inside another active run inherits that run's id as its
parent_run_id, with no manual threading — this links nested multi-agent runs
into a parent/child graph (the substrate DELEGATION_LOOP and HANDOFF_CONTEXT_LOSS
consume). An explicit parent_run_id always wins; a top-level or sequential run
has no parent.

Run: python -m unittest tests.test_parent_run_id -v
"""

from __future__ import annotations

import asyncio
import unittest

from dunetrace import Dunetrace
from dunetrace.context import _current_run
from dunetrace.models import EventType


class _Cap:
    def __init__(self):
        self.events = []

    def handle(self, e):
        self.events.append(e)

    def started(self):
        return [e for e in self.events if e.event_type == EventType.RUN_STARTED]


def _client(exp):
    c = Dunetrace(api_key="k", exporters=[exp])
    c._ship = lambda batch: None
    return c


def _parent_of(exp, agent_id):
    for e in exp.started():
        if e.agent_id == agent_id:
            return e.parent_run_id
    raise AssertionError(f"no run.started for {agent_id}")


class TestParentRunIdThreading(unittest.TestCase):
    def setUp(self):
        # These assertions turn on a clean ambient run; isolate from any run
        # another test in the suite left set in the _current_run contextvar.
        self._tok = _current_run.set(None)

    def tearDown(self):
        _current_run.reset(self._tok)

    def test_nested_run_inherits_parent_run_id(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("orchestrator") as parent:
            with c.run("researcher") as child:
                self.assertEqual(child._parent_run_id, parent.run_id)
        c.shutdown(timeout=1)
        # ...and the run.started event carries it, not just the RunContext.
        self.assertIsNone(_parent_of(exp, "orchestrator"))
        self.assertEqual(_parent_of(exp, "researcher"), parent.run_id)

    def test_explicit_parent_run_id_wins(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("orchestrator"):
            with c.run("researcher", parent_run_id="explicit-parent") as child:
                self.assertEqual(child._parent_run_id, "explicit-parent")
        c.shutdown(timeout=1)

    def test_top_level_run_has_no_parent(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("solo") as run:
            self.assertIsNone(run._parent_run_id)
        c.shutdown(timeout=1)

    def test_sequential_runs_do_not_link(self):
        # The first run has closed (contextvar reset) before the second opens.
        exp = _Cap()
        c = _client(exp)
        with c.run("agent-a"):
            pass
        with c.run("agent-b") as b:
            self.assertIsNone(b._parent_run_id)
        c.shutdown(timeout=1)

    def test_three_level_nesting_links_each_to_its_immediate_parent(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("a") as a:
            with c.run("b") as b:
                with c.run("c") as cc:
                    self.assertEqual(cc._parent_run_id, b.run_id)
                self.assertEqual(b._parent_run_id, a.run_id)
            self.assertIsNone(a._parent_run_id)
        c.shutdown(timeout=1)

    def test_sibling_runs_share_the_same_parent(self):
        exp = _Cap()
        c = _client(exp)
        with c.run("orchestrator") as parent:
            with c.run("worker-1") as w1:
                pass
            with c.run("worker-2") as w2:
                pass
        c.shutdown(timeout=1)
        self.assertEqual(w1._parent_run_id, parent.run_id)
        self.assertEqual(w2._parent_run_id, parent.run_id)

    def test_asyncio_child_task_inherits_parent(self):
        exp = _Cap()
        c = _client(exp)

        async def child_agent(parent_run_id):
            with c.run("child") as child:
                # asyncio copies the parent task's context, so the ambient run
                # propagates into the awaited child task.
                self.assertEqual(child._parent_run_id, parent_run_id)

        async def main():
            with c.run("parent") as parent:
                await child_agent(parent.run_id)

        asyncio.run(main())
        c.shutdown(timeout=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
