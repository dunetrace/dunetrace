"""
Framework memory auto-instrumentation (Phase 1.2).

LangGraph ``BaseStore`` put/get/delete and CrewAI memory save/search/reset become
memory.* events when a ``dt.run()`` is open. Uses the real InMemoryStore and the
real CrewAI Memory class (with a stub storage backend) — no network, no LLM.

Run: python -m unittest tests.test_memory_auto_instrument -v
"""

from __future__ import annotations

import unittest

from dunetrace import Dunetrace
from dunetrace.auto import (
    _patch_crewai_memory,
    _patch_langgraph_store,
    _wrap_crewai_memory_class,
)
from dunetrace.models import EventType


class _Cap:
    def __init__(self):
        self.events = []

    def handle(self, e):
        self.events.append(e)

    def of(self, event_type):
        return [e for e in self.events if e.event_type == event_type]


def _client(exp):
    c = Dunetrace(api_key="k", exporters=[exp])
    c._ship = lambda batch: None
    return c


class TestLangGraphStoreInstrumentation(unittest.TestCase):
    def test_put_get_delete_emit_memory_events(self):
        from langgraph.store.memory import InMemoryStore

        _patch_langgraph_store()
        exp = _Cap()
        c = _client(exp)
        store = InMemoryStore()
        with c.run("a"):
            store.put(("users", "u1"), "prefs", {"text": "likes dark mode"})
            got = store.get(("users", "u1"), "prefs")
            store.delete(("users", "u1"), "prefs")
        c.shutdown(timeout=1)

        # Return value preserved — the patch is transparent.
        self.assertEqual(got.value, {"text": "likes dark mode"})

        writes = exp.of(EventType.MEMORY_WRITTEN)
        reads = exp.of(EventType.MEMORY_READ)
        clears = exp.of(EventType.MEMORY_CLEARED)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].payload["key"], "users/u1:prefs")
        self.assertIn("dark mode", writes[0].payload["value"])
        self.assertNotIn("source", writes[0].payload)  # provenance unknown from store API
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0].payload["key"], "users/u1:prefs")
        self.assertEqual(len(clears), 1)
        self.assertEqual(clears[0].payload["key"], "users/u1:prefs")

    def test_store_outside_run_emits_nothing(self):
        from langgraph.store.memory import InMemoryStore

        _patch_langgraph_store()
        exp = _Cap()
        c = _client(exp)
        store = InMemoryStore()
        store.put(("x",), "k", {"v": 1})  # no run open
        c.shutdown(timeout=1)
        self.assertEqual(exp.of(EventType.MEMORY_WRITTEN), [])

    def test_memory_events_do_not_advance_step(self):
        from langgraph.store.memory import InMemoryStore

        _patch_langgraph_store()
        exp = _Cap()
        c = _client(exp)
        store = InMemoryStore()
        with c.run("a") as run:
            step_before = run.step
            store.put(("n",), "k", {"v": 1})
            self.assertEqual(run.step, step_before)  # memory writes are annotations
        c.shutdown(timeout=1)


class TestCrewAIMemoryInstrumentation(unittest.TestCase):
    def test_real_memory_class_save_emits(self):
        try:
            from crewai.memory.memory import Memory
        except ImportError:
            # Another test in the suite may leave a *stub* crewai in sys.modules
            # (import isolation). The Dummy-class test below covers the emit logic
            # order-independently; this one only runs against the genuine package.
            self.skipTest("real crewai.memory not importable (stub crewai in sys.modules)")

        class _FakeStorage:
            def save(self, *a, **k):
                return None

            def search(self, *a, **k):
                return []

            def reset(self, *a, **k):
                return None

        _patch_crewai_memory()
        m = Memory(storage=_FakeStorage())
        exp = _Cap()
        c = _client(exp)
        with c.run("a"):
            m.save("agent stored this fact")
        c.shutdown(timeout=1)

        writes = exp.of(EventType.MEMORY_WRITTEN)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].payload["key"], "memory")
        self.assertIn("agent stored", writes[0].payload["value"])

    def test_wrapper_covers_save_search_reset(self):
        # A dummy with CrewAI's memory method shape exercises all three verbs
        # (base Memory has no reset(); subclasses do) without needing an embedder.
        class Dummy:
            def save(self, value, metadata=None):
                return "saved"

            def search(self, query):
                return []

            def reset(self):
                return "reset"

        _wrap_crewai_memory_class(Dummy, "dummy")
        exp = _Cap()
        c = _client(exp)
        d = Dummy()
        with c.run("a"):
            saved = d.save("bad memory content")
            d.search("what is the plan")
            d.reset()
        c.shutdown(timeout=1)

        self.assertEqual(saved, "saved")  # return value passes through
        writes = exp.of(EventType.MEMORY_WRITTEN)
        reads = exp.of(EventType.MEMORY_READ)
        clears = exp.of(EventType.MEMORY_CLEARED)
        self.assertEqual(writes[0].payload["key"], "dummy")
        self.assertIn("bad memory content", writes[0].payload["value"])
        self.assertEqual(reads[0].payload["key"], "what is the plan")
        self.assertEqual(clears[0].payload, {"key": None})  # reset clears all


if __name__ == "__main__":
    unittest.main(verbosity=2)
