"""The client stamps ``dropped_events`` on run.completed / run.errored.

The buffer sheds whole runs when it overflows (see ``dunetrace.buffer`` and
``tests/test_buffer.py``); these tests check the *client* side of that
contract — the count reaches the terminal payload, the terminal itself is
never lost, and a healthy run's wire format is unchanged. No network: ``_ship``
is replaced, and parked behind a gate so the buffer genuinely overfills.

Run: cd packages/sdk-py && python -m pytest tests/test_client_dropped_events.py -q
"""

from __future__ import annotations

import threading
import time
import unittest

from dunetrace.client import DunetraceClient
from dunetrace.models import EventType

_TERMINAL = {EventType.RUN_COMPLETED, EventType.RUN_ERRORED}


class _GatedShip:
    """A ``_ship`` that parks the drain thread until released.

    The drain thread ships its first batch and then blocks here, so every
    later push lands in a buffer nothing is emptying — the overload the
    run-shedding rule exists for. Releasing the gate lets the thread finish
    and ``shutdown()`` flush the rest.
    """

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.emitted: list = []

    def __call__(self, batch) -> bool:
        self.gate.wait(10)
        self.emitted.extend(batch)
        return True


def _client(**kwargs) -> DunetraceClient:
    return DunetraceClient(api_key="dt_test", debug=False, **kwargs)


class TestTerminalDroppedEvents(unittest.TestCase):
    def _overloaded_run(self, *, raise_in_run: bool = False, n_tools: int = 30):
        """Drive one run through a 4-slot buffer with the drain thread parked.

        Returns ``(emitted, pushed)``: what shipped, and every event the client
        tried to push (counted on the agent thread, before the buffer decides).
        """
        ship = _GatedShip()
        client = _client(buffer_size=4)
        client._ship = ship
        pushed: list = []
        real_push = client._buffer.push

        def counting_push(item, **kw):
            pushed.append(item)
            return real_push(item, **kw)

        client._buffer.push = counting_push  # type: ignore[method-assign]
        try:
            with client.run("hello", tools=["calc"]) as run:
                for i in range(n_tools):
                    run.tool_called("calc", {"i": i})
                if raise_in_run:
                    raise ValueError("boom")
        except ValueError:
            pass
        finally:
            ship.gate.set()
            client.shutdown(timeout=5)
        return ship.emitted, pushed

    def test_run_completed_carries_dropped_events_after_overload(self):
        emitted, pushed = self._overloaded_run()

        terminals = [e for e in emitted if e.event_type == EventType.RUN_COMPLETED]
        self.assertEqual(len(terminals), 1, "the terminal must never be shed")
        dropped = terminals[0].payload["dropped_events"]
        self.assertGreater(dropped, 0)

        # Nothing is silently lost: every non-terminal event the client emitted
        # either shipped or is in the count.
        delivered = sum(1 for e in emitted if e.event_type not in _TERMINAL)
        attempted = sum(1 for e in pushed if e.event_type not in _TERMINAL)
        self.assertEqual(dropped + delivered, attempted)
        self.assertEqual(delivered + 1, len(emitted))

    def test_run_errored_carries_dropped_events_after_overload(self):
        emitted, pushed = self._overloaded_run(raise_in_run=True)

        terminals = [e for e in emitted if e.event_type in _TERMINAL]
        self.assertEqual([e.event_type for e in terminals], [EventType.RUN_ERRORED])
        dropped = terminals[0].payload["dropped_events"]
        self.assertGreater(dropped, 0)
        self.assertEqual(terminals[0].payload["error_type"], "ValueError")

        delivered = sum(1 for e in emitted if e.event_type not in _TERMINAL)
        attempted = sum(1 for e in pushed if e.event_type not in _TERMINAL)
        self.assertEqual(dropped + delivered, attempted)

    def test_healthy_run_omits_dropped_events(self):
        """Wire format is unchanged when nothing was dropped: no key, not 0."""
        emitted: list = []
        client = _client()
        client._ship = lambda batch: emitted.extend(batch)

        with client.run("hello", tools=["calc"]) as run:
            run.tool_called("calc", {"x": 1})
        try:
            with client.run("again", tools=["calc"]):
                raise ValueError("boom")
        except ValueError:
            pass
        client.shutdown(timeout=5)

        terminals = [e for e in emitted if e.event_type in _TERMINAL]
        self.assertEqual(len(terminals), 2)
        for term in terminals:
            self.assertNotIn("dropped_events", term.payload)

    def test_shed_run_logs_one_warning_naming_the_run(self):
        with self.assertLogs("dunetrace", level="WARNING") as cm:
            emitted, _ = self._overloaded_run()
        run_id = next(e.run_id for e in emitted if e.event_type == EventType.RUN_COMPLETED)
        shed_lines = [line for line in cm.output if "shed run" in line]
        self.assertEqual(len(shed_lines), 1)
        self.assertIn(run_id, shed_lines[0])

    def test_emit_stays_fast_under_sustained_overload(self):
        """The agent thread pays microseconds per event even when every push
        has to shed. Loose bound — two orders of magnitude off an O(n) sweep."""
        ship = _GatedShip()
        client = _client(buffer_size=200)
        client._ship = ship
        n = 5_000
        try:
            t0 = time.perf_counter()
            with client.run("hello", tools=["calc"]) as run:
                for i in range(n):
                    run.tool_called("calc", {"i": i})
            elapsed = time.perf_counter() - t0
        finally:
            ship.gate.set()
            client.shutdown(timeout=5)
        per_event_us = elapsed / n * 1e6
        self.assertLess(per_event_us, 500.0, f"{per_event_us:.1f}µs per event")


if __name__ == "__main__":
    unittest.main()
