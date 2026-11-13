"""Tests for the run-aware RingBuffer.

Run: cd packages/sdk-py && python -m pytest tests/test_buffer.py -q
"""

from __future__ import annotations

import logging
import threading
import time
import unittest
import unittest.mock

from dunetrace.buffer import RingBuffer


class _Ev:
    """Minimal stand-in for AgentEvent: the buffer only reads these three."""

    __slots__ = ("run_id", "event_type", "payload")

    def __init__(self, run_id: str, event_type: str = "tool.called", payload=None):
        self.run_id = run_id
        self.event_type = event_type
        self.payload = {} if payload is None else payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Ev({self.run_id}, {self.event_type})"


def _done(run_id: str, payload=None) -> _Ev:
    return _Ev(run_id, "run.completed", payload)


class TestRingBufferGeneric(unittest.TestCase):
    """Items without a run_id keep the plain ring-buffer contract."""

    def test_push_and_drain(self):
        buf = RingBuffer(maxsize=10)
        buf.push("a")
        buf.push("b")
        buf.push("c")
        result = buf.drain(10)
        self.assertEqual(result, ["a", "b", "c"])
        self.assertEqual(len(buf), 0)

    def test_drain_respects_n(self):
        buf = RingBuffer(maxsize=10)
        for i in range(5):
            buf.push(i)
        result = buf.drain(3)
        self.assertEqual(result, [0, 1, 2])
        self.assertEqual(len(buf), 2)

    def test_ring_drops_oldest_when_full(self):
        """No run_id to shed by, so the single oldest item goes — as before."""
        buf = RingBuffer(maxsize=3)
        buf.push(1)
        buf.push(2)
        buf.push(3)
        self.assertTrue(buf.push(4))  # 4 is enqueued, 1 was dropped
        result = buf.drain_all()
        self.assertEqual(result, [2, 3, 4])

    def test_drain_all_empties_buffer(self):
        buf = RingBuffer(maxsize=100)
        for i in range(50):
            buf.push(i)
        result = buf.drain_all()
        self.assertEqual(len(result), 50)
        self.assertEqual(len(buf), 0)

    def test_bool_and_len(self):
        buf = RingBuffer(maxsize=5)
        self.assertFalse(buf)
        buf.push("x")
        self.assertTrue(buf)
        self.assertEqual(len(buf), 1)

    def test_concurrent_push_drain(self):
        """Multiple threads pushing should not crash or lose the buffer invariant."""
        buf: RingBuffer[int] = RingBuffer(maxsize=1000)
        errors: list = []

        def pusher(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    buf.push(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertLessEqual(len(buf), 1000)


class TestRunAwareShedding(unittest.TestCase):
    def test_full_buffer_sheds_whole_oldest_run_not_single_event(self):
        """Two interleaved runs, capacity 4. The fifth push sheds ALL of run A —
        both its buffered events — so run B and the new event stay intact."""
        buf = RingBuffer(maxsize=4)
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))
        self.assertTrue(buf.push(_Ev("C")))

        drained = buf.drain_all()
        self.assertEqual([e.run_id for e in drained], ["B", "B", "C"])
        self.assertEqual(buf.dropped_count("A"), 2)
        self.assertEqual(buf.dropped_count("B"), 0)
        self.assertEqual(buf.shed_runs_total, 1)

    def test_shedding_frees_all_of_the_runs_slots(self):
        """Shedding a 3-event run makes room for 3 pushes before the next shed."""
        buf = RingBuffer(maxsize=3)
        for _ in range(3):
            buf.push(_Ev("A"))
        buf.push(_Ev("B"))  # sheds A (3 events), B enqueued -> live 1
        buf.push(_Ev("B"))
        buf.push(_Ev("B"))  # live 3, nothing shed
        self.assertEqual(buf.shed_runs_total, 1)
        self.assertEqual(buf.dropped_count("A"), 3)
        self.assertEqual([e.run_id for e in buf.drain_all()], ["B", "B", "B"])

    def test_later_events_of_a_shed_run_are_counted_and_dropped(self):
        buf = RingBuffer(maxsize=2)
        buf.push(_Ev("A"))
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))  # sheds A
        self.assertFalse(buf.push(_Ev("A")))
        self.assertFalse(buf.push(_Ev("A")))
        self.assertEqual(buf.dropped_count("A"), 4)
        self.assertEqual(len(buf), 1)
        self.assertEqual([e.run_id for e in buf.drain_all()], ["B"])

    def test_event_whose_own_run_is_shed_by_its_push_is_dropped(self):
        """A single run fills the buffer; its next event sheds the run itself
        and must not sneak in as the lone survivor of a run it just cut."""
        buf = RingBuffer(maxsize=3)
        for _ in range(3):
            buf.push(_Ev("A"))
        self.assertFalse(buf.push(_Ev("A")))
        self.assertEqual(buf.dropped_count("A"), 4)
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.drain_all(), [])

    def test_terminal_event_is_forced_through_a_full_buffer(self):
        buf = RingBuffer(maxsize=2)
        buf.push(_Ev("A"))
        buf.push(_Ev("A"))
        done = _done("B")
        self.assertTrue(buf.push(done))  # sheds A to make room
        self.assertEqual(buf.dropped_count("A"), 2)
        self.assertEqual(buf.drain_all(), [done])

    def test_terminal_event_of_a_shed_run_is_still_enqueued(self):
        buf = RingBuffer(maxsize=2)
        buf.push(_Ev("A"))
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))  # sheds A
        buf.push(_Ev("A"))  # dropped, counted
        done = _done("A")
        self.assertTrue(buf.push(done))
        drained = buf.drain_all()
        self.assertIn(done, drained)
        self.assertEqual([e.run_id for e in drained], ["B", "A"])

    def test_force_flag_enqueues_a_non_terminal_item(self):
        buf = RingBuffer(maxsize=1)
        buf.push(_Ev("A"))
        forced = _Ev("A", "tool.called")
        self.assertTrue(buf.push(forced, force=True))
        self.assertEqual(buf.drain_all(), [forced])

    def test_reserve_terminal_sheds_first_and_returns_the_count(self):
        """The client reads the count BEFORE building the terminal payload, so
        any shed needed to fit the terminal must already be in it."""
        buf = RingBuffer(maxsize=2)
        buf.push(_Ev("A"))
        buf.push(_Ev("A"))
        self.assertEqual(buf.reserve_terminal("A"), 2)
        self.assertEqual(len(buf), 0)
        self.assertTrue(buf.push(_done("A"), force=True))
        self.assertEqual(buf.reserve_terminal("never-shed"), 0)

    def test_record_is_released_when_the_terminal_drains(self):
        buf = RingBuffer(maxsize=2)
        buf.push(_Ev("A"))
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))
        self.assertEqual(buf.dropped_count("A"), 2)
        buf.push(_done("A"), force=True)
        buf.drain_all()
        self.assertEqual(buf.dropped_count("A"), 0)
        # A fresh run reusing the id is not treated as shed.
        self.assertTrue(buf.push(_Ev("A")))

    def test_forget_releases_record_and_its_tombstones(self):
        buf = RingBuffer(maxsize=2)
        buf.push(_Ev("A"))
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))  # A shed; its two tombstones still sit in the deque
        buf.forget("A")
        self.assertEqual(buf.dropped_count("A"), 0)
        self.assertEqual([e.run_id for e in buf.drain_all()], ["B"])

    def test_run_shed_after_its_terminal_was_buffered_gets_stamped_on_drain(self):
        """The client wrote no dropped_events (nothing was dropped yet), then
        the run was shed while it sat in the buffer. Its terminal must not
        ship looking healthy."""
        buf = RingBuffer(maxsize=3)
        buf.push(_Ev("A"))
        buf.push(_Ev("A"))
        done = _done("A", payload={"exit_reason": "completed"})
        buf.push(done)
        buf.push(_Ev("B"))  # full: sheds A (two non-terminal events)
        drained = buf.drain_all()
        self.assertEqual([e.run_id for e in drained], ["A", "B"])
        self.assertIs(drained[0], done)
        self.assertEqual(done.payload["dropped_events"], 2)
        self.assertEqual(buf.dropped_count("A"), 0)  # released on drain

    def test_a_larger_client_count_is_not_overwritten_by_the_stamp(self):
        buf = RingBuffer(maxsize=3)
        buf.push(_Ev("A"))
        done = _done("A", payload={"dropped_events": 7})
        buf.push(done)
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))  # sheds A
        buf.drain_all()
        self.assertEqual(done.payload["dropped_events"], 7)

    def test_shed_map_is_bounded_and_evicts_oldest(self):
        buf = RingBuffer(maxsize=1, max_shed_runs=3)
        for i in range(6):
            buf.push(_Ev(f"r{i}"))  # each push from r1 on sheds the previous run
        self.assertEqual(buf.shed_runs_total, 5)
        self.assertLessEqual(len(buf._shed), 3)
        self.assertEqual(buf.dropped_count("r0"), 0)  # evicted
        self.assertEqual(buf.dropped_count("r4"), 1)  # still tracked
        self.assertEqual(buf.dropped_count("r5"), 0)  # never shed

    def test_evicting_a_drop_record_leaves_its_tombstones_dead(self):
        """Evicting a run's drop record must not resurrect its tombstones.

        Tombstones live in ``_dead`` (keyed by sequence number), the drop
        record in ``_shed``. They used to be one map, so evicting the record
        un-tombstoned the run's buffered events and the eviction had to
        compact them out of the deque — the compaction that corrupted an
        in-flight drain. Keeping them apart means eviction touches nothing
        but the record."""
        buf = RingBuffer(maxsize=4, max_shed_runs=1)
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))
        buf.push(_Ev("C"))  # sheds A -> tombstones A,A in deque; map={A}
        buf.push(_Ev("C"))
        buf.push(_Ev("C"))  # sheds B -> map cap evicts A while A's tombstones remain
        self.assertEqual([e.run_id for e in buf.drain_all()], ["C", "C", "C"])

    def test_compaction_cannot_corrupt_an_in_flight_drain(self):
        """The audited four-push reproducer.

        A's terminal is buffered FIRST, then a late non-terminal for A, then
        run B — so the fourth push sheds A while its terminal sits ahead of
        the tombstone in the deque. Draining then released A's record, which
        rebound ``self._buf`` to a compacted deque underneath a loop holding
        the old one: every event shipped a second time and ``_live`` went to
        -3, at which point ``push``'s ``while self._live >= self._maxsize``
        gate could never fire again and the buffer grew without bound inside
        the customer's process."""
        buf = RingBuffer(maxsize=3)
        buf.push(_done("A"))  # A's terminal, buffered FIRST
        buf.push(_Ev("A", "llm.responded"))  # late event for A
        buf.push(_Ev("B", "run.started"))
        buf.push(_Ev("B", "tool.called"))  # overflow -> sheds oldest run A

        first = buf.drain_all()
        # A's terminal survives (protected) and carries the drop count; its
        # late non-terminal is a tombstone and is not shipped.
        self.assertEqual(
            [(e.run_id, e.event_type) for e in first],
            [("A", "run.completed"), ("B", "run.started"), ("B", "tool.called")],
        )
        self.assertEqual(first[0].payload["dropped_events"], 1)

        # Nothing is delivered twice, and the live counter lands exactly on 0.
        self.assertEqual(buf.drain_all(), [])
        self.assertEqual(len(buf), 0)
        self.assertEqual(len(buf._buf), 0)

        # len() was raising ValueError("__len__() should return >= 0") here.
        self.assertGreaterEqual(len(buf), 0)

        # And the capacity gate still works afterwards.
        for i in range(10):
            buf.push(_Ev(f"later{i}"))
        self.assertLessEqual(len(buf), 3)

    def test_no_event_is_ever_delivered_twice(self):
        """Identity check across a long interleaved workload: shedding,
        forced terminals, record eviction and compaction all at once."""
        buf = RingBuffer(maxsize=25, max_shed_runs=4)
        seen: list = []
        for i in range(400):
            run = f"r{i // 3}"
            buf.push(_Ev(run))
            if i % 3 == 2:
                buf.push(_done(run), force=True)
            if i % 7 == 0:
                seen.extend(buf.drain(10))
        seen.extend(buf.drain_all())
        ids = [id(e) for e in seen]
        self.assertEqual(len(ids), len(set(ids)), "an event was delivered twice")
        self.assertEqual(len(buf), 0)

    def test_live_counter_never_goes_negative_under_churn(self):
        buf = RingBuffer(maxsize=8, max_shed_runs=2)
        for i in range(500):
            run = f"r{i // 2}"
            buf.push(_Ev(run))
            buf.push(_done(run), force=True)
            if i % 5 == 0:
                buf.drain(3)
            self.assertGreaterEqual(len(buf), 0)
            self.assertLessEqual(len(buf), 8)
        buf.drain_all()
        self.assertEqual(len(buf), 0)

    def test_physical_deque_is_bounded_by_twice_capacity(self):
        buf = RingBuffer(maxsize=50)
        runs = [f"r{i}" for i in range(10)]
        for i in range(20_000):
            buf.push(_Ev(runs[i % 10]))
        self.assertLessEqual(len(buf._buf), 100)
        self.assertLessEqual(len(buf), 50)

    def test_mixed_generic_and_run_items(self):
        buf = RingBuffer(maxsize=2)
        buf.push("plain")
        buf.push(_Ev("A"))
        buf.push(_Ev("B"))  # head is generic -> just it is dropped
        self.assertEqual(buf.shed_runs_total, 0)
        self.assertEqual([getattr(e, "run_id", e) for e in buf.drain_all()], ["A", "B"])

    def test_one_warning_per_minute_with_count(self):
        buf = RingBuffer(maxsize=1)
        with self.assertLogs("dunetrace", level="WARNING") as cm:
            for i in range(5):
                buf.push(_Ev(f"r{i}"))
        self.assertEqual(len(cm.output), 1)
        self.assertIn("shed run r0", cm.output[0])
        # 4 runs shed, the first logged, the other 3 counted at the next line.
        buf._last_warn_ts = None  # "never warned" — see the sentinel note in buffer.py
        with self.assertLogs("dunetrace", level="WARNING") as cm:
            buf.push(_Ev("late"))
        self.assertIn("3 other run(s)", cm.output[0])
        logging.getLogger("dunetrace").handlers.clear()

    def test_first_warning_fires_on_a_freshly_booted_host(self):
        """time.monotonic()'s epoch is arbitrary (seconds since boot on Linux),
        so on a host up less than _WARN_INTERVAL_S the clock reads under 60.
        With 0.0 as the "never warned" sentinel that made the very first shed
        warning look like one issued moments ago, and it was dropped — on CI
        runners and freshly started containers, which is where a buffer
        overflowing most needs saying. Pin both rate limiters at t=3s.
        """
        for attr, make_overflow in (
            ("_last_warn_ts", lambda buf: [buf.push(_Ev(f"r{i}")) for i in range(2)]),
            (
                "_last_protected_warn_ts",
                lambda buf: [buf.push(_done(f"r{i}"), force=True) for i in range(3)],
            ),
        ):
            with self.subTest(limiter=attr):
                buf = RingBuffer(maxsize=1)
                self.assertIsNone(getattr(buf, attr), "sentinel must be None, not 0.0")
                with unittest.mock.patch("dunetrace.buffer.time.monotonic", return_value=3.0):
                    with self.assertLogs("dunetrace", level="WARNING") as cm:
                        make_overflow(buf)
                self.assertTrue(cm.output)
                logging.getLogger("dunetrace").handlers.clear()

    def test_push_stays_fast_under_sustained_overload(self):
        """Every push here sheds a run (worst case: one event per run) or
        one in ten (round-robin). Loose bound: well under 100µs a push on
        average, which is two orders of magnitude off an O(n) sweep."""
        buf = RingBuffer(maxsize=1000)
        n = 20_000
        events = [_Ev(f"r{i}") for i in range(n)]
        t0 = time.perf_counter()
        for ev in events:
            buf.push(ev)
        per_push_us = (time.perf_counter() - t0) / n * 1e6
        self.assertLess(per_push_us, 100.0, f"{per_push_us:.1f}µs per push")

        buf = RingBuffer(maxsize=1000)
        runs = [f"rr{i}" for i in range(10)]
        events = [_Ev(runs[i % 10]) for i in range(n)]
        t0 = time.perf_counter()
        for ev in events:
            buf.push(ev)
        per_push_us = (time.perf_counter() - t0) / n * 1e6
        self.assertLess(per_push_us, 100.0, f"{per_push_us:.1f}µs per push")


if __name__ == "__main__":
    unittest.main()
