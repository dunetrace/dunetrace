"""
Ring buffer for the outbound event queue.

Fixed capacity, never blocks the agent thread. When full it sheds the OLDEST
RUN — every buffered event of that run and every later non-terminal event it
emits — rather than the single oldest event. Dropping single events produced
partial runs that the detector could not tell apart from complete ones: a tool
loop with its early calls missing looked clean, and a run missing its
``run.started`` had no tools list. Shedding by run keeps every run that does
reach the detector whole, and the runs that were cut carry a visible
``dropped_events`` count on their terminal event (see ``reserve_terminal``)
so the detector can mark their verdicts as drawn from incomplete data.

Terminal events (``run.completed`` / ``run.errored``) are protected: they are
never tombstoned and never shed to make room for an ordinary event. They are
**not** exempt from the capacity bound. When every live item is a protected
terminal — a backend that has been unreachable long enough for ``maxsize``
runs to finish — the oldest terminal is evicted, and its run's drop record is
kept with the terminal folded into it. That loses a run, loudly; the
alternative, letting the protected set grow without limit, loses the
customer's process. See ``_drop_oldest_protected_locked``.

Two indexes make this cheap and, more importantly, correct:

* ``_dead`` — the tombstone set, keyed by push sequence number. Shedding a run
  marks its buffered events dead in O(k) (paid once per event, so O(1)
  amortised per push); a dead entry is skipped by ``drain`` and discarded when
  it reaches the head.
* ``_shed`` — the per-run *drop count*, the number that rides out on the run's
  terminal event. Bounded at ``max_shed_runs`` and evicted oldest-first.

They are deliberately separate. When they were one map, evicting a drop record
un-tombstoned that run's buffered events, so the record's eviction had to
physically compact them out of the deque — and that compaction rebound
``self._buf`` while ``_drain_locked`` was iterating a stale reference to the
old deque, shipping every event twice and driving the live counter negative.
A negative counter disables shedding (``push`` gates on ``_live >= _maxsize``)
and the buffer then grows without bound inside the customer's process.
Compaction now happens only in ``push``, never underneath a drain.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from threading import Lock
from typing import Any, Deque, Dict, Generic, Hashable, List, Optional, Set, Tuple, TypeVar

T = TypeVar("T")

logger = logging.getLogger("dunetrace")

# Wire names, not the EventType enum, so this module stays free of any model
# import and keeps working for items that are plain objects with ``event_type``
# strings. ``EventType`` is a ``str`` Enum, so membership works for both.
TERMINAL_EVENT_TYPES = frozenset({"run.completed", "run.errored"})

_WARN_INTERVAL_S = 60.0


def _run_key(item: Any) -> Optional[Hashable]:
    """The run an item belongs to, or ``None`` for items that carry no run_id.

    Items without a ``run_id`` (the generic fallback — tests push ints and
    strings) are shed one at a time, oldest first, as a plain ring buffer would.
    """
    return getattr(item, "run_id", None)


def _is_terminal(item: Any) -> bool:
    return getattr(item, "event_type", None) in TERMINAL_EVENT_TYPES


class RingBuffer(Generic[T]):
    """
    Fixed-capacity FIFO, run-aware.

    Capacity counts *live* items — the ones a drain will ship — and is a hard
    bound: ``len(buffer) <= maxsize`` always holds, protected items included.

    Shedding a run does not scan the deque: the run's buffered events are left
    in place as tombstones (their sequence numbers go into ``_dead``), skipped
    when drained and discarded when they reach the head. A per-run FIFO of the
    run's own live entries makes both the shed and the choice of victim O(1)
    amortised — there is no scan for a sheddable run, which is what used to
    turn ``push`` into an O(n) sweep once the head filled with unsheddable
    terminals. The physical deque can hold more than ``maxsize`` entries
    (tombstones are lazy); it is compacted (one O(n) sweep) only if it exceeds
    twice the capacity, which happens at most once per ``maxsize`` pushes.

    Once a run is shed, its later non-terminal events are counted and
    discarded on arrival. Its terminal event is always enqueued and, when it
    drains, the run's drop record is released. A shed run whose terminal never
    comes must not pin memory forever: ``_shed`` is capped at ``max_shed_runs``
    and the oldest record is evicted first. Because tombstones are tracked
    separately, evicting a record leaves the deque alone.

    The single lock keeps the critical sections tiny (a few dict operations),
    so the agent thread is never parked for more than microseconds behind a
    drain.
    """

    def __init__(self, maxsize: int = 10_000, *, max_shed_runs: int = 4096) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        # (seq, item) in push order. ``seq`` is a monotonic per-buffer counter:
        # it identifies an entry unambiguously, where ``id(item)`` cannot (the
        # same object — a small int, an interned string — can be pushed twice).
        self._buf: Deque[Tuple[int, T]] = deque()
        self._maxsize = maxsize
        self._max_shed_runs = max(1, max_shed_runs)
        self._lock = Lock()
        self._seq = 0
        self._live = 0  # items a drain will ship (buffered minus tombstones)
        # run_id -> FIFO of that run's live, sheddable (non-protected) entries,
        # in run-arrival order. The first key is the oldest sheddable run and
        # its head entry carries that run's oldest sequence number, which is
        # what makes picking a shed victim O(1).
        self._run_items: "OrderedDict[Hashable, Deque[Tuple[int, T]]]" = OrderedDict()
        # Live, sheddable entries carrying no run_id, in push order.
        self._keyless: Deque[Tuple[int, T]] = deque()
        # seqs of tombstoned entries still physically in the deque.
        self._dead: Set[int] = set()
        # seqs of non-terminal entries pushed with force=True that are still in
        # the deque. Protected like terminals: never tombstoned, never dropped
        # to make room for an ordinary event.
        self._forced: Set[int] = set()
        # run_id -> events dropped for that run (buffered at shed time + later
        # arrivals). Insertion order is shed order, so eviction is oldest-first.
        self._shed: "OrderedDict[Hashable, int]" = OrderedDict()
        self._shed_runs_total = 0
        self._protected_dropped_total = 0
        # None, not 0.0, means "never warned". time.monotonic()'s epoch is
        # arbitrary — on Linux it is time since boot — so on a host that has
        # been up for less than _WARN_INTERVAL_S, `now - 0.0` is itself under
        # the interval and the sentinel reads as a warning issued moments ago.
        # That swallowed the FIRST shed warning on a freshly booted machine,
        # which is exactly when a buffer overflowing most needs saying.
        self._last_warn_ts: Optional[float] = None
        self._sheds_since_warn = 0
        self._last_protected_warn_ts: Optional[float] = None
        self._protected_drops_since_warn = 0

    # ── Producer side ────────────────────────────────────────────────────────

    def push(self, item: T, *, force: bool = False) -> bool:
        """Append *item*. Never blocks for more than a few dict operations.

        Returns ``True`` if the item was enqueued, ``False`` if it was dropped.

        A non-terminal event of a run that has already been shed is counted
        against that run and dropped. When the buffer is full the oldest run is
        shed to make room (see ``_make_room_locked``). Terminal events — or any
        item pushed with ``force=True`` — are never shed to make room for an
        ordinary event, and will evict the oldest protected item rather than
        push the buffer past its capacity; they are what carries the drop count
        to the server, so losing one is a last resort, not a routine one.
        """
        key = _run_key(item)
        is_terminal = _is_terminal(item)
        protected = force or is_terminal
        with self._lock:
            if key is not None and key in self._shed and not protected:
                self._shed[key] += 1
                return False
            while self._live >= self._maxsize:
                if not self._make_room_locked(allow_protected=protected):
                    break
            if not protected:
                if key is not None and key in self._shed:
                    # The eviction just shed this item's own run.
                    self._shed[key] += 1
                    return False
                if self._live >= self._maxsize:
                    # Only protected items are left, and an ordinary event is
                    # not worth a terminal. Drop this item — and shed its run,
                    # so it is cut whole, not piecemeal.
                    if key is not None:
                        self._shed_run_locked(key, extra=1)
                    return False
            self._append_locked(key, item, protected=protected, is_terminal=is_terminal)
            if len(self._buf) > 2 * self._maxsize:
                self._compact_locked()
            return True

    def reserve_terminal(self, run_id: Hashable) -> int:
        """Make room for *run_id*'s terminal event and return its drop count.

        Called by the client just before it builds ``run.completed`` /
        ``run.errored``: any shedding needed to fit the terminal happens *now*,
        so the count it returns already includes it. The event should then be
        pushed with ``force=True``. Returns 0 for a run nothing was dropped
        from.
        """
        with self._lock:
            while self._live >= self._maxsize:
                if not self._make_room_locked(allow_protected=True):
                    break
            return self._shed.get(run_id, 0)

    def dropped_count(self, run_id: Hashable) -> int:
        """Events dropped so far for *run_id* (0 if the run was never shed)."""
        with self._lock:
            return self._shed.get(run_id, 0)

    def forget(self, run_id: Hashable) -> None:
        """Release *run_id*'s drop record explicitly.

        The record is released on its own when the run's terminal event drains;
        this is for callers that know the terminal will never come. Tombstones
        the run still has in the deque stay dead — they are tracked in
        ``_dead``, not here — so nothing has to be swept out of the deque.
        """
        with self._lock:
            self._shed.pop(run_id, None)

    # ── Consumer side ────────────────────────────────────────────────────────

    def drain(self, n: int = 100) -> List[T]:
        """Return up to *n* live items from the front, removing them from the buffer."""
        with self._lock:
            return self._drain_locked(n)

    def drain_all(self) -> List[T]:
        """Drain every live item currently in the buffer."""
        with self._lock:
            return self._drain_locked(len(self._buf))

    def __len__(self) -> int:
        return self._live

    def __bool__(self) -> bool:
        return self._live > 0

    @property
    def shed_runs_total(self) -> int:
        """Number of runs shed since construction (for tests and diagnostics)."""
        return self._shed_runs_total

    @property
    def protected_dropped_total(self) -> int:
        """Protected items (terminals, forced pushes) evicted to hold capacity.

        Non-zero means whole runs were lost with nothing on the wire to say so
        — the buffer was full of terminals, i.e. the backend has been
        unreachable for at least ``maxsize`` completed runs.
        """
        return self._protected_dropped_total

    # ── Internals (caller holds the lock) ────────────────────────────────────

    def _append_locked(
        self, key: Optional[Hashable], item: T, *, protected: bool, is_terminal: bool
    ) -> None:
        seq = self._seq
        self._seq += 1
        entry = (seq, item)
        self._buf.append(entry)
        self._live += 1
        if protected:
            # Terminals are recognised from the item itself; only forced
            # non-terminals need recording.
            if not is_terminal:
                self._forced.add(seq)
        elif key is not None:
            run = self._run_items.get(key)
            if run is None:
                run = self._run_items[key] = deque()
            run.append(entry)
        else:
            self._keyless.append(entry)

    def _drain_locked(self, n: int) -> List[T]:
        batch: List[T] = []
        while self._buf and len(batch) < n:
            seq, item = self._buf.popleft()
            if seq in self._dead:
                self._dead.discard(seq)
                continue
            key = _run_key(item)
            if _is_terminal(item):
                if key is not None and key in self._shed:
                    # The run was shed after this terminal was already
                    # buffered, so the count the client stamped is stale (or
                    # absent). The wire must say how much is missing, or the
                    # detector reads a partial run as a whole one.
                    self._stamp_terminal(item, self._shed[key])
                    self._shed.pop(key, None)
            elif seq in self._forced:
                self._forced.discard(seq)
            elif key is not None:
                self._pop_run_entry_locked(key, seq)
            else:
                self._pop_keyless_locked(seq)
            self._live -= 1
            batch.append(item)
        return batch

    def _make_room_locked(self, *, allow_protected: bool) -> bool:
        """Reclaim at least one live slot. ``False`` if nothing could be freed.

        Sheddable work always goes first; a protected item is evicted only when
        the caller is itself pushing a protected item and there is nothing else
        left to give.
        """
        self._trim_dead_head_locked()
        if self._shed_oldest_locked():
            return True
        if allow_protected:
            return self._drop_oldest_protected_locked()
        return False

    def _trim_dead_head_locked(self) -> None:
        """Discard tombstones sitting at the head. O(1) amortised."""
        buf = self._buf
        dead = self._dead
        while buf and buf[0][0] in dead:
            dead.discard(buf.popleft()[0])

    def _shed_oldest_locked(self) -> bool:
        """Shed the oldest sheddable thing — a whole run, or one keyless item.

        O(1): the victim is the head of ``_run_items`` (run-arrival order) or
        the head of ``_keyless``, whichever entered the buffer first. Returns
        ``False`` when every live item is protected.
        """
        run_key: Optional[Hashable] = None
        run_seq: Optional[int] = None
        while self._run_items:
            candidate = next(iter(self._run_items))
            entries = self._run_items[candidate]
            if not entries:  # pragma: no cover - defensive
                del self._run_items[candidate]
                continue
            run_key, run_seq = candidate, entries[0][0]
            break
        keyless_seq = self._keyless[0][0] if self._keyless else None
        if run_seq is None and keyless_seq is None:
            return False
        if keyless_seq is not None and (run_seq is None or keyless_seq < run_seq):
            # Generic item: no run to shed, drop just this one.
            seq, _ = self._keyless.popleft()
            self._dead.add(seq)
            self._live -= 1
            return True
        if run_key is None:  # pragma: no cover - narrowed by run_seq is not None
            return False
        self._shed_run_locked(run_key)
        return True

    def _drop_oldest_protected_locked(self) -> bool:
        """Evict the oldest protected item. The capacity bound's last resort.

        Reached only when every live item is a terminal (or a forced push) and
        another one is arriving — i.e. the drain side has been stuck for at
        least ``maxsize`` completed runs. The evicted run is then shed whole
        and its drop count kept, with the terminal folded into it (``extra=1``)
        so ``dropped_count`` still reports the true size of the loss; there is
        no longer any event of that run on which to ship it.

        The head of the deque is the oldest protected item: ``_make_room_locked``
        trims dead entries first, and every live non-protected entry is indexed
        in ``_run_items``/``_keyless``, both of which the caller found empty.
        """
        if not self._buf:
            return False
        seq, item = self._buf.popleft()
        self._live -= 1
        self._forced.discard(seq)
        self._protected_dropped_total += 1
        key = _run_key(item)
        if key is not None:
            self._shed_run_locked(key, warn=False, extra=1)
        self._warn_protected_drop_locked(key)
        return True

    def _shed_run_locked(self, key: Hashable, *, warn: bool = True, extra: int = 0) -> None:
        """Tombstone every buffered event of *key* and start counting drops for it."""
        entries = self._run_items.pop(key, None)
        count = 0
        if entries:
            count = len(entries)
            self._live -= count
            dead = self._dead
            for entry_seq, _ in entries:
                dead.add(entry_seq)
        # pop-then-set moves the record to the end, so a re-shed refreshes its
        # position and eviction stays oldest-shed-first.
        self._shed[key] = self._shed.pop(key, 0) + count + extra
        self._shed_runs_total += 1
        while len(self._shed) > self._max_shed_runs:
            self._shed.popitem(last=False)
        if warn:
            self._warn_locked(key, count)

    def _pop_run_entry_locked(self, key: Hashable, seq: int) -> None:
        """Remove a drained entry from its run's index.

        A run's live entries sit in ``_run_items[key]`` in push order and leave
        it only from the front (a drain) or all at once (a shed), so the entry
        being drained is always the head.
        """
        entries = self._run_items.get(key)
        if entries is None:
            return
        if entries and entries[0][0] == seq:
            entries.popleft()
        if not entries:
            del self._run_items[key]

    def _pop_keyless_locked(self, seq: int) -> None:
        if self._keyless and self._keyless[0][0] == seq:
            self._keyless.popleft()

    def _compact_locked(self) -> None:
        """Physically remove every tombstone. O(n).

        Called from ``push`` only — never while a drain holds a reference into
        the deque, which is what made the old per-run compaction unsound.
        """
        if not self._dead:
            return
        dead = self._dead
        self._buf = deque(entry for entry in self._buf if entry[0] not in dead)
        dead.clear()

    @staticmethod
    def _stamp_terminal(item: Any, dropped: int) -> None:
        payload = getattr(item, "payload", None)
        if isinstance(payload, dict):
            try:
                current = int(payload.get("dropped_events") or 0)
            except (TypeError, ValueError):
                current = 0
            if dropped > current:
                payload["dropped_events"] = dropped

    def _warn_locked(self, key: Hashable, count: int) -> None:
        """One WARNING per shed run, rate-limited to one line a minute."""
        self._sheds_since_warn += 1
        now = time.monotonic()
        if self._last_warn_ts is not None and now - self._last_warn_ts < _WARN_INTERVAL_S:
            return
        suppressed = self._sheds_since_warn - 1
        self._last_warn_ts = now
        self._sheds_since_warn = 0
        logger.warning(
            "Dunetrace: event buffer full (%d) — shed run %s (%d buffered events dropped); "
            "%d other run(s) shed since the last warning. The run will reach the server "
            "with dropped_events set and its signals held in shadow.",
            self._maxsize,
            key,
            count,
            suppressed,
        )

    def _warn_protected_drop_locked(self, key: Optional[Hashable]) -> None:
        """One WARNING per evicted terminal, rate-limited to one line a minute."""
        self._protected_drops_since_warn += 1
        now = time.monotonic()
        if (
            self._last_protected_warn_ts is not None
            and now - self._last_protected_warn_ts < _WARN_INTERVAL_S
        ):
            return
        suppressed = self._protected_drops_since_warn - 1
        self._last_protected_warn_ts = now
        self._protected_drops_since_warn = 0
        logger.warning(
            "Dunetrace: event buffer full of terminal events (%d) — dropped run %s's "
            "terminal event; %d other terminal(s) dropped since the last warning. "
            "Those runs are lost entirely and nothing on the wire will say so. "
            "The ingest endpoint has been unreachable long enough for %d runs to "
            "finish — check connectivity and DUNETRACE_INGEST_URL.",
            self._maxsize,
            key,
            suppressed,
            self._maxsize,
        )
