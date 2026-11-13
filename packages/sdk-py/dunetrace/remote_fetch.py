"""
Per-agent bookkeeping for anything the SDK refreshes from the server in the
background: the remote policy bundle (``PolicyEngine``) and the detector
configuration (``DetectorConfigStore``). One implementation so the two
cannot drift — the same 60s success TTL, the same 2s/4s/8s/15s failure
backoff, the same in-flight guard, the same staleness verdict.

The contract the client's fetch helpers drive:

    needs_fetch()       — is a fetch due for this agent right now?
    begin_fetch()       — claim the in-flight slot (False if already claimed)
    mark_fetched()      — a load succeeded: start the TTL, clear any streak
    mark_fetch_failed() — a load failed: start/extend the backoff streak
    end_fetch()         — release the in-flight slot (always, in a ``finally``)
    bundle_status()     — ``(stale, age_s)`` for the policy.evaluated payload

A *failed* fetch never earns the success TTL. It is retried on an exponential
backoff — 2s, 4s, 8s, then capped — so an agent whose server is briefly
unreachable runs on stale data for seconds, not a full TTL. The cap is
deliberately well under the TTL.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Set, Tuple


class RemoteFetchState:
    """Thread-safe fetch bookkeeping keyed by ``agent_id``. Subclasses share
    ``self._lock`` for their own state (one lock per store, not two)."""

    _FETCH_TTL = 60.0  # seconds between remote refreshes per agent_id
    _FETCH_BACKOFF_BASE = 2.0
    _FETCH_BACKOFF_MAX = 15.0

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        # agent_id → monotonic time of the last SUCCESSFUL remote load. Only
        # mark_fetched() writes it; a failed fetch never does.
        self._fetch_times: Dict[str, float] = {}
        # agent_id → (monotonic time of last failure, consecutive failures).
        # Cleared by mark_fetched(). Drives needs_fetch()'s backoff.
        self._fetch_failures: Dict[str, Tuple[float, int]] = {}
        # Agents with a fetch currently on the wire. needs_fetch() is False
        # for these, and begin_fetch() is a test-and-set so two runs starting
        # together still produce one request, not two.
        self._fetch_in_flight: Set[str] = set()

    def mark_fetched(self, agent_id: str) -> None:
        """Record a **successful** remote load for ``agent_id``.

        Only call this once the data is actually in memory. It used to be
        called before the request as a stampede guard, which meant a failed
        fetch inherited the full success TTL and the agent ran unguarded for
        a minute with nothing above DEBUG in the logs. begin_fetch() is the
        stampede guard now.
        """
        with self._lock:
            self._fetch_times[agent_id] = time.monotonic()
            self._fetch_failures.pop(agent_id, None)

    def mark_fetch_failed(self, agent_id: str) -> int:
        """Record a failed remote fetch. Returns the length of the current
        failure streak (1 for the first failure), so the caller can decide
        how loudly to log."""
        with self._lock:
            prev = self._fetch_failures.get(agent_id)
            streak = prev[1] + 1 if prev is not None else 1
            self._fetch_failures[agent_id] = (time.monotonic(), streak)
            return streak

    def fetch_failure_streak(self, agent_id: str) -> int:
        """Consecutive failed fetches for ``agent_id`` since the last success."""
        with self._lock:
            prev = self._fetch_failures.get(agent_id)
            return prev[1] if prev is not None else 0

    @classmethod
    def fetch_backoff(cls, streak: int) -> float:
        """Seconds to wait after the ``streak``-th consecutive failure:
        2, 4, 8, then capped at _FETCH_BACKOFF_MAX."""
        if streak <= 0:
            return 0.0
        return min(cls._FETCH_BACKOFF_MAX, cls._FETCH_BACKOFF_BASE * (2.0 ** (streak - 1)))

    def begin_fetch(self, agent_id: str) -> bool:
        """Claim the in-flight slot for ``agent_id``. Returns False if another
        fetch for the same agent is already on the wire — the caller should
        then do nothing. Always pair with end_fetch() in a ``finally``."""
        with self._lock:
            if agent_id in self._fetch_in_flight:
                return False
            self._fetch_in_flight.add(agent_id)
            return True

    def end_fetch(self, agent_id: str) -> None:
        with self._lock:
            self._fetch_in_flight.discard(agent_id)

    def needs_fetch(self, agent_id: str) -> bool:
        """True when a remote fetch for ``agent_id`` is due.

        - never loaded and never failed → True
        - a fetch already in flight → False (no stampede)
        - last attempt failed → True once the backoff for that streak has
          elapsed (2s, 4s, 8s, … capped), regardless of the success TTL
        - last attempt succeeded → True once _FETCH_TTL has elapsed
        """
        now = time.monotonic()
        with self._lock:
            if agent_id in self._fetch_in_flight:
                return False
            failure = self._fetch_failures.get(agent_id)
            if failure is not None:
                failed_at, streak = failure
                return (now - failed_at) >= self.fetch_backoff(streak)
            last = self._fetch_times.get(agent_id)
            if last is None:
                return True
            return (now - last) > self._FETCH_TTL

    def bundle_status(self, agent_id: str, remote_configured: bool) -> Tuple[bool, Optional[float]]:
        """``(stale, age_s)`` for the policy.evaluated payload.

        ``age_s`` is seconds since the last successful remote load for this
        agent (None if there has never been one). ``stale`` is True when
        remote fetching is configured and either no load has ever succeeded
        — the agent is running on local data, a disk cache, or nothing — or a
        refresh is overdue (age past _FETCH_TTL). With no remote configured
        there is nothing to be stale relative to: (False, None).
        """
        if not remote_configured:
            return False, None
        with self._lock:
            last = self._fetch_times.get(agent_id)
        if last is None:
            return True, None
        age = time.monotonic() - last
        return age > self._FETCH_TTL, age
