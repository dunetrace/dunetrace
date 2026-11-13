"""
OTLP receiver observability counters (Phase 5).

Per-(org, hour) tallies of what the receiver did: spans received, events
translated, spans rejected (by reason), auth failures, and rate-limit hits. Kept
in memory on the hot path and drained to the otel_receiver_stats table by the
maintenance loop, so the dashboard can show receiver health per org without the
accept path ever touching the DB.

Reasons that happen before a request is attributed to an org (a bad key, an
oversized or malformed body) are tallied under the "_system" bucket, since there
is no org to attribute them to; everything post-auth is per real org.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Tuple

_SYSTEM = "_system"


def _new_bucket() -> dict:
    return {
        "batches_received": 0,
        "spans_received": 0,
        "events_translated": 0,
        "spans_rejected": 0,
        "auth_failures": 0,
        "rate_limit_hits": 0,
        "rejections": defaultdict(int),  # reason -> count
    }


class OtelStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[Tuple[str, int], dict] = defaultdict(_new_bucket)

    @staticmethod
    def _hour() -> int:
        return int(time.time() // 3600) * 3600

    def _bucket(self, org_id: str) -> dict:
        return self._buckets[(org_id or _SYSTEM, self._hour())]

    def record_received(self, org_id: str, spans: int) -> None:
        with self._lock:
            b = self._bucket(org_id)
            b["batches_received"] += 1
            b["spans_received"] += spans

    def record_translated(self, org_id: str, events: int) -> None:
        with self._lock:
            self._bucket(org_id)["events_translated"] += events

    def record_rejected(self, org_id: str, reason: str) -> None:
        with self._lock:
            b = self._bucket(org_id)
            b["spans_rejected"] += 1
            b["rejections"][reason] += 1

    def record_rate_limit_hit(self, org_id: str) -> None:
        with self._lock:
            self._bucket(org_id)["rate_limit_hits"] += 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self._bucket(_SYSTEM)["auth_failures"] += 1

    def drain(self) -> Dict[Tuple[str, int], dict]:
        """Return and clear the accumulated buckets, for flushing to the DB.
        Empty when nothing happened since the last drain."""
        with self._lock:
            drained = self._buckets
            self._buckets = defaultdict(_new_bucket)
        # Convert the inner defaultdict(rejections) to a plain dict for callers.
        for bucket in drained.values():
            bucket["rejections"] = dict(bucket["rejections"])
        return drained


_stats: OtelStats | None = None


def get_otel_stats() -> OtelStats:
    global _stats
    if _stats is None:
        _stats = OtelStats()
    return _stats
