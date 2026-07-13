"""
Per-org pack activation cache (Phase 1.0). detector_svc is a long-running
worker process (not stateless-per-request), so a plain in-process TTL cache
is enough to avoid a DB round trip on every run's detector selection — no
Postgres LISTEN/NOTIFY or Redis needed. Neither exists anywhere else in this
codebase; the closest real precedent is the SDK's own remote-policy cache
(60s TTL per agent, see docs/policies.md). An org's newly-activated pack
takes effect within _TTL_SECS, same staleness window that cache already
accepts.
"""

from __future__ import annotations

import time
from typing import Dict, Set, Tuple

from detector_svc.db import fetch_org_enabled_packs

_TTL_SECS = 60

# org_id -> (enabled pack names, cached_at monotonic timestamp)
_cache: Dict[str, Tuple[Set[str], float]] = {}


async def get_enabled_packs(org_id: str) -> Set[str]:
    cached = _cache.get(org_id)
    if cached is not None and (time.monotonic() - cached[1]) < _TTL_SECS:
        return cached[0]
    packs = set(await fetch_org_enabled_packs(org_id))
    _cache[org_id] = (packs, time.monotonic())
    return packs


def _invalidate(org_id: str) -> None:
    """Test-only escape hatch — production code relies on TTL expiry, never
    calls this. Exists so tests can assert fresh-read behavior without
    sleeping for _TTL_SECS."""
    _cache.pop(org_id, None)
