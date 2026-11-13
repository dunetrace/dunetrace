"""
GET /v1/detector-config assembly: server-authoritative thresholds for the
SDK's client-side detector pass.

The detector worker builds each detector from detectors.yml overrides
(default section + the agent's category section), the org's enabled packs and
per-agent P75 baselines. The SDK's in-path pass (run_detectors(context=
"policy")) historically used class defaults for all three, so a threshold
tuned in detectors.yml never reached it and the two disagreed. This module
returns what the server WOULD apply for an agent; the SDK overlays it on its
class defaults and falls back to them for anything missing.

Caching: the yaml parse happens once, on first use (the file is mounted
read-only and the detector/alerts workers also only read it at startup;
restart ingest after editing it). Packs are cached per org and baselines per
(org, agent, version) for TTL_S — the same 60s staleness the detector's own
pack cache and the SDK's policy cache already accept. Every DB read fails
open: a failure is logged at WARNING and served as "no packs" / "no
baselines", never as an error response — the SDK must still get its yaml
overrides when the baseline table is unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from dunetrace_schemas.detector_config import (
    BUILTIN_DETECTOR_KEYS,
    effective_detector_kwargs,
    json_safe_kwargs,
    load_detector_kwargs,
)
from ingest_svc.config import settings
from ingest_svc.db.postgres import (
    fetch_agent_baselines,
    fetch_latest_agent_version,
    fetch_org_enabled_packs,
)

logger = logging.getLogger("dunetrace.ingest.detector_config")

SCHEMA_VERSION = 1

# How long the SDK may keep a response, and how long packs/baselines are cached
# here before the next request re-reads them.
TTL_S = 60

# Cache entries are evicted lazily; this caps how many (org, agent, version)
# keys can accumulate before a sweep of expired ones runs.
_MAX_CACHE_ENTRIES = 5000

# Sentinel for "not loaded yet" — distinct from {} (loaded, file missing/empty).
_UNLOADED: dict = {}
_yaml_config: Optional[dict[str, dict[str, dict[str, Any]]]] = None

_packs_cache: dict[str, tuple[list[str], float]] = {}
_baselines_cache: dict[
    tuple[str, str, Optional[str]], tuple[Optional[dict], Optional[str], float]
] = {}


def get_yaml_config() -> dict[str, dict[str, dict[str, Any]]]:
    """detectors.yml parsed once, into {category: {detector: {KWARG: value}}}.
    A missing file, missing PyYAML or a parse error all yield {} (logged by the
    parser) — the endpoint then serves an empty override map."""
    global _yaml_config
    if _yaml_config is None:
        _yaml_config = load_detector_kwargs(
            settings.DETECTOR_CONFIG, known_detectors=set(BUILTIN_DETECTOR_KEYS)
        )
    return _yaml_config


def reset_caches() -> None:
    """Drop the parsed yaml and every packs/baselines entry. Test hook; the
    service relies on TTL expiry and never calls this."""
    global _yaml_config
    _yaml_config = None
    _packs_cache.clear()
    _baselines_cache.clear()


def _sweep(cache: dict, now: float) -> None:
    if len(cache) < _MAX_CACHE_ENTRIES:
        return
    for key in [k for k, v in cache.items() if now - v[-1] >= TTL_S]:
        del cache[key]


async def _enabled_packs(org_id: str) -> list[str]:
    now = time.monotonic()
    cached = _packs_cache.get(org_id)
    if cached is not None and now - cached[1] < TTL_S:
        return cached[0]
    try:
        packs = sorted(await fetch_org_enabled_packs(org_id))
    except Exception as exc:
        logger.warning(
            "detector-config: could not read enabled packs for org=%s — serving none: %s",
            org_id,
            exc,
        )
        packs = []
    _sweep(_packs_cache, now)
    _packs_cache[org_id] = (packs, now)
    return packs


async def _baselines(
    org_id: str, agent_id: str, agent_version: Optional[str]
) -> tuple[Optional[dict], Optional[str]]:
    """(baselines, version they were computed for). Version resolution and the
    aggregate are both inside the fail-open guard; a failure caches None for
    TTL_S so a broken table warns once per minute per agent, not per request."""
    now = time.monotonic()
    key = (org_id, agent_id, agent_version)
    cached = _baselines_cache.get(key)
    if cached is not None and now - cached[2] < TTL_S:
        return cached[0], cached[1]
    baselines: Optional[dict] = None
    version = agent_version
    try:
        if version is None:
            version = await fetch_latest_agent_version(org_id, agent_id)
        if version is not None:
            baselines = await fetch_agent_baselines(org_id, agent_id, version)
    except Exception as exc:
        logger.warning(
            "detector-config: baseline read failed for org=%s agent=%s — serving null: %s",
            org_id,
            agent_id,
            exc,
        )
        baselines = None
    _sweep(_baselines_cache, now)
    _baselines_cache[key] = (baselines, version, now)
    return baselines, version


async def build_detector_config(
    org_id: str, agent_id: str, agent_version: Optional[str] = None
) -> dict[str, Any]:
    """The response body for GET /v1/detector-config."""
    detectors = json_safe_kwargs(effective_detector_kwargs(get_yaml_config(), agent_id))
    packs = await _enabled_packs(org_id)
    baselines, version = await _baselines(org_id, agent_id, agent_version)
    return {
        "schema_version": SCHEMA_VERSION,
        "agent_id": agent_id,
        "agent_version": version,
        "generated_at": time.time(),
        "ttl_s": TTL_S,
        "detectors": detectors,
        "packs": packs,
        "baselines": baselines,
    }
