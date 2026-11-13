"""
Server-authoritative detector configuration for the SDK's in-path pass.

The detector worker builds every built-in detector from ``detectors.yml``
(the ``default`` section with the agent's own section merged over it), adds
the packs the org has enabled, and seeds each run's ``RunState.baseline_*``
fields from the agent's P75 history. The SDK's in-path pass — the
``run_detectors(context="policy")`` call a ``trigger="signal"`` policy drives
from ``run_context.py`` — historically ran the ``TIER1_DETECTORS`` module
singletons with class defaults for all three, so a threshold tuned server-side
never reached it and the two passes disagreed about the same run.

This module closes that gap. The ingest service serves what the worker WOULD
apply for an agent at ``GET /v1/detector-config`` (assembled by
``services/ingest/ingest_svc/detector_config.py`` from the shared parser in
``dunetrace_schemas.detector_config``)::

    {
      "schema_version": 1,
      "agent_id": "...", "agent_version": "..." | null,
      "generated_at": <unix seconds>, "ttl_s": 60,
      "detectors": {<yaml section key>: {<UPPERCASE kwarg>: value, ...}, ...},
      "packs": ["voice", ...],
      "baselines": {"p75_steps": 41.0, "p75_token_growth": 2.1, ...} | null
    }

``build_detectors`` turns one such response into per-agent detector instances;
``DetectorConfigStore`` keeps them per agent with the same TTL / backoff /
in-flight / staleness bookkeeping the policy bundle uses (``RemoteFetchState``);
``apply_baselines`` copies the response's baselines onto a ``RunState`` so the
baseline-driven detectors (STEP_COUNT_INFLATION, SLOW_STEP, CONTEXT_BLOAT,
REASONING_STALL, COST_SPIKE, SESSION_LATENCY) work in-path as they do
server-side.

**In-path budget caps can only be lowered by the server, never raised.** A
few ``TIER1_DETECTORS`` entries are constructed with caps far below their
class defaults (``UngroundedDestinationDetector`` ``MAX_SCAN_NS`` /
``MAX_SURFACE_CHARS`` / ``MAX_ARGS_CHARS``, ``UnresolvedAmbiguityDetector``
``MAX_SCAN_NS`` / ``MAX_OUTPUT_CHARS``) because the in-path pass runs once per
step *inside the customer's agent* — it is their request latency that pays
for every byte scanned. ``detectors.yml`` tunes the server-side instance,
which runs on Dunetrace's own worker with nothing waiting on it; a value that
is right there can be 50x too expensive here. So for the keys in
``BUDGET_CAPS`` the server's value applies only when it is *lower* than what
the in-path instance already has (see ``_merge_kwargs``).

Everything here fails open. A missing or malformed response yields the plain
``TIER1_DETECTORS`` list; an unknown kwarg is skipped with one WARNING per
(detector, key); a detector whose constructor rejects the merged kwargs falls
back to its in-path instance. Zero dependencies — stdlib and ``dunetrace``
only, like the rest of the SDK.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from dunetrace.detectors import DETECTOR_KEYS, TIER1_DETECTORS, BaseDetector
from dunetrace.models import RunState, Severity
from dunetrace.remote_fetch import RemoteFetchState

logger = logging.getLogger("dunetrace.detector_config")

# The response shape this module understands. A response carrying a different
# version is still applied best-effort (unknown keys are skipped, not fatal)
# but logged once per agent so a mismatch is visible.
SCHEMA_VERSION = 1

# Per-detector scan budgets the server may lower but never raise — see the
# module docstring. MAX_COST_NS is the cross-cutting BaseDetector budget
# (settable for any detector via detectors.yml's ``max_cost_ns``); the rest are
# the input bounds on the two content-scanning detectors.
BUDGET_CAPS: frozenset = frozenset(
    {
        "MAX_SCAN_NS",
        "MAX_COST_NS",
        "MAX_SURFACE_CHARS",
        "MAX_ARGS_CHARS",
        "MAX_OUTPUT_CHARS",
        "MAX_CANDIDATES_PER_RUN",
        "MAX_DEPTH",
        "MAX_NODES",
    }
)

# ``baselines`` response key → ``RunState`` attribute. The response names
# follow the detector each baseline feeds; the state fields are what those
# detectors read. Mirrors ``dunetrace_schemas.baselines.BASELINE_RESPONSE_KEYS``
# (the SDK cannot import it — zero dependencies); the detector test-suite's
# parity test fails the build if the two key sets diverge.
BASELINE_FIELDS: Dict[str, str] = {
    "p75_steps": "baseline_p75_steps",
    "p75_latency_tool_ms": "baseline_p75_latency_tool",
    "p75_latency_llm_ms": "baseline_p75_latency_llm",
    "p75_token_growth": "baseline_p75_token_growth",
    "p75_llm_tool_ratio": "baseline_p75_llm_tool_ratio",
    "p75_total_tokens": "baseline_p75_total_tokens",
    "p75_duration_s": "baseline_p75_duration_s",
}

# The in-path instance per detector class. Only detectors present here are
# ever built client-side: PromptInjectionDetector, HandoffContextLossDetector
# and DelegationLoopDetector are absent from TIER1_DETECTORS on purpose (they
# need raw input or a second run's data), so a server override for them is
# ignored, not applied to a detector that cannot run here.
_INPATH_BY_CLASS: Dict[type, BaseDetector] = {type(d): d for d in TIER1_DETECTORS}

# (yaml key, kwarg) pairs already warned about — one WARNING per pair per
# process, not one per fetch every 60s.
_warned_kwargs: Set[Tuple[str, str]] = set()


def _tunable_attrs(cls: type) -> Set[str]:
    """The UPPERCASE attributes ``cls(**kwargs)`` accepts — the same rule
    BaseDetector.__init__ applies, computed here so an unknown key can be
    skipped with a warning instead of a TypeError."""
    tunable: Set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            break
        tunable.update(k for k in vars(klass) if k.isupper())
    return tunable


def _inpath_overrides(instance: BaseDetector) -> Dict[str, Any]:
    """The non-default UPPERCASE attributes the in-path instance was built
    with — BaseDetector.__init__ setattr()s overrides on the instance, so they
    are exactly the instance's own UPPERCASE ``__dict__`` entries."""
    return {k: v for k, v in vars(instance).items() if k.isupper()}


def _warn_once(det_key: str, kwarg: str, message: str, *args: Any) -> None:
    if (det_key, kwarg) in _warned_kwargs:
        logger.debug(message, *args)
        return
    _warned_kwargs.add((det_key, kwarg))
    logger.warning(message, *args)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_kwargs(
    det_key: str, cls: type, base: BaseDetector, server: Dict[str, Any]
) -> Dict[str, Any]:
    """The kwargs to construct this agent's instance with: the in-path
    instance's overrides (its caps) with the server's kwargs overlaid, except
    that a BUDGET_CAPS key is taken from the server only when it lowers the
    in-path value. Unknown or unusable keys are skipped with one warning."""
    kwargs = _inpath_overrides(base)
    tunable = _tunable_attrs(cls)
    for key, value in server.items():
        if not isinstance(key, str) or key not in tunable:
            _warn_once(
                det_key,
                str(key),
                "Dunetrace: detector config for %r carries unknown parameter %r — "
                "ignored (this SDK's %s does not have it; the server may be newer)",
                det_key,
                key,
                cls.__name__,
            )
            continue
        if key in BUDGET_CAPS:
            current = getattr(base, key, None)
            if not _is_number(value):
                _warn_once(
                    det_key,
                    key,
                    "Dunetrace: detector config for %r has non-numeric %s=%r — ignored",
                    det_key,
                    key,
                    value,
                )
                continue
            if _is_number(current) and value >= current:
                # The in-path budget is the customer's request latency; the
                # server tunes its own worker's instance and may not spend
                # more of ours. Lowering is always honoured.
                logger.debug(
                    "Dunetrace: detector config for %r asks %s=%r, in-path cap is %r — "
                    "caps may only be lowered client-side; keeping %r",
                    det_key,
                    key,
                    value,
                    current,
                    current,
                )
                continue
        if key == "SEVERITY" and isinstance(value, str):
            try:
                value = Severity(value.upper())
            except ValueError:
                _warn_once(
                    det_key,
                    key,
                    "Dunetrace: detector config for %r has invalid severity %r — ignored",
                    det_key,
                    value,
                )
                continue
        kwargs[key] = value
    return kwargs


def build_detectors(config: Optional[Dict[str, Any]]) -> List[BaseDetector]:
    """Per-agent detector instances for one ``GET /v1/detector-config``
    response, in the server's evaluation order.

    ``None`` (nothing fetched yet, or the fetch is failing with nothing
    retained) returns the plain ``TIER1_DETECTORS`` list — the pre-existing
    behaviour, untouched. Otherwise, for every built-in detector that has an
    in-path instance, a fresh instance is constructed from its class with the
    in-path caps overlaid by the server's kwargs (``_merge_kwargs``); pack
    detectors for every pack the response lists are appended with class
    defaults, exactly as the worker builds them. Unknown detector keys are
    ignored at DEBUG (a newer server may know detectors this SDK does not).
    """
    if not isinstance(config, dict):
        if config is not None:
            logger.warning(
                "Dunetrace: detector config is %s, expected object — using detector defaults",
                type(config).__name__,
            )
        return TIER1_DETECTORS

    overrides = config.get("detectors")
    if not isinstance(overrides, dict):
        overrides = {}
    for key in overrides:
        if key not in DETECTOR_KEYS:
            logger.debug("Dunetrace: detector config names unknown detector %r — ignored", key)

    detectors: List[BaseDetector] = []
    for det_key, cls in DETECTOR_KEYS.items():
        base = _INPATH_BY_CLASS.get(cls)
        if base is None:
            continue  # never evaluated client-side; see _INPATH_BY_CLASS
        server = overrides.get(det_key)
        kwargs = _merge_kwargs(det_key, cls, base, server if isinstance(server, dict) else {})
        try:
            detectors.append(cls(**kwargs))
        except Exception as exc:
            # A detector may validate its tunables in __init__ (Scattershot
            # rejects a float MIN_DISTINCT_TOOLS). The worker falls back to
            # class defaults for the same reason; here the equivalent is the
            # in-path instance, caps included.
            logger.warning(
                "Dunetrace: detector config for %r could not be applied (%s: %s) — "
                "keeping this detector's defaults",
                det_key,
                type(exc).__name__,
                exc,
            )
            detectors.append(base)

    packs = config.get("packs")
    if isinstance(packs, list) and packs:
        detectors.extend(_build_pack_detectors(packs))
    return detectors


def _build_pack_detectors(pack_names: List[Any]) -> List[BaseDetector]:
    # Imported lazily: dunetrace.packs registers the voice pack on import,
    # and nothing else on the in-path route needs it until a config lists one.
    from dunetrace.packs import PACK_REGISTRY

    detectors: List[BaseDetector] = []
    for name in pack_names:
        pack = PACK_REGISTRY.get(name) if isinstance(name, str) else None
        if pack is None:
            logger.debug("Dunetrace: detector config lists unknown pack %r — ignored", name)
            continue
        for cls in pack.detectors:
            try:
                detectors.append(cls())
            except Exception as exc:
                logger.warning(
                    "Dunetrace: pack detector %s (pack=%s) could not be built (%s: %s) — skipped",
                    cls.__name__,
                    pack.name,
                    type(exc).__name__,
                    exc,
                )
    return detectors


def apply_baselines(state: RunState, baselines: Optional[Dict[str, Any]]) -> None:
    """Seed ``state.baseline_*`` from a response's ``baselines`` mapping —
    only the fields that are still ``None``, so a value the caller set
    explicitly is never overwritten. Non-numeric or null entries are
    skipped (null is the server's "insufficient history" for that column)."""
    if not isinstance(baselines, dict):
        return
    for key, attr in BASELINE_FIELDS.items():
        value = baselines.get(key)
        if value is None or not _is_number(value):
            continue
        if getattr(state, attr, None) is None:
            setattr(state, attr, float(value))


@dataclass
class DetectorConfigEntry:
    """What the store holds per agent."""

    config: Dict[str, Any]
    detectors: List[BaseDetector]
    baselines: Optional[Dict[str, Any]]
    fetched_at: float  # time.monotonic() when the response was loaded


class DetectorConfigStore(RemoteFetchState):
    """Per-agent detector configuration on the Dunetrace client.

    ``load()`` is called by the client's background fetch with the parsed
    response; ``detectors_for()`` is what the in-path pass runs (falling back
    to ``TIER1_DETECTORS`` until a load has happened). Fetch scheduling —
    TTL, failure backoff, in-flight guard — and the ``(stale, age_s)``
    verdict come from ``RemoteFetchState``, the same implementation the
    policy engine uses, so ``detector_config_stale`` on a policy.evaluated
    event means exactly what ``policy_bundle_stale`` does.

    ``generation`` increments on every load so a run's cached detector
    subset can be regenerated when the configuration changes mid-run.
    """

    def __init__(self) -> None:
        super().__init__()
        self._entries: Dict[str, DetectorConfigEntry] = {}
        self._generation: int = 0
        self._schema_warned: Set[str] = set()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def load(self, agent_id: str, config: Dict[str, Any]) -> DetectorConfigEntry:
        """Build and store the detectors for ``agent_id`` from ``config``.
        Does not mark the fetch succeeded — the client does that once the
        load returns, mirroring ``PolicyEngine.load`` / ``mark_fetched``."""
        if not isinstance(config, dict):
            raise ValueError(f"detector config is {type(config).__name__}, expected object")
        version = config.get("schema_version")
        if version != SCHEMA_VERSION and agent_id not in self._schema_warned:
            self._schema_warned.add(agent_id)
            logger.warning(
                "Dunetrace: detector config for agent %r has schema_version %r; this SDK "
                "understands %d — applying what it recognises",
                agent_id,
                version,
                SCHEMA_VERSION,
            )
        detectors = build_detectors(config)
        baselines = config.get("baselines")
        entry = DetectorConfigEntry(
            config=config,
            detectors=detectors,
            baselines=baselines if isinstance(baselines, dict) else None,
            fetched_at=time.monotonic(),
        )
        with self._lock:
            self._entries[agent_id] = entry
            self._generation += 1
        logger.debug(
            "Dunetrace: detector config loaded for agent %r: %d detector(s), %d override(s), "
            "packs=%s, baselines=%s",
            agent_id,
            len(detectors),
            len(config.get("detectors") or {}),
            config.get("packs") or [],
            "yes" if entry.baselines else "none",
        )
        return entry

    def has_config(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._entries

    def entry_for(self, agent_id: str) -> Optional[DetectorConfigEntry]:
        with self._lock:
            return self._entries.get(agent_id)

    def detectors_for(self, agent_id: str) -> List[BaseDetector]:
        """The detector list the in-path pass should run for ``agent_id`` —
        the server-configured instances once loaded, ``TIER1_DETECTORS``
        before that (and while a fetch keeps failing with nothing retained)."""
        entry = self.entry_for(agent_id)
        return entry.detectors if entry is not None else TIER1_DETECTORS

    def baselines_for(self, agent_id: str) -> Optional[Dict[str, Any]]:
        entry = self.entry_for(agent_id)
        return entry.baselines if entry is not None else None

    def apply_baselines(self, agent_id: str, state: RunState) -> None:
        apply_baselines(state, self.baselines_for(agent_id))

    def is_stale(self, agent_id: str, remote_configured: bool = True) -> bool:
        """The stale flag for ``agent_id``: no successful fetch yet, or the
        last one older than the TTL. See ``RemoteFetchState.bundle_status``."""
        return self.bundle_status(agent_id, remote_configured)[0]
