"""
Loads the active detector list and thresholds from detectors.yml.
Edit that file to tune - no code changes or rebuild needed.

Restart to pick up changes:
    docker compose restart detector

Also merges in any third-party Python-class custom detectors loaded by
custom_python_detectors.py (see that module's docstring) — registered via
BaseDetector.__init_subclass__, not hardcoded here like _DETECTOR_CLASSES
below. get_detectors() is called fresh per run (see worker.py), so a plugin
file dropped in mid-process-lifetime is picked up on the next call, without
needing detector_svc restarted — only the plugin-loading pass itself
(load_custom_detector_plugins(), called once at worker startup) needs a
restart to notice a brand-new file.
"""

from __future__ import annotations

import logging

from dunetrace.detectors import (
    CUSTOM_DETECTOR_REGISTRY,
    BaseDetector,
    CascadingToolFailureDetector,
    ContextBloatDetector,
    CostSpikeDetector,
    EmptyLlmResponseDetector,
    FirstStepFailureDetector,
    GoalAbandonmentDetector,
    LlmTruncationLoopDetector,
    PrematureTerminationDetector,
    PromptInjectionDetector,
    RagEmptyRetrievalDetector,
    ReasoningSpinDetector,
    RetryStormDetector,
    SessionLatencyDetector,
    SlowStepDetector,
    HandoffContextLossDetector,
    RetrievedContentInjectionDetector,
    RunawayIterationDetector,
    StepCountInflationDetector,
    ToolArgumentFabricationDetector,
    ToolAvoidanceDetector,
    ToolLoopDetector,
    ToolThrashingDetector,
    UnreadToolErrorDetector,
)
from detector_svc.config_loader import load_detector_kwargs

logger = logging.getLogger("dunetrace.detector.detectors")

# Maps YAML section key → detector class
_DETECTOR_CLASSES: dict[str, type[BaseDetector]] = {
    "tool_loop": ToolLoopDetector,
    "tool_thrashing": ToolThrashingDetector,
    "tool_avoidance": ToolAvoidanceDetector,
    "goal_abandonment": GoalAbandonmentDetector,
    "prompt_injection_signal": PromptInjectionDetector,
    "rag_empty_retrieval": RagEmptyRetrievalDetector,
    "llm_truncation_loop": LlmTruncationLoopDetector,
    "context_bloat": ContextBloatDetector,
    "slow_step": SlowStepDetector,
    "retry_storm": RetryStormDetector,
    "empty_llm_response": EmptyLlmResponseDetector,
    "step_count_inflation": StepCountInflationDetector,
    "cascading_tool_failure": CascadingToolFailureDetector,
    "first_step_failure": FirstStepFailureDetector,
    "reasoning_stall": ReasoningSpinDetector,
    "cost_spike": CostSpikeDetector,
    "session_latency": SessionLatencyDetector,
    "premature_termination": PrematureTerminationDetector,
    "unread_tool_error": UnreadToolErrorDetector,
    "tool_argument_fabrication": ToolArgumentFabricationDetector,
    "retrieved_content_injection": RetrievedContentInjectionDetector,
    "handoff_context_loss": HandoffContextLossDetector,
    "runaway_iteration": RunawayIterationDetector,
}

# Load config once at import time
_CONFIG = load_detector_kwargs()


def _build_plugin_detectors() -> list[BaseDetector]:
    """Instantiate every currently-registered third-party detector class with
    no overrides — plugin classes aren't in _DETECTOR_CLASSES, so they have no
    detectors.yml section to read tunables from. A plugin author sets its
    defaults directly on the class instead (same UPPERCASE-attribute
    mechanism every built-in detector already uses).

    Read fresh from CUSTOM_DETECTOR_REGISTRY on every call (not cached at
    import time like _CONFIG) — see this module's docstring for why that
    matters for when a newly-loaded plugin file actually takes effect.
    """
    detectors: list[BaseDetector] = []
    for cls in CUSTOM_DETECTOR_REGISTRY.values():
        try:
            detectors.append(cls())
        except Exception:
            logger.exception(
                "Failed to instantiate custom detector plugin class %s — skipping it.",
                cls.__name__,
            )
    return detectors


def _build_detectors(category: str) -> list[BaseDetector]:
    category_cfg = _CONFIG.get(category, {})
    detectors = []
    for key, cls in _DETECTOR_CLASSES.items():
        kwargs = category_cfg.get(key, {})
        detectors.append(cls(**kwargs))
    detectors.extend(_build_plugin_detectors())
    return detectors


def _build_pack_detectors(enabled_packs: set[str]) -> list[BaseDetector]:
    """Instantiate every detector class belonging to a pack this org has
    activated, with no overrides — same reasoning as _build_plugin_detectors:
    pack detectors aren't part of the per-category detectors.yml tuning
    system built-ins use. A pack author sets defaults directly on the class.
    Imported here (not at module level) so importing detector_svc.detectors
    doesn't require dunetrace.packs to exist yet at import time — mirrors
    how _seed_packs in db.py defers the same import for the same reason."""
    from dunetrace.packs import PACK_REGISTRY

    detectors: list[BaseDetector] = []
    for pack in PACK_REGISTRY.values():
        if pack.name not in enabled_packs:
            continue
        for cls in pack.detectors:
            try:
                detectors.append(cls())
            except Exception:
                logger.exception(
                    "Failed to instantiate pack detector %s (pack=%s) — skipping it.",
                    cls.__name__,
                    pack.name,
                )
    return detectors


async def get_detectors(agent_category: str, org_id: str) -> list[BaseDetector]:
    """Return the configured detector list for the given agent_category and org.

    Falls back to "default" if no category-specific config exists.
    Thresholds are loaded from detectors.yml — edit that file to tune.
    Always includes any registered custom Python-class detector plugins,
    regardless of category — they aren't part of the per-category tuning
    system built-in detectors use. Additionally includes every detector
    belonging to a pack this org has activated (see detector_svc/packs.py) —
    org_enabled_packs is checked via a 60s TTL-cached read, not a DB hit on
    every call.
    """
    from detector_svc.packs import get_enabled_packs

    enabled_packs = await get_enabled_packs(org_id)

    if agent_category in _CONFIG and agent_category != "default":
        # Merge: use category overrides on top of default kwargs
        default_cfg = _CONFIG.get("default", {})
        category_cfg = _CONFIG.get(agent_category, {})
        detectors = []
        for key, cls in _DETECTOR_CLASSES.items():
            kwargs = {**default_cfg.get(key, {}), **category_cfg.get(key, {})}
            detectors.append(cls(**kwargs))
        detectors.extend(_build_plugin_detectors())
        detectors.extend(_build_pack_detectors(enabled_packs))
        return detectors

    detectors = _build_detectors("default")
    detectors.extend(_build_pack_detectors(enabled_packs))
    return detectors
