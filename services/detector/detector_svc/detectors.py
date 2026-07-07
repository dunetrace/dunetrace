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


def get_detectors(agent_category: str = "default") -> list[BaseDetector]:
    """Return the configured detector list for the given agent_category.

    Falls back to "default" if no category-specific config exists.
    Thresholds are loaded from detectors.yml — edit that file to tune.
    Always includes any registered custom Python-class detector plugins,
    regardless of category — they aren't part of the per-category tuning
    system built-in detectors use.
    """
    if agent_category in _CONFIG and agent_category != "default":
        # Merge: use category overrides on top of default kwargs
        default_cfg = _CONFIG.get("default", {})
        category_cfg = _CONFIG.get(agent_category, {})
        detectors = []
        for key, cls in _DETECTOR_CLASSES.items():
            kwargs = {**default_cfg.get(key, {}), **category_cfg.get(key, {})}
            detectors.append(cls(**kwargs))
        detectors.extend(_build_plugin_detectors())
        return detectors

    return _build_detectors("default")
