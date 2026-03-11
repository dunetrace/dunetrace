"""
services/detector/detector_svc/detectors.py

Builds the active detector list from detectors.yml (repo root).
Edit detectors.yml to tune thresholds i.e. no code change needed.

Restart the detector after any change:
    docker compose restart detector
"""
from __future__ import annotations

from dunetrace.detectors import (
    BaseDetector,
    CascadingToolFailureDetector,
    ContextBloatDetector,
    EmptyLlmResponseDetector,
    FirstStepFailureDetector,
    GoalAbandonmentDetector,
    LlmTruncationLoopDetector,
    PromptInjectionDetector,
    RagEmptyRetrievalDetector,
    RetryStormDetector,
    SlowStepDetector,
    StepCountInflationDetector,
    ToolAvoidanceDetector,
    ToolLoopDetector,
    ToolThrashingDetector,
)
from detector_svc.config_loader import load_detector_kwargs

# Maps YAML section key → detector class
_DETECTOR_CLASSES: dict[str, type[BaseDetector]] = {
    "tool_loop":               ToolLoopDetector,
    "tool_thrashing":          ToolThrashingDetector,
    "tool_avoidance":          ToolAvoidanceDetector,
    "goal_abandonment":        GoalAbandonmentDetector,
    "prompt_injection_signal": PromptInjectionDetector,
    "rag_empty_retrieval":     RagEmptyRetrievalDetector,
    "llm_truncation_loop":     LlmTruncationLoopDetector,
    "context_bloat":           ContextBloatDetector,
    "slow_step":               SlowStepDetector,
    "retry_storm":             RetryStormDetector,
    "empty_llm_response":      EmptyLlmResponseDetector,
    "step_count_inflation":    StepCountInflationDetector,
    "cascading_tool_failure":  CascadingToolFailureDetector,
    "first_step_failure":      FirstStepFailureDetector,
}

# Load config once at import time
_CONFIG = load_detector_kwargs()


def _build_detectors(category: str) -> list[BaseDetector]:
    category_cfg = _CONFIG.get(category, {})
    detectors = []
    for key, cls in _DETECTOR_CLASSES.items():
        kwargs = category_cfg.get(key, {})
        detectors.append(cls(**kwargs))
    return detectors


def get_detectors(agent_category: str = "default") -> list[BaseDetector]:
    """Return the configured detector list for the given agent_category.

    Falls back to "default" if no category-specific config exists.
    Thresholds are loaded from detectors.yml — edit that file to tune.
    """
    if agent_category in _CONFIG and agent_category != "default":
        # Merge: use category overrides on top of default kwargs
        default_cfg = _CONFIG.get("default", {})
        category_cfg = _CONFIG.get(agent_category, {})
        detectors = []
        for key, cls in _DETECTOR_CLASSES.items():
            kwargs = {**default_cfg.get(key, {}), **category_cfg.get(key, {})}
            detectors.append(cls(**kwargs))
        return detectors

    return _build_detectors("default")
