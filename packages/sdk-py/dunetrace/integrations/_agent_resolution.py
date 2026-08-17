"""
Shared agent_id resolution for framework-level auto-instrumentation
(LangChain, CrewAI).

The SDK-level patches in ``dunetrace.auto`` (openai, anthropic, mistral, httpx,
requests) never need this — they only ever attach to an already-active
``dt.run()`` and never decide an agent_id themselves. Framework-level
integrations are different: they can open their *own* run when no ``dt.run()``
is active, so something has to decide what agent_id that run gets. Guessing
wrong fails silently in the worst way — the run still gets recorded, just
under the wrong agent, so it quietly vanishes from the dashboard the user is
looking at. This module exists so that failure mode is loud instead.

Resolution order (first non-empty value wins):
  1. An already-active ``dt.run()`` context — checked by the caller, not
     here (see each integration's run-boundary code), since it means "attach
     to that run" rather than "resolve an agent_id for a new one".
  2. ``per_call_agent_id`` — an explicit override for this one invocation
     (LangChain: ``config={"metadata": {"agent_id": "..."}}``; CrewAI:
     ``kickoff(inputs={"agent_id": "..."})``).
  3. ``framework_native_agent_id`` — an identity the framework already knows
     (CrewAI: the ``Agent.role`` of the agent making the call).
  4. ``default_agent_id`` — set once via ``dt.init(agent_id=...)`` or the
     ``DUNETRACE_AGENT_ID`` environment variable.
  5. Loud fallback — never silently mis-attribute. Logs a warning naming the
     integration and pointing at the docs, then returns a clearly-fake
     placeholder so the run is still recorded (better than crashing the
     caller's agent) but is easy to spot and fix.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("dunetrace.auto")

DOCS_URL = (
    "https://github.com/dunetrace/dunetrace/blob/main/docs/integrations/"
    "auto-instrumentation.md#agent-id-resolution"
)

FALLBACK_AGENT_ID = "unattributed-agent"


def resolve_agent_id(
    *,
    per_call_agent_id: Optional[str] = None,
    framework_native_agent_id: Optional[str] = None,
    default_agent_id: Optional[str] = None,
    integration: str = "auto_instrument",
) -> str:
    """Resolve the agent_id for a new run (tiers 2-4 + loud fallback).

    Callers must check for an already-active ``dt.run()`` themselves before
    calling this — that's tier 1, and it means "attach to that run", not
    "pick an agent_id for a new one", which is all this function does.
    """
    if per_call_agent_id:
        return per_call_agent_id
    if framework_native_agent_id:
        return framework_native_agent_id
    if default_agent_id:
        return default_agent_id
    logger.warning(
        "Dunetrace: %s could not determine an agent_id for this run — no "
        "active dt.run(), no per-call override, no framework-native identity, "
        "and no default_agent_id set via dt.init(agent_id=...) or the "
        "DUNETRACE_AGENT_ID environment variable. Recording it under %r so "
        "nothing crashes, but it will be hard to find in the dashboard. "
        "See %s",
        integration,
        FALLBACK_AGENT_ID,
        DOCS_URL,
    )
    return FALLBACK_AGENT_ID
