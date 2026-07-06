"""
Classifies a signal's fix into one of two categories, and — for the
Dunetrace-native category — deterministically derives a ready-to-submit
policy from the signal's own evidence (no LLM call needed for this part;
the policy IS the evidence, just reshaped into PolicyCondition/PolicyAction
form).

- **dunetrace_native**: the fix is a runtime guardrail Dunetrace can enforce
  itself (a Policy — see packages/sdk-py/dunetrace/policies.py) without any
  change to the customer's prompt or code. Only failure types whose evidence
  maps unambiguously onto an existing PolicyCondition trigger
  (tool_call_count / step_count / error_count) are classified this way —
  see the docstring on each entry in _NATIVE_POLICY_BUILDERS for why a given
  detector's evidence was or wasn't judged a clean fit. The suggested policy
  is shaped exactly like services/api/api_svc/routers/policies.py's
  PolicyCreate body, so the dashboard can submit it to the existing
  POST /v1/policies endpoint unchanged after the user confirms it.

- **customer_code**: the fix touches the customer's system prompt, tool
  schema, or actual code — Dunetrace has no write access to any of that, so
  it only ever produces a diff (fix_content/fix_patch, from _call_llm) for
  the customer to apply manually, or to push to a connected external store
  (see prompt_stores.py) if one exists.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

FixCategory = str  # "dunetrace_native" | "customer_code"


def _tool_loop_policy(signal: dict) -> Optional[Dict[str, Any]]:
    """TOOL_LOOP's evidence.count is how many times *one specific tool* was
    called — the only available guardrail trigger is tool_call_count, which
    caps *total* tool calls across the whole run. That's a coarser net than
    "stop calling this one tool," but it's still a real, useful stop
    condition, so this is included."""
    count = signal.get("evidence", {}).get("count")
    if not isinstance(count, (int, float)) or count <= 0:
        return None
    return {
        "condition": {"trigger": "tool_call_count", "operator": "gte", "value": int(count)},
        "action": {"type": "stop"},
    }


def _retry_storm_policy(signal: dict) -> Optional[Dict[str, Any]]:
    consecutive = signal.get("evidence", {}).get("consecutive_fails")
    if not isinstance(consecutive, (int, float)) or consecutive <= 0:
        return None
    return {
        "condition": {"trigger": "error_count", "operator": "gte", "value": int(consecutive)},
        "action": {"type": "stop"},
    }


def _cascading_tool_failure_policy(signal: dict) -> Optional[Dict[str, Any]]:
    consecutive = signal.get("evidence", {}).get("consecutive_failures")
    if not isinstance(consecutive, (int, float)) or consecutive <= 0:
        return None
    return {
        "condition": {"trigger": "error_count", "operator": "gte", "value": int(consecutive)},
        "action": {"type": "stop"},
    }


def _step_count_inflation_policy(signal: dict) -> Optional[Dict[str, Any]]:
    current_steps = signal.get("evidence", {}).get("current_steps")
    if not isinstance(current_steps, (int, float)) or current_steps <= 0:
        return None
    return {
        "condition": {"trigger": "step_count", "operator": "gte", "value": int(current_steps)},
        "action": {"type": "stop"},
    }


# Only detectors whose evidence maps unambiguously onto an existing
# PolicyCondition trigger are here. Explicitly excluded, and why:
#   TOOL_THRASHING     — evidence is an alternating A/B *pattern*, not a
#                        count threshold; no trigger captures "oscillation."
#   COST_SPIKE         — evidence is in tokens (total_tokens); the only cost
#                        trigger is cost_usd, and converting tokens to a
#                        dollar figure requires the per-model price table,
#                        which is a real estimate, not the observed fact —
#                        judged too indirect to auto-suggest as a guardrail.
#   everything else    — genuinely about prompt wording, missing behavior,
#                        or code/infra (RAG_EMPTY_RETRIEVAL, CONTEXT_BLOAT,
#                        TOOL_AVOIDANCE, GOAL_ABANDONMENT, ...) — a "stop
#                        after N" policy can't fix any of those; the agent
#                        needs different instructions or different code.
_NATIVE_POLICY_BUILDERS = {
    "TOOL_LOOP": _tool_loop_policy,
    "RETRY_STORM": _retry_storm_policy,
    "CASCADING_TOOL_FAILURE": _cascading_tool_failure_policy,
    "STEP_COUNT_INFLATION": _step_count_inflation_policy,
}


def classify_fix(signal: dict) -> FixCategory:
    """dunetrace_native only if this failure type has a builder AND that
    builder can actually produce a policy from this signal's evidence —
    a detector in _NATIVE_POLICY_BUILDERS whose evidence is missing/malformed
    degrades to customer_code rather than offering a broken "apply" button."""
    builder = _NATIVE_POLICY_BUILDERS.get(signal.get("failure_type", ""))
    if builder is not None and builder(signal) is not None:
        return "dunetrace_native"
    return "customer_code"


def build_suggested_policy(signal: dict) -> Optional[Dict[str, Any]]:
    """Returns a PolicyCreate-shaped dict (see routers/policies.py) ready to
    submit to POST /v1/policies once the user confirms it, or None if this
    signal isn't dunetrace_native (or its evidence didn't actually support
    building one)."""
    builder = _NATIVE_POLICY_BUILDERS.get(signal.get("failure_type", ""))
    if builder is None:
        return None
    shape = builder(signal)
    if shape is None:
        return None
    return {
        "name": f"Auto-suggested: stop recurring {signal.get('failure_type', 'failure')}",
        "agent_id": signal.get("agent_id", "*"),
        "condition": shape["condition"],
        "action": shape["action"],
        "priority": 100,
        "enabled": True,
    }
