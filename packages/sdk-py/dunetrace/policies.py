"""
Policy engine: runtime guardrails evaluated during agent runs.

Define rules (condition → action) that fire mid-run before a failure
propagates. Three ways to load policies:

  1. Local config — always works, no network:
       dt.add_policy(
           name="stop at 5 tools",
           condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
           action={"type": "stop"},
       )

  2. Remote fetch — auto-pulled from the ingest endpoint at run start:
       dt = Dunetrace(api_key="dt_live_...", endpoint="https://ingest.dunetrace.com")
       # Policies defined in the dashboard apply automatically.

Supported triggers:
  tool_call_count  — total tool calls so far (int)
  step_count       — current step index (int)
  cost_usd         — accumulated LLM cost in USD (float)
  error_count      — failed tool calls (int)
  finish_reason    — latest LLM finish_reason string
  llm_latency_ms   — latest LLM call latency in ms (int)
  signal           — detector signal name e.g. "TOOL_LOOP"

Supported operators: gt  gte  lt  lte  eq  neq  contains

Supported actions:
  stop          — raises PolicyViolation; run exits with exit_reason="policy_violation"
  switch_model  — sets run.model_override (str); agent code reads it between steps
  inject_prompt — appends to run.prompt_additions (list); agent code prepends to messages
  log           — emits policy.triggered event, no interruption
"""

from __future__ import annotations

import hashlib
import hmac
import json as _json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Required, TypedDict


class PolicyCondition(TypedDict, total=False):
    trigger: Required[
        str
    ]  # tool_call_count | step_count | cost_usd | error_count | finish_reason | llm_latency_ms | signal
    operator: str  # gt | gte | lt | lte | eq | neq | contains  (default: gt)
    value: Any  # threshold value


class PolicyAction(TypedDict, total=False):
    type: Required[Literal["stop", "switch_model", "inject_prompt", "log"]]
    params: Dict[str, Any]  # e.g. {"model": "gpt-4o-mini"} for switch_model


logger = logging.getLogger("dunetrace.policies")

# ── Token pricing (USD per token, as of 2025) ─────────────────────────────────
# Matched by prefix — longest match wins. Falls back to _DEFAULT_PRICE.

_MODEL_PRICES: Dict[str, Dict[str, float]] = {
    "claude-opus-4": {"input": 15.00e-6, "output": 75.00e-6},
    "claude-sonnet-4": {"input": 3.00e-6, "output": 15.00e-6},
    "claude-haiku-4": {"input": 0.80e-6, "output": 4.00e-6},
    "claude-3-5-sonnet": {"input": 3.00e-6, "output": 15.00e-6},
    "claude-3-5-haiku": {"input": 0.80e-6, "output": 4.00e-6},
    "claude-3-opus": {"input": 15.00e-6, "output": 75.00e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
    "gpt-4o": {"input": 5.00e-6, "output": 15.00e-6},
    "gpt-4-turbo": {"input": 10.00e-6, "output": 30.00e-6},
    "gpt-3.5-turbo": {"input": 0.50e-6, "output": 1.50e-6},
}
_DEFAULT_PRICE: Dict[str, float] = {"input": 3.00e-6, "output": 12.00e-6}


def _price_for(model: str) -> Dict[str, float]:
    for key, price in _MODEL_PRICES.items():
        if model.startswith(key) or key in model:
            return price
    return _DEFAULT_PRICE


def compute_run_cost(llm_calls: list) -> float:
    """Sum USD cost across all LLM calls tracked in a RunState."""
    total = 0.0
    for lc in llm_calls:
        p = int(getattr(lc, "prompt_tokens", None) or 0)
        c = int(getattr(lc, "completion_tokens", None) or 0)
        price = _price_for(getattr(lc, "model", "") or "")
        total += p * price["input"] + c * price["output"]
    return total


def build_metrics(
    state: Any,
    step: int,
    *,
    error_count: Optional[int] = None,
    cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute all policy-evaluable metrics from the current RunState.

    Returns a dict keyed by trigger name. Values are None when the metric
    is not yet available (e.g. no LLM calls have happened yet).

    Pass pre-computed ``error_count`` and ``cost_usd`` to skip the O(n) scans
    when the caller maintains running totals.
    """
    llm_calls = getattr(state, "llm_calls", []) or []
    tool_calls = getattr(state, "tool_calls", []) or []

    last_llm = llm_calls[-1] if llm_calls else None

    return {
        "tool_call_count": len(tool_calls),
        "step_count": step,
        "cost_usd": cost_usd if cost_usd is not None else compute_run_cost(llm_calls),
        "error_count": (
            error_count
            if error_count is not None
            else sum(1 for tc in tool_calls if getattr(tc, "success", None) is False)
        ),
        "finish_reason": getattr(last_llm, "finish_reason", None),
        "llm_latency_ms": getattr(last_llm, "latency_ms", None),
        # "signal" is filled in by the engine when a detector-based policy exists
    }


# ── Condition evaluation ──────────────────────────────────────────────────────

_OPERATORS: Dict[str, Any] = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
    "contains": lambda a, b: b in (a if isinstance(a, (list, tuple, set)) else [a]),
}


# ── Core data types ───────────────────────────────────────────────────────────


class PolicyViolation(RuntimeError):
    """Raised when a 'stop' policy fires. Carries the policy name and action."""

    def __init__(self, policy_name: str, action: dict, message: str = "") -> None:
        self.policy_name = policy_name
        self.action = action
        super().__init__(message or f"Policy '{policy_name}' triggered — run stopped")


@dataclass
class Policy:
    name: str
    condition: PolicyCondition
    action: PolicyAction
    agent_id: str = "*"  # "*" matches all agents
    enabled: bool = True
    priority: int = 100
    id: Optional[int] = None

    @property
    def key(self) -> str:
        """Stable identifier used to deduplicate triggers within a run."""
        return str(self.id) if self.id is not None else self.name

    def matches(self, metrics: Dict[str, Any]) -> bool:
        trigger = self.condition.get("trigger", "")
        operator = self.condition.get("operator", "gt")
        value = self.condition.get("value")
        current = metrics.get(trigger)
        if current is None:
            return False
        op_fn = _OPERATORS.get(operator)
        if op_fn is None:
            return False
        try:
            return bool(op_fn(current, value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        return cls(
            id=d.get("id"),
            agent_id=d.get("agent_id", "*"),
            name=d.get("name", ""),
            condition=dict(d.get("condition") or {}),
            action=dict(d.get("action") or {}),
            enabled=bool(d.get("enabled", True)),
            priority=int(d.get("priority", 100)),
        )


def _verify_policy_signature(policy: dict, secret: str) -> bool:
    """Return True if the policy's HMAC-SHA256 signature matches. Always True when secret is empty.

    Policies with an empty signature and a non-empty secret are treated as unsigned
    (e.g., created before signing was enabled) — they are loaded with a warning rather
    than silently dropped, to support zero-downtime migration when POLICY_SIGNING_SECRET
    is first set.
    """
    if not secret:
        return True
    actual_sig = policy.get("signature") or ""
    if not actual_sig:
        # Unsigned policy — emit a warning but allow through during migration.
        # Set POLICY_SIGNING_SECRET and re-save all policies to enforce verification.
        logger.warning(
            "Policy '%s' (id=%s) has no signature — loaded without verification. "
            "Re-save this policy to sign it.",
            policy.get("name"),
            policy.get("id"),
        )
        return True
    # Use null-byte as separator — safe against colons in agent_id or name.
    canonical = "\x00".join(
        [
            str(policy.get("id", "")),
            policy.get("agent_id", ""),
            policy.get("name", ""),
            _json.dumps(policy.get("condition", {}), sort_keys=True),
            _json.dumps(policy.get("action", {}), sort_keys=True),
            str(policy.get("enabled", True)),
            str(policy.get("priority", 100)),
        ]
    )
    expected_sig = hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_sig, actual_sig)


# ── Engine ────────────────────────────────────────────────────────────────────


class PolicyEngine:
    """
    Thread-safe policy evaluator. One instance lives on the Dunetrace client
    and is shared across all concurrent runs.
    """

    _FETCH_TTL = 60.0  # seconds between remote refreshes per agent_id

    def __init__(self) -> None:
        self._policies: List[Policy] = []
        self._lock: threading.Lock = threading.Lock()
        self._fetch_times: Dict[str, float] = {}  # agent_id → last fetch monotonic
        self._generation: int = (
            0  # incremented on every load/add so RunContext can detect staleness
        )

    # ── Config API ────────────────────────────────────────────────────────────

    def add(self, policy: Policy) -> None:
        with self._lock:
            self._policies.append(policy)
            self._policies.sort(key=lambda p: p.priority)
            self._generation += 1

    def load(self, raw: List[dict], secret: str = "") -> None:
        """Replace remote-sourced policies with a fresh list from the API.

        When ``secret`` is set, each policy's HMAC-SHA256 signature is verified
        before it is loaded. Policies that fail verification are skipped and logged
        as warnings — a tampered or replayed policy never reaches the agent.
        """
        verified = []
        for p in raw:
            if secret:
                if not _verify_policy_signature(p, secret):
                    logger.warning(
                        "Policy '%s' (id=%s) failed signature verification — skipped",
                        p.get("name"),
                        p.get("id"),
                    )
                    continue
            verified.append(Policy.from_dict(p))

        with self._lock:
            local = [p for p in self._policies if p.id is None]
            self._policies = sorted(local + verified, key=lambda p: p.priority)
            self._generation += 1
        logger.debug("Policies loaded: %d total (%d remote)", len(self._policies), len(verified))

    def mark_fetched(self, agent_id: str) -> None:
        self._fetch_times[agent_id] = time.monotonic()

    def needs_fetch(self, agent_id: str) -> bool:
        last = self._fetch_times.get(agent_id)
        if last is None:
            return True
        return (time.monotonic() - last) > self._FETCH_TTL

    def __len__(self) -> int:
        with self._lock:
            return len(self._policies)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        triggered_already: set,  # policy keys already fired in this run
    ) -> Optional[tuple]:
        """
        Return (policy, action_dict) for the highest-priority matching policy
        that hasn't already fired in this run, or None.

        Policies with action.type == "log" are returned but also re-evaluatable
        (they don't get added to triggered_already by the caller).
        """
        with self._lock:
            snapshot = list(self._policies)

        for policy in snapshot:
            if not policy.enabled:
                continue
            if policy.agent_id not in ("*", agent_id):
                continue
            if policy.key in triggered_already:
                continue
            if policy.matches(metrics):
                return policy, dict(policy.action)

        return None
