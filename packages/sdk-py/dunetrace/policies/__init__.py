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
  before_tool_call — the name of the tool about to be called (str). Only
                     meaningful with a require_approval action — see below.
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

Voice-agent actions (Phase 1.3) — same "Dunetrace sets intent, the agent's own
loop enforces it between turns" contract as switch_model. Intended for voice
agents; a text agent that never reads the attribute simply ignores it.
  stop_current_tts       — sets run.stop_tts (bool); agent halts in-progress TTS
  escalate_to_human      — sets run.escalate_to_human (bool) + run.escalation_reason
                           (str|None from params.reason); agent transfers the call
  inject_recovery_prompt — sets run.recovery_prompt (str, last-wins) from
                           params.prompt; agent speaks/prepends it next turn
  slow_response_pace     — sets run.response_pace (str, last-wins) from params.pace
                           (default "slow"); agent slows its turn (filler token
                           while thinking, lower TTS rate) until it reads "normal"

Human-in-the-loop action (Capability 2) — pairs with the before_tool_call
trigger:
  require_approval — blocks the guarded tool call until a human approves
                     (via Slack/dashboard/API) or it times out. Raises
                     ApprovalDenied on a deny or timeout (fail-closed). Unlike
                     the other actions this one is evaluated per tool call
                     (not fire-once): every invocation of a guarded tool is
                     approved independently. params.timeout_s sets the wait
                     (default 300). A require_approval policy's condition
                     matches on the tool name, e.g.
                     {"trigger": "before_tool_call", "operator": "eq",
                      "value": "wire_money"}.
"""

from __future__ import annotations

import hashlib
import hmac
import json as _json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Required, TypedDict, cast

# Expression-based conditions (condition.match). Parser + immutable tree; the
# evaluator lands in Phase 2. Re-exported here so callers keep importing from
# the stable `dunetrace.policies` namespace.
from dunetrace.policies.expressions import (  # noqa: E402,F401
    And,
    Comparison,
    ConditionExpression,
    EXPRESSION_OPERATORS,
    ExpressionError,
    FIELD_PREFIXES,
    MAX_DEPTH,
    Or,
    parse_condition,
    parse_match_block,
)
from dunetrace.policies.evaluator import (  # noqa: E402,F401
    ComparisonTrace,
    EvaluationContext,
    MISSING,
    evaluate,
)
from dunetrace.policies.observability import (  # noqa: E402,F401
    EVAL_LOGGER_NAME,
    EvaluationRateLimiter,
    PolicyEvaluationRecord,
    build_reason,
    eval_logger,
)


class PolicyCondition(TypedDict, total=False):
    trigger: Required[
        str
    ]  # tool_call_count | step_count | cost_usd | error_count | finish_reason | llm_latency_ms | signal
    operator: str  # gt | gte | lt | lte | eq | neq | contains  (default: gt)
    value: Any  # threshold value


class PolicyAction(TypedDict, total=False):
    type: Required[
        Literal[
            "stop",
            "switch_model",
            "inject_prompt",
            "log",
            # Voice-agent actions (Phase 1.3)
            "stop_current_tts",
            "escalate_to_human",
            "inject_recovery_prompt",
            "slow_response_pace",
            # Human-in-the-loop (Capability 2)
            "require_approval",
        ]
    ]
    params: Dict[str, Any]  # e.g. {"model": "gpt-4o-mini"} for switch_model


logger = logging.getLogger("dunetrace.policies")

# Actions a REMOTE policy may only take when its signature verified.
#
# These are the ones that change what a customer's production agent does rather
# than what Dunetrace records: halting a live run, blocking a tool call on a
# human, handing the conversation to a person, rewriting the prompt. The stated
# threat for policy signing is "a compromised or spoofed server halts a live
# agent", and with no configured secret there is nothing to verify against — so
# an unverifiable remote policy is downgraded to log-only rather than obeyed.
#
# Deliberately remote-only. A policy the customer registered in their own
# process via dt.add_policy() is their own code and needs no signature.
#
# NOTE ON THE RESIDUAL RISK: signing is a shared HMAC secret, so a server that
# is itself compromised holds the signing key and can still mint a valid stop
# policy. Closing that needs asymmetric signing (server signs with a private
# key, SDK verifies with a public one), which needs a crypto library — and the
# core SDK is deliberately zero-dependency. What this constant buys is that a
# party who cannot answer with a *correctly signed* payload — a spoofed
# endpoint, a hijacked DNS entry, a MITM — cannot stop anyone's agent.
ENFORCING_ACTIONS = frozenset(
    {
        "stop",
        "require_approval",
        "escalate_to_human",
        "stop_current_tts",
        "switch_model",
        "inject_prompt",
        "inject_recovery_prompt",
    }
)

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
    # 2.50/10.00, matching explainer_svc/cost.py. The SDK table carried the
    # original May-2024 launch price long after it was cut, so a cost_usd
    # policy threshold evaluated in-process fired at half the real spend
    # while the server-side rollups for the same run disagreed.
    "gpt-4o": {"input": 2.50e-6, "output": 10.00e-6},
    "gpt-4-turbo": {"input": 10.00e-6, "output": 30.00e-6},
    "gpt-3.5-turbo": {"input": 0.50e-6, "output": 1.50e-6},
    # Mistral, verified against https://mistral.ai/pricing/api on 2026-08-08.
    # Order matters here: _price_for() walks this dict in insertion order and
    # falls back to a substring test, so "codestral-embed" has to be seen
    # before "codestral" or the chat rate would win for the embedding model.
    # Note mistral.ai/pricing (the FAQ page, not /pricing/api) still quotes the
    # retired $2/$6 Large rate. The per-model cards agree with the numbers here.
    # Reasoning (magistral) and coding (devstral) families. Verified against
    # https://mistral.ai/pricing/api on 2026-08-09. No substring collision with
    # the mistral-* rows: "magistral" does not contain "mistral".
    "magistral-medium": {"input": 2.00e-6, "output": 5.00e-6},
    "magistral-small": {"input": 0.50e-6, "output": 1.50e-6},
    "devstral-medium": {"input": 0.40e-6, "output": 2.00e-6},
    "devstral-small": {"input": 0.10e-6, "output": 0.30e-6},
    "mistral-medium": {"input": 1.50e-6, "output": 7.50e-6},
    "mistral-large": {"input": 0.50e-6, "output": 1.50e-6},
    "mistral-small": {"input": 0.15e-6, "output": 0.60e-6},
    "ministral-3b": {"input": 0.10e-6, "output": 0.10e-6},
    "ministral-8b": {"input": 0.15e-6, "output": 0.15e-6},
    "ministral-14b": {"input": 0.20e-6, "output": 0.20e-6},
    "codestral-embed": {"input": 0.15e-6, "output": 0.0},
    "codestral": {"input": 0.30e-6, "output": 0.90e-6},
    # Embeddings bill on input only, so output is 0 rather than absent. Leaving
    # it out would fall through to _DEFAULT_PRICE and charge 12.00e-6 per
    # output token against a response that has none.
    "mistral-embed": {"input": 0.10e-6, "output": 0.0},
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


class ApprovalDenied(RuntimeError):
    """Raised when a human-in-the-loop approval (Capability 2) resolves to
    anything other than 'granted' — an explicit deny, or a timeout (which is
    fail-closed: no human decided in time, so the guarded tool is blocked).

    Attributes:
      status      the terminal approval status: "denied" | "timeout"
      note        what the deciding human wrote, or None. None on a timeout by
                  construction — nobody decided, so nobody wrote anything.
      decided_by  who decided, when the deciding surface attributed it.
      tool_name / approval_id

    `note` is the whole point of catching this: it is a human correction
    delivered into the run with provenance ("wrong Chen — it's Sarah,
    CUST_8834"). Feed it into the next planning step and retry *within the same
    run*. Dunetrace does not auto-append it to the prompt — steering stays the
    agent's job, consistent with every other advisory action.

    Naming note for anyone reading a diff: this attribute was `reason` and held
    the status. It is `status` now, and `note` is the human's text. One name per
    concept, no alias — `reason` was the wrong word the moment grants started
    carrying notes too.
    """

    def __init__(
        self,
        tool_name: str,
        status: str,
        approval_id: int,
        message: str = "",
        *,
        note: Optional[str] = None,
        decided_by: Optional[str] = None,
    ) -> None:
        self.tool_name = tool_name
        self.status = status  # "denied" | "timeout"
        self.approval_id = approval_id
        self.note = note
        self.decided_by = decided_by
        detail = f"status: {status}"
        if note:
            detail += f"; note: {note}"
        super().__init__(message or f"Approval for tool '{tool_name}' was not granted ({detail})")


@dataclass
class Policy:
    name: str
    condition: PolicyCondition
    action: PolicyAction
    agent_id: str = "*"  # "*" matches all agents
    enabled: bool = True
    priority: int = 100
    id: Optional[int] = None
    # Parsed form of condition["match"] (the expression-condition block), or None
    # for a legacy flat condition. Parsed once at construction so evaluation
    # never re-parses. Not an init arg and excluded from equality/repr — it's
    # derived state, fully determined by `condition`.
    match_expr: Optional[ConditionExpression] = field(
        default=None, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        # Fail-fast: a malformed `match` raises ExpressionError here (at load),
        # never at fire time. PolicyEngine.load() catches and skips such policies.
        self.match_expr = parse_condition(dict(self.condition), policy_name=self.name)

    @property
    def key(self) -> str:
        """Stable identifier used to deduplicate triggers within a run."""
        return str(self.id) if self.id is not None else self.name

    def matches(
        self, metrics: Dict[str, Any], context: "Optional[EvaluationContext]" = None
    ) -> bool:
        """True iff this policy's condition matches. The legacy flat trigger
        (metric/signal) and the new expression block (``condition.match``) are
        ANDed: both must hold. ``context`` supplies the expression's field
        namespaces (args/run/agent/org/event); when a policy has an expression
        but no context is available (e.g. a caller that can't provide args), the
        expression cannot match and the policy does not fire."""
        if not self._legacy_matches(metrics):
            return False
        if self.match_expr is not None:
            if context is None:
                return False
            return bool(evaluate(self.match_expr, context))
        return True

    def _legacy_matches(self, metrics: Dict[str, Any]) -> bool:
        trigger = self.condition.get("trigger", "")
        if trigger == "expression":
            # Pure-expression policy: the flat part is a no-op and match_expr
            # carries the whole condition. Guard against a malformed always-on
            # policy (trigger="expression" with no valid match block).
            return self.match_expr is not None
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
            condition=cast(PolicyCondition, dict(d.get("condition") or {})),
            action=cast(PolicyAction, dict(d.get("action") or {})),
            enabled=bool(d.get("enabled", True)),
            priority=int(d.get("priority", 100)),
        )


# ── HMAC signing canonical form (versioned) ───────────────────────────────────
#
# The canonical string that the HMAC-SHA256 policy signature is computed over.
# Versioned so the field set can evolve without invalidating already-signed
# policies. MUST stay in exact sync with api_svc's _policy_canonical / _sign_policy.
#
#   v1 — the original 7-field form (id, agent_id, name, condition, action,
#        enabled, priority), null-byte separated. condition is JSON-dumped with
#        sort_keys, so the `match` expression block — which lives *inside*
#        condition — is already covered: a tampered match fails verification.
#   v2 — identical fields, but a "v2" domain-separation token is prepended and
#        bound into the hash. Used for policies that carry a condition.match
#        block. Because the version token is authenticated, a v2 policy cannot be
#        downgraded to v1 (or vice-versa) to slip past verification.
#
# The current CURRENT_SIG_VERSION is what new capability-using policies sign as;
# legacy policies stay v1 forever (byte-identical), so old signatures and older
# SDKs keep working. Verification is driven by each policy's own sig_version.

CURRENT_SIG_VERSION = 2


def sig_version_for_condition(condition: dict) -> int:
    """The minimum canonical-form version that can represent this condition.
    A condition using the new `match` block signs as v2; everything else stays
    v1 (byte-identical to pre-feature policies)."""
    if isinstance(condition, dict) and condition.get("match") is not None:
        return CURRENT_SIG_VERSION
    return 1


def _policy_canonical(
    version: int,
    policy_id: Any,
    agent_id: str,
    name: str,
    condition: dict,
    action: dict,
    enabled: Any,
    priority: Any,
) -> str:
    # Null-byte separator — safe against colons in agent_id or name.
    fields = [
        str(policy_id),
        agent_id,
        name,
        _json.dumps(condition, sort_keys=True),
        _json.dumps(action, sort_keys=True),
        str(enabled),
        str(priority),
    ]
    if version >= 2:
        fields.insert(0, "v%d" % version)  # authenticated domain separation
    return "\x00".join(fields)


def _verify_policy_signature(policy: dict, secret: str) -> bool:
    """Return True if the policy's HMAC-SHA256 signature matches. Always True when secret is empty.

    An empty signature with a non-empty secret is a FAILURE, not a migration
    case. Accepting unsigned policies "during migration" meant an attacker who
    could answer the policy fetch simply omitted the signature field and was
    obeyed — the verification was optional for exactly the party it was meant
    to constrain.

    The policy's own ``sig_version`` (default 1, for legacy policies with no such
    field) selects the canonical form — so v1-signed policies keep verifying under
    this newer code while v2 policies (those using condition.match) verify against
    the version-bound form.
    """
    if not secret:
        return True
    actual_sig = policy.get("signature") or ""
    if not actual_sig:
        logger.warning(
            "Policy '%s' (id=%s) is unsigned but a policy secret is configured — "
            "rejected. Re-save the policy so the server signs it.",
            policy.get("name"),
            policy.get("id"),
        )
        return False
    try:
        version = int(policy.get("sig_version", 1) or 1)
    except (TypeError, ValueError):
        version = 1
    canonical = _policy_canonical(
        version,
        policy.get("id", ""),
        policy.get("agent_id", ""),
        policy.get("name", ""),
        policy.get("condition", {}),
        policy.get("action", {}),
        policy.get("enabled", True),
        policy.get("priority", 100),
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
        # Remote policies keyed by the agent_id they were fetched FOR, not by
        # the agent_id on the policy — a wildcard ("*") policy comes back on
        # every agent's fetch and so appears under several keys. Keeping them
        # separate is what stops one agent's fetch from wiping another's: the
        # fetch is already per-agent (see needs_fetch/mark_fetched), so a load
        # that replaced every remote policy left each agent in a two-agent
        # process unguarded for a fetch interval at a time, alternating.
        self._remote_by_agent: Dict[str, List[Policy]] = {}
        self._generation: int = (
            0  # incremented on every load/add so RunContext can detect staleness
        )

    # ── Config API ────────────────────────────────────────────────────────────

    def add(self, policy: Policy) -> None:
        with self._lock:
            self._policies.append(policy)
            self._policies.sort(key=lambda p: p.priority)
            self._generation += 1

    def load(self, raw: List[dict], secret: str = "", agent_id: str = "") -> None:
        """Replace the remote-sourced policies **for one agent** with a fresh
        list from the API.

        agent_id is the agent the fetch was made for. Policies fetched for other
        agents are untouched, so a process running several agents keeps every
        agent's guardrails loaded at once. Omitting it keeps the historical
        replace-everything behaviour, which is correct for a single-agent
        process and for callers that hold no per-agent notion.

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
            if not secret:
                action_type = (p.get("action") or {}).get("type", "log")
                if action_type in ENFORCING_ACTIONS:
                    # No secret means no way to tell this policy came from the
                    # real server, and this action would change what the
                    # customer's agent does. Record it, don't obey it.
                    logger.warning(
                        "Remote policy '%s' (id=%s) requests %r but no policy secret is "
                        "configured, so its origin cannot be verified — downgraded to "
                        "log-only. Set DUNETRACE_POLICY_SECRET (and POLICY_SIGNING_SECRET "
                        "on the server) to enable enforcing remote policies.",
                        p.get("name"),
                        p.get("id"),
                        action_type,
                    )
                    p = {**p, "action": {"type": "log"}}
            try:
                verified.append(Policy.from_dict(p))
            except ExpressionError as exc:
                # A malformed condition.match block — skip this one policy rather
                # than failing the whole load. A bad policy never reaches runtime.
                logger.warning(
                    "Policy '%s' (id=%s) has an invalid condition expression — skipped: %s",
                    p.get("name"),
                    p.get("id"),
                    exc,
                )

        with self._lock:
            local = [p for p in self._policies if p.id is None]
            if agent_id:
                self._remote_by_agent[agent_id] = verified
            else:
                self._remote_by_agent = {"": verified}
            # Dedupe by policy id: a wildcard policy is returned by every
            # agent's fetch, and evaluating it twice would double-count.
            remote: List[Policy] = []
            seen_ids = set()
            for bucket in self._remote_by_agent.values():
                for policy in bucket:
                    if policy.id is not None and policy.id in seen_ids:
                        continue
                    if policy.id is not None:
                        seen_ids.add(policy.id)
                    remote.append(policy)
            self._policies = sorted(local + remote, key=lambda p: p.priority)
            self._generation += 1
        logger.debug(
            "Policies loaded for %r: %d total (%d remote across %d agent(s))",
            agent_id or "*",
            len(self._policies),
            len(remote),
            len(self._remote_by_agent),
        )

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

    @staticmethod
    def _evaluate_traced(
        policy: "Policy",
        metrics: Dict[str, Any],
        context: "Optional[EvaluationContext]",
    ) -> tuple:
        """Evaluate a policy while capturing an observability trace. Returns
        ``(trigger_matched, trace, fired)`` where ``trace`` is a list of
        ComparisonTrace for the expression block (empty for legacy-only policies).
        Semantically identical to ``policy.matches`` — same AND composition — but
        instrumented."""
        legacy_ok = policy._legacy_matches(metrics)
        trace: list = []
        if policy.match_expr is None:
            expr_ok = True
        elif context is None:
            expr_ok = False
        else:
            expr_ok = bool(evaluate(policy.match_expr, context, trace=trace))
        return legacy_ok, trace, (legacy_ok and expr_ok)

    def evaluate(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        triggered_already: set,  # policy keys already fired in this run
        context: "Optional[EvaluationContext]" = None,
        observer=None,
    ) -> Optional[tuple]:
        """
        Return (policy, action_dict) for the highest-priority matching policy
        that hasn't already fired in this run, or None.

        ``context`` (when supplied) carries the field namespaces expression
        conditions evaluate against. In this metric-driven path raw tool ``args``
        are not available, so a policy whose ``match`` references ``args.*`` will
        not fire here — that is the ``before_tool_call`` gate's job.

        ``observer`` (optional) enables evaluation observability: when set, every
        applicable policy is evaluated with a trace and reported via
        ``observer(policy, trigger_matched, trace, fired)``, and enforcement no
        longer short-circuits at the first match (so lower-priority policies are
        still observed). When None (the default, production hot path), behavior
        and cost are exactly as before — short-circuit at the first match, no
        traces built.

        Policies with action.type == "log" are returned but also re-evaluatable
        (they don't get added to triggered_already by the caller).
        """
        with self._lock:
            snapshot = list(self._policies)

        winner = None
        for policy in snapshot:
            if not policy.enabled:
                continue
            if policy.agent_id not in ("*", agent_id):
                continue
            already = policy.key in triggered_already
            if observer is None:
                if already:
                    continue
                if policy.matches(metrics, context):
                    return policy, dict(policy.action)
                continue
            # Observed path: evaluate with a trace and report every policy.
            # before_tool_call policies are gate-only — they can never fire in the
            # metric path (no such metric), so observing them here would emit a
            # misleading "trigger did not match" record every tick. They're
            # observed at the approval gate (find_approval_policy) instead.
            if policy.condition.get("trigger") == "before_tool_call":
                continue
            trigger_matched, trace, fired = self._evaluate_traced(policy, metrics, context)
            observer(policy, trigger_matched, trace, fired)
            if fired and not already and winner is None:
                winner = (policy, dict(policy.action))

        return winner

    def find_approval_policy(
        self,
        agent_id: str,
        tool_name: str,
        context: "Optional[EvaluationContext]" = None,
        observer=None,
    ) -> Optional["Policy"]:
        """Return the highest-priority enabled require_approval policy whose
        before_tool_call condition matches this tool name, or None.

        Deliberately NOT deduplicated against triggered_already like evaluate()
        is — approval is a per-call gate: every invocation of a guarded tool
        must be approved independently, so a second call to the same tool in one
        run is gated again.

        ``context`` carries this call's raw ``args`` (plus run/event metadata),
        so a policy with a ``match`` block like ``{args.amount: {gt: 10000}}``
        gates only the calls whose arguments actually match.

        ``observer`` mirrors ``evaluate``: when set, every candidate approval
        policy is evaluated with a trace and reported, and the first match is
        still returned as the gate decision."""
        metrics = {"before_tool_call": tool_name}
        with self._lock:
            snapshot = list(self._policies)
        winner = None
        for policy in snapshot:
            if not policy.enabled:
                continue
            if policy.agent_id not in ("*", agent_id):
                continue
            if policy.condition.get("trigger") != "before_tool_call":
                continue
            if policy.action.get("type") != "require_approval":
                continue
            if observer is None:
                if policy.matches(metrics, context):
                    return policy
                continue
            trigger_matched, trace, fired = self._evaluate_traced(policy, metrics, context)
            observer(policy, trigger_matched, trace, fired)
            if fired and winner is None:
                winner = policy
        return winner
