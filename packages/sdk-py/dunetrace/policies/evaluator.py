"""
Condition expression evaluator (Capability: expression-based policy conditions).

Evaluates a parsed ``ConditionExpression`` (see ``expressions.py``) against an
``EvaluationContext`` — the five field-path namespaces (``args``, ``run``,
``agent``, ``org``, ``event``) — and returns a bool.

Design guarantees:
  - **Deterministic.** Same expression + same context → same result. No clock
    reads, no randomness, no I/O. (``run.duration_ms`` etc. are supplied *in*
    the context by the caller, never computed here.)
  - **Safe.** No ``eval``, no ``getattr`` on caller data, no dynamic dispatch
    into Python internals. Field resolution is explicit ``Mapping`` key lookups;
    operators are an explicit whitelist table. A malformed policy cannot reach
    anything but dict indexing and the whitelisted comparators.
  - **Fast.** Field paths are pre-split at parse time and operators pre-resolved
    to functions, so the hot path is dict walks + one function call per
    comparison. Target <100µs per policy evaluation (see
    ``scripts/bench_policy_expressions.py``).
  - **Graceful on missing fields.** ``exists``/``not_exists`` answer presence
    directly; every other operator against an absent field is ``False`` with a
    DEBUG log line (so "why didn't my policy fire?" is answerable without source
    reading).

Type coercion (documented, deterministic — see docs/policies/condition-expressions.md):
  - Ordered ops (``gt``/``gte``/``lt``/``lte``): numbers compare numerically; a
    number vs a numeric string coerces the string to float; two strings compare
    lexicographically; anything else incomparable → ``False``.
  - ``eq``/``ne``: exact equality, plus number↔numeric-string coercion. Booleans
    only ever equal booleans (``eq: true`` never matches the integer ``1``).
    Lists and tuples compare by element (so the parser's list→tuple
    normalization is invisible).
  - ``in``/``not_in``/``contains``: membership using the same ``eq`` semantics.
  - ``matches``: regex (ReDoS-safe ``regex`` engine, per-call timeout) against
    string values only.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from dunetrace.policies.expressions import And, Comparison, ConditionExpression, Or

logger = logging.getLogger("dunetrace.policies.evaluator")

#: Sentinel for "field path did not resolve". Distinct from ``None`` because a
#: field may legitimately hold ``None`` (JSON null) — that is *present*, absent
#: is not.
MISSING: Any = object()

#: Default per-call regex budget for the ``matches`` operator.
_DEFAULT_REGEX_TIMEOUT_MS: float = 5.0


# ── Evaluation context ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationContext:
    """The runtime data a condition expression is evaluated against. One frozen
    struct with exactly five namespaces, matching the whitelisted field-path
    prefixes. Callers populate whichever they have; unpopulated namespaces make
    their fields evaluate as absent.

    - ``args``  — the tool call's arguments (``args.amount`` → ``args["amount"]``)
    - ``run``   — run metadata (``run.agent_id``, ``run.duration_ms``, ``run.event_count``)
    - ``agent`` — agent metadata (no SDK source yet; see BACKLOG.md)
    - ``org``   — org metadata (no SDK source yet; see BACKLOG.md)
    - ``event`` — the current event's fields (``event.type``, ``event.severity``)
    """

    args: Mapping[str, Any] = field(default_factory=dict)
    run: Mapping[str, Any] = field(default_factory=dict)
    agent: Mapping[str, Any] = field(default_factory=dict)
    org: Mapping[str, Any] = field(default_factory=dict)
    event: Mapping[str, Any] = field(default_factory=dict)

    def _root(self, prefix: str) -> Mapping[str, Any]:
        # Explicit dispatch — no getattr on a variable prefix. Prefix is already
        # validated against the whitelist at parse time.
        if prefix == "args":
            return self.args
        if prefix == "run":
            return self.run
        if prefix == "agent":
            return self.agent
        if prefix == "org":
            return self.org
        return self.event  # "event"


# ── Observability trace ───────────────────────────────────────────────────────


@dataclass
class ComparisonTrace:
    """One recorded comparison, for "why did/didn't this fire?" debugging.
    Populated only when a ``trace`` list is passed to ``evaluate`` — the hot
    path (``trace=None``) records nothing. ``actual`` is ``"<absent>"`` when the
    field did not resolve. Phase 5 serializes these into the evaluation log."""

    field_path: str
    operator: str
    expected: Any
    actual: Any
    result: bool
    present: bool

    def to_dict(self) -> dict:
        return {
            "field_path": self.field_path,
            "operator": self.operator,
            "expected": self.expected,
            "actual": self.actual,
            "result": self.result,
            "present": self.present,
        }


def _record(
    trace: Optional[List[ComparisonTrace]],
    node: "Comparison",
    actual: Any,
    result: bool,
) -> bool:
    if trace is not None:
        present = actual is not MISSING
        trace.append(
            ComparisonTrace(
                field_path=node.field_path,
                operator=node.operator,
                expected=None if node.operator in ("exists", "not_exists") else node.value,
                actual="<absent>" if not present else actual,
                result=result,
                present=present,
            )
        )
    return result


# ── Field resolution ──────────────────────────────────────────────────────────


def _resolve(ctx: EvaluationContext, prefix: str, path) -> Any:
    """Walk ``path`` through the namespace mapping via explicit key lookups.
    Returns ``MISSING`` if any segment is absent or a non-mapping is hit."""
    cur: Any = ctx._root(prefix)
    for seg in path:
        if isinstance(cur, Mapping) and seg in cur:
            cur = cur[seg]
        else:
            return MISSING
    return cur


# ── Value helpers (deterministic coercion) ────────────────────────────────────


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_number(s: Any):
    """float(s) if s is a numeric string, else None."""
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_numeric_pair(a: Any, b: Any):
    """If exactly one side is a number and the other a numeric string, coerce
    the string to float. Otherwise return the pair unchanged."""
    if _is_number(a) and not _is_number(b):
        nb = _as_number(b)
        if nb is not None:
            return a, nb
    elif _is_number(b) and not _is_number(a):
        na = _as_number(a)
        if na is not None:
            return na, b
    return a, b


def _values_equal(a: Any, b: Any) -> bool:
    """eq semantics: exact, with number↔numeric-string coercion; booleans equal
    only booleans; sequences compare element-wise (list/tuple agnostic)."""
    if a is MISSING:
        return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    a_bool, b_bool = isinstance(a, bool), isinstance(b, bool)
    if a_bool or b_bool:
        return a_bool and b_bool and a == b
    if a == b:
        return True
    x, y = _coerce_numeric_pair(a, b)
    if x is not a or y is not b:
        try:
            return bool(x == y)
        except TypeError:
            return False
    return False


def _ordered(a: Any, b: Any, cmp) -> bool:
    """gt/gte/lt/lte: numeric coercion then native comparison; incomparable → False."""
    x, y = _coerce_numeric_pair(a, b)
    try:
        return bool(cmp(x, y))
    except TypeError:
        return False


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str):
        return isinstance(expected, str) and expected in actual
    if isinstance(actual, (list, tuple)):
        return any(_values_equal(item, expected) for item in actual)
    return False


@functools.lru_cache(maxsize=512)
def _compile(pattern: str):
    """Compile (and cache) a regex with the ReDoS-safe ``regex`` engine, falling
    back to stdlib ``re`` only if ``regex`` is unavailable. Patterns are already
    compile-validated at parse time; this just memoizes the compiled object."""
    try:
        import regex as _re

        return _re.compile(pattern), True  # True → supports timeout kwarg
    except ImportError:  # pragma: no cover - regex is a declared dependency
        import re as _re

        return _re.compile(pattern), False


def _matches(actual: Any, pattern: str, timeout_ms: float) -> bool:
    if not isinstance(actual, str):
        return False
    compiled, has_timeout = _compile(pattern)
    try:
        if has_timeout:
            return compiled.search(actual, timeout=timeout_ms / 1000.0) is not None
        return compiled.search(actual) is not None  # pragma: no cover - fallback
    except TimeoutError:
        logger.warning(
            "Regex 'matches' timed out after %.1fms on pattern %r — treated as "
            "no match. Simplify the pattern.",
            timeout_ms,
            pattern,
        )
        return False


# ── Operator table (explicit whitelist; no dynamic dispatch) ──────────────────
#
# Each entry maps a canonical operator to a comparator (actual, expected) -> bool.
# ``matches`` is handled separately because it needs the regex timeout. Presence
# operators (exists/not_exists) are handled before this table (they must see the
# MISSING sentinel, which every other operator treats as an automatic False).

_BINARY_OPS = {
    "eq": _values_equal,
    "ne": lambda a, b: not _values_equal(a, b),
    "gt": lambda a, b: _ordered(a, b, lambda x, y: x > y),
    "gte": lambda a, b: _ordered(a, b, lambda x, y: x >= y),
    "lt": lambda a, b: _ordered(a, b, lambda x, y: x < y),
    "lte": lambda a, b: _ordered(a, b, lambda x, y: x <= y),
    "in": lambda a, b: any(_values_equal(a, item) for item in b),
    "not_in": lambda a, b: not any(_values_equal(a, item) for item in b),
    "contains": _contains,
    "starts_with": lambda a, b: isinstance(a, str) and a.startswith(b),
    "ends_with": lambda a, b: isinstance(a, str) and a.endswith(b),
}


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    expr: ConditionExpression,
    ctx: EvaluationContext,
    *,
    trace: Optional[List[ComparisonTrace]] = None,
    regex_timeout_ms: float = _DEFAULT_REGEX_TIMEOUT_MS,
) -> bool:
    """Evaluate ``expr`` against ``ctx``, returning True iff it matches.

    Pass a ``trace`` list to collect a ``ComparisonTrace`` per leaf (for
    observability). When ``trace is None`` the And/Or nodes short-circuit;
    when a trace is requested every leaf is evaluated so the record is complete.
    """
    return _eval(expr, ctx, trace, regex_timeout_ms)


def _eval(
    node: ConditionExpression,
    ctx: EvaluationContext,
    trace: Optional[List[ComparisonTrace]],
    rt: float,
) -> bool:
    if type(node) is Comparison:
        return _eval_comparison(node, ctx, trace, rt)
    if type(node) is And:
        if trace is None:
            return all(_eval(c, ctx, None, rt) for c in node.children)
        # Force full evaluation (list, not generator) so the trace is complete.
        return all([_eval(c, ctx, trace, rt) for c in node.children])
    if type(node) is Or:
        if trace is None:
            return any(_eval(c, ctx, None, rt) for c in node.children)
        return any([_eval(c, ctx, trace, rt) for c in node.children])
    # Unreachable for well-formed trees; fail closed rather than raise on the
    # runtime path.
    logger.error("Unknown expression node type %r — evaluated as False", type(node))
    return False


def _eval_comparison(
    node: Comparison,
    ctx: EvaluationContext,
    trace: Optional[List[ComparisonTrace]],
    rt: float,
) -> bool:
    op = node.operator
    actual = _resolve(ctx, node.prefix, node.path)

    # Presence operators reason about MISSING directly.
    if op == "exists":
        return _record(trace, node, actual, actual is not MISSING)
    if op == "not_exists":
        return _record(trace, node, actual, actual is MISSING)

    # Every other operator against an absent field is False (with a debug note).
    if actual is MISSING:
        logger.debug(
            "Field %r absent while evaluating '%s' — comparison is False.",
            node.field_path,
            op,
        )
        return _record(trace, node, actual, False)

    if op == "matches":
        return _record(trace, node, actual, _matches(actual, node.value, rt))

    comparator = _BINARY_OPS.get(op)
    if comparator is None:  # pragma: no cover - operators validated at parse time
        logger.error("Unknown operator %r — comparison is False", op)
        return _record(trace, node, actual, False)
    return _record(trace, node, actual, comparator(actual, node.value))
