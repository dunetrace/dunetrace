"""
Condition expression parser (Capability: expression-based policy conditions).

Parses the ``condition.match`` block of a policy into an immutable
``ConditionExpression`` tree. This module is *parse and validate only* — it
contains no evaluation logic (see ``evaluator.py``) and makes no runtime data
access. Parsing is deterministic and side-effect free.

The ``match`` block sits alongside the existing flat ``{trigger, operator,
value}`` condition and is fully additive: policies without a ``match`` key are
unaffected. Example::

    condition:
      trigger: before_tool_call        # existing fields, unchanged
      value: refund_customer
      match:                           # NEW — parsed by this module
        args.amount: {gt: 10000}
        or:
          - agent.tier: {eq: "trial"}
          - org.plan:   {in: ["free", "starter"]}

evaluates as ``amount > 10000 AND (agent.tier == "trial" OR org.plan in
["free","starter"])``.

Grammar (a *block* is a mapping):
  - ``"<prefix>.<path>": {<op>: <value>, ...}`` — one Comparison per operator;
    multiple operators on the same field AND together (e.g. a range check
    ``{gt: 10, lt: 100}``).
  - ``or: [<block>, ...]`` — the sub-blocks OR together.
  - ``and: [<block>, ...]`` — the sub-blocks AND together (explicit; useful for
    grouping ORs).
  - All keys in one block AND together. An empty block is rejected.

Whitelists (v1 — no wildcards, no additions):
  operators   — eq ne gt gte lt lte in not_in contains starts_with ends_with
                matches exists not_exists  (``neq`` is accepted as an alias of
                ``ne`` for muscle-memory parity with the legacy flat condition)
  field paths — args.*  run.*  agent.*  org.*  event.*
                (``agent.*`` / ``org.*`` parse and validate but have no runtime
                source yet — they always evaluate as *absent*; see BACKLOG.md)

Nesting is capped at ``MAX_DEPTH`` (3) levels of blocks; deeper policies are
rejected at parse time so a bad policy can never reach the runtime path.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

# ── Whitelists (v1 — closed sets, no wildcards) ───────────────────────────────

#: Field-path prefixes. Anything not starting with one of these is rejected.
FIELD_PREFIXES: FrozenSet[str] = frozenset({"args", "run", "agent", "org", "event"})

#: The 14 whitelisted comparison operators (canonical spellings).
EXPRESSION_OPERATORS: FrozenSet[str] = frozenset(
    {
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "not_in",
        "contains",
        "starts_with",
        "ends_with",
        "matches",
        "exists",
        "not_exists",
    }
)

#: Reserved block keys that combine sub-blocks rather than name a field.
_COMBINATORS: FrozenSet[str] = frozenset({"and", "or"})

#: Operators that ignore their value (presence checks). Value is stored as None.
_NO_VALUE_OPERATORS: FrozenSet[str] = frozenset({"exists", "not_exists"})

#: Operators whose value must be a list/tuple (membership tests).
_LIST_VALUE_OPERATORS: FrozenSet[str] = frozenset({"in", "not_in"})

#: Operators whose value must be a string (affix tests).
_STRING_VALUE_OPERATORS: FrozenSet[str] = frozenset({"starts_with", "ends_with"})

#: Accepted operator aliases → canonical. Per product decision, the expression
#: namespace tolerates the legacy ``neq`` spelling as well as ``ne``.
_OPERATOR_ALIASES: Dict[str, str] = {"neq": "ne"}

#: Curated "did you mean" hints for common operator typos (difflib alone misses
#: word-shaped mistakes like "greaterthan"). Consulted before difflib.
_OP_SUGGESTIONS: Dict[str, str] = {
    "greaterthan": "gt",
    "greater_than": "gt",
    ">": "gt",
    "greaterorequal": "gte",
    "greaterthanorequal": "gte",
    ">=": "gte",
    "gteq": "gte",
    "lessthan": "lt",
    "<": "lt",
    "lessorequal": "lte",
    "lessthanorequal": "lte",
    "<=": "lte",
    "lteq": "lte",
    "equal": "eq",
    "equals": "eq",
    "==": "eq",
    "=": "eq",
    "is": "eq",
    "notequal": "ne",
    "notequals": "ne",
    "not_equal": "ne",
    "!=": "ne",
    "<>": "ne",
    "included_in": "in",
    "member_of": "in",
    "notin": "not_in",
    "not_included": "not_in",
    "includes": "contains",
    "has": "contains",
    "startswith": "starts_with",
    "prefix": "starts_with",
    "beginswith": "starts_with",
    "endswith": "ends_with",
    "suffix": "ends_with",
    "regex": "matches",
    "regex_matches": "matches",
    "match": "matches",
    "re": "matches",
    "present": "exists",
    "is_present": "exists",
    "isset": "exists",
    "absent": "not_exists",
    "missing": "not_exists",
    "is_absent": "not_exists",
}

#: Maximum block-nesting depth. The root block is depth 1; each ``or``/``and``
#: list descends one level. Blocks deeper than this are rejected.
MAX_DEPTH: int = 3


class ExpressionError(ValueError):
    """Raised when a ``condition.match`` block is malformed. The message names
    the policy, the offending token, and (where possible) a suggested fix, so a
    customer can debug a rejected policy without reading source."""


# ── Expression tree (immutable, hashable, serializable) ───────────────────────


class ConditionExpression:
    """Base for the three node kinds. Instances are immutable and hashable so a
    parsed tree can be cached, compared, and serialized. Subclasses are frozen
    dataclasses; this base only declares the shared interface."""

    __slots__ = ()

    def field_paths(self) -> FrozenSet[str]:
        """Every ``prefix.path`` string referenced anywhere in this subtree."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Canonical, JSON-serializable form. Deterministic for a given tree —
        used for debugging, observability, and (Phase 4) signing cross-checks."""
        raise NotImplementedError

    def describe(self) -> str:
        """Compact human-readable rendering for logs/observability."""
        raise NotImplementedError


@dataclass(frozen=True)
class Comparison(ConditionExpression):
    """A single ``field_path <operator> value`` leaf.

    ``value`` is normalized to a hashable form at construction: list values
    become tuples, and presence operators (``exists``/``not_exists``) store
    ``None`` (their value is ignored).
    """

    field_path: str  # e.g. "args.amount" — the original dotted string
    prefix: str  # e.g. "args"
    path: Tuple[str, ...]  # e.g. ("amount",) or ("customer", "email")
    operator: str  # canonical operator, e.g. "gt", "ne"
    value: Any  # hashable; None for exists/not_exists

    def field_paths(self) -> FrozenSet[str]:
        return frozenset({self.field_path})

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {"field": self.field_path, "op": self.operator}
        if self.operator not in _NO_VALUE_OPERATORS:
            v = self.value
            out["value"] = list(v) if isinstance(v, tuple) else v
        return out

    def describe(self) -> str:
        if self.operator in _NO_VALUE_OPERATORS:
            return f"{self.field_path} {self.operator}"
        return f"{self.field_path} {self.operator} {self.value!r}"


@dataclass(frozen=True)
class And(ConditionExpression):
    """All children must match."""

    children: Tuple[ConditionExpression, ...]

    def field_paths(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for c in self.children:
            out |= c.field_paths()
        return out

    def to_dict(self) -> dict:
        return {"and": [c.to_dict() for c in self.children]}

    def describe(self) -> str:
        return "(" + " AND ".join(c.describe() for c in self.children) + ")"


@dataclass(frozen=True)
class Or(ConditionExpression):
    """Any child may match."""

    children: Tuple[ConditionExpression, ...]

    def field_paths(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for c in self.children:
            out |= c.field_paths()
        return out

    def to_dict(self) -> dict:
        return {"or": [c.to_dict() for c in self.children]}

    def describe(self) -> str:
        return "(" + " OR ".join(c.describe() for c in self.children) + ")"


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_condition(condition: dict, *, policy_name: str = "") -> Optional[ConditionExpression]:
    """Extract and parse the ``match`` block from a full policy ``condition``.

    Returns ``None`` when the condition has no ``match`` key (a legacy flat
    condition), so callers can cheaply detect "no expression here". Raises
    ``ExpressionError`` if a ``match`` block is present but malformed.
    """
    if not isinstance(condition, dict):
        return None
    raw = condition.get("match")
    if raw is None:
        return None
    return parse_match_block(raw, policy_name=policy_name)


def parse_match_block(raw: Any, *, policy_name: str = "", _depth: int = 1) -> ConditionExpression:
    """Parse one block (mapping) into a ConditionExpression tree.

    A block's keys AND together. A key is either a ``prefix.path`` field
    (mapping to a ``{operator: value}`` dict) or a combinator (``and``/``or``
    mapping to a list of sub-blocks). Raises ``ExpressionError`` on any
    malformed structure, unknown operator, unknown field prefix, empty block,
    or nesting past ``MAX_DEPTH``.
    """
    where = f" in policy {policy_name!r}" if policy_name else ""

    if _depth > MAX_DEPTH:
        raise ExpressionError(
            f"Condition nesting exceeds the maximum depth of {MAX_DEPTH}{where}. "
            f"Flatten the policy or split it into multiple policies."
        )
    if not isinstance(raw, dict):
        raise ExpressionError(
            f"A condition block must be a mapping{where}, got {type(raw).__name__}."
        )
    if not raw:
        raise ExpressionError(
            f"Empty condition block{where}. A block must contain at least one "
            f"field comparison or combinator; an empty block would match "
            f"unconditionally, which is never what you want."
        )

    children: List[ConditionExpression] = []
    for key, val in raw.items():
        if not isinstance(key, str):
            raise ExpressionError(
                f"Condition keys must be strings{where}, got {key!r} ({type(key).__name__})."
            )
        if key in _COMBINATORS:
            children.append(_parse_combinator(key, val, policy_name, _depth))
        else:
            children.extend(_parse_field(key, val, policy_name))

    if len(children) == 1:
        return children[0]
    return And(tuple(children))


def _parse_combinator(key: str, val: Any, policy_name: str, depth: int) -> ConditionExpression:
    where = f" in policy {policy_name!r}" if policy_name else ""
    if not isinstance(val, list):
        raise ExpressionError(
            f"'{key}' must map to a list of condition blocks{where}, got {type(val).__name__}."
        )
    if not val:
        raise ExpressionError(
            f"Empty '{key}' list{where}. '{key}' needs at least one sub-block "
            f"(and typically two — a single-element '{key}' is redundant)."
        )
    subs = tuple(
        parse_match_block(block, policy_name=policy_name, _depth=depth + 1) for block in val
    )
    return Or(subs) if key == "or" else And(subs)


def _parse_field(field_path: str, comp: Any, policy_name: str) -> List[Comparison]:
    where = f" in policy {policy_name!r}" if policy_name else ""
    prefix, path = _split_field_path(field_path, policy_name)

    if not isinstance(comp, dict):
        raise ExpressionError(
            f"Field {field_path!r} must map to an operator mapping like "
            f"{{gt: 10}}{where}, got {type(comp).__name__}."
        )
    if not comp:
        raise ExpressionError(
            f"Field {field_path!r} has no operators{where}. Specify at least one, "
            f"e.g. {{{field_path}: {{eq: ...}}}}."
        )

    out: List[Comparison] = []
    for op_raw, value in comp.items():
        op = _canonical_operator(op_raw, field_path, policy_name)
        norm_value = _validate_value(op, value, field_path, policy_name)
        out.append(Comparison(field_path, prefix, path, op, norm_value))
    return out


def _split_field_path(fp: Any, policy_name: str) -> Tuple[str, Tuple[str, ...]]:
    where = f" in policy {policy_name!r}" if policy_name else ""
    if not isinstance(fp, str) or "." not in fp:
        raise ExpressionError(f"Field path {fp!r} must be dotted like 'args.amount'{where}.")
    prefix, _, rest = fp.partition(".")
    if prefix not in FIELD_PREFIXES:
        suggestion = _closest(prefix, FIELD_PREFIXES)
        hint = f" Did you mean '{suggestion}.…'?" if suggestion else ""
        raise ExpressionError(
            f"Unknown field path prefix {prefix!r}{where}. "
            f"Allowed prefixes: {sorted(FIELD_PREFIXES)}.{hint}"
        )
    if not rest:
        raise ExpressionError(f"Field path {fp!r} is missing a sub-path after '{prefix}.'{where}.")
    path = tuple(rest.split("."))
    if any(seg == "" for seg in path):
        raise ExpressionError(
            f"Field path {fp!r} has an empty segment{where} — no '..' or trailing '.'."
        )
    return prefix, path


def _canonical_operator(op_raw: Any, field_path: str, policy_name: str) -> str:
    where = f" in policy {policy_name!r}" if policy_name else ""
    if not isinstance(op_raw, str):
        raise ExpressionError(
            f"Operator for field {field_path!r} must be a string{where}, got "
            f"{op_raw!r} ({type(op_raw).__name__})."
        )
    op = _OPERATOR_ALIASES.get(op_raw, op_raw)
    if op not in EXPRESSION_OPERATORS:
        suggestion = _OP_SUGGESTIONS.get(op_raw.lower()) or _closest(op_raw, EXPRESSION_OPERATORS)
        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
        raise ExpressionError(f"Unknown operator {op_raw!r} for field {field_path!r}{where}.{hint}")
    return op


def _validate_value(op: str, value: Any, field_path: str, policy_name: str) -> Any:
    """Validate and normalize a comparison value for its operator. Returns a
    hashable value (lists become tuples; presence ops return None)."""
    where = f" in policy {policy_name!r}" if policy_name else ""

    if op in _NO_VALUE_OPERATORS:
        return None  # value is ignored for exists / not_exists

    if op in _LIST_VALUE_OPERATORS:
        if not isinstance(value, (list, tuple)):
            raise ExpressionError(
                f"Operator '{op}' on {field_path!r} requires a list value{where}, "
                f"got {type(value).__name__}."
            )
        return tuple(value)

    if op in _STRING_VALUE_OPERATORS:
        if not isinstance(value, str):
            raise ExpressionError(
                f"Operator '{op}' on {field_path!r} requires a string value{where}, "
                f"got {type(value).__name__}."
            )
        return value

    if op == "matches":
        if not isinstance(value, str):
            raise ExpressionError(
                f"Operator 'matches' on {field_path!r} requires a string regex "
                f"pattern{where}, got {type(value).__name__}."
            )
        _validate_regex(value, field_path, policy_name)
        return value

    # eq / ne / gt / gte / lt / lte / contains — scalar comparand. Normalize any
    # list to a tuple so the Comparison stays hashable.
    if isinstance(value, list):
        return tuple(value)
    return value


def _validate_regex(pattern: str, field_path: str, policy_name: str) -> None:
    """Compile-check a ``matches`` pattern at parse time so an invalid regex is
    rejected when the policy loads, not when it first fires. Uses the ReDoS-safe
    ``regex`` engine when available (the codebase standard), falling back to the
    stdlib ``re`` for the compile check only — actual matching, with a timeout,
    lives in the evaluator."""
    try:
        import regex as _re  # ReDoS-safe engine (also used by custom detectors)
    except ImportError:  # pragma: no cover - regex is a declared dependency
        import re as _re  # brief-sanctioned restricted fallback
    where = f" in policy {policy_name!r}" if policy_name else ""
    try:
        _re.compile(pattern)
    except Exception as exc:  # re.error / regex.error
        raise ExpressionError(f"Invalid regex for 'matches' on {field_path!r}{where}: {exc}")


def _closest(word: Any, candidates: FrozenSet[str]) -> Optional[str]:
    """Nearest whitelist entry to ``word`` via difflib, or None. Deterministic."""
    matches = difflib.get_close_matches(str(word).lower(), sorted(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None
