"""
Policy evaluation observability (Phase 5).

Answers "why did (or didn't) my policy fire?" without reading source. Every
policy evaluation can produce a structured ``PolicyEvaluationRecord`` — the
policy, whether the trigger matched, each condition checked with the value
compared vs. expected, and the overall result — surfaced two ways:

  1. **Structured logs** on the ``dunetrace.policies.evaluation`` logger (DEBUG).
     Always available: raise that logger to DEBUG/INFO and every evaluation is
     logged as one structured record (``extra["policy_evaluation"]``), queryable
     by a log pipeline. Zero cost when the logger is below DEBUG.
  2. **Dashboard** via ``GET /v1/policies/{id}/evaluations`` — records are shipped
     (opt-in, ``policy_evaluation_reporting=True`` on the client) through the
     normal event transport as ``policy.evaluated`` events and routed by ingest
     into the ``policy_evaluations`` table.

Both are bounded by a per-policy rate limiter (100/min, deterministic sampling
beyond) so a hot loop can't flood logs or the network.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Structured-log logger. Customers set this to DEBUG to capture evaluations
#: locally; it is independent of the dashboard-reporting opt-in.
EVAL_LOGGER_NAME = "dunetrace.policies.evaluation"
eval_logger = logging.getLogger(EVAL_LOGGER_NAME)

#: Rate-limit defaults: record up to this many evaluations per policy per window…
DEFAULT_LIMIT_PER_MINUTE = 100
#: …then keep 1-in-N beyond the limit (deterministic, not random — preserves
#: reproducibility). N = 1 / sample_rate.
DEFAULT_SAMPLE_RATE = 0.1
_WINDOW_SECONDS = 60.0


@dataclass
class PolicyEvaluationRecord:
    """One structured evaluation outcome. JSON-serializable via ``to_dict``."""

    policy_name: str
    policy_id: Optional[int]
    agent_id: str
    run_id: str
    trigger: str
    trigger_matched: bool  # did the legacy (metric/signal/tool-name) part match
    fired: bool  # overall result (this policy matched)
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # ComparisonTrace dicts
    reason: str = ""
    sampled: bool = False  # True if emitted under beyond-limit sampling
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_id": self.policy_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "trigger": self.trigger,
            "trigger_matched": self.trigger_matched,
            "fired": self.fired,
            "conditions": self.conditions,
            "reason": self.reason,
            "sampled": self.sampled,
            "ts": self.ts,
        }


def build_reason(
    trigger: str, trigger_matched: bool, fired: bool, conditions: List[Dict[str, Any]]
) -> str:
    """A short human explanation of the outcome, for logs and the dashboard."""
    if fired:
        return "fired: all conditions matched"
    if not trigger_matched:
        return f"did not fire: trigger '{trigger}' did not match"
    # Trigger matched but an expression condition failed — point at the first
    # failing comparison, the usual "why didn't it fire" answer.
    for c in conditions:
        if not c.get("result", True):
            actual = c.get("actual")
            return (
                f"did not fire: {c.get('field_path')} {c.get('operator')} "
                f"{c.get('expected')!r} — actual {actual!r}"
            )
    return "did not fire"


class EvaluationRateLimiter:
    """Per-policy sliding-window limiter. Up to ``limit`` records per policy per
    60s window are admitted unsampled; beyond that, a deterministic 1-in-N are
    admitted and flagged sampled. Thread-safe; shared across all runs of a
    client so the cap is per process, not per run."""

    def __init__(
        self,
        limit_per_minute: int = DEFAULT_LIMIT_PER_MINUTE,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self._limit = max(0, int(limit_per_minute))
        self._n = max(1, int(round(1.0 / sample_rate))) if sample_rate > 0 else 0
        self._lock = threading.Lock()
        # key -> [window_start_monotonic, count_in_window, beyond_limit_seen]
        self._state: Dict[str, list] = {}

    def admit(self, key: str, *, now: Optional[float] = None) -> tuple:
        """Return ``(admit: bool, sampled: bool)`` for one evaluation of ``key``."""
        t = time.monotonic() if now is None else now
        with self._lock:
            st = self._state.get(key)
            if st is None or (t - st[0]) >= _WINDOW_SECONDS:
                self._state[key] = [t, 1, 0]
                return (self._limit > 0), False
            if st[1] < self._limit:
                st[1] += 1
                return True, False
            # Beyond the limit: deterministic 1-in-N sampling.
            st[2] += 1
            if self._n and (st[2] % self._n == 0):
                return True, True
            return False, True
