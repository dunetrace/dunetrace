"""
Pure math for Phase 1.5's semantic usage projection — no DB, no I/O.
Deliberately simple linear extrapolation, not a forecasting model: usage
rarely tracks linearly in practice, but a rough "at this rate you'll hit N by
month-end" figure is more useful for a quota dashboard than nothing, and
anything fancier is unjustified complexity for a first version.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone


def current_month(now: datetime | None = None) -> str:
    """'YYYY-MM' UTC bucket — same convention semantic_svc's usage counters
    use (services/semantic/semantic_svc/worker.py::_current_month)."""
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m")


def project_month_end(used_so_far: float, now: datetime | None = None) -> float:
    """Linearly extrapolates used_so_far (evaluations or cost, either works —
    same formula) to a month-end estimate, using now.day as a rough proxy for
    "days elapsed" (slightly overcounts early in the day, undercounts late in
    it — good enough for an estimate, not exact billing).
    """
    now = now or datetime.now(timezone.utc)
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    days_elapsed = max(now.day, 1)
    return used_so_far / days_elapsed * days_in_month
