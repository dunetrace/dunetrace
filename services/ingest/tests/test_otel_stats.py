"""Phase 5 unit tests for the OTLP receiver stats collector."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingest_svc.otel_stats import OtelStats


def _bucket(drained, org):
    return next(v for (o, _hour), v in drained.items() if o == org)


def test_records_per_org_and_drains():
    s = OtelStats()
    s.record_received("org-1", 10)
    s.record_received("org-1", 5)
    s.record_translated("org-1", 12)
    s.record_rejected("org-1", "rate_limited")
    s.record_rejected("org-1", "malformed")
    s.record_rate_limit_hit("org-1")

    drained = s.drain()
    b = _bucket(drained, "org-1")
    assert b["batches_received"] == 2
    assert b["spans_received"] == 15
    assert b["events_translated"] == 12
    assert b["spans_rejected"] == 2
    assert b["rejections"] == {"rate_limited": 1, "malformed": 1}
    assert b["rate_limit_hits"] == 1


def test_unattributed_events_go_to_system_bucket():
    s = OtelStats()
    s.record_auth_failure()
    s.record_rejected(None, "oversized")

    drained = s.drain()
    b = _bucket(drained, "_system")
    assert b["auth_failures"] == 1
    assert b["rejections"] == {"oversized": 1}


def test_drain_clears_state():
    s = OtelStats()
    s.record_received("org", 1)
    assert s.drain()  # non-empty
    assert s.drain() == {}  # cleared


def test_rejections_are_plain_dicts_after_drain():
    # drain() must hand back plain dicts, not defaultdicts, so the DB flush and
    # JSON serialization behave predictably.
    s = OtelStats()
    s.record_rejected("org", "malformed")
    b = _bucket(s.drain(), "org")
    assert type(b["rejections"]) is dict
