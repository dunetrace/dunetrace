"""
Phase 5 tests for OTel receiver observability: the anomaly-detection / totals
functions and the dashboard endpoint (route handler called directly with a
mocked DB, matching this codebase's test pattern).
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from api_svc.otel_receiver_health import detect_anomalies, summarize_totals
from api_svc.routers.otel_receiver import (
    OtelIngestionToggle,
    get_otel_ingestion_enabled,
    otel_receiver_stats,
    set_otel_ingestion_enabled,
)


def _hour(**kw):
    base = {
        "batches_received": 0,
        "spans_received": 0,
        "events_translated": 0,
        "spans_rejected": 0,
        "auth_failures": 0,
        "rate_limit_hits": 0,
        "rejections": {},
    }
    base.update(kw)
    return base


class TestSummarizeTotals(unittest.TestCase):
    def test_sums_and_merges_rejections(self):
        series = [
            _hour(
                spans_received=100,
                events_translated=90,
                spans_rejected=2,
                rejections={"malformed": 2},
            ),
            _hour(
                spans_received=50,
                events_translated=48,
                spans_rejected=3,
                rejections={"malformed": 1, "rate_limited": 2},
            ),
        ]
        totals = summarize_totals(series)
        self.assertEqual(totals["spans_received"], 150)
        self.assertEqual(totals["events_translated"], 138)
        self.assertEqual(totals["rejections"], {"malformed": 3, "rate_limited": 2})

    def test_translation_success_rate(self):
        series = [_hour(events_translated=90, spans_rejected=10)]
        self.assertEqual(summarize_totals(series)["translation_success_rate"], 0.9)

    def test_empty_series(self):
        totals = summarize_totals([])
        self.assertEqual(totals["spans_received"], 0)
        self.assertIsNone(totals["translation_success_rate"])


class TestDetectAnomalies(unittest.TestCase):
    def test_steady_traffic_has_no_anomalies(self):
        series = [
            _hour(batches_received=100, spans_received=1000, spans_rejected=1) for _ in range(6)
        ]
        self.assertEqual(detect_anomalies(series), [])

    def test_high_rejection_rate(self):
        series = [
            _hour(batches_received=100, spans_received=1000, spans_rejected=1) for _ in range(3)
        ]
        series.append(
            _hour(batches_received=40, spans_received=400, spans_rejected=60)
        )  # 60% rejected
        types = {a["type"] for a in detect_anomalies(series)}
        self.assertIn("high_rejection_rate", types)

    def test_traffic_drop(self):
        series = [_hour(batches_received=100, spans_received=1000) for _ in range(3)]
        series.append(_hour(batches_received=1, spans_received=5))  # near-zero vs ~1000
        types = {a["type"] for a in detect_anomalies(series)}
        self.assertIn("traffic_drop", types)

    def test_traffic_spike(self):
        series = [_hour(batches_received=10, spans_received=100) for _ in range(3)]
        series.append(_hour(batches_received=100, spans_received=2000))  # 20x baseline
        types = {a["type"] for a in detect_anomalies(series)}
        self.assertIn("traffic_spike", types)

    def test_low_volume_not_flagged(self):
        # A couple of rejected requests in a quiet hour is not an anomaly.
        series = [_hour(batches_received=1, spans_received=5, spans_rejected=2)]
        self.assertEqual(detect_anomalies(series), [])


class TestEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_returns_series_totals_and_anomalies(self):
        series = [
            _hour(
                batches_received=100, spans_received=1000, events_translated=990, spans_rejected=1
            )
            for _ in range(3)
        ]
        series.append(
            _hour(
                batches_received=40,
                spans_received=400,
                spans_rejected=60,
                rejections={"rate_limited": 60},
            )
        )
        with patch(
            "api_svc.routers.otel_receiver.fetch_otel_receiver_stats",
            AsyncMock(return_value=series),
        ):
            result = await otel_receiver_stats(hours=24, org_id="org-1")

        self.assertEqual(result["org_id"], "org-1")
        self.assertEqual(len(result["series"]), 4)
        self.assertEqual(result["totals"]["rejections"]["rate_limited"], 60)
        self.assertTrue(any(a["type"] == "high_rejection_rate" for a in result["anomalies"]))

    async def test_empty_when_no_data(self):
        with patch(
            "api_svc.routers.otel_receiver.fetch_otel_receiver_stats",
            AsyncMock(return_value=[]),
        ):
            result = await otel_receiver_stats(hours=24, org_id="org-1")
        self.assertEqual(result["series"], [])
        self.assertEqual(result["anomalies"], [])


class TestEnablementToggle(unittest.IsolatedAsyncioTestCase):
    async def test_get_reflects_stored_value(self):
        with patch(
            "api_svc.routers.otel_receiver.get_org_otel_ingestion_enabled",
            AsyncMock(return_value=False),
        ):
            result = await get_otel_ingestion_enabled(org_id="org-1")
        self.assertEqual(result, {"enabled": False})

    async def test_put_sets_value_for_own_org(self):
        setter = AsyncMock()
        with patch("api_svc.routers.otel_receiver.set_org_otel_ingestion_enabled", setter):
            result = await set_otel_ingestion_enabled(
                OtelIngestionToggle(enabled=False), org_id="org-1"
            )
        setter.assert_awaited_once_with("org-1", False)
        self.assertEqual(result, {"enabled": False})
