"""
Anomaly detection and totals for OTel receiver stats (Phase 5).

Pure functions over an ascending hourly series of receiver counters (from the
otel_receiver_stats table). Kept dependency-free so the dashboard endpoint and
its tests can call them directly.
"""

from __future__ import annotations

from typing import Any, Dict, List


def summarize_totals(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum the counters across the series, merging the per-reason rejection maps.
    The rejection breakdown is the receiver's troubleshooting view: it says why
    spans were turned away, so a customer whose spans are missing can see whether
    it is auth, rate limiting, or malformed payloads."""
    totals = {
        "spans_received": 0,
        "events_translated": 0,
        "spans_rejected": 0,
        "auth_failures": 0,
        "rate_limit_hits": 0,
        "batches_received": 0,
    }
    rejections: Dict[str, int] = {}
    for hour in series:
        for key in totals:
            totals[key] += int(hour.get(key, 0) or 0)
        for reason, count in (hour.get("rejections") or {}).items():
            rejections[reason] = rejections.get(reason, 0) + int(count or 0)
    totals["rejections"] = rejections
    handled = totals["events_translated"]
    translated_and_rejected = handled + totals["spans_rejected"]
    totals["translation_success_rate"] = (
        round(handled / translated_and_rejected, 4) if translated_and_rejected else None
    )
    return totals


def detect_anomalies(
    series: List[Dict[str, Any]],
    *,
    min_volume: int = 20,
    rejection_rate: float = 0.2,
    drop_ratio: float = 0.25,
    spike_ratio: float = 5.0,
) -> List[Dict[str, str]]:
    """Flag abnormal receiver patterns in the most recent hour against the
    preceding hours as a baseline: a high rejection rate, a sudden traffic drop
    (a customer integration broke), or a sudden spike."""
    anomalies: List[Dict[str, str]] = []
    if not series:
        return anomalies

    latest = series[-1]
    accepted = int(latest.get("batches_received", 0) or 0)
    rejected = int(latest.get("spans_rejected", 0) or 0)
    requests = accepted + rejected
    if requests >= min_volume:
        rate = rejected / requests
        if rate >= rejection_rate:
            anomalies.append(
                {
                    "type": "high_rejection_rate",
                    "severity": "high",
                    "detail": f"{rate:.0%} of requests rejected in the last hour",
                }
            )

    baseline = series[:-1]
    if baseline:
        avg_received = sum(int(h.get("spans_received", 0) or 0) for h in baseline) / len(baseline)
        current = int(latest.get("spans_received", 0) or 0)
        if avg_received >= min_volume and current < avg_received * drop_ratio:
            anomalies.append(
                {
                    "type": "traffic_drop",
                    "severity": "medium",
                    "detail": f"spans/hour fell from ~{avg_received:.0f} to {current}",
                }
            )
        elif avg_received >= 1 and current >= min_volume and current > avg_received * spike_ratio:
            anomalies.append(
                {
                    "type": "traffic_spike",
                    "severity": "medium",
                    "detail": f"spans/hour jumped from ~{avg_received:.0f} to {current}",
                }
            )
    return anomalies
