"""
services/alerts/app/worker.py

The alert delivery loop.

Every POLL_INTERVAL seconds:
  1. Fetch unalerted live signals from the DB
  2. For each: reconstruct FailureSignal → explain() → format → send
  3. On successful delivery: mark as alerted

Pipeline per signal:
    DB row
      → FailureSignal (models)
      → Explanation   (explainer)
      → Slack payload + webhook payload (formatters)
      → HTTP delivery (sender)
      → mark alerted  (DB)

Severity filter:
    Only signals at or above SLACK_MIN_SEVERITY are sent to Slack.
    All live signals are sent to the generic webhook (recipient can filter).

Delivery is at-least-once:
    If the process crashes between send and mark_alerted,
    the signal will be re-sent on the next restart.
    Idempotency is the receiver's responsibility.

Run:
    cd services/alerts
    python -m app.worker
"""
from __future__ import annotations

import asyncio
import logging
import sys
import os
import time

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports ────────────────────────────────────────────────────────────────────
from dunetrace.models import FailureSignal, FailureType, Severity

from alerts_svc.explainer import explain  # bridge                             # type: ignore
# Explanation is in dunetrace.models (added there for cross-service use)
from alerts_svc.formatters.slack   import format_slack              # type: ignore
from alerts_svc.formatters.webhook import build_signed_request      # type: ignore
from alerts_svc.sender  import send_slack, send_webhook, SendResult
from alerts_svc.db      import init_pool, close_pool, fetch_unalerted_signals, mark_alerted_batch
from alerts_svc.config  import settings, SEVERITY_ORDER

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.alerts")


# ── Signal reconstruction ──────────────────────────────────────────────────────

def _row_to_signal(row: dict) -> FailureSignal:
    """Reconstruct a FailureSignal from a DB row dict."""
    detected_at = row.get("detected_at")
    if hasattr(detected_at, "timestamp"):
        detected_at = detected_at.timestamp()
    elif detected_at is None:
        detected_at = time.time()

    return FailureSignal(
        failure_type  = FailureType(row["failure_type"]),
        severity      = Severity(row["severity"]),
        run_id        = row["run_id"],
        agent_id      = row["agent_id"],
        agent_version = row["agent_version"],
        step_index    = row["step_index"],
        confidence    = row["confidence"],
        evidence      = row.get("evidence") or {},
        detected_at   = detected_at,
    )


# ── Severity filter ────────────────────────────────────────────────────────────

def _meets_slack_threshold(severity: str) -> bool:
    return (
        SEVERITY_ORDER.get(severity, 0)
        >= SEVERITY_ORDER.get(settings.SLACK_MIN_SEVERITY, 2)
    )


# ── Per-signal delivery ────────────────────────────────────────────────────────

def deliver(explanation: Explanation) -> dict[str, SendResult]:
    """
    Send an explanation to all configured destinations.
    Returns {destination: SendResult} for logging/metrics.
    Synchronous — called from asyncio.to_thread to avoid blocking the loop.
    """
    results = {}

    # ── Slack ──────────────────────────────────────────────────────────────────
    if settings.slack_enabled:
        if _meets_slack_threshold(explanation.severity):
            payload = format_slack(explanation)
            results["slack"] = send_slack(payload)
        else:
            logger.debug(
                "Severity %s below Slack threshold %s — skipping Slack. run_id=%s",
                explanation.severity, settings.SLACK_MIN_SEVERITY, explanation.run_id,
            )

    # ── Generic webhook ────────────────────────────────────────────────────────
    if settings.webhook_enabled:
        body, headers = build_signed_request(explanation, settings.WEBHOOK_SECRET)
        results["webhook"] = send_webhook(body, headers)

    return results


# ── Poll cycle ─────────────────────────────────────────────────────────────────

async def poll_once() -> tuple[int, int]:
    """
    One poll cycle. Returns (signals_found, signals_delivered).
    """
    rows = await fetch_unalerted_signals(limit=settings.BATCH_SIZE)
    if not rows:
        return 0, 0

    logger.info("Found %d unalerted signal(s)", len(rows))

    delivered_ids = []

    for row in rows:
        signal_id = row["id"]

        try:
            signal      = _row_to_signal(row)
            explanation = explain(signal)
        except Exception as exc:
            logger.error("Failed to build explanation for signal_id=%d: %s",
                         signal_id, exc)
            continue

        logger.info(
            "[%s] %s — run_id=%s agent_id=%s confidence=%s",
            explanation.severity,
            explanation.title,
            explanation.run_id,
            explanation.agent_id,
            explanation.confidence_pct(),
        )

        # Run synchronous HTTP in a thread so we don't block the event loop
        try:
            results = await asyncio.to_thread(deliver, explanation)
        except Exception as exc:
            logger.error("Delivery error for signal_id=%d: %s", signal_id, exc)
            continue

        # Mark as alerted only if at least one destination succeeded
        any_success = any(r.success for r in results.values()) if results else False
        no_destinations = not results  # nothing configured — still mark done

        if any_success or no_destinations:
            delivered_ids.append(signal_id)
            for dest, result in results.items():
                if not result.success:
                    logger.warning("Partial delivery failure. dest=%s signal_id=%d error=%s",
                                   dest, signal_id, result.error)
        else:
            logger.error("All destinations failed for signal_id=%d — will retry next cycle",
                         signal_id)

    if delivered_ids:
        await mark_alerted_batch(delivered_ids)
        logger.info("Marked %d signal(s) as alerted", len(delivered_ids))

    return len(rows), len(delivered_ids)


# ── Main loop ──────────────────────────────────────────────────────────────────

async def run_worker() -> None:
    await init_pool()

    enabled = []
    if settings.slack_enabled:
        enabled.append(f"Slack ({settings.SLACK_CHANNEL}, min={settings.SLACK_MIN_SEVERITY})")
    if settings.webhook_enabled:
        enabled.append(f"Webhook ({settings.WEBHOOK_URL[:40]}...)")
    if not enabled:
        logger.warning(
            "No destinations configured. "
            "Set SLACK_WEBHOOK_URL or WEBHOOK_URL to start delivering alerts."
        )
    else:
        logger.info("Alert destinations: %s", ", ".join(enabled))

    logger.info("Alert worker started. poll_interval=%ss", settings.POLL_INTERVAL)

    try:
        while True:
            try:
                found, delivered = await poll_once()
                if found:
                    logger.info("Cycle: found=%d delivered=%d", found, delivered)
            except Exception as exc:
                logger.error("Poll cycle error: %s", exc)

            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Worker cancelled — shutting down")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
