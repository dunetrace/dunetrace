"""
Alert delivery worker. Polls for unalerted live signals, explains them,
formats payloads, and sends to Slack and/or webhook.

Each signal goes through: DB row → FailureSignal → Explanation → formatted payload → HTTP send → mark alerted.

Slack only receives signals at or above SLACK_MIN_SEVERITY. The generic webhook
gets everything and can filter on its end.

Delivery is at-least-once: if the process crashes between send and mark_alerted,
the signal will be re-sent on the next restart.
    Idempotency is the receiver's responsibility.

Run:
    cd services/alerts
    python -m alerts_svc.worker
"""
from __future__ import annotations

import asyncio
import logging
import time

from dunetrace.models import FailureSignal, FailureType, Severity

from explainer_svc.explainer import explain
from explainer_svc.models import Explanation
from alerts_svc.formatters.slack   import format_slack
from alerts_svc.formatters.webhook import build_signed_request      # type: ignore
from alerts_svc.sender  import send_slack, send_webhook, SendResult
from alerts_svc.db      import init_pool, close_pool, fetch_unalerted_signals, mark_alerted_batch
from alerts_svc.config  import settings, SEVERITY_ORDER

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.alerts")


# Signal reconstruction

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


# Severity filter

def _meets_slack_threshold(severity: str) -> bool:
    return (
        SEVERITY_ORDER.get(severity, 0)
        >= SEVERITY_ORDER.get(settings.SLACK_MIN_SEVERITY, 2)
    )


# Per-signal delivery

def deliver(explanation: Explanation) -> dict[str, SendResult]:
    """
    Send an explanation to all configured destinations.
    Returns {destination: SendResult} for logging/metrics.
    Synchronous i.e. called from asyncio.to_thread to avoid blocking the loop.
    """
    results = {}

    # Slack
    if settings.slack_enabled:
        if _meets_slack_threshold(explanation.severity):
            payload = format_slack(explanation)
            results["slack"] = send_slack(payload)
        else:
            logger.debug(
                "Severity %s below Slack threshold %s — skipping Slack. run_id=%s",
                explanation.severity, settings.SLACK_MIN_SEVERITY, explanation.run_id,
            )

    # Generic webhook
    if settings.webhook_enabled:
        body, headers = build_signed_request(explanation, settings.WEBHOOK_SECRET)
        results["webhook"] = send_webhook(body, headers)

    return results


# Poll cycle

async def poll_once() -> tuple[int, int]:
    """
    One poll cycle. Returns (signals_found, signals_delivered).
    """
    rows = await fetch_unalerted_signals(limit=settings.BATCH_SIZE)
    if not rows:
        return 0, 0

    logger.info("Found %d unalerted signal(s)", len(rows))

    # Build explanations first (fast, synchronous)
    work: list[tuple[int, Explanation]] = []
    for row in rows:
        try:
            signal      = _row_to_signal(row)
            explanation = explain(signal)
            work.append((row["id"], explanation))
        except Exception as exc:
            logger.error("Failed to build explanation for signal_id=%d: %s", row["id"], exc)

    if not work:
        return len(rows), 0

    # Deliver all signals concurrently
    async def _deliver_one(signal_id: int, explanation: Explanation) -> int | None:
        logger.info(
            "[%s] %s — run_id=%s agent_id=%s confidence=%s",
            explanation.severity, explanation.title,
            explanation.run_id, explanation.agent_id, explanation.confidence_pct(),
        )
        try:
            results = await asyncio.to_thread(deliver, explanation)
        except Exception as exc:
            logger.error("Delivery error for signal_id=%d: %s", signal_id, exc)
            return None

        any_success    = any(r.success for r in results.values()) if results else False
        no_destinations = not results

        if any_success or no_destinations:
            for dest, result in results.items():
                if not result.success:
                    logger.warning("Partial delivery failure. dest=%s signal_id=%d error=%s",
                                   dest, signal_id, result.error)
            return signal_id
        else:
            logger.error("All destinations failed for signal_id=%d — will retry next cycle",
                         signal_id)
            return None

    outcomes = await asyncio.gather(*[_deliver_one(sid, exp) for sid, exp in work])
    delivered_ids = [sid for sid in outcomes if sid is not None]

    if delivered_ids:
        await mark_alerted_batch(delivered_ids)
        logger.info("Marked %d signal(s) as alerted", len(delivered_ids))

    return len(rows), len(delivered_ids)


# Main loop

async def run_worker() -> None:
    await init_pool()

    enabled = []
    if settings.slack_enabled:
        enabled.append(f"Slack ({settings.SLACK_CHANNEL}, min={settings.SLACK_MIN_SEVERITY})")
    if settings.webhook_enabled:
        enabled.append(f"Webhook ({settings.WEBHOOK_URL[:40]}...)")
        if not settings.WEBHOOK_SECRET:
            logger.warning(
                "WEBHOOK_URL is set but WEBHOOK_SECRET is empty — "
                "payloads will be sent unsigned. Set WEBHOOK_SECRET for HMAC-SHA256 signing."
            )
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
        logger.info("Worker cancelled i.e. shutting down gracefully")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
