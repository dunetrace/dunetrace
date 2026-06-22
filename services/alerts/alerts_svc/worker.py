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
import json
import logging
import time

from dunetrace.models import FailureSignal, FailureType, Severity

from explainer_svc.explainer import explain
from explainer_svc.models import Explanation
from alerts_svc.formatters.slack import format_slack
from alerts_svc.formatters.webhook import build_signed_request  # type: ignore
from alerts_svc.sender import send_slack, send_webhook, SendResult
from alerts_svc.db import (
    init_pool,
    close_pool,
    fetch_unalerted_signals,
    mark_alerted_batch,
    fetch_signal_rate_context,
    fetch_run_tokens,
    ensure_digest_schema,
    ensure_dedup_schema,
    fetch_dedup_states,
    record_alert_sent,
    increment_suppressed_count,
    evaluate_alert_policy,
    fetch_agent_overrides,
)
from alerts_svc.config import load_alert_policies, get_alert_policy
from explainer_svc.cost import estimate_cost
from alerts_svc.digest import send_weekly_digest
from alerts_svc.config import settings, SEVERITY_ORDER

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

    evidence = (
        json.loads(row["evidence"])
        if isinstance(row.get("evidence"), str)
        else (row.get("evidence") or {})
    )

    raw_ft = row["failure_type"]
    try:
        failure_type = FailureType(raw_ft)
    except ValueError:
        # Custom detector signal — raw type not in the built-in FailureType enum.
        # Use the CUSTOM sentinel so the explainer's fallback template fires.
        # Preserve the raw name in evidence so the alert displays it correctly.
        failure_type = FailureType.CUSTOM
        evidence = dict(evidence)
        evidence.setdefault("raw_failure_type", raw_ft)

    return FailureSignal(
        failure_type=failure_type,
        severity=Severity(row["severity"]),
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        agent_version=row["agent_version"],
        step_index=row["step_index"],
        confidence=row["confidence"],
        evidence=evidence,
        detected_at=detected_at,
    )


# Severity filter


def _meets_slack_threshold(severity: str) -> bool:
    return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(settings.SLACK_MIN_SEVERITY, 2)


# Per-signal delivery


def deliver(
    explanation: Explanation,
    suppressed_count: int = 0,
    signal_id: int | None = None,
) -> dict[str, SendResult]:
    """Send an explanation to all configured destinations. Returns {destination: SendResult}. Synchronous — called via asyncio.to_thread to avoid blocking the event loop."""
    results = {}

    # Slack
    if settings.slack_enabled:
        if _meets_slack_threshold(explanation.severity):
            payload = format_slack(
                explanation,
                suppressed_count=suppressed_count,
                dedup_window=settings.ALERT_DEDUP_WINDOW,
                signal_id=signal_id,
            )
            results["slack"] = send_slack(payload)
        else:
            logger.debug(
                "Severity %s below Slack threshold %s — skipping Slack. run_id=%s",
                explanation.severity,
                settings.SLACK_MIN_SEVERITY,
                explanation.run_id,
            )

    # Generic webhook
    if settings.webhook_enabled:
        body, headers = build_signed_request(explanation, settings.WEBHOOK_SECRET)
        results["webhook"] = send_webhook(body, headers)

    return results


# Poll cycle


async def poll_once() -> tuple[int, int]:
    """One poll cycle. Returns (signals_found, signals_delivered)."""
    rows = await fetch_unalerted_signals(limit=settings.BATCH_SIZE)
    if not rows:
        return 0, 0

    logger.info("Found %d unalerted signal(s)", len(rows))

    # ── Group + policy + dedup ────────────────────────────────────────────────
    # Flow per (agent_id, failure_type) group:
    #   1. Alert policy check  — is this pattern confirmed across enough runs?
    #   2. Dedup window check  — have we already alerted about this recently?
    #   3. Deliver one alert   — highest-confidence signal in the batch

    from collections import defaultdict
    import time

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["agent_id"], row["failure_type"])].append(row)

    # Load alert policies from detectors.yml (fast — file read, no DB)
    alert_policies = load_alert_policies()

    # Batch-fetch false-positive overrides for all groups
    agent_overrides = await fetch_agent_overrides(list(groups.keys()))

    dedup_window = settings.ALERT_DEDUP_WINDOW
    dedup_states: dict[tuple[str, str], dict] = {}
    if dedup_window > 0:
        dedup_states = await fetch_dedup_states(list(groups.keys()))

    now = time.time()

    # to_deliver: (row, suppressed_since_last_alert)
    to_deliver: list[tuple[dict, int]] = []
    # IDs to mark alerted without sending (policy pending or dedup suppressed)
    silent_ids: list[int] = []
    duplicate_ids: list[int] = []
    suppressed_groups: list[tuple[str, str, int]] = []  # for dedup count increment

    for (agent_id, failure_type), group_rows in groups.items():
        best = max(group_rows, key=lambda r: r["confidence"])
        rest_ids = [r["id"] for r in group_rows if r["id"] != best["id"]]

        # ── Step 1: Alert policy ──────────────────────────────────────────────
        policy = get_alert_policy(alert_policies, failure_type)
        policy_met, policy_reason = await evaluate_alert_policy(
            agent_id,
            failure_type,
            mode=policy["mode"],
            threshold=policy["threshold"],
            window_runs=policy["window_runs"],
        )

        if not policy_met:
            logger.info(
                "Policy pending — %s on %s: %s (mode=%s)",
                failure_type,
                agent_id,
                policy_reason,
                policy["mode"],
            )
            silent_ids.extend(r["id"] for r in group_rows)
            continue

        # ── Step 2: False-positive confidence floor ───────────────────────────
        override = agent_overrides.get((agent_id, failure_type))
        if override:
            if override["silenced"]:
                logger.info(
                    "Silenced by false positives — %s on %s (%d FPs, manually reset to re-enable)",
                    failure_type,
                    agent_id,
                    override["fp_count"],
                )
                silent_ids.extend(r["id"] for r in group_rows)
                continue
            floor = override["confidence_floor"]
            if best["confidence"] <= floor:
                logger.info(
                    "Confidence %.2f below floor %.1f — %s on %s (%d FPs)",
                    best["confidence"],
                    floor,
                    failure_type,
                    agent_id,
                    override["fp_count"],
                )
                silent_ids.extend(r["id"] for r in group_rows)
                continue

        # ── Step 3: Dedup window ──────────────────────────────────────────────
        state = dedup_states.get((agent_id, failure_type))
        within_window = (
            dedup_window > 0
            and state is not None
            and (now - state["last_alerted_at"].timestamp()) < dedup_window
        )

        if within_window:
            suppressed_groups.append((agent_id, failure_type, len(group_rows)))
            silent_ids.extend(r["id"] for r in group_rows)
            logger.debug(
                "Dedup suppressed %d signal(s): agent=%s type=%s (%.0fs remaining)",
                len(group_rows),
                agent_id,
                failure_type,
                dedup_window - (now - state["last_alerted_at"].timestamp()),
            )
            continue

        # ── Step 3: Queue for delivery ────────────────────────────────────────
        suppressed_since = int(state["suppressed_count"]) if state else 0
        to_deliver.append((best, suppressed_since))
        duplicate_ids.extend(rest_ids)

    # Persist dedup counts before any network calls so they survive a crash
    for agent_id, failure_type, count in suppressed_groups:
        await increment_suppressed_count(agent_id, failure_type, count)

    # Mark policy-pending and dedup-suppressed signals as processed
    mark_now = silent_ids + duplicate_ids
    if mark_now:
        await mark_alerted_batch(mark_now)

    if not to_deliver:
        if silent_ids:
            policy_cnt = len(silent_ids) - sum(c for _, _, c in suppressed_groups)
            dedup_cnt = sum(c for _, _, c in suppressed_groups)
            logger.info(
                "No alerts to send: %d policy-pending, %d dedup-suppressed",
                policy_cnt,
                dedup_cnt,
            )
        return len(rows), 0

    # ── Build explanations ────────────────────────────────────────────────────
    signals_by_row: list[tuple[dict, object]] = []
    for row, _ in to_deliver:
        try:
            signals_by_row.append((row, _row_to_signal(row)))
        except Exception as exc:
            logger.error("Failed to reconstruct signal for signal_id=%d: %s", row["id"], exc)

    rate_contexts = await asyncio.gather(
        *[
            fetch_signal_rate_context(row["agent_id"], row["failure_type"])
            for row, _ in signals_by_row
        ]
    )

    run_token_map = await fetch_run_tokens([row["run_id"] for row, _ in signals_by_row])

    # work: (signal_id, explanation, suppressed_count)
    work: list[tuple[int, Explanation, int]] = []
    deliver_idx = {row["id"]: suppressed for row, suppressed in to_deliver}

    for (row, signal), rate_ctx in zip(signals_by_row, rate_contexts):
        try:
            explanation = explain(signal, rate_context=rate_ctx)
            tk = run_token_map.get(row["run_id"], {})
            if tk:
                pt = int(tk.get("prompt_tokens") or 0)
                ct = int(tk.get("completion_tokens") or 0)
                explanation.total_tokens = pt + ct or None
                explanation.cost_usd = estimate_cost(tk.get("model") or "", pt, ct) or None
            work.append((row["id"], explanation, deliver_idx.get(row["id"], 0)))
        except Exception as exc:
            logger.error("Failed to build explanation for signal_id=%d: %s", row["id"], exc)

    if not work:
        return len(rows), 0

    # ── Deliver concurrently ───────────────────────────────────────────────────
    async def _deliver_one(
        signal_id: int, explanation: Explanation, suppressed_count: int
    ) -> int | None:
        if suppressed_count > 0:
            logger.info(
                "[%s] %s — run_id=%s agent_id=%s confidence=%s (+%d suppressed)",
                explanation.severity,
                explanation.title,
                explanation.run_id,
                explanation.agent_id,
                explanation.confidence_pct(),
                suppressed_count,
            )
        else:
            logger.info(
                "[%s] %s — run_id=%s agent_id=%s confidence=%s",
                explanation.severity,
                explanation.title,
                explanation.run_id,
                explanation.agent_id,
                explanation.confidence_pct(),
            )
        try:
            results = await asyncio.to_thread(deliver, explanation, suppressed_count, signal_id)
        except Exception as exc:
            logger.error("Delivery error for signal_id=%d: %s", signal_id, exc)
            return None

        any_success = any(r.success for r in results.values()) if results else False
        no_destinations = not results

        if any_success or no_destinations:
            for dest, result in results.items():
                if not result.success:
                    logger.warning(
                        "Partial delivery failure. dest=%s signal_id=%d error=%s",
                        dest,
                        signal_id,
                        result.error,
                    )
            return signal_id
        else:
            logger.error(
                "All destinations failed for signal_id=%d — will retry next cycle",
                signal_id,
            )
            return None

    outcomes = await asyncio.gather(*[_deliver_one(sid, exp, sup) for sid, exp, sup in work])
    delivered_ids = [sid for sid in outcomes if sid is not None]

    if delivered_ids:
        await mark_alerted_batch(delivered_ids)
        # Update dedup records for successful deliveries
        for sid, exp, _ in work:
            if sid in delivered_ids:
                await record_alert_sent(exp.agent_id, exp.failure_type)
        logger.info("Marked %d signal(s) as alerted", len(delivered_ids))

    total_suppressed = sum(c for _, _, c in suppressed_groups)
    if total_suppressed:
        logger.info("Suppressed %d signal(s) within dedup window", total_suppressed)

    return len(rows), len(delivered_ids)


# Main loop


async def run_worker() -> None:
    await init_pool()
    await ensure_digest_schema()
    await ensure_dedup_schema()

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

    alert_policies = load_alert_policies()
    if alert_policies:
        logger.info(
            "Alert policies loaded: %d detectors with custom policy (from detectors.yml)",
            len(alert_policies),
        )
    else:
        logger.info("No alert policies found in detectors.yml — using immediate for all detectors")

    if settings.ALERT_DEDUP_WINDOW > 0:
        logger.info(
            "Alert dedup enabled: silence window=%ds per (agent, failure_type)",
            settings.ALERT_DEDUP_WINDOW,
        )
    else:
        logger.info("Alert dedup disabled (ALERT_DEDUP_WINDOW=0)")
    logger.info("Alert worker started. poll_interval=%ss", settings.POLL_INTERVAL)

    try:
        while True:
            try:
                found, delivered = await poll_once()
                if found:
                    logger.info("Cycle: found=%d delivered=%d", found, delivered)
            except Exception as exc:
                logger.error("Poll cycle error: %s", exc)

            try:
                await send_weekly_digest()
            except Exception as exc:
                logger.error("Digest cycle error: %s", exc)

            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Worker cancelled — shutting down gracefully")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
