"""
Alert delivery worker. Polls for unalerted live signals, explains them,
formats payloads, and sends to Slack and/or webhook.

Each signal goes through: DB row → FailureSignal → Explanation → formatted payload → HTTP send → mark alerted.

Slack only receives signals at or above SLACK_MIN_SEVERITY. The generic webhook
gets everything and can filter on its end.

Delivery is at-least-once: if the process crashes between send and mark_alerted,
the signal will be re-sent on the next restart.
    Idempotency is the receiver's responsibility.

Horizontally scalable by agent_id, the same way detector_svc is: set SHARD_COUNT=N
and run N replicas with SHARD_INDEX=0..N-1. Signals are also *claimed* before
delivery (failure_signals.alert_claimed_at), which is what makes a second replica
safe even when it is misconfigured onto the same shard — without the claim, two
workers both see alerted = FALSE and both deliver the same alert, because
`alerted` is only set after a successful send. See db.py::claim_unalerted_signals.

Run:
    cd services/alerts
    python -m alerts_svc.worker
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time

from dunetrace.models import FailureSignal, FailureType, Severity

from explainer_svc.explainer import coerce_failure_type, explain
from explainer_svc.models import Explanation
from alerts_svc.formatters.slack import format_slack
from alerts_svc.formatters.webhook import build_signed_request, sign_payload  # type: ignore
from alerts_svc.formatters.linear import format_linear_issue
from alerts_svc.formatters.approval import format_slack_approval, format_webhook_approval
from alerts_svc.sender import send_slack, send_webhook, send_linear, SendResult
from alerts_svc.crypto import decrypt_credentials
from alerts_svc.db import (
    init_pool,
    close_pool,
    claim_unalerted_signals,
    release_claims,
    ensure_alert_claim_columns,
    mark_alerted_batch,
    fetch_signal_rate_context,
    fetch_run_tokens,
    ensure_digest_schema,
    ensure_dedup_schema,
    ensure_semantic_signal_column,
    ensure_alert_integrations_schema,
    fetch_dedup_states,
    record_alert_sent,
    increment_suppressed_count,
    evaluate_alert_policy,
    fetch_agent_overrides,
    fetch_org_alert_integration,
    record_linear_issue_mapping,
    fetch_undelivered_approvals,
    mark_approval_delivered,
)
from alerts_svc.config import (
    load_alert_policies,
    get_alert_policy,
    load_semantic_confidence_floors,
    get_semantic_confidence_floor,
    load_detector_destinations,
    get_detector_destinations,
)
from explainer_svc.cost import estimate_cost
from alerts_svc.digest import send_weekly_digest
from alerts_svc.config import settings, SEVERITY_ORDER

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.alerts")

# Identifies this process in failure_signals.alert_claimed_by. Diagnostic only —
# correctness comes from the claim being set atomically, not from the value —
# but it's what tells you *which* replica is sitting on a stuck claim.
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:shard{settings.SHARD_INDEX}"


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

    # Custom detector, pack, and semantic signals store failure types outside the
    # built-in enum; the CUSTOM sentinel plus evidence["raw_failure_type"] is what
    # makes the explainer's fallback template render them correctly.
    failure_type, evidence = coerce_failure_type(row["failure_type"], evidence)

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
    org_id: str | None = None,
    destinations: list[str] | None = None,
    slack_webhook_url: str | None = None,
    linear_config: dict | None = None,
) -> dict[str, SendResult]:
    """Send an explanation to configured destinations. Returns {destination: SendResult}.
    Synchronous — called via asyncio.to_thread to avoid blocking the event loop.

    destinations: per-detector routing override (Phase 4.1, from detectors.yml)
    — when given, only destinations in this list are attempted, even if
    others are globally enabled. None means no override — behave exactly as
    before 4.1 (send to every globally-enabled destination).

    slack_webhook_url: Phase 4.1 — the caller's already-resolved per-org (or
    global-fallback) destination; see worker.py's _resolve_slack_destination.
    None here just means "no override," same as before this param existed —
    send_slack() itself still falls back to the global settings value.

    linear_config: Phase 4.1 — {api_key, webhook_secret, team_id, project_id},
    already decrypted by the caller (_resolve_linear_config). None means
    this org has no Linear integration configured — Linear is skipped
    entirely, not attempted with empty credentials.
    """
    results = {}

    # Slack
    slack_configured = bool(slack_webhook_url) or settings.slack_enabled
    if slack_configured and (destinations is None or "slack" in destinations):
        if _meets_slack_threshold(explanation.severity):
            payload = format_slack(
                explanation,
                suppressed_count=suppressed_count,
                dedup_window=settings.ALERT_DEDUP_WINDOW,
                signal_id=signal_id,
                org_id=org_id,
            )
            results["slack"] = send_slack(payload, webhook_url=slack_webhook_url)
        else:
            logger.debug(
                "Severity %s below Slack threshold %s — skipping Slack. run_id=%s",
                explanation.severity,
                settings.SLACK_MIN_SEVERITY,
                explanation.run_id,
            )

    # Generic webhook
    if settings.webhook_enabled and (destinations is None or "webhook" in destinations):
        body, headers = build_signed_request(explanation, settings.WEBHOOK_SECRET)
        results["webhook"] = send_webhook(body, headers)

    # Linear (Phase 4.1) — org-configured only, no global fallback (Linear
    # has no self-hosted single-tenant precedent to preserve the way Slack's
    # SLACK_WEBHOOK_URL does).
    if linear_config and (destinations is None or "linear" in destinations):
        title, description = format_linear_issue(explanation)
        results["linear"] = send_linear(
            api_key=linear_config["api_key"],
            team_id=linear_config["team_id"],
            project_id=linear_config.get("project_id") or None,
            title=title,
            description=description,
        )

    return results


# ── Per-org destination resolution (Phase 4.1) ────────────────────────────────


async def _resolve_slack_destination(org_id: str) -> str | None:
    """Per-org Slack webhook URL if configured, else None (deliver() itself
    falls back to the global .env SLACK_WEBHOOK_URL when this is None —
    preserves today's self-hosted single-tenant behavior for any org that
    hasn't configured its own destination)."""
    integration = await fetch_org_alert_integration(org_id, "slack")
    if integration is None:
        return None
    try:
        creds = decrypt_credentials(integration["encrypted_credentials"])
    except Exception as exc:
        logger.error("Failed to decrypt Slack credentials for org=%s: %s", org_id, exc)
        return None
    return creds.get("webhook_url")


async def _resolve_linear_config(org_id: str) -> dict | None:
    """Per-org Linear config (decrypted api_key/webhook_secret + team_id/
    project_id from config_json), or None if not configured. No global
    fallback — Linear has no self-hosted single-tenant precedent to
    preserve the way Slack's SLACK_WEBHOOK_URL does."""
    integration = await fetch_org_alert_integration(org_id, "linear")
    if integration is None:
        return None
    try:
        creds = decrypt_credentials(integration["encrypted_credentials"])
    except Exception as exc:
        logger.error("Failed to decrypt Linear credentials for org=%s: %s", org_id, exc)
        return None
    return {
        "api_key": creds.get("api_key"),
        "webhook_secret": creds.get("webhook_secret"),
        "team_id": integration["config"].get("team_id"),
        "project_id": integration["config"].get("project_id"),
    }


# Poll cycle


async def deliver_pending_approvals() -> int:
    """Notify a human about each pending, not-yet-delivered approval
    (Capability 2, Phase 2.3). Runs each poll cycle alongside signal alerts.
    Delivers over Slack (interactive Approve/Deny buttons) and/or the generic
    webhook, whichever the org has configured. Marks each approval delivered
    afterward so it's notified exactly once — including the no-channel case,
    where marking it delivered stops a hot re-attempt loop and the SDK's own
    fail-closed timeout takes over. Returns how many were delivered/handled."""
    approvals = await fetch_undelivered_approvals()
    if not approvals:
        return 0

    handled = 0
    for approval in approvals:
        org_id = approval["org_id"]
        slack_dest = await _resolve_slack_destination(org_id)
        slack_available = bool(slack_dest) or settings.slack_enabled
        webhook_available = settings.webhook_enabled

        if not slack_available and not webhook_available:
            logger.warning(
                "Approval %s for org=%s has no delivery channel configured — "
                "marking delivered; the SDK will fall back to its fail-closed timeout.",
                approval["id"],
                org_id,
            )
            await mark_approval_delivered(approval["id"])
            handled += 1
            continue

        # asyncio.to_thread, matching _deliver_one on the signal path below:
        # send_slack/send_webhook are BLOCKING urllib calls. Awaited inline they
        # stall the whole event loop, and this is the one delivery path with a
        # live agent waiting synchronously on the other end — a degraded Slack
        # endpoint would hold up every other pending approval behind it, turning
        # one slow destination into a fleet-wide agent stall.
        if slack_available:
            try:
                await asyncio.to_thread(
                    send_slack, format_slack_approval(approval), webhook_url=slack_dest
                )
            except Exception as exc:
                logger.error("Approval %s Slack delivery failed: %s", approval["id"], exc)

        if webhook_available:
            try:
                payload = format_webhook_approval(approval)
                body = json.dumps(payload, separators=(",", ":")).encode()
                headers = {
                    "Content-Type": "application/json",
                    "X-Dunetrace-Event": "approval_request",
                    "X-Dunetrace-Version": "1.0",
                }
                sig = sign_payload(body, settings.WEBHOOK_SECRET)
                if sig:
                    headers["X-Dunetrace-Signature"] = sig
                await asyncio.to_thread(send_webhook, body, headers)
            except Exception as exc:
                logger.error("Approval %s webhook delivery failed: %s", approval["id"], exc)

        await mark_approval_delivered(approval["id"])
        handled += 1

    return handled


async def poll_once() -> tuple[int, int]:
    """One poll cycle. Returns (signals_found, signals_delivered)."""
    rows = await claim_unalerted_signals(
        limit=settings.BATCH_SIZE,
        shard_count=settings.SHARD_COUNT,
        shard_index=settings.SHARD_INDEX,
        worker_id=WORKER_ID,
        claim_timeout_secs=settings.CLAIM_TIMEOUT_SECS,
    )
    if not rows:
        return 0, 0

    # Everything this poll took ownership of. Any of these not marked alerted by
    # the end must have its claim handed back, or it waits out CLAIM_TIMEOUT_SECS
    # before anyone retries it.
    claimed_ids = [r["id"] for r in rows]

    logger.info("Claimed %d unalerted signal(s)", len(rows))

    # ── Group + policy + dedup ────────────────────────────────────────────────
    # Flow per (agent_id, failure_type) group:
    #   1. Alert policy check  — is this pattern confirmed across enough runs?
    #   2. Dedup window check  — have we already alerted about this recently?
    #   3. Deliver one alert   — highest-confidence signal in the batch

    from collections import defaultdict
    import time

    # Grouped by (org_id, agent_id, failure_type) — agent_id alone is not unique
    # across orgs, so a 2-key group could merge two different orgs' signals and
    # silently drop one org's alert as a "duplicate" of another org's.
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["org_id"], row["agent_id"], row["failure_type"])].append(row)

    # Load alert policies from detectors.yml (fast — file read, no DB)
    alert_policies = load_alert_policies()
    semantic_confidence_floors = load_semantic_confidence_floors()
    detector_destinations = load_detector_destinations()

    # Batch-fetch false-positive overrides for all groups
    agent_overrides = await fetch_agent_overrides(list(groups.keys()))

    dedup_window = settings.ALERT_DEDUP_WINDOW
    dedup_states: dict[tuple[str, str, str], dict] = {}
    if dedup_window > 0:
        dedup_states = await fetch_dedup_states(list(groups.keys()))

    now = time.time()

    # to_deliver: (row, suppressed_since_last_alert)
    to_deliver: list[tuple[dict, int]] = []
    # IDs to mark alerted without sending (policy pending or dedup suppressed)
    silent_ids: list[int] = []
    duplicate_ids: list[int] = []
    suppressed_groups: list[tuple[str, str, str, int]] = []  # for dedup count increment

    for (org_id, agent_id, failure_type), group_rows in groups.items():
        best = max(group_rows, key=lambda r: r["confidence"])
        rest_ids = [r["id"] for r in group_rows if r["id"] != best["id"]]

        # ── Step 1: Alert policy ──────────────────────────────────────────────
        policy = get_alert_policy(alert_policies, failure_type)
        policy_met, policy_reason = await evaluate_alert_policy(
            org_id,
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
        override = agent_overrides.get((org_id, agent_id, failure_type))
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

            snoozed_until = override.get("snoozed_until")
            if snoozed_until and snoozed_until.timestamp() > now:
                logger.info(
                    "Snoozed until %s — %s on %s",
                    snoozed_until.isoformat(),
                    failure_type,
                    agent_id,
                )
                silent_ids.extend(r["id"] for r in group_rows)
                continue

        # ── Step 2.5: Semantic evaluator confidence floor ─────────────────────
        # Structural signals (source="structural") are unaffected — this only
        # gates semantic ones, per-evaluator, via docs/config/semantic-evaluators.yml.
        # Below-floor signals are still stored/visible in the dashboard, just
        # never alerted — distinct from the FP-override mechanism above, which
        # only kicks in after real human feedback accumulates.
        if best.get("source") == "semantic":
            floor = get_semantic_confidence_floor(semantic_confidence_floors, failure_type)
            if best["confidence"] < floor:
                logger.info(
                    "Semantic confidence %.2f below floor %.1f — %s on %s",
                    best["confidence"],
                    floor,
                    failure_type,
                    agent_id,
                )
                silent_ids.extend(r["id"] for r in group_rows)
                continue

        # ── Step 3: Dedup window ──────────────────────────────────────────────
        state = dedup_states.get((org_id, agent_id, failure_type))
        within_window = (
            dedup_window > 0
            and state is not None
            and (now - state["last_alerted_at"].timestamp()) < dedup_window
        )

        if within_window:
            suppressed_groups.append((org_id, agent_id, failure_type, len(group_rows)))
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
    for org_id, agent_id, failure_type, count in suppressed_groups:
        await increment_suppressed_count(org_id, agent_id, failure_type, count)

    # Mark policy-pending and dedup-suppressed signals as processed
    mark_now = silent_ids + duplicate_ids
    if mark_now:
        await mark_alerted_batch(mark_now)

    if not to_deliver:
        if silent_ids:
            policy_cnt = len(silent_ids) - sum(c for _, _, _, c in suppressed_groups)
            dedup_cnt = sum(c for _, _, _, c in suppressed_groups)
            logger.info(
                "No alerts to send: %d policy-pending, %d dedup-suppressed",
                policy_cnt,
                dedup_cnt,
            )
        await release_claims(claimed_ids)
        return len(rows), 0

    # ── Build explanations ────────────────────────────────────────────────────
    signals_by_row: list[tuple[dict, object]] = []
    for row, _ in to_deliver:
        try:
            signals_by_row.append((row, _row_to_signal(row)))
        except Exception as exc:
            logger.error("Failed to reconstruct signal for signal_id=%d: %s", row["id"], exc)

    # TODO: batch this into a single query — currently one DB round-trip per signal
    rate_contexts = await asyncio.gather(
        *[
            fetch_signal_rate_context(row["org_id"], row["agent_id"], row["failure_type"])
            for row, _ in signals_by_row
        ]
    )

    run_token_map = await fetch_run_tokens([row["run_id"] for row, _ in signals_by_row])
    org_by_signal_id = {row["id"]: row["org_id"] for row in rows}

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
        await release_claims(claimed_ids)
        return len(rows), 0

    # ── Deliver concurrently ───────────────────────────────────────────────────
    async def _deliver_one(
        signal_id: int, explanation: Explanation, suppressed_count: int
    ) -> int | None:
        org_id = org_by_signal_id.get(signal_id)
        slack_webhook_url = await _resolve_slack_destination(org_id) if org_id else None
        linear_config = await _resolve_linear_config(org_id) if org_id else None
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
        destinations = get_detector_destinations(detector_destinations, explanation.failure_type)
        try:
            results = await asyncio.to_thread(
                deliver,
                explanation,
                suppressed_count,
                signal_id,
                org_id,
                destinations,
                slack_webhook_url,
                linear_config,
            )
        except Exception as exc:
            logger.error("Delivery error for signal_id=%d: %s", signal_id, exc)
            return None

        linear_result = results.get("linear")
        if linear_result and linear_result.success and linear_result.metadata:
            await record_linear_issue_mapping(
                org_id, signal_id, linear_result.metadata["linear_issue_id"]
            )

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
                await record_alert_sent(org_by_signal_id[sid], exp.agent_id, exp.failure_type)
        logger.info("Marked %d signal(s) as alerted", len(delivered_ids))

    total_suppressed = sum(c for _, _, _, c in suppressed_groups)
    if total_suppressed:
        logger.info("Suppressed %d signal(s) within dedup window", total_suppressed)

    # Hand back claims on anything that didn't get delivered — rows dropped while
    # reconstructing the signal or building the explanation, and destinations that
    # failed outright. release_claims is guarded on alerted = FALSE, so the rows
    # just marked above are untouched. Without this they'd sit claimed until the
    # timeout expired instead of being retried on the next poll, which is the
    # behaviour this path had before claiming existed.
    await release_claims(claimed_ids)

    return len(rows), len(delivered_ids)


# Main loop


async def run_worker() -> None:
    await init_pool()
    await ensure_digest_schema()
    await ensure_dedup_schema()
    await ensure_semantic_signal_column()
    await ensure_alert_integrations_schema()
    await ensure_alert_claim_columns()

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

    detector_destinations = load_detector_destinations()
    if detector_destinations:
        logger.info(
            "Detector destination routing loaded: %d detectors with a custom destination list "
            "(from detectors.yml)",
            len(detector_destinations),
        )

    if settings.ALERT_DEDUP_WINDOW > 0:
        logger.info(
            "Alert dedup enabled: silence window=%ds per (agent, failure_type)",
            settings.ALERT_DEDUP_WINDOW,
        )
    else:
        logger.info("Alert dedup disabled (ALERT_DEDUP_WINDOW=0)")
    if settings.SHARD_COUNT > 1:
        # Sharding splits signals across replicas by agent_id, so a bucket with no
        # worker is a bucket whose alerts are never delivered — silently, since
        # nothing is failing. Worth a startup line because the most likely cause
        # is a SHARD_COUNT meant for the detector leaking in via the shared .env.
        logger.warning(
            "SHARD_COUNT=%d — this worker only alerts on agents hashing to bucket "
            "%d. All %d indices (0..%d) must be running or signals in the missing "
            "buckets will never be delivered. Set ALERTS_SHARD_COUNT to scale the "
            "alerts worker independently of the detector.",
            settings.SHARD_COUNT,
            settings.SHARD_INDEX,
            settings.SHARD_COUNT,
            settings.SHARD_COUNT - 1,
        )
    logger.info(
        "Alert worker started. poll_interval=%ss worker_id=%s shard=%d/%d claim_timeout=%ss",
        settings.POLL_INTERVAL,
        WORKER_ID,
        settings.SHARD_INDEX,
        settings.SHARD_COUNT,
        settings.CLAIM_TIMEOUT_SECS,
    )

    try:
        while True:
            try:
                found, delivered = await poll_once()
                if found:
                    logger.info("Cycle: found=%d delivered=%d", found, delivered)
            except Exception:
                logger.exception("Poll cycle error")

            try:
                approvals_delivered = await deliver_pending_approvals()
                if approvals_delivered:
                    logger.info("Approvals delivered: %d", approvals_delivered)
            except Exception:
                logger.exception("Approval delivery cycle error")

            try:
                sent = await send_weekly_digest()
                if sent:
                    logger.info("Weekly digests sent: %d org(s)", sent)
            except Exception:
                logger.exception("Digest cycle error")

            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Worker cancelled — shutting down gracefully")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
