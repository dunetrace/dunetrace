"""
HTTP delivery for Slack and generic webhooks. No external deps — stdlib urllib only.

Retries with exponential backoff. Times out at 5s connect / 10s read.
Returns a SendResult instead of raising so the caller decides what to do.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from alerts_svc.config import settings

logger = logging.getLogger("dunetrace.alerts.sender")


@dataclass
class SendResult:
    success: bool
    destination: str  # "slack" | "webhook" | "linear" | url
    attempts: int
    status_code: Optional[int] = None
    error: Optional[str] = None
    # Phase 4.1 — carries the created Linear issue id back to worker.py's
    # _deliver_one, which does the (async) linear_issue_signals write;
    # deliver()/send_linear() are both sync, so this is the only channel
    # for that id to reach the async caller.
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        if self.success:
            return f"<SendResult ok dest={self.destination} attempts={self.attempts}>"
        return (
            f"<SendResult FAILED dest={self.destination} "
            f"attempts={self.attempts} error={self.error!r}>"
        )


def _post(url: str, body: bytes, headers: dict) -> tuple[int, str]:
    """Single HTTP POST. Returns (status_code, response_body) or raises URLError/HTTPError."""
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read().decode(errors="replace")


def send_with_retry(
    url: str,
    body: bytes,
    headers: dict,
    destination: str,
    max_retries: int = None,
    retry_backoff: float = None,
) -> SendResult:
    """POST with exponential backoff. max_retries=3, backoff=2.0 → delays of 2s, 4s, 8s (~14s total before giving up)."""
    max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
    retry_backoff = retry_backoff if retry_backoff is not None else settings.RETRY_BACKOFF

    last_error = None
    last_status = None
    delay = retry_backoff

    for attempt in range(1, max_retries + 2):  # +1 for initial attempt
        try:
            status, response_body = _post(url, body, headers)

            # Slack returns 200 with body "ok" on success
            if 200 <= status < 300:
                logger.info(
                    "Alert sent. dest=%s attempt=%d status=%d",
                    destination,
                    attempt,
                    status,
                )
                return SendResult(
                    success=True,
                    destination=destination,
                    attempts=attempt,
                    status_code=status,
                )
            else:
                last_error = f"HTTP {status}: {response_body[:200]}"
                last_status = status
                logger.warning(
                    "Alert delivery failed (non-2xx). dest=%s attempt=%d status=%d body=%r",
                    destination,
                    attempt,
                    status,
                    response_body[:100],
                )

        except urllib.error.HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.reason}"
            last_status = exc.code
            logger.warning(
                "Alert HTTPError. dest=%s attempt=%d error=%s",
                destination,
                attempt,
                last_error,
            )

        except urllib.error.URLError as exc:
            last_error = f"URLError: {exc.reason}"
            logger.warning(
                "Alert URLError. dest=%s attempt=%d error=%s",
                destination,
                attempt,
                last_error,
            )

        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Alert unexpected error. dest=%s attempt=%d error=%s",
                destination,
                attempt,
                last_error,
            )

        # Don't sleep after the last attempt
        if attempt <= max_retries:
            logger.debug("Retrying in %.1fs. dest=%s", delay, destination)
            time.sleep(delay)
            delay *= 2

    logger.error(
        "Alert failed after %d attempts. dest=%s error=%s",
        max_retries + 1,
        destination,
        last_error,
    )
    return SendResult(
        success=False,
        destination=destination,
        attempts=max_retries + 1,
        status_code=last_status,
        error=last_error,
    )


# ── Destination-specific senders ───────────────────────────────────────────────


def send_slack(payload: dict, webhook_url: str | None = None) -> SendResult:
    """POST a Block Kit payload to a Slack webhook URL.

    webhook_url: Phase 4.1 — the caller's resolved destination (per-org
    config if the org has one, else the global .env fallback — see
    worker.py's _resolve_slack_destination). None (digest.py's existing
    call, not yet made per-org-aware — see BACKLOG.md) falls back to the
    global settings.SLACK_WEBHOOK_URL exactly as before this parameter
    existed.
    """
    url = webhook_url or settings.SLACK_WEBHOOK_URL
    if not url:
        logger.debug("Slack not configured — skipping")
        return SendResult(success=False, destination="slack", attempts=0, error="not_configured")

    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    return send_with_retry(
        url=url,
        body=body,
        headers=headers,
        destination="slack",
    )


def send_webhook(body: bytes, headers: dict) -> SendResult:
    """POST a signed JSON payload to the generic webhook URL."""
    if not settings.webhook_enabled:
        logger.debug("Webhook not configured — skipping")
        return SendResult(success=False, destination="webhook", attempts=0, error="not_configured")

    return send_with_retry(
        url=settings.WEBHOOK_URL,
        body=body,
        headers=headers,
        destination="webhook",
    )


def send_linear(
    api_key: str,
    team_id: str,
    project_id: str | None,
    title: str,
    description: str,
) -> SendResult:
    """Create a Linear issue for this signal. Not a retry_with_backoff HTTP
    POST like Slack/webhook — linear_client.create_issue already handles
    its own request/error handling and returns None on any failure, so
    there's nothing to retry here (a transient Linear outage just means
    this signal doesn't get an issue this cycle; the next unrelated signal
    isn't blocked waiting on retries)."""
    from alerts_svc.linear_client import create_issue

    issue_id = create_issue(api_key, team_id, title, description, project_id=project_id)
    if issue_id is None:
        return SendResult(
            success=False, destination="linear", attempts=1, error="create_issue_failed"
        )
    return SendResult(
        success=True, destination="linear", attempts=1, metadata={"linear_issue_id": issue_id}
    )
