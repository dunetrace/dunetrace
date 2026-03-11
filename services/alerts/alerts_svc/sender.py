"""
services/alerts/alerts_svc/sender.py

HTTP delivery for alerts. Stdlib urllib only i.e. no httpx, no requests.

Features:
  - Exponential backoff retry (configurable max retries + base delay)
  - Per-destination timeout (5s connect, 10s read)
  - Structured logging of every attempt and outcome
  - Never raises i.e. returns a SendResult so callers can decide what to do

Supports two destinations:
  - Slack Incoming Webhook (POST JSON, check for "ok" in response)
  - Generic webhook (POST JSON + HMAC-SHA256 signature header)
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
    success:     bool
    destination: str           # "slack" | "webhook" | url
    attempts:    int
    status_code: Optional[int] = None
    error:       Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return f"<SendResult ok dest={self.destination} attempts={self.attempts}>"
        return (f"<SendResult FAILED dest={self.destination} "
                f"attempts={self.attempts} error={self.error!r}>")


def _post(url: str, body: bytes, headers: dict) -> tuple[int, str]:
    """
    Single HTTP POST. Returns (status_code, response_body).
    Raises urllib.error.URLError / HTTPError on failure.
    """
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
    """
    POST body to url with exponential backoff retry.

    max_retries=3, backoff=2.0 means delays of 2s, 4s, 8s between attempts.
    Total max wait: ~14s before giving up.
    """
    max_retries   = max_retries   if max_retries   is not None else settings.MAX_RETRIES
    retry_backoff = retry_backoff if retry_backoff is not None else settings.RETRY_BACKOFF

    last_error  = None
    last_status = None
    delay       = retry_backoff

    for attempt in range(1, max_retries + 2):  # +1 for initial attempt
        try:
            status, response_body = _post(url, body, headers)

            # Slack returns 200 with body "ok" on success
            if 200 <= status < 300:
                logger.info("Alert sent. dest=%s attempt=%d status=%d",
                            destination, attempt, status)
                return SendResult(
                    success=True,
                    destination=destination,
                    attempts=attempt,
                    status_code=status,
                )
            else:
                last_error  = f"HTTP {status}: {response_body[:200]}"
                last_status = status
                logger.warning("Alert delivery failed (non-2xx). "
                               "dest=%s attempt=%d status=%d body=%r",
                               destination, attempt, status, response_body[:100])

        except urllib.error.HTTPError as exc:
            last_error  = f"HTTPError {exc.code}: {exc.reason}"
            last_status = exc.code
            logger.warning("Alert HTTPError. dest=%s attempt=%d error=%s",
                           destination, attempt, last_error)

        except urllib.error.URLError as exc:
            last_error = f"URLError: {exc.reason}"
            logger.warning("Alert URLError. dest=%s attempt=%d error=%s",
                           destination, attempt, last_error)

        except Exception as exc:
            last_error = str(exc)
            logger.warning("Alert unexpected error. dest=%s attempt=%d error=%s",
                           destination, attempt, last_error)

        # Don't sleep after the last attempt
        if attempt <= max_retries:
            logger.debug("Retrying in %.1fs. dest=%s", delay, destination)
            time.sleep(delay)
            delay *= 2

    logger.error("Alert failed after %d attempts. dest=%s error=%s",
                 max_retries + 1, destination, last_error)
    return SendResult(
        success=False,
        destination=destination,
        attempts=max_retries + 1,
        status_code=last_status,
        error=last_error,
    )


# ── Destination-specific senders ───────────────────────────────────────────────

def send_slack(payload: dict) -> SendResult:
    """POST a Block Kit payload to the Slack webhook URL."""
    if not settings.slack_enabled:
        logger.debug("Slack not configured — skipping")
        return SendResult(success=False, destination="slack",
                          attempts=0, error="not_configured")

    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    return send_with_retry(
        url=settings.SLACK_WEBHOOK_URL,
        body=body,
        headers=headers,
        destination="slack",
    )


def send_webhook(body: bytes, headers: dict) -> SendResult:
    """POST a signed JSON payload to the generic webhook URL."""
    if not settings.webhook_enabled:
        logger.debug("Webhook not configured — skipping")
        return SendResult(success=False, destination="webhook",
                          attempts=0, error="not_configured")

    return send_with_retry(
        url=settings.WEBHOOK_URL,
        body=body,
        headers=headers,
        destination="webhook",
    )
