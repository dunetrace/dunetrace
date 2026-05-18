"""
Slack interactive callback handler.

Receives button-click payloads from Slack (Mark resolved / Not a problem)
and updates the DB accordingly.

Setup required:
  1. Create a Slack app at https://api.slack.com/apps
  2. Enable "Interactivity & Shortcuts" and set Request URL to
     https://<your-host>/v1/slack/callback
  3. Set SLACK_SIGNING_SECRET in .env (from App Settings > Basic Information)

In local dev, expose the API with ngrok:
  ngrok http 8002
  then set the Request URL to https://<ngrok-id>.ngrok.io/v1/slack/callback
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from urllib.parse import unquote_plus

from fastapi import APIRouter, Request, Response

from api_svc.config import settings
from api_svc.db.queries import mark_signal_resolved, record_false_positive

logger = logging.getLogger("dunetrace.api.slack")

router = APIRouter(tags=["Slack"])


def _verify_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Return True if the request genuinely came from Slack."""
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
        base    = f"v0:{timestamp}:{body.decode()}"
        digest  = hmac.new(
            settings.SLACK_SIGNING_SECRET.encode(),
            base.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(f"v0={digest}", signature)
    except Exception:
        return False


@router.post("/v1/slack/callback", include_in_schema=False)
async def slack_callback(request: Request) -> Response:
    """Receive Slack interactive button payloads."""
    body = await request.body()

    # Signature verification — skip in dev mode when secret is not set
    if settings.SLACK_SIGNING_SECRET:
        ts  = request.headers.get("X-Slack-Request-Timestamp", "")
        sig = request.headers.get("X-Slack-Signature", "")
        if not _verify_signature(body, ts, sig):
            logger.warning("Slack callback: invalid signature — rejecting")
            return Response(status_code=403)
    else:
        logger.debug("SLACK_SIGNING_SECRET not set — skipping signature verification")

    # Parse payload (Slack sends application/x-www-form-urlencoded)
    try:
        form_body = body.decode()
        if form_body.startswith("payload="):
            payload = json.loads(unquote_plus(form_body[len("payload="):]))
        else:
            payload = json.loads(form_body)
    except Exception as exc:
        logger.error("Slack callback: failed to parse payload: %s", exc)
        return Response(status_code=400)

    actions = payload.get("actions", [])
    if not actions:
        return Response(status_code=200)

    action    = actions[0]
    action_id = action.get("action_id", "")
    user      = payload.get("user", {})
    user_name = user.get("name", "unknown")

    try:
        value = json.loads(action.get("value", "{}"))
    except Exception:
        logger.error("Slack callback: bad action value: %s", action.get("value"))
        return Response(status_code=400)

    signal_id    = int(value.get("signal_id", 0))
    agent_id     = value.get("agent_id", "")
    failure_type = value.get("failure_type", "")

    if not signal_id or not agent_id or not failure_type:
        return Response(status_code=400)

    if action_id == "mark_resolved":
        updated = await mark_signal_resolved(signal_id)
        logger.info(
            "Signal %d marked resolved by %s (agent=%s type=%s updated=%s)",
            signal_id, user_name, agent_id, failure_type, updated,
        )
        return Response(
            content='{"text": ":white_check_mark: Marked as resolved."}',
            media_type="application/json",
        )

    if action_id == "false_positive":
        override = await record_false_positive(signal_id, agent_id, failure_type)
        fp_count = override.get("fp_count", 1)
        silenced = override.get("silenced", False)
        floor    = override.get("confidence_floor", 0.1)
        logger.info(
            "False positive recorded by %s — agent=%s type=%s fp_count=%d floor=%.1f silenced=%s",
            user_name, agent_id, failure_type, fp_count, floor, silenced,
        )
        if silenced:
            msg = (
                f":mute: Got it — {failure_type} on `{agent_id}` is now silenced "
                f"after {fp_count} false positives. Reset via the API when ready."
            )
        else:
            msg = (
                f":thumbsup: Noted — confidence floor for {failure_type} on `{agent_id}` "
                f"raised to {floor:.1f} ({fp_count}/3 false positives)."
            )
        return Response(
            content=json.dumps({"text": msg}),
            media_type="application/json",
        )

    logger.warning("Slack callback: unknown action_id=%s", action_id)
    return Response(status_code=200)
