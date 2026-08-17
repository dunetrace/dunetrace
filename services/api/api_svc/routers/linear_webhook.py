"""
Linear webhook receiver (Phase 4.1) — bi-directional sync: an issue Dunetrace
created via a Linear alert getting moved to Done/Canceled marks the
corresponding Dunetrace signal resolved.

Unlike Slack's callback (one shared Dunetrace-app-level signing secret,
services/api/api_svc/routers/slack.py), each org's Linear webhook has its
OWN signing secret — Linear generates one per webhook, and each org brings
their own webhook (created manually in Linear's UI, pointed at this
service's per-org URL: POST /v1/webhooks/linear/{org_id}). The org_id in the
URL path is what makes signature-secret lookup possible at all before the
payload itself has been parsed.

Setup (customer-facing, documented in docs/integrations.md):
  1. Configure POST /v1/orgs/integrations/linear with your API key +
     a webhook_secret of your choosing.
  2. In Linear: Settings > API > Webhooks > New webhook.
     URL: https://<your-dunetrace-host>/v1/webhooks/linear/<your_org_id>
     Resource types: Issues
  3. Linear shows you a signing secret when you create the webhook — this
     must be the SAME value you configured as webhook_secret in step 1
     (Dunetrace has no way to read Linear's generated secret back out of
     Linear's UI, so the customer is the one who copies it into both places).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time

from fastapi import APIRouter, Request, Response

from api_svc.crypto import decrypt_credentials_for_webhook_verification
from api_svc.db.queries import (
    get_org_linear_webhook_secret,
    get_signal_id_for_linear_issue,
    mark_signal_resolved,
)
from api_svc.linear_client import fetch_workflow_state_type

logger = logging.getLogger("dunetrace.api.linear_webhook")

router = APIRouter(tags=["Linear"])

# LINEAR_WORKFLOW_STATE_ENUM_ASSUMED — "completed"/"canceled" are the
# WorkflowState.type values a closed/canceled issue is expected to have,
# per third-party (non-Linear-owned) documentation referenced during Phase
# 4.1 discovery. Not confirmed against Linear's own docs or a live
# workspace. See BACKLOG.md's Phase 4.1 exit criteria — the smoke test
# script will either confirm this or reveal the real values.
_RESOLVED_STATE_TYPES = {"completed", "canceled"}

_MAX_TIMESTAMP_SKEW_SECS = 60  # Linear's own recommended replay-protection window


def _verify_signature(body: bytes, secret: str, signature: str) -> bool:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


# org-id-path-ok: third-party webhook, no Dunetrace API key on this request —
# org_id in the path is how the per-org webhook secret gets looked up before
# the payload can be HMAC-verified. See scripts/check_endpoint_conventions.py.
@router.post("/v1/webhooks/linear/{org_id}", include_in_schema=False)
async def linear_webhook(org_id: str, request: Request) -> Response:
    body = await request.body()

    encrypted = await get_org_linear_webhook_secret(org_id)
    if not encrypted:
        logger.warning("Linear webhook for org=%s with no Linear integration configured", org_id)
        return Response(status_code=404)

    try:
        creds = decrypt_credentials_for_webhook_verification(encrypted)
    except Exception as exc:
        logger.error("Failed to decrypt Linear credentials for org=%s: %s", org_id, exc)
        return Response(status_code=500)

    signature = request.headers.get("Linear-Signature", "")
    if not _verify_signature(body, creds["webhook_secret"], signature):
        logger.warning("Linear webhook: invalid signature — org=%s", org_id)
        return Response(status_code=403)

    try:
        # LINEAR_WEBHOOK_PAYLOAD_SHAPE_ASSUMED — envelope keys (action, type,
        # data, updatedFrom, webhookTimestamp) are confirmed from Linear's
        # docs; the exact literal JSON body for an issue state-change event
        # is not (only the envelope structure was doc-confirmed, not a full
        # example). Parsing below assumes data.id and data.stateId exist on
        # every Issue-type update event.
        payload = json.loads(body)
    except Exception as exc:
        logger.error("Linear webhook: failed to parse payload: %s", exc)
        return Response(status_code=400)

    webhook_ts = payload.get("webhookTimestamp")
    if webhook_ts and abs(time.time() - webhook_ts / 1000) > _MAX_TIMESTAMP_SKEW_SECS:
        logger.warning("Linear webhook: stale webhookTimestamp — org=%s", org_id)
        return Response(status_code=400)

    if payload.get("type") != "Issue" or payload.get("action") != "update":
        return Response(status_code=200)

    data = payload.get("data") or {}
    updated_from = payload.get("updatedFrom") or {}
    if "stateId" not in updated_from:
        # Not a state-change update (e.g. a title/description edit) — nothing to sync.
        return Response(status_code=200)

    issue_id = data.get("id")
    new_state_id = data.get("stateId")
    if not issue_id or not new_state_id:
        return Response(status_code=200)

    mapping = await get_signal_id_for_linear_issue(issue_id)
    if mapping is None:
        # Not an issue Dunetrace created — ignore.
        return Response(status_code=200)

    # Defensive: confirm the new state is actually a resolved-type state via
    # a follow-up query, rather than trusting any state-name field the
    # webhook payload itself might carry (LINEAR_WEBHOOK_PAYLOAD_SHAPE_ASSUMED
    # above) — this is the one place the design deliberately does NOT trust
    # an unverified assumption, at the cost of one extra API call.
    try:
        state_type = await fetch_workflow_state_type(creds["api_key"], new_state_id)
    except Exception as exc:
        logger.error("Linear webhook: failed to fetch workflow state type: %s", exc)
        return Response(status_code=200)  # ack anyway — Linear retries on non-2xx

    if state_type not in _RESOLVED_STATE_TYPES:
        return Response(status_code=200)

    updated = await mark_signal_resolved(mapping["org_id"], mapping["signal_id"])
    logger.info(
        "Linear issue %s moved to %s — signal %s marked resolved (updated=%s)",
        issue_id,
        state_type,
        mapping["signal_id"],
        updated,
    )
    return Response(status_code=200)
