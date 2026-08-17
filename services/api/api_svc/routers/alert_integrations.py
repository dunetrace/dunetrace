"""Per-org Slack/Linear alert-destination config (Phase 4.1) — both
bring-your-own (a customer's own Slack incoming webhook, a customer's own
Linear API key + a webhook they create themselves pointed at this
service's /v1/webhooks/linear/{org_id}). Credentials are encrypted at rest
and never returned once stored, same pattern as Phase 2.1's
routers/integrations.py.

*Request model field names become the encrypted credentials dict's keys —
must exactly match alerts_svc's own provider client kwargs, since
alerts_svc decrypts and uses them directly (see alerts_svc/worker.py's
_resolve_slack_destination/_resolve_linear_config).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_svc.auth import require_org
from api_svc.crypto import encrypt_credentials
from api_svc.db.queries import (
    delete_org_alert_integration,
    get_org_alert_integration_status,
    upsert_org_alert_integration,
)
from api_svc.linear_client import LinearApiError, fetch_projects, fetch_teams

logger = logging.getLogger("dunetrace.api.alert_integrations")
router = APIRouter(prefix="/v1/orgs/integrations", tags=["Alert Integrations"])


class SlackIntegrationRequest(BaseModel):
    webhook_url: str
    channel: str = ""


class LinearIntegrationRequest(BaseModel):
    api_key: str
    webhook_secret: str
    team_id: str
    project_id: str = ""


class LinearPreviewTeamsRequest(BaseModel):
    api_key: str


class AlertIntegrationStatus(BaseModel):
    configured: bool
    enabled: bool | None = None
    config: dict | None = None


def _status_response(status: dict | None) -> AlertIntegrationStatus:
    if status is None:
        return AlertIntegrationStatus(configured=False)
    return AlertIntegrationStatus(
        configured=True, enabled=status["enabled"], config=status["config"]
    )


@router.post("/slack", summary="Configure a Slack alert destination for this org", status_code=201)
async def set_slack_integration(
    body: SlackIntegrationRequest,
    org_id: str = Depends(require_org),
) -> AlertIntegrationStatus:
    try:
        encrypted = encrypt_credentials({"webhook_url": body.webhook_url})
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    await upsert_org_alert_integration(org_id, "slack", encrypted, {"channel": body.channel})
    status = await get_org_alert_integration_status(org_id, "slack")
    return _status_response(status)


@router.get("/slack", summary="Get this org's Slack integration status")
async def get_slack_integration(org_id: str = Depends(require_org)) -> AlertIntegrationStatus:
    status = await get_org_alert_integration_status(org_id, "slack")
    return _status_response(status)


@router.delete("/slack", summary="Remove this org's Slack integration", status_code=204)
async def remove_slack_integration(org_id: str = Depends(require_org)):
    deleted = await delete_org_alert_integration(org_id, "slack")
    if not deleted:
        raise HTTPException(status_code=404, detail="No Slack integration configured.")


@router.post(
    "/linear/preview-teams",
    summary="List this Linear workspace's teams/projects before saving the integration",
)
async def preview_linear_teams(body: LinearPreviewTeamsRequest) -> dict:
    """Takes the API key directly in the request body — never persisted at
    this step (matches custom-detectors' preview-before-create pattern).
    Lets the dashboard build a team/project picker before the customer
    commits to saving the integration."""
    try:
        teams = await fetch_teams(body.api_key)
    except LinearApiError as exc:
        raise HTTPException(status_code=502, detail=f"Linear API error: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach Linear: {exc}")

    result = []
    for team in teams:
        try:
            projects = await fetch_projects(body.api_key, team["id"])
        except Exception as exc:
            logger.warning("Failed to fetch projects for team %s: %s", team["id"], exc)
            projects = []
        result.append({"id": team["id"], "name": team["name"], "projects": projects})
    return {"teams": result}


@router.post(
    "/linear", summary="Configure a Linear alert destination for this org", status_code=201
)
async def set_linear_integration(
    body: LinearIntegrationRequest,
    org_id: str = Depends(require_org),
) -> AlertIntegrationStatus:
    try:
        encrypted = encrypt_credentials(
            {"api_key": body.api_key, "webhook_secret": body.webhook_secret}
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    await upsert_org_alert_integration(
        org_id, "linear", encrypted, {"team_id": body.team_id, "project_id": body.project_id}
    )
    status = await get_org_alert_integration_status(org_id, "linear")
    return _status_response(status)


@router.get("/linear", summary="Get this org's Linear integration status")
async def get_linear_integration(org_id: str = Depends(require_org)) -> AlertIntegrationStatus:
    status = await get_org_alert_integration_status(org_id, "linear")
    return _status_response(status)


@router.delete("/linear", summary="Remove this org's Linear integration", status_code=204)
async def remove_linear_integration(org_id: str = Depends(require_org)):
    deleted = await delete_org_alert_integration(org_id, "linear")
    if not deleted:
        raise HTTPException(status_code=404, detail="No Linear integration configured.")
