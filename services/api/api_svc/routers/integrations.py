"""External evaluation integration config (Phase 2.1: Langfuse; Phase 2.2:
LangSmith; Phase 2.3: Braintrust — all under the same
/v1/orgs/integrations/{provider} shape). Credentials are encrypted at rest
and never returned once stored — GET/DELETE only ever expose configuration/
health status.

Each provider's *Request model's field names become the encrypted
credentials dict's keys — they must exactly match that provider's class
constructor kwargs in integrations_svc/providers/, since
integrations_svc.worker._poll_one calls provider_cls(endpoint_url, **creds)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api_svc.auth import require_org
from api_svc.crypto import encrypt_credentials
from api_svc.db.queries import (
    delete_external_integration,
    get_external_integration_status,
    upsert_external_integration,
)

logger = logging.getLogger("dunetrace.api.integrations")
router = APIRouter(prefix="/v1/orgs/integrations", tags=["Integrations"])


class LangfuseIntegrationRequest(BaseModel):
    endpoint_url: str
    public_key: str
    secret_key: str
    poll_interval_secs: int = Field(default=60, ge=10, le=3600)


class LangSmithIntegrationRequest(BaseModel):
    endpoint_url: str
    api_key: str
    project_name: str
    poll_interval_secs: int = Field(default=60, ge=10, le=3600)


class BraintrustIntegrationRequest(BaseModel):
    endpoint_url: str
    api_key: str
    project_id: str
    poll_interval_secs: int = Field(default=60, ge=10, le=3600)


class IntegrationStatus(BaseModel):
    configured: bool
    endpoint_url: str | None = None
    poll_interval_secs: int | None = None
    enabled: bool | None = None
    last_polled_at: float | None = None
    last_success_at: float | None = None
    consecutive_failures: int | None = None


def _status_response(status: dict | None) -> IntegrationStatus:
    if status is None:
        return IntegrationStatus(configured=False)
    return IntegrationStatus(
        configured=True,
        endpoint_url=status["endpoint_url"],
        poll_interval_secs=status["poll_interval_secs"],
        enabled=status["enabled"],
        last_polled_at=status["last_polled_at"].timestamp() if status["last_polled_at"] else None,
        last_success_at=status["last_success_at"].timestamp()
        if status["last_success_at"]
        else None,
        consecutive_failures=status["consecutive_failures"],
    )


@router.post(
    "/langfuse",
    summary="Configure a Langfuse pull integration for this org",
    response_model=IntegrationStatus,
    status_code=201,
)
async def set_langfuse_integration(
    body: LangfuseIntegrationRequest,
    org_id: str = Depends(require_org),
) -> IntegrationStatus:
    try:
        encrypted = encrypt_credentials(
            {"public_key": body.public_key, "secret_key": body.secret_key}
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    await upsert_external_integration(
        org_id, "langfuse", body.endpoint_url, encrypted, body.poll_interval_secs
    )
    status = await get_external_integration_status(org_id, "langfuse")
    return _status_response(status)


@router.get(
    "/langfuse",
    summary="Get this org's Langfuse integration status",
    response_model=IntegrationStatus,
)
async def get_langfuse_integration(org_id: str = Depends(require_org)) -> IntegrationStatus:
    status = await get_external_integration_status(org_id, "langfuse")
    return _status_response(status)


@router.delete(
    "/langfuse",
    summary="Remove this org's Langfuse integration",
    status_code=204,
)
async def remove_langfuse_integration(org_id: str = Depends(require_org)):
    deleted = await delete_external_integration(org_id, "langfuse")
    if not deleted:
        raise HTTPException(status_code=404, detail="No Langfuse integration configured.")


@router.post(
    "/langsmith",
    summary="Configure a LangSmith pull integration for this org",
    response_model=IntegrationStatus,
    status_code=201,
)
async def set_langsmith_integration(
    body: LangSmithIntegrationRequest,
    org_id: str = Depends(require_org),
) -> IntegrationStatus:
    try:
        encrypted = encrypt_credentials(
            {"api_key": body.api_key, "project_name": body.project_name}
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    await upsert_external_integration(
        org_id, "langsmith", body.endpoint_url, encrypted, body.poll_interval_secs
    )
    status = await get_external_integration_status(org_id, "langsmith")
    return _status_response(status)


@router.get(
    "/langsmith",
    summary="Get this org's LangSmith integration status",
    response_model=IntegrationStatus,
)
async def get_langsmith_integration(org_id: str = Depends(require_org)) -> IntegrationStatus:
    status = await get_external_integration_status(org_id, "langsmith")
    return _status_response(status)


@router.delete(
    "/langsmith",
    summary="Remove this org's LangSmith integration",
    status_code=204,
)
async def remove_langsmith_integration(org_id: str = Depends(require_org)):
    deleted = await delete_external_integration(org_id, "langsmith")
    if not deleted:
        raise HTTPException(status_code=404, detail="No LangSmith integration configured.")


@router.post(
    "/braintrust",
    summary="Configure a Braintrust pull integration for this org",
    response_model=IntegrationStatus,
    status_code=201,
)
async def set_braintrust_integration(
    body: BraintrustIntegrationRequest,
    org_id: str = Depends(require_org),
) -> IntegrationStatus:
    try:
        encrypted = encrypt_credentials({"api_key": body.api_key, "project_id": body.project_id})
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    await upsert_external_integration(
        org_id, "braintrust", body.endpoint_url, encrypted, body.poll_interval_secs
    )
    status = await get_external_integration_status(org_id, "braintrust")
    return _status_response(status)


@router.get(
    "/braintrust",
    summary="Get this org's Braintrust integration status",
    response_model=IntegrationStatus,
)
async def get_braintrust_integration(org_id: str = Depends(require_org)) -> IntegrationStatus:
    status = await get_external_integration_status(org_id, "braintrust")
    return _status_response(status)


@router.delete(
    "/braintrust",
    summary="Remove this org's Braintrust integration",
    status_code=204,
)
async def remove_braintrust_integration(org_id: str = Depends(require_org)):
    deleted = await delete_external_integration(org_id, "braintrust")
    if not deleted:
        raise HTTPException(status_code=404, detail="No Braintrust integration configured.")
