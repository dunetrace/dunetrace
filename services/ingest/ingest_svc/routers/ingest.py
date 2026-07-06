"""POST /v1/ingest — accepts event batches from the SDK.
GET  /v1/policies — returns runtime policies for the SDK to enforce.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, status, BackgroundTasks

from ingest_svc.auth import is_trusted
from ingest_svc.db import (
    get_event_store,
    insert_deploy_event,
    verify_api_key,
    create_api_key,
    fetch_policies,
)
from ingest_svc.schemas import (
    IngestRequest,
    IngestResponse,
    DeployRequest,
    DeployResponse,
    KeyCreateRequest,
    KeyCreateResponse,
)

logger = logging.getLogger("dunetrace.ingest")
router = APIRouter()


async def _resolve_org_id(request: Request, api_key: str) -> str:
    """Resolve the org_id for this request. Raises 401 if it can't be resolved.

    Trusted path: dunetrace-cloud's gateway has already authenticated the caller
    and forwards its org identity via a header — no api_keys lookup here.
    x-org-id is the current header name; x-customer-id is accepted as a fallback
    for callers running an older cloud gateway build (pre-v0.5.0 naming).

    Untrusted path (self-hosted): resolves org_id from the OSS api_keys table.
    Keys are org-scoped, not agent-scoped — a valid key may submit events for
    any agent_id under its org, discovered on first ingest. See
    docs/migrations/multi-tenancy-v0.5.0.md.
    """
    if is_trusted(request):
        org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id", "")
        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-org-id",
            )
        return org_id

    org_id = await verify_api_key(api_key)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    return org_id


@router.post(
    "/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of agent events",
)
async def ingest(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    org_id = await _resolve_org_id(request, body.api_key)

    # Accept immediately — 202 before any DB work
    batch_id = str(uuid.uuid4())
    n = len(body.events)

    logger.info(
        "Accepted. batch_id=%s org_id=%s agent_id=%s events=%d", batch_id, org_id, body.agent_id, n
    )

    # Persist after response is sent
    background_tasks.add_task(_persist, body.events, batch_id, org_id)

    return IngestResponse(accepted=n, batch_id=batch_id)


@router.get(
    "/v1/policies",
    summary="Fetch runtime policies for an agent (SDK-facing)",
    include_in_schema=False,
)
async def get_policies(
    request: Request,
    agent_id: str = Query(...),
    api_key: str = Query(...),
) -> dict:
    """
    Called by the SDK at run start to retrieve active policies.
    Authenticates via api_key query param — same key used for event ingestion.
    """
    org_id = await _resolve_org_id(request, api_key)
    policies = await fetch_policies(agent_id, org_id)
    return {"policies": policies}


@router.post(
    "/v1/deploy",
    response_model=DeployResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Record a deploy marker for an agent",
)
async def mark_deploy(request: Request, body: DeployRequest) -> DeployResponse:
    org_id = await _resolve_org_id(request, body.api_key)
    row_id = await insert_deploy_event(body.agent_id, body.version, body.meta, org_id)
    logger.info(
        "Deploy marked. org_id=%s agent_id=%s version=%s id=%d",
        org_id,
        body.agent_id,
        body.version,
        row_id,
    )
    return DeployResponse(
        id=row_id,
        agent_id=body.agent_id,
        version=body.version,
        deployed_at=time.time(),
    )


@router.post(
    "/v1/keys",
    response_model=KeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new org-scoped API key",
    include_in_schema=False,
)
async def create_key(body: KeyCreateRequest) -> KeyCreateResponse:
    """Admin-only endpoint. Requires ADMIN_API_KEY env var to match body.admin_key."""
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key or not secrets.compare_digest(body.admin_key, admin_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin key")
    name = body.org_name or body.org_id
    key = await create_api_key(body.org_id, org_name=name, rate_limit_rpm=body.rate_limit_rpm)
    logger.info(
        "API key created. org_id=%s org_name=%s rpm=%d", body.org_id, name, body.rate_limit_rpm
    )
    return KeyCreateResponse(key=key, org_id=body.org_id, org_name=name)


async def _persist(events: list, batch_id: str, org_id: str) -> None:
    try:
        inserted = await get_event_store().insert_events(events, batch_id, org_id)
        if inserted == len(events):
            logger.debug("Persisted. batch_id=%s inserted=%d", batch_id, inserted)
        else:
            logger.error(
                "Persist shortfall. batch_id=%s inserted=%d expected=%d — events lost",
                batch_id,
                inserted,
                len(events),
            )
    except Exception as exc:
        logger.error("Persist failed. batch_id=%s error=%s", batch_id, exc)
