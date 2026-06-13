"""POST /v1/ingest — accepts event batches from the SDK.
GET  /v1/policies — returns runtime policies for the SDK to enforce.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, status, BackgroundTasks

from ingest_svc.auth import is_trusted
from ingest_svc.db import (
    insert_events,
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


@router.post(
    "/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of agent events",
)
async def ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    # Auth
    agent_id = await verify_api_key(body.api_key)
    if agent_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )

    # Accept immediately — 202 before any DB work
    batch_id = str(uuid.uuid4())
    n = len(body.events)

    logger.info("Accepted. batch_id=%s agent_id=%s events=%d", batch_id, body.agent_id, n)

    # Persist after response is sent
    background_tasks.add_task(_persist, body.events, batch_id, body.agent_id)

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
    Trusted internal requests (from auth service) skip DB validation.
    """
    if not is_trusted(request):
        resolved = await verify_api_key(api_key)
        if resolved is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
    policies = await fetch_policies(agent_id)
    return {"policies": policies}


@router.post(
    "/v1/deploy",
    response_model=DeployResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Record a deploy marker for an agent",
)
async def mark_deploy(body: DeployRequest) -> DeployResponse:
    agent_id = await verify_api_key(body.api_key)
    if agent_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    row_id = await insert_deploy_event(agent_id, body.version, body.meta)
    logger.info("Deploy marked. agent_id=%s version=%s id=%d", agent_id, body.version, row_id)
    return DeployResponse(
        id=row_id,
        agent_id=agent_id,
        version=body.version,
        deployed_at=time.time(),
    )


@router.post(
    "/v1/keys",
    response_model=KeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new API key for an agent",
    include_in_schema=False,
)
async def create_key(body: KeyCreateRequest) -> KeyCreateResponse:
    """Admin-only endpoint. Requires ADMIN_API_KEY env var to match body.admin_key."""
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key or body.admin_key != admin_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin key")
    name = body.company_name or body.customer_id
    key = await create_api_key(
        body.agent_id, body.customer_id, company_name=name, rate_limit_rpm=body.rate_limit_rpm
    )
    logger.info(
        "API key created. agent_id=%s customer_id=%s company=%s rpm=%d",
        body.agent_id,
        body.customer_id,
        name,
        body.rate_limit_rpm,
    )
    return KeyCreateResponse(
        key=key, agent_id=body.agent_id, customer_id=body.customer_id, company_name=name
    )


async def _persist(events: list, batch_id: str, agent_id: str) -> None:
    try:
        inserted = await insert_events(events, batch_id)
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
