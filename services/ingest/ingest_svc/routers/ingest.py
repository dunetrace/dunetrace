"""POST /v1/ingest — accepts event batches from the SDK."""
from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from ingest_svc.db import insert_events, verify_api_key
from ingest_svc.schemas import IngestRequest, IngestResponse

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

    logger.info("Accepted. batch_id=%s agent_id=%s events=%d",
                batch_id, body.agent_id, n)

    # Persist after response is sent
    background_tasks.add_task(_persist, body.events, batch_id, body.agent_id)

    return IngestResponse(accepted=n, batch_id=batch_id)


async def _persist(events: list, batch_id: str, agent_id: str) -> None:
    try:
        inserted = await insert_events(events, batch_id)
        if inserted == len(events):
            logger.debug("Persisted. batch_id=%s inserted=%d", batch_id, inserted)
        else:
            logger.error(
                "Persist shortfall. batch_id=%s inserted=%d expected=%d — events lost",
                batch_id, inserted, len(events),
            )
    except Exception as exc:
        logger.error("Persist failed. batch_id=%s error=%s", batch_id, exc)
