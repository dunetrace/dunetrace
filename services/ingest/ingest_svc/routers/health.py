"""
services/ingest/ingest_svc/routers/health.py

GET /health
"""
from fastapi import APIRouter
from ingest_svc.db import check_db
from ingest_svc.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health() -> HealthResponse:
    return HealthResponse(db=await check_db())
