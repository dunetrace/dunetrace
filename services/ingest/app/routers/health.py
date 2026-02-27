"""
services/ingest/app/routers/health.py

GET /health
"""
from fastapi import APIRouter
from app.db import check_db
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health() -> HealthResponse:
    return HealthResponse(db=await check_db())
