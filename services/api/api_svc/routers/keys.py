"""API key management endpoints — list, create, revoke."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api_svc.auth import require_customer
from api_svc.db import queries

logger = logging.getLogger("dunetrace.api.keys")
router = APIRouter(prefix="/v1/keys", tags=["Keys"])


# ── Schemas ────────────────────────────────────────────────────────────────────


class KeyOut(BaseModel):
    id: int
    key_prefix: str
    agent_id: str
    customer_id: str
    company_name: Optional[str]
    active: bool
    rate_limit_rpm: int
    created_at: str


class KeyCreateBody(BaseModel):
    agent_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    company_name: Optional[str] = None
    rate_limit_rpm: int = Field(default=600, ge=1, le=100_000)


class KeyCreateResponse(BaseModel):
    id: int
    key: str  # full key — only returned once
    key_prefix: str
    agent_id: str
    customer_id: str
    company_name: str
    rate_limit_rpm: int
    created_at: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.get("", response_model=List[KeyOut])
async def list_keys(
    agent_id: Optional[str] = Query(None),
    customer_id: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
    _customer: str = Depends(require_customer),
):
    return await queries.list_api_keys(
        agent_id=agent_id, customer_id=customer_id, active_only=active_only, limit=limit
    )


@router.post("", response_model=KeyCreateResponse, status_code=201)
async def create_key(body: KeyCreateBody, _customer: str = Depends(require_customer)):
    return await queries.create_api_key(
        agent_id=body.agent_id,
        customer_id=body.customer_id,
        company_name=body.company_name,
        rate_limit_rpm=body.rate_limit_rpm,
    )


@router.delete("/{key_id}", status_code=204)
async def revoke_key(key_id: int, _customer: str = Depends(require_customer)):
    ok = await queries.revoke_api_key(key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found")
