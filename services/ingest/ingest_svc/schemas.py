"""
Pydantic v2 request and response models for the ingest API.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from dunetrace_schemas import AgentEventSchema, VALID_EVENT_TYPES

# IngestEvent is the canonical wire-format event, defined once in
# dunetrace-schemas and shared with any future consumer of that boundary.
# Kept as a name here (rather than importing AgentEventSchema directly at
# every call site) so nothing else in this service needs to change.
IngestEvent = AgentEventSchema


class IngestRequest(BaseModel):
    api_key: str = Field(default="")
    agent_id: str = Field(min_length=1)
    events: List[IngestEvent] = Field(min_length=1)

    @field_validator("events")
    @classmethod
    def check_batch_size(cls, v: list) -> list:
        from ingest_svc.config import settings

        if len(v) > settings.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum of {settings.MAX_BATCH_SIZE}")
        return v


class IngestResponse(BaseModel):
    accepted: int
    batch_id: str
    queued_at: float = Field(default_factory=time.time)


class DeployRequest(BaseModel):
    api_key: str = Field(default="")
    agent_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    meta: Dict[str, Any] = Field(default_factory=dict)


class DeployResponse(BaseModel):
    id: int
    agent_id: str
    version: str
    deployed_at: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    """GET /health — liveness. No dependency state here by design; see
    ReadyResponse for that."""

    status: str = "ok"
    version: str = "0.5.0"


class ReadyResponse(BaseModel):
    """GET /ready — the readiness verdict and the inputs it was made from
    (dunetrace_schemas.metrics.db_ready's info dict, plus the build version).
    Served with 200 when status is "ok", 503 when it is "not_ready"."""

    status: str  # "ok" | "not_ready"
    version: str
    db: str  # "ok" | "no_pool" | the exception class name the check hit
    schema_version: Optional[int] = None  # applied migration version; None when unreachable
    required: int  # CURRENT_SCHEMA_VERSION this build needs
    pool: Optional[Dict[str, Any]] = None  # size/min/max/idle when the pool exposes them


class KeyCreateRequest(BaseModel):
    org_id: str = Field(min_length=1)
    admin_key: str = Field(min_length=1)
    org_name: Optional[str] = None
    rate_limit_rpm: int = Field(default=600, ge=1, le=100_000)
    # Scopes to grant (see dunetrace_schemas.scopes). None — the field omitted —
    # mints an ``admin`` key: this is the operator's bootstrap endpoint, and an
    # admin key is the one thing a fresh install cannot get any other way. Pass
    # a list (e.g. ``["ingest"]``) to mint something narrower; unknown names
    # are dropped and an all-unknown or empty list falls back to ``ingest``.
    # See routers/ingest.py::create_key for why this default differs from the
    # Customer API's.
    scopes: Optional[List[str]] = None


class KeyCreateResponse(BaseModel):
    key: str  # full key — returned once, never stored or logged
    key_prefix: str  # non-secret leading characters, as listed by GET /v1/keys
    org_id: str
    org_name: str
    scopes: List[str]  # what was actually written, after normalisation
    created_at: float = Field(default_factory=time.time)


class PruneEventsRequest(BaseModel):
    admin_key: str = Field(min_length=1)
    retention_days: Optional[int] = None  # defaults to EVENT_RETENTION_DAYS if omitted


class PruneEventsResponse(BaseModel):
    partitions_dropped: int
    signals_scrubbed: int = 0
    retention_days: int


class QuotaSetRequest(BaseModel):
    admin_key: str = Field(min_length=1)
    quota_pct: float = Field(gt=0, le=1)  # fraction of the key's rpm, e.g. 0.2 = 20%


class QuotaResponse(BaseModel):
    key_id: int
    agent_id: str
    quota_pct: float
    is_override: bool  # False when quota_pct is the default, not an explicit override
