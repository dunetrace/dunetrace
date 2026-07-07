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
    status: str = "ok"
    version: str = "0.1.0"
    db: str = "unknown"


class KeyCreateRequest(BaseModel):
    org_id: str = Field(min_length=1)
    admin_key: str = Field(min_length=1)
    org_name: Optional[str] = None
    rate_limit_rpm: int = Field(default=600, ge=1, le=100_000)


class KeyCreateResponse(BaseModel):
    key: str
    org_id: str
    org_name: str
    created_at: float = Field(default_factory=time.time)


class PruneEventsRequest(BaseModel):
    admin_key: str = Field(min_length=1)
    retention_days: Optional[int] = None  # defaults to EVENT_RETENTION_DAYS if omitted


class PruneEventsResponse(BaseModel):
    partitions_dropped: int
    retention_days: int


class QuotaSetRequest(BaseModel):
    admin_key: str = Field(min_length=1)
    quota_pct: float = Field(gt=0, le=1)  # fraction of the key's rpm, e.g. 0.2 = 20%


class QuotaResponse(BaseModel):
    key_id: int
    agent_id: str
    quota_pct: float
    is_override: bool  # False when quota_pct is the default, not an explicit override
