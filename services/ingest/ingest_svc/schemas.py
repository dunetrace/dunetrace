"""
Pydantic v2 request and response models for the ingest API.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

VALID_EVENT_TYPES = {
    "run.started",
    "run.completed",
    "run.errored",
    "llm.called",
    "llm.responded",
    "tool.called",
    "tool.responded",
    "retrieval.called",
    "retrieval.responded",
    "external.signal",
    "policy.triggered",
}


class IngestEvent(BaseModel):
    event_type: str
    run_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    timestamp: float = Field(default_factory=time.time)
    payload: Dict[str, Any] = Field(default_factory=dict)
    parent_run_id: Optional[str] = None

    @field_validator("event_type")
    @classmethod
    def valid_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(f"Unknown event_type {v!r}. Valid: {sorted(VALID_EVENT_TYPES)}")
        return v


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
    agent_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    admin_key: str = Field(min_length=1)
    company_name: Optional[str] = None
    rate_limit_rpm: int = Field(default=600, ge=1, le=100_000)


class KeyCreateResponse(BaseModel):
    key: str
    agent_id: str
    customer_id: str
    company_name: str
    created_at: float = Field(default_factory=time.time)
