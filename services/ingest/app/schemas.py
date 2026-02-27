"""
services/ingest/app/schemas.py

Pydantic v2 request/response models.
Replaces the hand-rolled validation.py entirely.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


VALID_EVENT_TYPES = {
    "run.started", "run.completed", "run.errored",
    "llm.called", "llm.responded",
    "tool.called", "tool.responded",
    "retrieval.called", "retrieval.responded",
}


class IngestEvent(BaseModel):
    event_type:    str
    run_id:        str            = Field(min_length=1)
    agent_id:      str            = Field(min_length=1)
    agent_version: str            = Field(min_length=1)
    step_index:    int            = Field(ge=0)
    timestamp:     float          = Field(default_factory=time.time)
    payload:       Dict[str, Any] = Field(default_factory=dict)
    parent_run_id: Optional[str]  = None

    @field_validator("event_type")
    @classmethod
    def valid_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {v!r}. Valid: {sorted(VALID_EVENT_TYPES)}"
            )
        return v


class IngestRequest(BaseModel):
    api_key:  str            = Field(min_length=1)
    agent_id: str            = Field(min_length=1)
    events:   List[IngestEvent] = Field(min_length=1)

    @field_validator("events")
    @classmethod
    def check_batch_size(cls, v: list) -> list:
        from app.config import settings
        if len(v) > settings.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of {settings.MAX_BATCH_SIZE}"
            )
        return v


class IngestResponse(BaseModel):
    accepted:  int
    batch_id:  str
    queued_at: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "0.1.0"
    db:      str = "unknown"
