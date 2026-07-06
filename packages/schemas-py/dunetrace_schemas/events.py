"""
Canonical wire-format model for a single agent instrumentation event.

This is the shape validated at the ingest boundary — one per event in an
ingest batch. Field-for-field equivalent to ingest_svc's former hand-rolled
IngestEvent; ingest_svc now imports AgentEventSchema directly instead of
maintaining its own copy of this validation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from dunetrace_schemas.enums import EventType

VALID_EVENT_TYPES = {e.value for e in EventType}


class AgentEventSchema(BaseModel):
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
