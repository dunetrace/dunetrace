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

from pydantic import BaseModel, Field

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
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    # audit Finding 14: client-generated dedup id. Optional so older SDK clients
    # (which don't send it) still validate — those events simply aren't deduped.
    event_id: Optional[str] = None

    # event_type is deliberately NOT validated against a closed set. A newer SDK
    # may emit event types an older ingest doesn't know yet (memory.*, future
    # kinds); rejecting them here would fail the whole IngestRequest and drop the
    # entire batch, not just the unknown event. Accepting any string makes new
    # event types rolling-deploy safe — downstream services already ignore types
    # they don't handle. VALID_EVENT_TYPES stays exported as the set of types
    # this build knows about (used by the SDK↔schema parity tests).
