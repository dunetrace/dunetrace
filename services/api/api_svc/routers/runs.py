"""Runs endpoints — list and inspect individual agent runs."""

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from api_svc.auth import require_org
from api_svc.config import settings
from api_svc.db.queries import list_runs, get_run_detail
from api_svc.run_states import reconstruct_states
from api_svc.schemas import (
    RunDetail,
    RunEvent,
    RunListResponse,
    RunSignal,
    RunSummary,
    Page,
)

router = APIRouter(tags=["Runs"])


def _ts(v):
    if v is None:
        return None
    return v.timestamp() if hasattr(v, "timestamp") else float(v)


async def _runs_response(
    org_id: str,
    agent_id: Optional[str],
    offset: int,
    limit: int,
    has_signals: Optional[bool],
    per_agent_limit: Optional[int] = None,
) -> RunListResponse:
    """Shared body for the per-agent and org-wide list endpoints."""
    rows, total = await list_runs(
        org_id, agent_id, offset, limit, has_signals, per_agent_limit=per_agent_limit
    )

    runs = [
        RunSummary(
            run_id=r["run_id"],
            agent_id=r["agent_id"],
            agent_version=r["agent_version"],
            started_at=_ts(r.get("started_at")),
            completed_at=_ts(r.get("completed_at")),
            exit_reason=r.get("exit_reason"),
            step_count=r.get("step_count") or 0,
            total_tokens=r.get("total_tokens"),
            cost_usd=r.get("cost_usd"),
            signal_count=r.get("signal_count") or 0,
            has_signals=(r.get("signal_count") or 0) > 0,
        )
        for r in rows
    ]
    return RunListResponse(
        runs=runs,
        page=Page(total=total, offset=offset, limit=limit, has_more=(offset + limit) < total),
    )


@router.get(
    "/v1/runs",
    response_model=RunListResponse,
    summary="List runs across every agent in the org",
)
async def get_all_runs(
    offset: int = Query(0, ge=0),
    limit: int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    agent_id: Optional[str] = Query(None, description="Optional: restrict to one agent"),
    has_signals: Optional[bool] = Query(
        None, description="Filter to runs that do (true) or don't (false) have signals"
    ),
    per_agent_limit: Optional[int] = Query(
        None,
        ge=1,
        description=(
            "Return each agent's newest N runs instead of the newest N overall. "
            "Required to replace a per-agent fan-out: a plain global window is "
            "filled by the busiest agents and leaves the rest empty."
        ),
    ),
    org_id: str = Depends(require_org),
) -> RunListResponse:
    return await _runs_response(org_id, agent_id, offset, limit, has_signals, per_agent_limit)


@router.get(
    "/v1/agents/{agent_id}/runs",
    response_model=RunListResponse,
    summary="List runs for an agent",
)
async def get_runs(
    agent_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    has_signals: Optional[bool] = Query(
        None, description="Filter to runs that do (true) or don't (false) have signals"
    ),
    org_id: str = Depends(require_org),
) -> RunListResponse:
    return await _runs_response(org_id, agent_id, offset, limit, has_signals)


@router.get(
    "/v1/runs/{run_id}",
    response_model=RunDetail,
    summary="Get full run detail with events and signals",
)
async def get_run(
    run_id: str,
    include_shadow: bool = False,
    org_id: str = Depends(require_org),
) -> RunDetail:
    data = await get_run_detail(org_id, run_id, include_shadow=include_shadow)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id!r} not found"
        )

    return RunDetail(
        run_id=data["run_id"],
        agent_id=data["agent_id"],
        agent_version=data["agent_version"],
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        exit_reason=data.get("exit_reason"),
        step_count=data["step_count"],
        total_tokens=data.get("total_tokens"),
        cost_usd=data.get("cost_usd"),
        events=[RunEvent(**e) for e in data["events"]],
        signals=[RunSignal(**s) for s in data["signals"]],
        conversation_id=data.get("conversation_id"),
    )


@router.get(
    "/v1/runs/{run_id}/states",
    summary="Reconstruct the run's state-machine timeline from its events",
)
async def get_run_states(
    run_id: str,
    org_id: str = Depends(require_org),
) -> dict:
    # org_id from require_org(), not the URL — reuses get_run_detail's own
    # org-scoping (returns None for another org's run). See
    # scripts/check_endpoint_conventions.py.
    data = await get_run_detail(org_id, run_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id!r} not found"
        )
    return reconstruct_states(data["events"])
