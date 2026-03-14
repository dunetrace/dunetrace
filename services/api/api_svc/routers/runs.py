"""Runs endpoints — list and inspect individual agent runs."""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from api_svc.auth import require_customer
from api_svc.config import settings
from api_svc.db.queries import list_runs, get_run_detail
from api_svc.schemas import RunDetail, RunEvent, RunListResponse, RunSignal, RunSummary, Page

router = APIRouter(tags=["Runs"])


def _ts(v):
    if v is None:
        return None
    return v.timestamp() if hasattr(v, "timestamp") else float(v)


@router.get(
    "/v1/agents/{agent_id}/runs",
    response_model=RunListResponse,
    summary="List runs for an agent",
)
async def get_runs(
    agent_id:    str,
    offset:      int           = Query(0, ge=0),
    limit:       int           = Query(settings.PAGE_SIZE_DEFAULT, ge=1,
                                       le=settings.PAGE_SIZE_MAX),
    has_signals: Optional[bool] = Query(None,
        description="Filter to runs that do (true) or don't (false) have signals"),
    _customer:   str           = Depends(require_customer),
) -> RunListResponse:
    rows, total = await list_runs(agent_id, offset, limit, has_signals)

    runs = [
        RunSummary(
            run_id=r["run_id"],
            agent_id=r["agent_id"],
            agent_version=r["agent_version"],
            started_at=_ts(r.get("started_at")),
            completed_at=_ts(r.get("completed_at")),
            exit_reason=r.get("exit_reason"),
            step_count=r.get("step_count") or 0,
            signal_count=r.get("signal_count") or 0,
            has_signals=(r.get("signal_count") or 0) > 0,
        )
        for r in rows
    ]
    return RunListResponse(
        runs=runs,
        page=Page(total=total, offset=offset, limit=limit,
                  has_more=(offset + limit) < total),
    )


@router.get(
    "/v1/runs/{run_id}",
    response_model=RunDetail,
    summary="Get full run detail with events and signals",
)
async def get_run(
    run_id:   str,
    _customer: str = Depends(require_customer),
) -> RunDetail:
    data = await get_run_detail(run_id)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Run {run_id!r} not found")

    return RunDetail(
        run_id=data["run_id"],
        agent_id=data["agent_id"],
        agent_version=data["agent_version"],
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        exit_reason=data.get("exit_reason"),
        step_count=data["step_count"],
        events=[RunEvent(**e) for e in data["events"]],
        signals=[RunSignal(**s) for s in data["signals"]],
    )
