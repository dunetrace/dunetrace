"""
Human-in-the-loop approvals (Capability 2). The SDK-facing surface of the
approval flow:

  POST /v1/approvals               create a pending approval (SDK, before a
                                   guarded tool runs) — org from the API key
  GET  /v1/approvals/{id}          poll an approval's status (SDK blocks on this)
  POST /v1/approvals/{id}/decision record a decision (dashboard / API / the SDK
                                   itself marking a timeout)

Every endpoint is org-scoped from the caller's API key (registered with the
require_org dependency in main.py) — an approval id alone never lets one org
read or decide another org's approval; the queries filter on org_id too.

Delivery of the request to a human (Slack buttons, webhook, email) is Phase
2.3 and lives elsewhere; this module only stores and reports state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_svc.approvals import DECISION_STATUSES, ApprovalStatus
from api_svc.auth import require_org, require_scope
from api_svc.db.queries import (
    create_approval,
    get_approval,
    list_approvals,
    set_approval_decision,
)

logger = logging.getLogger("dunetrace.api.approvals")
router = APIRouter(tags=["Approvals"])

# Bounds on the SDK-requested timeout, so a bad client value can't create an
# approval that effectively never expires (or expires instantly).
_MIN_TIMEOUT_S = 1
_MAX_TIMEOUT_S = 3600


class ApprovalCreate(BaseModel):
    run_id: str
    agent_id: str
    tool_name: str
    tool_args: Optional[str] = None
    timeout_seconds: int = 300


class ApprovalDecision(BaseModel):
    decision: str  # granted | denied | timeout
    decided_by: Optional[str] = None
    decision_channel: Optional[str] = None


def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
    """Datetimes → ISO strings so the JSON body is stable regardless of DB
    driver representation."""
    out = dict(row)
    for k in ("requested_at", "expires_at", "decided_at"):
        v = out.get(k)
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    return out


@router.post("/v1/approvals", summary="Create a pending approval", status_code=201)
async def post_create_approval(
    body: ApprovalCreate,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    # org_id derived from require_org(), not the body — the SDK cannot create an
    # approval under another org by spoofing a field. Same rule every org-scoped
    # endpoint here follows; see scripts/check_endpoint_conventions.py.
    timeout = max(_MIN_TIMEOUT_S, min(_MAX_TIMEOUT_S, body.timeout_seconds))
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)
    row = await create_approval(
        org_id=org_id,
        run_id=body.run_id,
        agent_id=body.agent_id,
        tool_name=body.tool_name,
        tool_args=body.tool_args,
        expires_at=expires_at,
    )
    if row is None:
        raise HTTPException(status_code=503, detail="Approval store unavailable.")
    return _serialize(row)


@router.get("/v1/approvals", summary="List this org's approvals")
async def list_org_approvals(
    status: Optional[str] = None,
    org_id: str = Depends(require_org),
) -> list:
    # org_id from require_org(), not the URL — the dashboard only ever sees its
    # own org's approvals. See scripts/check_endpoint_conventions.py.
    if status is not None:
        try:
            ApprovalStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status {status!r}. Valid: "
                f"{sorted(s.value for s in ApprovalStatus)}",
            )
    rows = await list_approvals(org_id, status)
    return [_serialize(r) for r in rows]


@router.get("/v1/approvals/{approval_id}", summary="Poll an approval's status")
async def get_one_approval(
    approval_id: int,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    row = await get_approval(org_id, approval_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Approval not found.")
    return _serialize(row)


@router.post("/v1/approvals/{approval_id}/decision", summary="Record an approval decision")
async def post_approval_decision(
    approval_id: int,
    body: ApprovalDecision,
    # 'approve', not the ambient org key. The agent process blocked on this
    # request holds an ingest-scoped key and sends it on every call, so
    # require_org here meant the thing being gated could grant its own gate.
    # A decision must come from a credential the agent runtime does not have:
    # a dashboard/operator key, or the Slack path, which verifies Slack's own
    # signature rather than a Dunetrace key at all.
    org_id: str = Depends(require_scope("approve")),
) -> Dict[str, Any]:
    try:
        target = ApprovalStatus(body.decision)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid decision {body.decision!r}. Valid: "
            f"{sorted(s.value for s in DECISION_STATUSES)}",
        )
    if target not in DECISION_STATUSES:
        # e.g. 'pending' is a valid ApprovalStatus but not a decision.
        raise HTTPException(
            status_code=422,
            detail=f"{body.decision!r} is not a decision. Valid: "
            f"{sorted(s.value for s in DECISION_STATUSES)}",
        )

    updated = await set_approval_decision(
        org_id=org_id,
        approval_id=approval_id,
        new_status=target.value,
        decided_by=body.decided_by,
        decision_channel=body.decision_channel,
    )
    if updated is not None:
        return _serialize(updated)

    # set returned None: either the approval doesn't exist, or it's already in a
    # terminal state (the status='pending' guard blocked the write — a late
    # click or a decision racing the SDK's own timeout). Distinguish so the
    # caller learns the actual outcome instead of a bare failure.
    existing = await get_approval(org_id, approval_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Approval not found.")
    raise HTTPException(
        status_code=409,
        detail=f"Approval already decided: {existing['status']}.",
    )
