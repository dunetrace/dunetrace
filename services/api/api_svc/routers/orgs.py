"""Org-level settings. Phase 1.4.3: the semantic feedback loop opt-in toggle.
Phase 1.5: semantic evaluation quota/usage reporting."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api_svc.auth import require_org
from api_svc.db.queries import (
    get_organization_semantic_feedback,
    get_org_semantic_usage,
    update_organization_semantic_feedback,
)
from api_svc.semantic_usage import current_month, project_month_end

logger = logging.getLogger("dunetrace.api.orgs")
router = APIRouter(prefix="/v1/orgs", tags=["Orgs"])


class SemanticFeedbackSettings(BaseModel):
    enabled: bool
    auto_suppress: bool


@router.get(
    "/semantic-feedback",
    summary="Get this org's semantic feedback loop settings",
    response_model=SemanticFeedbackSettings,
)
async def get_semantic_feedback_settings(
    org_id: str = Depends(require_org),
) -> SemanticFeedbackSettings:
    settings = await get_organization_semantic_feedback(org_id)
    if settings is None:
        # Org row missing is unexpected (require_org implies it exists), but
        # the safe default is "off" rather than a 404 for a settings read.
        return SemanticFeedbackSettings(enabled=False, auto_suppress=False)
    return SemanticFeedbackSettings(**settings)


@router.patch(
    "/semantic-feedback",
    summary="Enable/disable the semantic feedback loop for this org",
    response_model=SemanticFeedbackSettings,
)
async def set_semantic_feedback_settings(
    body: SemanticFeedbackSettings,
    org_id: str = Depends(require_org),
) -> SemanticFeedbackSettings:
    """Opt-in, per the Phase 1.4 brief: both fields default to FALSE until an
    org explicitly turns this on via the dashboard. auto_suppress only takes
    effect once semantic_svc sees 3+ false-positive marks for a given
    recurring pattern — see semantic_svc/worker.py."""
    await update_organization_semantic_feedback(org_id, body.enabled, body.auto_suppress)
    return body


class SemanticUsage(BaseModel):
    quota: int
    used_this_month: int
    remaining: int
    allow_overage: bool
    projected_month_end_usage: int
    projected_month_end_cost_usd: float


@router.get(
    "/semantic-usage",
    summary="This org's semantic evaluation quota and usage for the current month",
    response_model=SemanticUsage,
)
async def get_semantic_usage(org_id: str = Depends(require_org)) -> SemanticUsage:
    """Org identity comes from require_org, not a path param — same
    convention as /v1/orgs/semantic-feedback, and this codebase's universal
    pattern of deriving org scope from auth rather than a URL param (so one
    org can never query another's usage/cost by guessing an id).
    """
    month = current_month()
    usage = await get_org_semantic_usage(org_id, month)

    remaining = max(0, usage["quota"] - usage["used_this_month"])
    projected_usage = project_month_end(usage["used_this_month"])
    projected_cost = project_month_end(usage["cost_so_far_usd"])

    return SemanticUsage(
        quota=usage["quota"],
        used_this_month=usage["used_this_month"],
        remaining=remaining,
        allow_overage=usage["allow_overage"],
        projected_month_end_usage=round(projected_usage),
        projected_month_end_cost_usd=round(projected_cost, 2),
    )
