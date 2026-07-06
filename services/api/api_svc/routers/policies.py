"""Policy CRUD endpoints — create, list, update, and delete runtime guardrails."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_svc.auth import require_org
from api_svc.db.queries import (
    list_policies,
    get_policy_by_id,
    create_policy,
    update_policy,
    delete_policy,
    log_policy_audit,
)

logger = logging.getLogger("dunetrace.api.policies")
router = APIRouter(prefix="/v1/policies", tags=["Policies"])

_VALID_TRIGGERS = {
    "tool_call_count",
    "step_count",
    "cost_usd",
    "error_count",
    "finish_reason",
    "llm_latency_ms",
    "signal",
}
_VALID_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "neq", "contains"}
_VALID_ACTIONS = {"stop", "switch_model", "inject_prompt", "log"}


class ConditionModel(BaseModel):
    trigger: str
    operator: str
    value: Any


class ActionModel(BaseModel):
    type: str
    params: Optional[Dict[str, Any]] = None


class PolicyCreate(BaseModel):
    name: str
    agent_id: str = "*"
    condition: ConditionModel
    action: ActionModel
    priority: int = 100
    enabled: bool = True


class PolicyUpdate(BaseModel):
    name: Optional[str] = None
    agent_id: Optional[str] = None
    condition: Optional[ConditionModel] = None
    action: Optional[ActionModel] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None


def _check_prompt_injection(text: str) -> list:
    """Return matched pattern labels if text contains injection signatures, else []."""
    from dunetrace.detectors import _INJECTION_PATTERNS_COMPILED

    return [label for label, pattern in _INJECTION_PATTERNS_COMPILED if pattern.search(text)]


def _validate(condition: ConditionModel, action: ActionModel) -> None:
    if condition.trigger not in _VALID_TRIGGERS:
        raise HTTPException(
            422,
            f"Invalid trigger {condition.trigger!r}. Valid: {sorted(_VALID_TRIGGERS)}",
        )
    if condition.operator not in _VALID_OPERATORS:
        raise HTTPException(
            422,
            f"Invalid operator {condition.operator!r}. Valid: {sorted(_VALID_OPERATORS)}",
        )
    if action.type not in _VALID_ACTIONS:
        raise HTTPException(
            422, f"Invalid action type {action.type!r}. Valid: {sorted(_VALID_ACTIONS)}"
        )
    if action.type == "switch_model" and not (action.params or {}).get("model"):
        raise HTTPException(422, "switch_model action requires params.model")
    if action.type == "inject_prompt":
        prompt = (action.params or {}).get("prompt", "")
        if not prompt:
            raise HTTPException(422, "inject_prompt action requires params.prompt")
        matched = _check_prompt_injection(prompt)
        if matched:
            raise HTTPException(
                422,
                f"inject_prompt content rejected: matches injection pattern(s): {matched}",
            )


@router.get("", response_model=List[Dict[str, Any]], summary="List all policies")
async def list_all_policies(
    agent_id: Optional[str] = None,
    org_id: str = Depends(require_org),
) -> List[Dict[str, Any]]:
    return await list_policies(org_id, agent_id=agent_id)


@router.post("", response_model=Dict[str, Any], summary="Create a policy")
async def create(
    body: PolicyCreate,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    _validate(body.condition, body.action)
    row = await create_policy(
        org_id=org_id,
        name=body.name,
        agent_id=body.agent_id,
        condition=body.condition.model_dump(),
        action=body.action.model_dump(exclude_none=True),
        priority=body.priority,
        enabled=body.enabled,
    )
    await log_policy_audit(row["id"], "created", org_id, before=None, after=row)
    return row


@router.get("/{policy_id}", response_model=Dict[str, Any], summary="Get a policy")
async def get_one(
    policy_id: int,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    row = await get_policy_by_id(org_id, policy_id)
    if row is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    return row


@router.put("/{policy_id}", response_model=Dict[str, Any], summary="Update a policy")
async def update(
    policy_id: int,
    body: PolicyUpdate,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    existing = await get_policy_by_id(org_id, policy_id)
    if existing is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    effective_condition = body.condition or ConditionModel(**existing["condition"])
    effective_action = body.action or ActionModel(**existing["action"])
    _validate(effective_condition, effective_action)
    row = await update_policy(org_id, policy_id, body.model_dump(exclude_none=True))
    await log_policy_audit(policy_id, "updated", org_id, before=existing, after=row)
    return row


@router.delete("/{policy_id}", response_model=Dict[str, Any], summary="Delete a policy")
async def delete(
    policy_id: int,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    existing = await get_policy_by_id(org_id, policy_id)
    if existing is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    await delete_policy(org_id, policy_id)
    await log_policy_audit(policy_id, "deleted", org_id, before=existing, after=None)
    return {"deleted": True, "policy_id": policy_id}


@router.patch(
    "/{policy_id}/toggle",
    response_model=Dict[str, Any],
    summary="Toggle enabled/disabled",
)
async def toggle(
    policy_id: int,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    existing = await get_policy_by_id(org_id, policy_id)
    if existing is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    row = await update_policy(org_id, policy_id, {"enabled": not existing["enabled"]})
    await log_policy_audit(policy_id, "toggled", org_id, before=existing, after=row)
    return row
