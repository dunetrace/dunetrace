"""Policy CRUD endpoints — create, list, update, and delete runtime guardrails."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_svc.auth import require_customer
from api_svc.db.queries import (
    list_policies,
    get_policy_by_id,
    create_policy,
    update_policy,
    delete_policy,
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


def _validate(condition: ConditionModel, action: ActionModel) -> None:
    if condition.trigger not in _VALID_TRIGGERS:
        raise HTTPException(
            422, f"Invalid trigger {condition.trigger!r}. Valid: {sorted(_VALID_TRIGGERS)}"
        )
    if condition.operator not in _VALID_OPERATORS:
        raise HTTPException(
            422, f"Invalid operator {condition.operator!r}. Valid: {sorted(_VALID_OPERATORS)}"
        )
    if action.type not in _VALID_ACTIONS:
        raise HTTPException(
            422, f"Invalid action type {action.type!r}. Valid: {sorted(_VALID_ACTIONS)}"
        )
    if action.type == "switch_model" and not (action.params or {}).get("model"):
        raise HTTPException(422, "switch_model action requires params.model")
    if action.type == "inject_prompt" and not (action.params or {}).get("prompt"):
        raise HTTPException(422, "inject_prompt action requires params.prompt")


@router.get("", response_model=List[Dict[str, Any]], summary="List all policies")
async def list_all_policies(
    agent_id: Optional[str] = None,
    _customer: str = Depends(require_customer),
) -> List[Dict[str, Any]]:
    return await list_policies(agent_id=agent_id)


@router.post("", response_model=Dict[str, Any], summary="Create a policy")
async def create(
    body: PolicyCreate,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    _validate(body.condition, body.action)
    row = await create_policy(
        name=body.name,
        agent_id=body.agent_id,
        condition=body.condition.model_dump(),
        action=body.action.model_dump(exclude_none=True),
        priority=body.priority,
        enabled=body.enabled,
    )
    return row


@router.get("/{policy_id}", response_model=Dict[str, Any], summary="Get a policy")
async def get_one(
    policy_id: int,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    row = await get_policy_by_id(policy_id)
    if row is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    return row


@router.put("/{policy_id}", response_model=Dict[str, Any], summary="Update a policy")
async def update(
    policy_id: int,
    body: PolicyUpdate,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    existing = await get_policy_by_id(policy_id)
    if existing is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    if body.condition:
        _validate(body.condition, body.action or ActionModel(type=existing["action"]["type"]))
    row = await update_policy(policy_id, body.model_dump(exclude_none=True))
    return row


@router.delete("/{policy_id}", response_model=Dict[str, Any], summary="Delete a policy")
async def delete(
    policy_id: int,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    existing = await get_policy_by_id(policy_id)
    if existing is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    await delete_policy(policy_id)
    return {"deleted": True, "policy_id": policy_id}


@router.patch(
    "/{policy_id}/toggle", response_model=Dict[str, Any], summary="Toggle enabled/disabled"
)
async def toggle(
    policy_id: int,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    existing = await get_policy_by_id(policy_id)
    if existing is None:
        raise HTTPException(404, f"Policy {policy_id} not found")
    row = await update_policy(policy_id, {"enabled": not existing["enabled"]})
    return row
