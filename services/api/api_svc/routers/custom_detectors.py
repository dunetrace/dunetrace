"""Custom detector CRUD — create, preview, list, update, delete."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from api_svc.custom_detector_translator import translate_description, SUPPORTED_METRICS
from api_svc.db.queries import (
    create_custom_detector,
    delete_custom_detector,
    get_custom_detector,
    get_custom_detector_shadow_stats,
    list_custom_detectors,
    update_custom_detector_status,
)

logger = logging.getLogger("dunetrace.api.custom_detectors")
router = APIRouter(prefix="/v1/custom-detectors", tags=["Custom Detectors"])

_VALID_STATUSES = {"shadow", "active", "paused"}
_VALID_OPERATORS = {">=", "<=", ">", "<", "==", "!="}


class PreviewRequest(BaseModel):
    description: str
    agent_id: str = "*"

    @field_validator("description")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("description must not be empty")
        if len(v) > 1000:
            raise ValueError("description must be ≤ 1000 characters")
        return v


class CreateRequest(BaseModel):
    description: str
    agent_id: str = "*"
    config: dict  # structured config returned by the preview endpoint


class StatusUpdate(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _valid(cls, v: str) -> str:
        if v not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}")
        return v


def _validate_config(config: dict) -> None:
    """Raise HTTPException(422) if the detector config is structurally invalid."""
    name = config.get("detector_name")
    if not name or not name.startswith("CUSTOM_"):
        raise HTTPException(422, "config.detector_name must start with CUSTOM_")
    if config.get("requires_content"):
        raise HTTPException(422, "Cannot save a detector that requires content access")

    conditions = config.get("conditions")
    if not isinstance(conditions, list) or len(conditions) == 0:
        raise HTTPException(422, "config.conditions must be a non-empty list")

    for i, cond in enumerate(conditions):
        for field in ("metric", "operator", "threshold"):
            if field not in cond:
                raise HTTPException(422, f"conditions[{i}] is missing required field '{field}'")
        if cond["metric"] not in SUPPORTED_METRICS:
            raise HTTPException(
                422,
                f"conditions[{i}].metric '{cond['metric']}' is not a supported metric. "
                f"Valid values: {sorted(SUPPORTED_METRICS)}",
            )
        if cond["operator"] not in _VALID_OPERATORS:
            raise HTTPException(
                422,
                f"conditions[{i}].operator '{cond['operator']}' is invalid. "
                f"Valid operators: {sorted(_VALID_OPERATORS)}",
            )
        try:
            float(cond["threshold"])
        except (TypeError, ValueError):
            raise HTTPException(422, f"conditions[{i}].threshold must be a number")


@router.post("/preview")
async def preview_detector(body: PreviewRequest):
    """Translate plain-English description to structured config via LLM.

    Returns either:
      - {detector_name, conditions, severity, evidence_template, fix_template, requires_content: false}
      - {requires_content: true, reason: "..."}
    """
    try:
        config = await translate_description(body.description)
    except ValueError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("LLM translation failed: %s", exc)
        raise HTTPException(502, "Translation service unavailable — check LLM API key")
    return config


@router.get("")
async def list_detectors(agent_id: Optional[str] = None):
    """List all custom detectors, optionally filtered by agent_id."""
    return await list_custom_detectors(agent_id)


@router.post("", status_code=201)
async def create_detector(body: CreateRequest):
    """Save a new custom detector (always starts in shadow mode)."""
    _validate_config(body.config)
    detector = await create_custom_detector(
        agent_id=body.agent_id,
        name=body.config["detector_name"],
        description=body.description,
        config=body.config,
    )
    return detector


@router.get("/{detector_id}")
async def get_detector(detector_id: int):
    det = await get_custom_detector(detector_id)
    if not det:
        raise HTTPException(404, "Custom detector not found")
    return det


@router.get("/{detector_id}/shadow-stats")
async def shadow_stats(detector_id: int):
    """Return shadow evaluation results — how often the detector would have fired."""
    det = await get_custom_detector(detector_id)
    if not det:
        raise HTTPException(404, "Custom detector not found")
    return await get_custom_detector_shadow_stats(detector_id)


@router.patch("/{detector_id}")
async def update_detector(detector_id: int, body: StatusUpdate):
    """Activate, pause, or return a detector to shadow mode."""
    try:
        updated = await update_custom_detector_status(detector_id, body.status)
    except RuntimeError:
        raise HTTPException(503, "Database unavailable")
    if not updated:
        raise HTTPException(404, "Custom detector not found")
    return updated


@router.delete("/{detector_id}", status_code=204)
async def remove_detector(detector_id: int):
    try:
        deleted = await delete_custom_detector(detector_id)
    except RuntimeError:
        raise HTTPException(503, "Database unavailable")
    if not deleted:
        raise HTTPException(404, "Custom detector not found")
