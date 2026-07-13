"""Detector pack activation (Phase 1.0). A pack is a Dunetrace-owned bundle
of detector classes (voice, and future browser/coding/etc. packs) an org
activates as a whole — see packages/sdk-py/dunetrace/packs/base.py and
detector_svc/detectors.py::get_detectors() for how activation actually
changes which detectors run for that org's traffic.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_svc.auth import require_org
from api_svc.db.queries import (
    activate_pack,
    deactivate_pack,
    list_all_packs,
    list_org_enabled_packs,
    pack_exists,
)

logger = logging.getLogger("dunetrace.api.packs")
router = APIRouter(tags=["Packs"])


class PackInfo(BaseModel):
    name: str
    description: str
    detector_names: List[str]


class EnabledPack(BaseModel):
    pack_name: str
    enabled_at: str
    enabled_by: Optional[str] = None


@router.get("/v1/packs", response_model=List[PackInfo], summary="List all available packs")
async def get_all_packs() -> List[PackInfo]:
    """No auth — this is a static catalog, not org-scoped data."""
    packs = await list_all_packs()
    return [
        PackInfo(name=p["name"], description=p["description"], detector_names=p["detector_names"])
        for p in packs
    ]


@router.get(
    "/v1/orgs/packs",
    response_model=List[EnabledPack],
    summary="List this org's activated packs",
)
async def get_org_packs(org_id: str = Depends(require_org)) -> List[EnabledPack]:
    # org_id derived from require_org(), not a URL param — prevents cross-org
    # config manipulation by editing the URL. Every org-scoped endpoint in
    # this codebase follows this same rule; see scripts/check_endpoint_
    # conventions.py, which greps for violations of it.
    rows = await list_org_enabled_packs(org_id)
    return [
        EnabledPack(
            pack_name=r["pack_name"],
            enabled_at=r["enabled_at"].isoformat(),
            enabled_by=r["enabled_by"],
        )
        for r in rows
    ]


@router.post(
    "/v1/orgs/packs/{pack_name}",
    summary="Activate a pack for this org",
    status_code=201,
)
async def post_activate_pack(pack_name: str, org_id: str = Depends(require_org)) -> dict:
    # org_id derived from require_org(), not a URL param — see comment on
    # get_org_packs above. Authorization here is the same org-scoped API key
    # every other write endpoint in this codebase accepts (POST /v1/policies,
    # POST /v1/orgs/integrations/github, etc.) — there is no admin/owner role
    # concept anywhere in this codebase's auth model (api_keys has no role
    # column), so "requires admin" isn't enforceable today without inventing
    # a role system disproportionate to a single per-org boolean toggle. Any
    # valid key for this org can activate/deactivate a pack, matching every
    # other org-scoped write endpoint.
    if not await pack_exists(pack_name):
        raise HTTPException(status_code=404, detail=f"Unknown pack: {pack_name!r}")

    # enabled_by: no mechanism in this codebase's auth model identifies
    # *which* key/caller made a request beyond org_id (require_org resolves
    # only an org_id, never a key id or user identity) — recording the raw
    # Authorization header here would leak a live secret into a plain-text
    # audit column, so this stays None until/unless the auth model grows a
    # real caller-identity concept. Not silently faked.
    await activate_pack(org_id, pack_name, enabled_by=None)
    return {"pack_name": pack_name, "enabled": True}


@router.delete(
    "/v1/orgs/packs/{pack_name}",
    summary="Deactivate a pack for this org",
    status_code=204,
)
async def delete_deactivate_pack(pack_name: str, org_id: str = Depends(require_org)):
    # org_id derived from require_org(), not a URL param — see comment on
    # get_org_packs above.
    deleted = await deactivate_pack(org_id, pack_name)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Pack {pack_name!r} is not activated for this org."
        )
