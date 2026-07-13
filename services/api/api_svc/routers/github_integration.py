"""Per-org GitHub App config (Phase 4.3) — install flow + repos/reviewers.

Unlike Slack/Linear (Phase 4.1), there's no encrypted credential to store
here at all: the GitHub App itself is one Dunetrace-owned, operator-level
registration (GITHUB_APP_ID/GITHUB_APP_PRIVATE_KEY), and installation_id
(what distinguishes one org's installation from another's) isn't a secret —
it's just an identifier GitHub hands back after the customer installs the
App onto their chosen repos.

Flow:
  1. GET .../github/install-url  -> customer clicks it, GitHub shows the
     App's permission request for the customer's own repos.
  2. GitHub redirects to .../github/callback?installation_id=...&state=...
     — state round-trips org_id (same CSRF-safe pattern Linear's/Slack's
     own OAuth-style flows would use, applied here even though the App
     itself needs no OAuth token exchange).
  3. POST .../github  sets which repos/branches/reviewers this org wants
     Dunetrace to use, from among what the App is actually installed on.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_svc.auth import require_org
from api_svc.db.queries import (
    delete_org_github_integration,
    get_org_github_integration,
    set_org_github_config,
    upsert_org_github_installation,
)
from api_svc.github_app_auth import build_install_url

logger = logging.getLogger("dunetrace.api.github_integration")
router = APIRouter(prefix="/v1/orgs/integrations/github", tags=["GitHub Integration"])


class RepoConfig(BaseModel):
    repo: str
    base_branch: str = "main"


class GitHubConfigRequest(BaseModel):
    repos: List[RepoConfig]
    reviewers: List[str] = []


class GitHubIntegrationStatus(BaseModel):
    configured: bool
    installation_id: Optional[int] = None
    repos: List[RepoConfig] = []
    reviewers: List[str] = []


@router.get("/install-url", summary="Get the URL to install the Dunetrace GitHub App")
async def get_install_url(org_id: str = Depends(require_org)) -> dict:
    try:
        return {"install_url": build_install_url(state=org_id)}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/callback", include_in_schema=False)
async def install_callback(installation_id: int, state: str) -> dict:
    """GitHub redirects the customer's browser here after they install the
    App — there is no Dunetrace API key on this request at all (it isn't
    an authenticated API call, it's a browser redirect GitHub controls), so
    org_id necessarily comes from `state` here, not require_org. `state` is
    exactly the value install-url above was called with — GitHub passes it
    back unchanged, the same CSRF-safe round-trip Linear's/Slack's own
    OAuth-style flows would use."""
    await upsert_org_github_installation(state, installation_id)
    return {"installed": True, "org_id": state, "installation_id": installation_id}


@router.post(
    "", summary="Configure repos/branches/reviewers for this org's GitHub App installation"
)
async def set_github_config(
    body: GitHubConfigRequest, org_id: str = Depends(require_org)
) -> GitHubIntegrationStatus:
    ok = await set_org_github_config(
        org_id,
        [r.model_dump() for r in body.repos],
        body.reviewers,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail="No GitHub App installation found for this org — install it first.",
        )
    integration = await get_org_github_integration(org_id)
    return GitHubIntegrationStatus(
        configured=True,
        installation_id=integration["installation_id"],
        repos=integration["repos"],
        reviewers=integration["reviewers"],
    )


@router.get("", summary="Get this org's GitHub integration status")
async def get_github_config(org_id: str = Depends(require_org)) -> GitHubIntegrationStatus:
    integration = await get_org_github_integration(org_id)
    if integration is None:
        return GitHubIntegrationStatus(configured=False)
    return GitHubIntegrationStatus(
        configured=True,
        installation_id=integration["installation_id"],
        repos=integration["repos"],
        reviewers=integration["reviewers"],
    )


@router.delete("", summary="Remove this org's GitHub integration", status_code=204)
async def remove_github_config(org_id: str = Depends(require_org)):
    deleted = await delete_org_github_integration(org_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No GitHub integration configured.")
