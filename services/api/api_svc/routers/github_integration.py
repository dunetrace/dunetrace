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
    consume_github_install_state,
    mint_github_install_state,
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
        # A fresh single-use nonce per click, not the org_id: this endpoint is
        # authenticated, the callback below cannot be, so the state is the only
        # thing that carries org identity across GitHub's redirect.
        state = await mint_github_install_state(org_id)
        return {"install_url": build_install_url(state=state)}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/callback", include_in_schema=False)
async def install_callback(installation_id: int, state: str) -> dict:
    """GitHub redirects the customer's browser here after they install the App.
    There is no Dunetrace API key on this request — it is a browser redirect
    GitHub controls, not an authenticated API call — so `state` is the only
    thing that establishes which org is installing.

    That makes the state a capability, and it is treated as one: a random
    single-use nonce minted by the authenticated /install-url call above,
    consumed here, and expiring after 30 minutes. It previously carried the raw
    org_id, which is stable and guessable rather than a nonce, so any caller
    could bind any org to any installation_id — in either direction: redirecting
    a victim org's fix PRs into an attacker's repo, or binding an attacker's org
    to a victim's (enumerable) installation_id to gain write access to their
    private repos."""
    org_id = await consume_github_install_state(state)
    if org_id is None:
        # Deliberately not distinguishing unknown / expired / already-used:
        # all three mean the same thing to a legitimate caller, and telling
        # them apart would confirm which states exist.
        raise HTTPException(
            status_code=400,
            detail="This install link is invalid or has expired. Start the install again.",
        )
    await upsert_org_github_installation(org_id, installation_id)
    return {"installed": True, "org_id": org_id, "installation_id": installation_id}


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
