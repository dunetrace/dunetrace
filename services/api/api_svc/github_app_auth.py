"""
GitHub App authentication (Phase 4.3). Two-step token exchange, standard
GitHub App flow:

  1. Sign a short-lived JWT with the App's own private key (RS256,
     iss=app_id, iat/exp within GitHub's required ~10min window) —
     proves "this request comes from the Dunetrace App itself."
  2. Exchange that JWT for an installation access token, scoped to one
     specific installation (one customer's chosen repos) — this is the
     token actually used for repo API calls (contents/pulls/issues).

Installation tokens are short-lived (~1 hour) and are not cached here —
PR creation is infrequent enough (one signal at a time) that fetching a
fresh token per call is simpler and avoids stale-token edge cases.

The App itself is a one-time, operator-level registration in GitHub's own
UI (Settings > Developer settings > GitHub Apps) — GITHUB_APP_ID and
GITHUB_APP_PRIVATE_KEY come from that registration, shared across every
org's installation. Real GitHub App permissions requested at registration
time: contents:write, pull_requests:write, issues:write (GitHub's own
taxonomy — there is no separate "read:code" scope; contents:write already
implies read).
"""

from __future__ import annotations

import time

import httpx
import jwt

from api_svc.config import settings

_GITHUB_API = "https://api.github.com"


def generate_app_jwt() -> str:
    if not settings.github_app_configured:
        raise ValueError(
            "GitHub App not configured — set GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY."
        )
    now = int(time.time())
    payload = {
        "iat": now - 30,  # small backdate for clock skew tolerance, per GitHub's own docs
        "exp": now + (9 * 60),  # GitHub's max is 10 minutes
        "iss": settings.GITHUB_APP_ID,
    }
    return jwt.encode(payload, settings.GITHUB_APP_PRIVATE_KEY, algorithm="RS256")


async def get_installation_token(installation_id: int) -> str:
    """Exchanges the App JWT for a short-lived token scoped to one
    installation. Raises httpx.HTTPStatusError on failure (invalid
    installation_id, revoked installation, etc.)."""
    app_jwt = generate_app_jwt()
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{_GITHUB_API}/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        r.raise_for_status()
        return r.json()["token"]


def build_install_url(state: str) -> str:
    """The URL a customer clicks to install the Dunetrace GitHub App onto
    their chosen repos. `state` round-trips org_id through GitHub's install
    flow — GitHub passes it back unchanged to the callback URL, the
    standard CSRF-safe pattern for this kind of redirect."""
    if not settings.GITHUB_APP_SLUG:
        raise ValueError("GITHUB_APP_SLUG is not configured.")
    return f"https://github.com/apps/{settings.GITHUB_APP_SLUG}/installations/new?state={state}"
