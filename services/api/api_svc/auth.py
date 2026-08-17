"""Authentication dependency for FastAPI routes."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Header, Request, status

from api_svc.config import settings
from api_svc.db.queries import verify_api_key, verify_api_key_with_scopes


def is_trusted(request: Request) -> bool:
    """True when the request carries a valid internal token from an upstream auth layer.

    INTERNAL_TOKEN must be non-empty (not set in dev/self-hosted) for this path to
    activate, preventing accidental trust escalation. Mirrors ingest_svc.auth.is_trusted.
    """
    if not settings.INTERNAL_TOKEN:
        return False
    token = request.headers.get("x-internal-token", "")
    return secrets.compare_digest(token, settings.INTERNAL_TOKEN)


async def require_org(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """FastAPI dependency that resolves the caller's org_id. Raises 401 if invalid.

    Trusted path: an upstream auth layer has already validated the caller and forwards
    identity via x-internal-token + x-org-id headers (x-customer-id accepted as a
    legacy fallback) — no DB lookup here.

    Direct path (self-hosted / dev): validates the Authorization header against this
    service's own api_keys table, as before. In dev mode (AUTH_MODE=dev), auth is
    skipped entirely and the caller is scoped to the 'default' org. Header format:
    "Bearer dt_live_..." or just the key.
    """
    if is_trusted(request):
        org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id", "")
        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-org-id",
            )
        return org_id

    if settings.is_dev:
        return "default"

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # Support "Bearer <key>" or just "<key>"
    parts = authorization.strip().split()
    api_key = parts[-1] if parts else authorization

    org_id = await verify_api_key(api_key)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )

    return org_id


async def resolve_scopes(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> tuple:
    """(org_id, scopes) for the caller. Same resolution order as require_org."""
    from dunetrace_schemas.scopes import ADMIN, DEFAULT_SCOPES

    if is_trusted(request):
        org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id", "")
        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-org-id",
            )
        # An upstream auth layer has already decided who this is; it carries the
        # scopes it granted, defaulting to ingest-only rather than to everything.
        header = request.headers.get("x-scopes", "")
        scopes = tuple(s.strip() for s in header.split(",") if s.strip()) or DEFAULT_SCOPES
        return (org_id, scopes)

    if settings.is_dev:
        return ("default", (ADMIN,))

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    parts = authorization.strip().split()
    api_key = parts[-1] if parts else authorization
    resolved = await verify_api_key_with_scopes(api_key)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    return resolved


def require_scope(scope: str):
    """Dependency factory: resolve the org and assert the key carries `scope`.

    Returns org_id, so a route can swap `Depends(require_org)` for
    `Depends(require_scope("approve"))` without changing its signature.
    """

    async def _dependency(resolved: tuple = Depends(resolve_scopes)) -> str:
        from dunetrace_schemas.scopes import has_scope

        org_id, scopes = resolved
        if not has_scope(scopes, scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"This API key lacks the {scope!r} scope. Agent keys are "
                    f"issued with 'ingest' only — a decision must come from a "
                    f"credential the agent runtime does not hold."
                ),
            )
        return org_id

    return _dependency
