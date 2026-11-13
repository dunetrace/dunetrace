"""Authentication dependencies for FastAPI routes.

Two questions are answered here, and they are deliberately separate:

* ``require_org``   — *who* is calling (an org id). Every authenticated route
  carries it, usually at router level.
* ``require_scope`` — *what* the credential may do. A write route that manages
  keys, policies, integrations, packs or org settings adds
  ``Depends(require_scope("admin"))``; the approval decision endpoint adds
  ``require_scope("approve")``. Reads stay ingest-accessible.

Both resolve the caller exactly once per request: the first dependency to run
memoises ``(org_id, scopes)`` on ``request.state``, so a router-level
``require_org`` stacked with a route-level ``require_scope`` costs one
``api_keys`` lookup, not two.

Scope semantics live in ``dunetrace_schemas.scopes``: ``admin`` implies
everything, nothing else implies anything, and an absent or unknown scope list
fails closed to ingest-only.

``AUTH_MODE=dev`` (local Docker only) disables authentication entirely and
resolves every request to the default org **with admin** — the local
quickstart has no key to present, so withholding scopes there would break the
dashboard's config pages without protecting anything. It is the one place a
missing credential means "admin", and it is opt-in (an unset AUTH_MODE is
``prod``).
"""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Header, Request, status

from api_svc.config import settings
from api_svc.db.queries import verify_api_key_with_scopes
from dunetrace_schemas.scopes import ADMIN, has_scope, normalise

#: ``request.state`` attribute holding the memoised ``(org_id, scopes)``.
_STATE_ATTR = "dunetrace_caller"


def is_trusted(request: Request) -> bool:
    """True when the request carries a valid internal token from an upstream auth layer.

    INTERNAL_TOKEN must be non-empty (not set in dev/self-hosted) for this path to
    activate, preventing accidental trust escalation. Mirrors ingest_svc.auth.is_trusted.
    """
    if not settings.INTERNAL_TOKEN:
        return False
    token = request.headers.get("x-internal-token", "")
    return secrets.compare_digest(token, settings.INTERNAL_TOKEN)


def scopes_from_trusted_header(header: Optional[str]) -> tuple:
    """Scopes an upstream gateway granted, from its ``x-scopes`` header.

    Comma-separated, case-insensitive, unknown values dropped — the same
    ``normalise`` the key store applies, so ``ADMIN,bogus`` is ``("admin",)``
    and ``bogus`` alone is ingest-only. A missing header *and* an explicitly
    empty one both resolve to ``DEFAULT_SCOPES`` (ingest): the gateway has to
    say ``admin`` to get admin. It never defaults up.
    """
    if not header:
        return normalise(None)
    return normalise(header.split(","))


async def _resolve_caller_uncached(request: Request, authorization: Optional[str]) -> tuple:
    if is_trusted(request):
        org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id", "")
        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-org-id",
            )
        # An upstream auth layer has already decided who this is; it carries the
        # scopes it granted, defaulting to ingest-only rather than to everything.
        return (org_id, scopes_from_trusted_header(request.headers.get("x-scopes")))

    if settings.is_dev:
        # Auth is disabled in dev mode by design — see the module docstring.
        return ("default", (ADMIN,))

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # Support "Bearer <key>" or just "<key>"
    parts = authorization.strip().split()
    api_key = parts[-1] if parts else authorization

    resolved = await verify_api_key_with_scopes(api_key)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    return resolved


async def _resolve_caller(request: Request, authorization: Optional[str]) -> tuple:
    """``(org_id, scopes)`` for this request, resolved at most once.

    Memoised on ``request.state`` — per request, so nothing leaks across
    callers. The ``isinstance`` guard matters: tests hand these dependencies a
    ``MagicMock`` request, whose ``state`` returns a mock for any attribute.
    """
    cached = getattr(getattr(request, "state", None), _STATE_ATTR, None)
    if isinstance(cached, tuple):
        return cached
    resolved = await _resolve_caller_uncached(request, authorization)
    setattr(request.state, _STATE_ATTR, resolved)
    return resolved


async def require_org(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """FastAPI dependency that resolves the caller's org_id. Raises 401 if invalid.

    Trusted path: an upstream auth layer has already validated the caller and forwards
    identity via x-internal-token + x-org-id headers (x-customer-id accepted as a
    legacy fallback) — no DB lookup here.

    Direct path (self-hosted / dev): validates the Authorization header against this
    service's own api_keys table. In dev mode (AUTH_MODE=dev), auth is skipped
    entirely and the caller is scoped to the 'default' org. Header format:
    "Bearer dt_live_..." or just the key.

    Answers only *who*; a route that needs a particular capability adds
    ``require_scope``.
    """
    org_id, _scopes = await _resolve_caller(request, authorization)
    return org_id


async def resolve_scopes(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> tuple:
    """(org_id, scopes) for the caller. Same resolution order as require_org."""
    return await _resolve_caller(request, authorization)


def require_scope(scope: str):
    """Dependency factory: resolve the org and assert the key carries `scope`.

    Returns org_id, so a route can swap `Depends(require_org)` for
    `Depends(require_scope("admin"))` without changing its signature.
    """

    async def _dependency(resolved: tuple = Depends(resolve_scopes)) -> str:
        org_id, scopes = resolved
        if not has_scope(scopes, scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"This API key lacks the {scope!r} scope. Agent keys are "
                    f"issued with 'ingest' only — this operation needs a "
                    f"credential the agent runtime does not hold."
                ),
            )
        return org_id

    # Lets a test walk the route table and prove every write route names a
    # scope, without parsing source — closures are otherwise anonymous.
    _dependency.required_scope = scope  # type: ignore[attr-defined]
    return _dependency
