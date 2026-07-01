"""Authentication dependency for FastAPI routes."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Header, Request, status

from api_svc.config import settings
from api_svc.db.queries import verify_api_key


def is_trusted(request: Request) -> bool:
    """True when the request carries a valid internal token from an upstream auth layer.

    INTERNAL_TOKEN must be non-empty (not set in dev/self-hosted) for this path to
    activate, preventing accidental trust escalation. Mirrors ingest_svc.auth.is_trusted.
    """
    if not settings.INTERNAL_TOKEN:
        return False
    token = request.headers.get("x-internal-token", "")
    return secrets.compare_digest(token, settings.INTERNAL_TOKEN)


async def require_customer(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """FastAPI dependency that resolves the caller's customer_id. Raises 401 if invalid.

    Trusted path: an upstream auth layer has already validated the caller and forwards
    identity via x-internal-token + x-customer-id headers — no DB lookup here.

    Direct path (self-hosted / dev): validates the Authorization header against this
    service's own api_keys table, as before. In dev mode (AUTH_MODE=dev), auth is
    skipped entirely. Header format: "Bearer dt_live_..." or just the key.
    """
    if is_trusted(request):
        customer_id = request.headers.get("x-customer-id", "")
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-customer-id",
            )
        return customer_id

    if settings.is_dev:
        return "dev_customer"

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # Support "Bearer <key>" or just "<key>"
    parts = authorization.strip().split()
    api_key = parts[-1] if parts else authorization

    customer_id = await verify_api_key(api_key)
    if customer_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )

    return customer_id
