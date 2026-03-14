"""Authentication dependency for FastAPI routes."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Header, status

from api_svc.config import settings
from api_svc.db.queries import verify_api_key


async def require_customer(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """
    FastAPI dependency: extract API key from Authorization header and verify.
    Returns customer_id if valid, raises 401 if not.

    In dev mode (AUTH_MODE=dev), auth is skipped entirely.
    Header format: "Bearer dt_live_..." or "dt_dev_..."
    """
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
