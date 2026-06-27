from __future__ import annotations

import secrets

from fastapi import Request

from ingest_svc.config import settings


def is_trusted(request: Request) -> bool:
    """True when the request carries a valid internal token from the auth service.

    INTERNAL_TOKEN must be non-empty (not set in dev) for this path to activate,
    preventing accidental trust escalation in local environments.
    """
    if not settings.INTERNAL_TOKEN:
        return False
    token = request.headers.get("x-internal-token", "")
    return secrets.compare_digest(token, settings.INTERNAL_TOKEN)
