"""Minimal ElevenLabs API client used by api_svc, for one job only: validating
a customer-supplied API key at config time (Phase 4.1).

api_svc never polls ElevenLabs. That is elevenlabs_worker's job (Phase 4.3),
which has its own richer client. This module exists so POST
/v1/orgs/integrations/elevenlabs can reject a bad key immediately, at save
time, instead of storing it and letting the poller fail silently five minutes
later. The two clients are deliberately separate: they live in separate
services and share nothing but the base URL and header name, which are stable.

The base URL and auth header were verified against the live ElevenLabs API,
not just the docs: GET https://api.elevenlabs.io/v1/history with header
`xi-api-key: <key>`.
"""

from __future__ import annotations

import httpx

ELEVENLABS_API_BASE = "https://api.elevenlabs.io"
# Validate against the history endpoint specifically, not /v1/user: it is the
# exact capability the poller needs (read generation history), so a key that
# authenticates but lacks history access is caught here rather than later.
_VALIDATE_PATH = "/v1/history"
_VALIDATE_TIMEOUT_SECS = 10.0


class ElevenLabsAuthError(Exception):
    """The key was reached but rejected by ElevenLabs (401/403), or the request
    was otherwise refused in a way that means the key will not work. Surfaced to
    the customer as a 400 so they fix the key."""


class ElevenLabsUnreachable(Exception):
    """ElevenLabs could not be reached, or returned a server-side error, so we
    cannot tell whether the key is valid. Surfaced as a 502 so the customer
    knows to retry rather than assuming their key is wrong."""


async def validate_api_key(api_key: str) -> None:
    """Make one authenticated request to confirm the key works and can read
    generation history. Returns None on success; raises ElevenLabsAuthError if
    the key is rejected, ElevenLabsUnreachable if ElevenLabs cannot be reached
    or fails on its own side. Never returns the response body: this is a
    pass/fail check, nothing is stored from it."""
    url = f"{ELEVENLABS_API_BASE}{_VALIDATE_PATH}"
    try:
        async with httpx.AsyncClient(timeout=_VALIDATE_TIMEOUT_SECS) as client:
            resp = await client.get(
                url,
                params={"page_size": 1},
                headers={"xi-api-key": api_key},
            )
    except httpx.HTTPError as exc:
        raise ElevenLabsUnreachable(str(exc)) from exc

    if resp.status_code in (401, 403):
        raise ElevenLabsAuthError("ElevenLabs rejected the API key.")
    if resp.status_code >= 500:
        raise ElevenLabsUnreachable(f"ElevenLabs returned HTTP {resp.status_code}.")
    if resp.status_code >= 400:
        # A 4xx that is not 401/403 (e.g. 400/422) still means this request as
        # authenticated will not succeed, so treat it as a key/permission
        # problem the customer must fix, not a transient outage.
        raise ElevenLabsAuthError(f"ElevenLabs rejected the request (HTTP {resp.status_code}).")
    # 2xx: the key authenticates and can read history.
