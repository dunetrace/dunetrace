"""
Bounded request-body reads for the ingest service.

Every JSON endpoint here buffers its body in full before it is parsed: FastAPI
does it for the route, and main.py's middlewares do it earlier still to find
the rate-limit bucket and org context. A size cap applied *after* that
buffering protects nothing, so the cap lives in the reader itself:
Content-Length is checked before a single byte is read, and the stream is
capped as it arrives so a missing or lying Content-Length (chunked transfer
encoding, a hostile client) cannot slip an oversized body past.

Used by main.py's outermost middleware for POST /v1/ingest and /v1/deploy
(INGEST_MAX_BODY_BYTES) and by the OTLP receiver route (OTLP_MAX_BODY_BYTES).
"""

from __future__ import annotations

from starlette.requests import Request


class BodyTooLarge(Exception):
    """A request body exceeds a configured byte limit. Callers map it to 413.

    ``declared`` is the Content-Length header as sent (None when absent or not
    an integer); ``received`` is how many bytes had arrived when the cap was
    hit (0 when the header alone was enough to reject, so nothing was read).
    Neither carries body content, so both are safe to log.
    """

    def __init__(self, limit: int = 0, declared: int | None = None, received: int = 0) -> None:
        self.limit = limit
        self.declared = declared
        self.received = received
        super().__init__(f"request body exceeds {limit} bytes")


def declared_content_length(request: Request) -> int | None:
    """Content-Length as an int, or None when absent or malformed. A malformed
    value is left for the ASGI server to reject; the reader then relies on the
    streaming cap, which never trusts the header anyway."""
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def read_bounded_body(request: Request, limit: int) -> bytes:
    """Read the request body, refusing to buffer more than ``limit`` bytes.

    Raises BodyTooLarge either at once, when Content-Length exceeds the limit
    (before any of the body is read), or mid-stream the moment the bytes
    received exceed it. Either way no more than ``limit`` bytes plus one
    chunk are ever held.

    On success the bytes are cached on the request exactly as
    ``await request.body()`` would have cached them, so ``request.body()``,
    ``request.json()`` and any app downstream all see the body. That last
    part is what makes this safe to call from a BaseHTTPMiddleware:
    Starlette's ``_CachedRequest`` forwards a *cached* body to the wrapped
    app, but a stream that was merely consumed is forwarded as an empty body
    (``starlette/middleware/base.py``, ``wrapped_receive``). The cache
    attribute is Starlette's own ``_body`` — the same one ``Request.body()``
    writes and ``Request.stream()`` reads. The ingest tests exercise the full
    middleware -> route path, so a Starlette upgrade that renamed it would
    fail every 202 test rather than silently blank the body.
    """
    declared = declared_content_length(request)
    if declared is not None and declared > limit:
        raise BodyTooLarge(limit=limit, declared=declared, received=0)

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise BodyTooLarge(limit=limit, declared=declared, received=total)
        chunks.append(chunk)
    body = b"".join(chunks)
    request._body = body
    return body
