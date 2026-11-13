"""Thin synchronous wrapper around the Dunetrace Customer API."""

from __future__ import annotations

import os
import re
from typing import Any, NoReturn
from urllib.parse import quote

import httpx

_DEFAULT_URL = "http://localhost:8002"
_DEFAULT_KEY = "dt_dev_test"


def _api_url() -> str:
    return os.environ.get("DUNETRACE_API_URL", _DEFAULT_URL).rstrip("/")


def seg(value: Any) -> str:
    """Percent-encode one caller-supplied value for use as a single path segment.

    Every id these tools take is attacker-influenced. The model driving this
    MCP server reads untrusted agent output — signal evidence, tool arguments,
    LLM text — so "the run id" can be whatever that text says it is, and the
    request it produces carries the operator's own bearer token (admin, under
    AUTH_MODE=dev).

    Two properties of the URL layer make an unescaped id a request-forgery
    primitive:

      * httpx resolves relative path segments before sending, so
        f"/v1/runs/{run_id}" with run_id="../../v1/keys" is literally
        GET /v1/keys — the read-only-gated server happily hands the model the
        org's API keys.
      * "?" starts a query string and "#" a fragment, so an id can also inject
        or truncate parameters.

    quote(safe="") handles "/", "?", "#", "&", ";" and whitespace. It leaves
    "." alone (unreserved), so a segment that *is* "." or ".." — which httpx
    would still resolve away — has its dots encoded explicitly.

    A legitimate id keeps working: "run/2026-01" round-trips as "run%2F2026-01",
    which Starlette decodes back to "run/2026-01" for the path parameter.
    """
    encoded = quote(str(value), safe="")
    if encoded in (".", ".."):
        encoded = encoded.replace(".", "%2E")
    return encoded


# A path element that is "." or ".." — the two httpx resolves away. Anything
# that reached here through seg() cannot match.
_RELATIVE_SEGMENT = re.compile(r"(?:^|/)\.{1,2}(?:/|$)")


def _url(path: str) -> str:
    """Join the API base to `path`, refusing one that was not escaped.

    Backstop for the rule above: every interpolated segment goes through
    seg(), and a call site that forgets gets a loud RuntimeError instead of a
    silently redirected, credentialed request. Query parameters belong in the
    `params` kwarg, never in the path.
    """
    if not path.startswith("/"):
        raise RuntimeError(f"API path must be absolute, got {path!r}")
    if "?" in path or "#" in path or _RELATIVE_SEGMENT.search(path):
        raise RuntimeError(
            f"Refusing unescaped API path {path!r} — "
            "interpolate ids with client.seg() and pass query parameters as kwargs."
        )
    return _api_url() + path


def _headers() -> dict[str, str]:
    key = os.environ.get("DUNETRACE_API_KEY", _DEFAULT_KEY)
    return {"Authorization": f"Bearer {key}"}


def _raise_http(e: httpx.ConnectError | httpx.HTTPStatusError) -> NoReturn:
    if isinstance(e, httpx.ConnectError):
        raise RuntimeError(
            f"Dunetrace API unreachable at {_api_url()}. "
            "Is the backend running? (docker compose up -d)"
        ) from e
    detail = ""
    try:
        detail = e.response.json().get("detail", "")
    except Exception:
        detail = e.response.text[:200]
    raise RuntimeError(
        f"API error {e.response.status_code}: {detail or e.response.text[:200]}"
    ) from e


def get(path: str, **params: Any) -> Any:
    url = _url(path)
    with httpx.Client(timeout=15) as c:
        try:
            r = c.get(
                url,
                headers=_headers(),
                params={k: v for k, v in params.items() if v is not None},
            )
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            _raise_http(e)


def post(path: str, body: dict | None = None) -> Any:
    url = _url(path)
    with httpx.Client(timeout=15) as c:
        try:
            r = c.post(url, headers=_headers(), json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            _raise_http(e)


def patch(path: str, body: dict | None = None) -> Any:
    url = _url(path)
    with httpx.Client(timeout=15) as c:
        try:
            r = c.patch(url, headers=_headers(), json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            _raise_http(e)


def delete(path: str) -> Any:
    url = _url(path)
    with httpx.Client(timeout=15) as c:
        try:
            r = c.delete(url, headers=_headers())
            r.raise_for_status()
            if r.status_code == 204:
                return {}
            return r.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            _raise_http(e)
