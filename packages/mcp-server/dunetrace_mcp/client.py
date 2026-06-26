"""Thin synchronous wrapper around the Dunetrace Customer API."""

from __future__ import annotations

import os
from typing import Any, NoReturn

import httpx

_DEFAULT_URL = "http://localhost:8002"
_DEFAULT_KEY = "dt_dev_test"


def _api_url() -> str:
    return os.environ.get("DUNETRACE_API_URL", _DEFAULT_URL).rstrip("/")


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
    url = _api_url() + path
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
    url = _api_url() + path
    with httpx.Client(timeout=15) as c:
        try:
            r = c.post(url, headers=_headers(), json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            _raise_http(e)


def patch(path: str, body: dict | None = None) -> Any:
    url = _api_url() + path
    with httpx.Client(timeout=15) as c:
        try:
            r = c.patch(url, headers=_headers(), json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            _raise_http(e)


def delete(path: str) -> Any:
    url = _api_url() + path
    with httpx.Client(timeout=15) as c:
        try:
            r = c.delete(url, headers=_headers())
            r.raise_for_status()
            if r.status_code == 204:
                return {}
            return r.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            _raise_http(e)
