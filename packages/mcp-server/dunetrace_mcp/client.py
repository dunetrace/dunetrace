"""Thin synchronous wrapper around the Dunetrace Customer API."""

from __future__ import annotations

import os
from typing import Any

import httpx

_DEFAULT_URL = "http://localhost:8002"
_DEFAULT_KEY = "dt_dev_test"


def _api_url() -> str:
    return os.environ.get("DUNETRACE_API_URL", _DEFAULT_URL).rstrip("/")


def _headers() -> dict[str, str]:
    key = os.environ.get("DUNETRACE_API_KEY", _DEFAULT_KEY)
    return {"Authorization": f"Bearer {key}"}


def get(path: str, **params: Any) -> Any:
    url = _api_url() + path
    with httpx.Client(timeout=15) as c:
        r = c.get(
            url,
            headers=_headers(),
            params={k: v for k, v in params.items() if v is not None},
        )
        r.raise_for_status()
        return r.json()
