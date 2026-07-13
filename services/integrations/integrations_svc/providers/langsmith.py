"""
LangSmith pull integration (Phase 2.2). Verified against LangSmith's live,
public, unauthenticated OpenAPI spec (api.smith.langchain.com/openapi.json)
AND a real LangSmith project (not just the spec — the spec's "all params
optional" was misleading, see below):

- Auth: X-Api-Key header (confirmed from the spec's securitySchemes — a
  genuinely different model from Langfuse's HTTP Basic Auth).
- GET {base_url}/api/v1/feedback requires at least one of `run`, `key`, or
  `session` as a query param — the live API 400s without one
  ("At least one of 'run', 'key', or 'session' must be provided"), even
  though the OpenAPI spec marks all three as individually optional (a
  cross-field constraint the spec alone doesn't capture). This provider
  always supplies `session` (LangSmith's internal name for a project).
- A project's `session` id isn't its name — resolved once via
  GET {base_url}/api/v1/sessions?name={project_name}, cached on the
  instance (a project's id doesn't change during this worker's lifetime).
- Feedback fields used: id, run_id, session_id, trace_id, key, score,
  value, comment, created_at. `created_at` is naive ISO (no timezone
  suffix at all, confirmed against a real response) — implicitly UTC.
- Response shape is a plain JSON array, NOT Langfuse's {"data": [...],
  "meta": {...}} envelope — pagination is limit/offset, not page-based.
- Real, measured indexing lag: ~5s for a freshly-created feedback item to
  appear in the list endpoint (faster than Langfuse's ~20-30s, but this
  provider still uses the same generous worker.py-wide overlap window).
- No verified private-dashboard URL pattern exists (only a *public*-share
  link format, `{ui_host}/public/{trace_id}/r`, which would be wrong for
  the vast majority of real customer traces that aren't explicitly shared)
  — source_url is always None here rather than a fabricated, likely-broken
  link. Revisit if LangSmith's feedback response ever includes enough
  (organization id + a documented private URL scheme) to build one for real.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from integrations_svc.providers.base import ExternalEvaluation

logger = logging.getLogger("dunetrace.integrations.langsmith")

_PAGE_LIMIT = 100
_MAX_PAGES = 50  # hard safety cap — a runaway loop must not poll forever


def _parse_timestamp(raw: str | None) -> float:
    if not raw:
        return datetime.now(timezone.utc).timestamp()
    # LangSmith's created_at has no timezone suffix at all (confirmed live)
    # — a naive datetime.fromisoformat() would assume the SYSTEM's local
    # timezone, which is wrong; the value is UTC, just not marked as such.
    # audit Finding 27: never raise on a malformed/odd-typed timestamp.
    try:
        if not isinstance(raw, str):
            return datetime.now(timezone.utc).timestamp()
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).timestamp()


def _format_min_created_at(since: float) -> str:
    return datetime.fromtimestamp(since, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _numeric_score(score) -> float | None:
    # score's schema is anyOf[number, integer, boolean, null] — bool is an
    # int subclass in Python, so check it explicitly rather than let
    # isinstance(score, (int, float)) silently accept True/False as 1/0
    # without a deliberate decision to do so.
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _string_value(score, value) -> str | None:
    if _numeric_score(score) is not None:
        return None
    if isinstance(value, str):
        return value
    if value is not None:
        return str(value)
    return None


class LangSmithProvider:
    name = "langsmith"

    def __init__(self, endpoint_url: str, api_key: str, project_name: str):
        self._base = endpoint_url.rstrip("/")
        self._headers = {"X-Api-Key": api_key}
        self._project_name = project_name
        self._session_id: str | None = None  # resolved lazily, cached on this instance

    async def _resolve_session_id(self, client: httpx.AsyncClient) -> str | None:
        if self._session_id is not None:
            return self._session_id
        resp = await client.get(
            f"{self._base}/api/v1/sessions", params={"name": self._project_name}
        )
        resp.raise_for_status()
        sessions = resp.json()
        if not sessions:
            return None
        self._session_id = sessions[0]["id"]
        return self._session_id

    async def fetch_new_evaluations(self, since: float) -> list[ExternalEvaluation]:
        min_created_at = _format_min_created_at(since)
        evaluations: list[ExternalEvaluation] = []

        async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
            session_id = await self._resolve_session_id(client)
            if session_id is None:
                logger.warning(
                    "LangSmith project %r not found — skipping this poll cycle",
                    self._project_name,
                )
                return []

            offset = 0
            for _ in range(_MAX_PAGES):
                resp = await client.get(
                    f"{self._base}/api/v1/feedback",
                    params={
                        "session": session_id,
                        "min_created_at": min_created_at,
                        "limit": _PAGE_LIMIT,
                        "offset": offset,
                    },
                )
                resp.raise_for_status()
                items = resp.json()

                for fb in items:
                    # audit Finding 27: skip a malformed record, don't sink the batch.
                    try:
                        trace_id = fb.get("trace_id")
                        external_id = fb.get("id")
                        if not trace_id or not external_id:
                            continue
                        evaluations.append(
                            ExternalEvaluation(
                                external_id=external_id,
                                trace_id=trace_id,
                                name=fb.get("key") or "feedback",
                                value=_numeric_score(fb.get("score")),
                                string_value=_string_value(fb.get("score"), fb.get("value")),
                                comment=fb.get("comment"),
                                timestamp=_parse_timestamp(fb.get("created_at")),
                                source_url=None,
                            )
                        )
                    except Exception as exc:
                        logger.warning("langsmith: skipping malformed feedback record: %s", exc)
                        continue

                if len(items) < _PAGE_LIMIT:
                    break
                offset += _PAGE_LIMIT

        return evaluations
