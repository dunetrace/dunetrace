"""
Langfuse pull integration (Phase 2.1). Verified directly against a real
Langfuse project's API (not just docs, which gave conflicting signals about
whether a GET scores endpoint even exists — see BACKLOG.md):

- GET {base_url}/api/public/scores?fromTimestamp=<ISO8601>&limit=<n>&page=<n>
- HTTP Basic Auth: public key as username, secret key as password.
- Response: {"data": [Score, ...], "meta": {"page", "limit", "totalItems", "totalPages"}}.
- Score fields used here: id, traceId, name, value, stringValue, comment,
  timestamp, projectId. (Also present but unused: environment, source,
  authorUserId, dataType, configId, queueId, executionTraceId, observationId.)
- Dashboard click-through URL is {base_url}/project/{projectId}/traces/{traceId}
  (confirmed via a real trace object's own htmlPath field, same shape).
- Real, measured indexing lag: a freshly-created score can take ~20-30s to
  appear in this list endpoint (individual by-id fetches are faster) — see
  worker.py's _OVERLAP_SECS, which is deliberately much larger than that
  measured lag for margin.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from integrations_svc.providers.base import ExternalEvaluation

logger = logging.getLogger("dunetrace.integrations.langfuse")

_PAGE_LIMIT = 100
_MAX_PAGES = 50  # hard safety cap — a runaway loop must not poll forever


def _parse_timestamp(raw: str | None) -> float:
    if not raw:
        return datetime.now(timezone.utc).timestamp()
    # Langfuse emits "...Z" — normalize to a form every supported Python
    # version's fromisoformat() accepts (3.11+ handles "Z" natively, but
    # don't rely on that here).
    # audit Finding 27: never raise on a malformed/odd-typed timestamp — a single
    # bad record must not sink the whole batch. Fall back to now().
    try:
        if not isinstance(raw, str):
            return datetime.now(timezone.utc).timestamp()
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).timestamp()


def _format_from_timestamp(since: float) -> str:
    return datetime.fromtimestamp(since, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class LangfuseProvider:
    name = "langfuse"

    def __init__(self, endpoint_url: str, public_key: str, secret_key: str):
        self._base = endpoint_url.rstrip("/")
        self._auth = (public_key, secret_key)

    async def fetch_new_evaluations(self, since: float) -> list[ExternalEvaluation]:
        from_ts = _format_from_timestamp(since)
        evaluations: list[ExternalEvaluation] = []

        async with httpx.AsyncClient(auth=self._auth, timeout=30.0) as client:
            page = 1
            while page <= _MAX_PAGES:
                resp = await client.get(
                    f"{self._base}/api/public/scores",
                    params={"fromTimestamp": from_ts, "limit": _PAGE_LIMIT, "page": page},
                )
                resp.raise_for_status()
                body = resp.json()

                for score in body.get("data", []):
                    # audit Finding 27: parse each record defensively — skip a bad
                    # one, don't let it discard the batch (which would wedge the
                    # integration since `since` never advances past it).
                    try:
                        trace_id = score.get("traceId")
                        external_id = score.get("id")
                        if not trace_id or not external_id:
                            # No traceId (session/observation score) or no id —
                            # nothing to correlate/dedup; skip.
                            continue
                        project_id = score.get("projectId", "")
                        evaluations.append(
                            ExternalEvaluation(
                                external_id=external_id,
                                trace_id=trace_id,
                                name=score.get("name") or "score",
                                value=score.get("value"),
                                string_value=score.get("stringValue"),
                                comment=score.get("comment"),
                                timestamp=_parse_timestamp(score.get("timestamp")),
                                source_url=f"{self._base}/project/{project_id}/traces/{trace_id}",
                            )
                        )
                    except Exception as exc:
                        logger.warning("langfuse: skipping malformed score record: %s", exc)
                        continue

                meta = body.get("meta") or {}
                total_pages = meta.get("totalPages", 1) or 1
                if page >= total_pages:
                    break
                page += 1

        return evaluations
