"""
Braintrust pull integration (Phase 2.3). Verified against Braintrust's real
public OpenAPI spec AND a real test project (inserted + fetched a live event)
— architecturally different from both Langfuse and LangSmith, not just a
different auth scheme:

- Auth: Authorization: Bearer <api_key> (a third distinct scheme).
- There is NO separate scores/feedback read endpoint. The one endpoint whose
  name suggests it (POST /v1/project_logs/{project_id}/feedback) is
  write-only ("Log feedback for a set of project logs events" — pushes
  feedback INTO Braintrust, not a source of it). Scores live embedded on
  each event returned by GET/POST /v1/project_logs/{project_id}/fetch, as a
  `scores: {name: number}` dict (0-1 range) — confirmed live by inserting an
  event with two named scores and fetching it back unchanged.
- The fetch endpoint has NO since=timestamp filter at all — only `cursor`
  (opaque pagination token) and `version`/`max_xact_id` (as-of snapshot
  filters, not "since" filters). Results are always returned in descending
  time order (latest _xact_id first, confirmed live with two inserted
  events). So "new since X" is implemented client-side here: page backwards
  via `cursor` and stop as soon as an event's `created` timestamp is older
  than `since` — everything after that point in the descending order is
  necessarily even older.
- Each event can carry multiple named scores, so one event maps to multiple
  ExternalEvaluations here — external_id is f"{event_id}:{score_name}" (an
  event id alone isn't a unique dedup key the way a single Langfuse score id
  or LangSmith feedback id is).
- Trace correlation field is `root_span_id` (required on every event,
  confirmed via schema and live fetch) — analogous to Langfuse's `traceId`
  and LangSmith's `trace_id`.
- No measured indexing lag: a freshly-inserted event was fetchable
  immediately in live testing (unlike Langfuse's ~20-30s or LangSmith's
  ~5s) — still uses the same worker-wide _OVERLAP_SECS for consistency and
  because this was only confirmed for one small test project, not at scale.
- A bad/inaccessible project_id returns 403, not 404 (confirmed live) —
  indistinguishable from a real permission error; both surface the same way
  through raise_for_status() and the existing consecutive-failure handling.
- No verified private-dashboard URL pattern was found for a single event in
  the public spec — source_url is always None here, same reasoning as
  LangSmith's provider (omit rather than fabricate a likely-wrong link).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from integrations_svc.providers.base import ExternalEvaluation

logger = logging.getLogger("dunetrace.integrations.braintrust")

_PAGE_LIMIT = 100
_MAX_PAGES = 50  # hard safety cap — a runaway loop must not poll forever


def _parse_timestamp(raw: str | None) -> float:
    if not raw:
        return datetime.now(timezone.utc).timestamp()
    # Braintrust emits "...Z" — normalize for fromisoformat() the same way
    # Langfuse's provider does. audit Finding 27: never raise on bad input.
    try:
        if not isinstance(raw, str):
            return datetime.now(timezone.utc).timestamp()
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).timestamp()


class BraintrustProvider:
    name = "braintrust"

    def __init__(self, endpoint_url: str, api_key: str, project_id: str):
        self._base = endpoint_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._project_id = project_id

    async def fetch_new_evaluations(self, since: float) -> list[ExternalEvaluation]:
        evaluations: list[ExternalEvaluation] = []

        async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
            cursor: str | None = None
            for _ in range(_MAX_PAGES):
                params = {"limit": _PAGE_LIMIT}
                if cursor is not None:
                    params["cursor"] = cursor
                resp = await client.get(
                    f"{self._base}/v1/project_logs/{self._project_id}/fetch",
                    params=params,
                )
                resp.raise_for_status()
                body = resp.json()
                events = body.get("events", [])

                hit_watermark = False
                for event in events:
                    # audit Finding 27: skip a malformed event, don't sink the batch.
                    try:
                        event_ts = _parse_timestamp(event.get("created"))
                        if event_ts < since:
                            hit_watermark = True
                            break

                        scores = event.get("scores") or {}
                        trace_id = event.get("root_span_id")
                        event_id = event.get("id")
                        if not trace_id or not event_id:
                            continue
                        for score_name, score_value in scores.items():
                            if score_value is None:
                                continue
                            try:
                                numeric = float(score_value)
                            except (TypeError, ValueError):
                                continue
                            evaluations.append(
                                ExternalEvaluation(
                                    external_id=f"{event_id}:{score_name}",
                                    trace_id=trace_id,
                                    name=score_name,
                                    value=numeric,
                                    string_value=None,
                                    comment=None,
                                    timestamp=event_ts,
                                    source_url=None,
                                )
                            )
                    except Exception as exc:
                        logger.warning("braintrust: skipping malformed event: %s", exc)
                        continue

                if hit_watermark:
                    break
                cursor = body.get("cursor")
                if not cursor:
                    break

        return evaluations
