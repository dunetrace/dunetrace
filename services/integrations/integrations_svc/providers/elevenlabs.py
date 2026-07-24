"""ElevenLabs pull provider (Phase 4.3). Fetches TTS generation history so
elevenlabs_worker can store it and Phase 4.4 can correlate it to Dunetrace
tts.generated events.

Deliberately NOT an EvaluationProvider (langfuse/langsmith/braintrust): those
return scores that correlate to a run by trace_id and become failure_signals.
ElevenLabs returns generations we store verbatim and later correlate to events
by timestamp / character-count / voice. So it does not implement that Protocol
or the ExternalEvaluation shape, and it runs under its own worker
(elevenlabs_worker.py), not worker.py.

Verified against the live ElevenLabs API during Phase 0 discovery, not just docs:
- GET https://api.elevenlabs.io/v1/history
- Header: xi-api-key: <key>
- Query: page_size (<= 1000), start_after_history_item_id (cursor), date_after_unix,
  sort_direction (default desc).
- Response: {"history": [item, ...], "last_history_item_id": str, "has_more": bool}.
- Per-item character count is the delta character_count_change_to - _from: the raw
  fields are running quota markers, not per-item counts.
- Rate limiting is concurrency-based (not requests-per-minute); a 429 carries
  too_many_concurrent_requests or system_busy. We poll one org's history
  sequentially, so we sit at concurrency 1, and still back off on any 429.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import httpx

logger = logging.getLogger("dunetrace.integrations.elevenlabs")

ELEVENLABS_API_BASE = "https://api.elevenlabs.io"
_PAGE_SIZE = 100
_MAX_PAGES = 50  # hard safety cap — a runaway loop must never paginate forever
_MAX_RETRIES = 5  # per-request 429/5xx backoff attempts before giving up
_BASE_BACKOFF_SECS = 1.0
_MAX_BACKOFF_SECS = 30.0


@dataclass(frozen=True)
class ElevenLabsGeneration:
    """One TTS generation pulled from ElevenLabs, reduced to the fields the
    worker stores. Nothing provider-specific leaks past this point."""

    generation_id: str  # history_item_id — dedup key and deterministic correlation key
    voice_id: str | None
    voice_name: str | None
    model: str | None  # model_id
    character_count: int
    text: str | None  # nullable on the ElevenLabs side; enables exact-text matching
    source: str | None  # TTS / ConvAI / etc — lets Phase 5 flag the ConvAI case
    generated_at: float  # date_unix (epoch seconds), same domain as events.timestamp


def _char_count(item: dict) -> int:
    """Per-item character count = character_count_change_to - _from. Those fields
    are running quota markers; the delta is what this generation billed. Falls
    back to len(text), then 0, when the markers are missing or nonsensical, so a
    weird record yields 0 rather than a negative or a crash."""
    to = item.get("character_count_change_to")
    frm = item.get("character_count_change_from")
    if isinstance(to, int) and isinstance(frm, int) and to >= frm:
        return to - frm
    text = item.get("text")
    return len(text) if isinstance(text, str) else 0


class ElevenLabsProvider:
    name = "elevenlabs"

    def __init__(self, api_key: str):
        # Matches the decrypted credentials dict key (see api_svc storing
        # {"api_key": ...}), so elevenlabs_worker can do Provider(**creds).
        self._headers = {"xi-api-key": api_key}

    async def _get_with_backoff(
        self, client: httpx.AsyncClient, url: str, params: dict
    ) -> httpx.Response:
        """One GET, retried on 429 and 5xx with exponential backoff plus jitter,
        honoring Retry-After when present. Raises after _MAX_RETRIES so the
        worker's per-org failure isolation records the failure and moves on
        rather than this hanging the whole cycle."""
        attempt = 0
        while True:
            resp = await client.get(url, params=params, headers=self._headers)
            # Only 429 and 5xx are retryable. Anything else (2xx success, or a
            # 4xx that won't fix itself) is returned/raised immediately.
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
                return resp
            attempt += 1
            if attempt > _MAX_RETRIES:
                resp.raise_for_status()  # exhausted — surface the HTTP error
            delay = self._retry_delay(resp, attempt)
            logger.warning(
                "elevenlabs: HTTP %d, backing off %.1fs (attempt %d/%d)",
                resp.status_code,
                delay,
                attempt,
                _MAX_RETRIES,
            )
            await asyncio.sleep(delay)

    def _retry_delay(self, resp: httpx.Response, attempt: int) -> float:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), _MAX_BACKOFF_SECS)
            except (ValueError, TypeError):
                pass
        backoff = min(_BASE_BACKOFF_SECS * (2 ** (attempt - 1)), _MAX_BACKOFF_SECS)
        return backoff + random.uniform(0, backoff * 0.25)  # jitter, avoid lockstep retries

    async def fetch_generations(self, since_unix: float) -> list[ElevenLabsGeneration]:
        """Every generation with date_unix >= since_unix, newest first. Paginates
        the desc-ordered history with the cursor, stopping as soon as a page
        crosses below since_unix (older items only follow in desc order) or
        has_more is false. date_after_unix is also passed as a server-side hint;
        the client-side crossing check is the belt-and-suspenders guarantee."""
        url = f"{ELEVENLABS_API_BASE}/v1/history"
        since_floor = int(since_unix)
        out: list[ElevenLabsGeneration] = []
        cursor: str | None = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(_MAX_PAGES):
                params: dict = {"page_size": _PAGE_SIZE, "date_after_unix": since_floor}
                if cursor:
                    params["start_after_history_item_id"] = cursor
                resp = await self._get_with_backoff(client, url, params)
                body = resp.json()
                items = body.get("history") or []

                crossed = False
                for item in items:
                    gen = self._parse(item)
                    if gen is None:
                        continue
                    if gen.generated_at < since_unix:
                        crossed = True
                        continue
                    out.append(gen)

                if crossed or not body.get("has_more"):
                    break
                cursor = body.get("last_history_item_id")
                if not cursor:
                    break

        return out

    def _parse(self, item: dict) -> ElevenLabsGeneration | None:
        # Parse each record defensively — skip a malformed one, never let it
        # discard the batch (which would wedge the poll, since the high-water
        # mark would never advance past it).
        try:
            gen_id = item.get("history_item_id")
            if not gen_id:
                return None
            date_unix = item.get("date_unix")
            generated_at = float(date_unix) if date_unix is not None else 0.0
            return ElevenLabsGeneration(
                generation_id=gen_id,
                voice_id=item.get("voice_id"),
                voice_name=item.get("voice_name"),
                model=item.get("model_id"),
                character_count=_char_count(item),
                text=item.get("text"),
                source=item.get("source"),
                generated_at=generated_at,
            )
        except Exception as exc:
            logger.warning("elevenlabs: skipping malformed history item: %s", exc)
            return None
