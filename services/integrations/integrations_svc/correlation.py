"""Correlation logic (Phase 4.4): match a stored ElevenLabs generation to the
Dunetrace tts.generated event that produced it.

The matching algorithm (match_generation) is pure and side-effect free so it can
be unit-tested exhaustively without a database. correlate_once wraps it with the
DB reads/writes and the pending-vs-give-up policy.

Signal priority, strongest first. The method recorded on a match feeds Phase 5's
honesty indicator, and the whole point of the ordering is to never claim a
match more confident than the evidence supports:

  1. generation_id  — the event captured ElevenLabs' own generation id. Deterministic.
  2. exact_text     — event text equals the synthesized text. Near-certain.
  3. voice_char_time— character count within tolerance AND voice id agrees.
  4. char_time      — character count within tolerance, inside the time window.

When more than one candidate survives and we cannot separate them honestly, we
do NOT guess. We return ambiguous and the generation is recorded as unmatched
drift (constraint: no fabricated correlation).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass

from integrations_svc.config import settings
from integrations_svc.db import (
    fetch_candidate_tts_events,
    fetch_uncorrelated_generations,
    mark_generation_correlated,
    mark_generation_unmatched,
)

logger = logging.getLogger("dunetrace.elevenlabs.correlation")

# How many generations one correlation pass will handle. Bounds the work so a
# large backlog can never starve the poll loop; leftovers are picked up next pass.
_PASS_LIMIT = 500


@dataclass(frozen=True)
class CorrelationOutcome:
    matched: bool
    event_id: int | None = None
    method: str | None = None
    confidence: float | None = None
    reason: str | None = None  # set only when matched is False


def _char_ok(candidate_text: str | None, gen_char: int, tolerance: float) -> bool:
    cc = len(candidate_text or "")
    if gen_char <= 0:
        return cc == 0
    return abs(cc - gen_char) <= tolerance * gen_char


def match_generation(
    *,
    generation_id: str,
    text: str | None,
    voice_id: str | None,
    character_count: int,
    generated_at: float,
    candidates: list[dict],
    char_tolerance: float,
) -> CorrelationOutcome:
    """Decide which candidate tts.generated event (if any) produced this
    generation. candidates are events already restricted to the org and the
    timestamp window; each is a dict with id, timestamp, text, voice_id,
    provider_generation_id."""
    if not candidates:
        return CorrelationOutcome(matched=False, reason="no_candidate_events")

    # 1. Deterministic: the event captured ElevenLabs' own generation id.
    gid = [
        c
        for c in candidates
        if c.get("provider_generation_id") and c["provider_generation_id"] == generation_id
    ]
    if gid:
        return CorrelationOutcome(
            True, event_id=gid[0]["id"], method="generation_id", confidence=1.0
        )

    # 2. Exact text. Near-certain; on the rare tie of identical text in-window,
    # take the nearest timestamp and mark the confidence a notch lower.
    if text:
        text_matches = [c for c in candidates if c.get("text") and c["text"] == text]
        if len(text_matches) == 1:
            return CorrelationOutcome(
                True, event_id=text_matches[0]["id"], method="exact_text", confidence=0.97
            )
        if len(text_matches) > 1:
            best = min(text_matches, key=lambda c: abs(c["timestamp"] - generated_at))
            return CorrelationOutcome(
                True, event_id=best["id"], method="exact_text", confidence=0.90
            )

    # 3. Character-count fallback, within tolerance.
    char_matches = [
        c for c in candidates if _char_ok(c.get("text"), character_count, char_tolerance)
    ]
    if not char_matches:
        return CorrelationOutcome(matched=False, reason="no_char_match")
    if len(char_matches) == 1:
        return CorrelationOutcome(
            True, event_id=char_matches[0]["id"], method="char_time", confidence=0.70
        )

    # 4. Still several: narrow by voice id when both sides carry one.
    if voice_id:
        voice_matches = [c for c in char_matches if c.get("voice_id") and c["voice_id"] == voice_id]
        if len(voice_matches) == 1:
            return CorrelationOutcome(
                True, event_id=voice_matches[0]["id"], method="voice_char_time", confidence=0.85
            )
        if len(voice_matches) > 1:
            char_matches = voice_matches  # narrowed, but see below

    # Cannot separate the survivors honestly. Do not guess.
    return CorrelationOutcome(matched=False, reason="ambiguous_multiple_matches")


async def correlate_once() -> dict:
    """One correlation pass over pending generations across all orgs. Matches are
    written immediately; a generation that cannot match yet stays pending and is
    retried next pass, unless it is older than CORRELATION_GIVEUP_SECS, in which
    case it is recorded as unmatched drift with its reason.

    Returns a summary of this pass: {processed, matched, unmatched, still_pending}.
    Per-generation errors are isolated so one bad row never sinks the pass."""
    window = settings.CORRELATION_WINDOW_SECS
    tolerance = settings.CORRELATION_CHAR_TOLERANCE
    giveup = settings.CORRELATION_GIVEUP_SECS

    rows = await fetch_uncorrelated_generations(_PASS_LIMIT)
    now = time.time()
    matched = 0
    unmatched = 0
    still_pending = 0
    methods: Counter = Counter()

    for row in rows:
        try:
            candidates = await fetch_candidate_tts_events(
                row["org_id"],
                row["generated_at"] - window,
                row["generated_at"] + window,
            )
            outcome = match_generation(
                generation_id=row["generation_id"],
                text=row["text"],
                voice_id=row["voice_id"],
                character_count=row["character_count"],
                generated_at=row["generated_at"],
                candidates=candidates,
                char_tolerance=tolerance,
            )
            if outcome.matched:
                # Denormalize the matched event's run/agent onto the generation
                # (Phase 5.1) so downstream read queries avoid an events.id join.
                matched_event = next((c for c in candidates if c["id"] == outcome.event_id), {})
                await mark_generation_correlated(
                    row["id"],
                    outcome.event_id,
                    matched_event.get("run_id"),
                    matched_event.get("agent_id"),
                    outcome.method,
                    outcome.confidence,
                )
                matched += 1
                methods[outcome.method] += 1
            elif (now - row["generated_at"]) > giveup:
                # Old enough that a matching event should have arrived. Record it
                # honestly as drift rather than retrying forever.
                await mark_generation_unmatched(row["id"], outcome.reason)
                unmatched += 1
            else:
                still_pending += 1  # try again next pass — the event may be in flight
        except Exception as exc:  # isolation: one bad row must not sink the pass
            logger.warning("correlation: skipping generation id=%s: %s", row.get("id"), exc)
            still_pending += 1

    if matched or unmatched:
        logger.info(
            "correlation pass — processed=%d matched=%d (%s) unmatched=%d still_pending=%d",
            len(rows),
            matched,
            dict(methods),
            unmatched,
            still_pending,
        )
    return {
        "processed": len(rows),
        "matched": matched,
        "unmatched": unmatched,
        "still_pending": still_pending,
    }
