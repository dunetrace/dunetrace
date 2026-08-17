"""Pure aggregation for the three ElevenLabs cross-tool analytics (Phase 6.1).

The DB layer returns raw grouped rows; these functions turn them into the
answer to each customer question, including rates, USD pricing, and honest
edge-case handling (no data, samples too small to trust). Kept pure so the
"computes correctly" and "edge cases handled" tests need no database.
"""

from __future__ import annotations

from typing import Any, Dict, List

from api_svc.voice_pricing import tts_cost_usd

# Signals that read as user frustration or call abandonment, for Analytic 3 and
# as the "negative outcome" lens. VOICE_TTS_TRUNCATION is deliberately excluded:
# it is the cause under study, not the downstream effect.
FRUSTRATION_SIGNALS: List[str] = [
    "VOICE_LATENCY_INDUCED_HANGUP",
    "VOICE_BARGE_IN_FAILURE",
    "VOICE_TURN_TAKING_COLLISION",
    "VOICE_SILENCE_TIMEOUT",
    "GOAL_ABANDONMENT",
]

# A voice with fewer correlated runs than this is reported but flagged: its rate
# is not yet a trustworthy comparison.
_MIN_RUNS_FOR_CONFIDENCE = 20


def summarize_cost_by_outcome(rows: List[dict], pricing: Dict[str, Any]) -> dict:
    """Analytic 1: how much did we spend on TTS for calls that did not go well.
    A call is 'unsuccessful' when it carries at least one non-shadow signal."""
    calls = []
    total_usd = 0.0
    unsuccessful_usd = 0.0
    total_credits = 0
    unsuccessful_credits = 0
    unsuccessful_count = 0

    for r in rows:
        chars = int(r.get("chars") or 0)
        credits = int(r.get("credits") or 0)
        cost_usd = tts_cost_usd(chars, "elevenlabs", pricing)
        unsuccessful = int(r.get("signal_count") or 0) > 0

        total_usd += cost_usd
        total_credits += credits
        if unsuccessful:
            unsuccessful_usd += cost_usd
            unsuccessful_credits += credits
            unsuccessful_count += 1

        calls.append(
            {
                "conversation_id": r["conversation_id"],
                "external_id": r.get("external_id"),
                "agent_id": r.get("agent_id"),
                "generation_count": int(r.get("gen_count") or 0),
                "credits": credits,
                "cost_usd": cost_usd,
                "signal_count": int(r.get("signal_count") or 0),
                "unsuccessful": unsuccessful,
            }
        )

    return {
        "summary": {
            "call_count": len(rows),
            "unsuccessful_call_count": unsuccessful_count,
            "total_cost_usd": total_usd,
            "unsuccessful_cost_usd": unsuccessful_usd,
            "successful_cost_usd": total_usd - unsuccessful_usd,
            "total_credits": total_credits,
            "unsuccessful_credits": unsuccessful_credits,
            # Share of spend that went to calls Dunetrace flagged. None when there
            # is no spend to take a share of.
            "wasted_share": (unsuccessful_usd / total_usd) if total_usd > 0 else None,
        },
        "calls": calls,
    }


def summarize_voice_impact(rows: List[dict]) -> dict:
    """Analytic 2: does one voice correlate with worse outcomes than another.
    signal_rate is the fraction of that voice's runs carrying a non-shadow
    signal (lower is better); voices below the sample threshold are flagged."""
    voices = []
    for r in rows:
        run_count = int(r.get("run_count") or 0)
        with_signals = int(r.get("runs_with_signals") or 0)
        voices.append(
            {
                "voice_id": r["voice_id"],
                "voice_name": r.get("voice_name"),
                "run_count": run_count,
                "runs_with_signals": with_signals,
                "signal_rate": (with_signals / run_count) if run_count else None,
                "insufficient_data": run_count < _MIN_RUNS_FOR_CONFIDENCE,
            }
        )
    return {"voices": voices}


def summarize_truncation_impact(row: dict) -> dict:
    """Analytic 3: when TTS gets truncated, do candidates notice? Compares the
    frustration/abandonment rate of truncated runs against non-truncated runs in
    the same ElevenLabs-correlated population. lift is the difference; positive
    means truncation coincides with more frustration."""
    truncated_runs = int(row.get("truncated_runs") or 0)
    truncated_frust = int(row.get("truncated_with_frustration") or 0)
    clean_runs = int(row.get("clean_runs") or 0)
    clean_frust = int(row.get("clean_with_frustration") or 0)

    truncated_rate = (truncated_frust / truncated_runs) if truncated_runs else None
    clean_rate = (clean_frust / clean_runs) if clean_runs else None
    # Both sides need runs before a comparison means anything.
    insufficient = truncated_runs < _MIN_RUNS_FOR_CONFIDENCE or clean_runs == 0
    lift = (
        (truncated_rate - clean_rate)
        if (truncated_rate is not None and clean_rate is not None)
        else None
    )

    return {
        "population": truncated_runs + clean_runs,
        "truncated_runs": truncated_runs,
        "truncated_frustration_rate": truncated_rate,
        "clean_runs": clean_runs,
        "clean_frustration_rate": clean_rate,
        "lift": lift,
        "insufficient_data": insufficient,
    }
