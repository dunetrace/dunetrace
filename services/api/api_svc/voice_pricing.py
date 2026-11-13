"""Voice cost pricing (Phase 2.2).

Loads transcription (STT), text-to-speech (TTS), and telephony rates from
voice-pricing.yml, with the same defaults baked in as a fallback so cost
tracking works even when the file is not mounted. LLM cost is not here: it comes
per token from the shared cost table (explainer_svc.cost.estimate_cost), keyed on
the model recorded on each run.

Rates are list-price defaults. A startup check warns if voice-pricing.yml's
rates_as_of is more than 90 days old, to nudge customers to verify their own
contracted rates.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("dunetrace.voice_pricing")

# Baked-in defaults. Kept value-for-value in sync with the shipped voice-pricing.yml
# so behavior is identical whether or not the file is mounted.
DEFAULT_PRICING: Dict[str, Any] = {
    "rates_as_of": "2026-07-22",
    "stt": {
        "default": "deepgram_nova3",
        "providers": {
            "deepgram_nova3": {"per_minute": 0.0048},
            "deepgram_nova3_multi": {"per_minute": 0.0058},
            "openai_gpt4o_transcribe": {"per_minute": 0.006},
            "openai_gpt4o_mini_transcribe": {"per_minute": 0.003},
        },
    },
    "tts": {
        "default": "deepgram_aura2",
        "providers": {
            "deepgram_aura2": {"per_1k_chars": 0.030},
            "deepgram_aura1": {"per_1k_chars": 0.015},
            "openai_tts1": {"per_1k_chars": 0.015},
            "elevenlabs": {"per_1k_chars": 0.30},
        },
    },
    "telephony": {
        "default": "none",
        "providers": {
            "twilio_pstn": {"per_minute": 0.0085},
            "vapi_transport": {"per_minute": 0.05},
        },
    },
    "agent_overrides": {},
}

_STALE_AFTER_DAYS = 90

_cache: Optional[Dict[str, Any]] = None


def _config_path() -> str:
    return os.environ.get("VOICE_PRICING_CONFIG", "/app/voice-pricing.yml")


def load_pricing(force: bool = False) -> Dict[str, Any]:
    """Return the pricing config: voice-pricing.yml when present and parseable,
    otherwise the baked-in defaults. Cached after first load."""
    global _cache
    if _cache is not None and not force:
        return _cache

    pricing = DEFAULT_PRICING
    path = _config_path()
    try:
        import yaml

        if os.path.exists(path):
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict) and loaded:
                pricing = loaded
                logger.info("voice pricing loaded from %s", path)
    except ImportError:
        logger.info("PyYAML unavailable; using built-in voice pricing defaults")
    except Exception as exc:
        logger.warning("failed to parse %s (%s); using built-in defaults", path, exc)

    _cache = pricing
    return pricing


def check_pricing_staleness(pricing: Optional[Dict[str, Any]] = None) -> None:
    """Log a warning if rates_as_of is more than 90 days old. Call once at startup.
    Unchanged pricing keeps the shipped rates_as_of, so this fires exactly when a
    customer has not revisited the defaults."""
    pricing = pricing or load_pricing()
    raw = pricing.get("rates_as_of")
    if not raw:
        return
    try:
        as_of = datetime.date.fromisoformat(str(raw))
    except ValueError:
        logger.warning("voice-pricing.yml rates_as_of=%r is not an ISO date", raw)
        return
    age_days = (datetime.date.today() - as_of).days
    if age_days > _STALE_AFTER_DAYS:
        logger.warning(
            "voice-pricing.yml rates_as_of is %d days old (%s). Voice cost figures "
            "use list-price defaults that may have drifted; verify against your "
            "contracted rates and bump rates_as_of.",
            age_days,
            raw,
        )


def providers_for(agent_id: str, pricing: Dict[str, Any]) -> Dict[str, str]:
    """The stt/tts/telephony provider keys for an agent: a per-agent override
    when present, else the category default."""
    overrides = (pricing.get("agent_overrides") or {}).get(agent_id, {}) or {}
    return {
        cat: overrides.get(cat) or (pricing.get(cat) or {}).get("default", "none")
        for cat in ("stt", "tts", "telephony")
    }


def _rate(pricing: Dict[str, Any], category: str, provider: str, key: str) -> Optional[float]:
    if not provider or provider == "none":
        return None
    prov = ((pricing.get(category) or {}).get("providers") or {}).get(provider)
    if not prov:
        return None
    val = prov.get(key)
    return float(val) if val is not None else None


def stt_cost_usd(audio_seconds: float, provider: str, pricing: Dict[str, Any]) -> float:
    rate = _rate(pricing, "stt", provider, "per_minute")
    if rate is None or audio_seconds <= 0:
        return 0.0
    return (audio_seconds / 60.0) * rate


def tts_cost_usd(chars: int, provider: str, pricing: Dict[str, Any]) -> float:
    rate = _rate(pricing, "tts", provider, "per_1k_chars")
    if rate is None or chars <= 0:
        return 0.0
    return (chars / 1000.0) * rate


def telephony_cost_usd(minutes: float, provider: str, pricing: Dict[str, Any]) -> float:
    rate = _rate(pricing, "telephony", provider, "per_minute")
    if rate is None or minutes <= 0:
        return 0.0
    return minutes * rate
