"""
Model pricing table and cost estimation utilities.
Prices are per 1M tokens (input/output) in USD, as of June 2026.
These are estimates — actual costs vary with caching, batching, and provider agreements.

Sources:
- OpenAI pricing: https://platform.openai.com/docs/pricing
- Together AI pricing/model pages: https://www.together.ai/pricing
- Fireworks serverless pricing: https://docs.fireworks.ai/serverless/pricing
"""

from __future__ import annotations

# (input_per_1M_usd, output_per_1M_usd)
_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3": (2.00, 8.00),
    "o3-pro": (20.00, 80.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Meta / Alibaba / DeepSeek models via common hosted inference providers
    "llama-3.3-70b": (0.88, 0.88),
    "llama-3.1-405b": (3.50, 3.50),
    "deepseek-r1": (3.00, 7.00),
    "deepseek-v3": (1.25, 1.25),
    "qwen2.5-72b": (1.20, 1.20),
    # Anthropic
    "claude-opus-4-7": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Google
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    # Mistral, verified against https://mistral.ai/pricing/api on 2026-08-08.
    # The previous mistral-large (3.00, 9.00) and mistral-small (0.20, 0.60)
    # rows predated the Large 3 / Small 4 price cuts and overcharged Large by
    # 6x. mistral.ai/pricing (the FAQ page) still quotes the retired $2/$6
    # Large rate; the per-model cards agree with the numbers here.
    # _normalise() sorts keys longest-first, so "codestral-embed" wins over
    # "codestral" without needing a particular order in this dict.
    # Reasoning (magistral) and coding (devstral) families. Verified against
    # https://mistral.ai/pricing/api on 2026-08-09.
    "magistral-medium": (2.00, 5.00),
    "magistral-small": (0.50, 1.50),
    "devstral-medium": (0.40, 2.00),
    "devstral-small": (0.10, 0.30),
    "mistral-medium": (1.50, 7.50),
    "mistral-large": (0.50, 1.50),
    "mistral-small": (0.15, 0.60),
    "ministral-3b": (0.10, 0.10),
    "ministral-8b": (0.15, 0.15),
    "ministral-14b": (0.20, 0.20),
    "codestral": (0.30, 0.90),
    # Embeddings bill on input only. Output stays 0 so an embedding call can't
    # pick up the _DEFAULT output rate.
    "mistral-embed": (0.10, 0.0),
    "codestral-embed": (0.15, 0.0),
}

_DEFAULT = (1.00, 4.00)  # fallback for unknown/unlisted models


def _normalise(model: str) -> str:
    """Strip vendor prefixes and version suffixes to produce a lookup key."""
    m = (model or "").lower().strip()
    # Strip common vendor prefixes like "openai/" or "anthropic/"
    for prefix in (
        "openai/",
        "anthropic/",
        "google/",
        "mistral/",
        "meta/",
        "meta-llama/",
        "qwen/",
        "deepseek-ai/",
    ):
        if m.startswith(prefix):
            m = m[len(prefix) :]
            break
    if m.startswith("meta-llama-"):
        m = m[len("meta-") :]
    # Longest-prefix match against the price table
    for key in sorted(_PRICES, key=len, reverse=True):
        if m == key or m.startswith(key + "-") or m.startswith(key + "."):
            return key
    return m


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
) -> float:
    """Return estimated cost in USD. Returns 0.0 when prompt_tokens and completion_tokens are both zero.

    This is the single source of truth for token pricing across every
    Dunetrace service — do not fork this table (see
    services/semantic/semantic_svc/evaluators, which computes semantic
    evaluation cost by calling this directly rather than trusting a
    third-party library's own bundled — and possibly stale or divergent —
    per-model pricing).
    """
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    billed_output_tokens = completion_tokens
    if not prompt_tokens and not billed_output_tokens:
        return 0.0
    inp, out = _PRICES.get(_normalise(model), _DEFAULT)
    return (prompt_tokens * inp + billed_output_tokens * out) / 1_000_000
