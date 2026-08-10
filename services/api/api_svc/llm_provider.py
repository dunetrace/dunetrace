"""
Which provider the Customer API's *own* LLM features run on.

Four features call an LLM on the customer's behalf — native explain's root-cause
analysis (`routers/signals.py`), fix diff generation (`diff_generation.py`),
custom-detector translation (`custom_detector_translator.py`), and issue
summarisation (`routers/issues.py`). Each used to inline the same
anthropic-then-openai dispatch, so a deployment running everything else on a
European provider still sent run text, source files and detector descriptions to
a US API. This centralises the choice and adds `mistral` as a third option.

`SEMANTIC_LLM_PROVIDER` deliberately does *not* control this. That one selects
the Tier 2 evaluator provider in a different service; conflating them would make
turning on semantic evaluation silently repoint these four features too. Set
`API_LLM_PROVIDER` to pin these, and set both if you want everything in one
provider.

Mistral is reached through the ``openai`` package pointed at Mistral's
OpenAI-compatible ``/v1/chat/completions`` endpoint, rather than by adding
``mistralai`` as a dependency of this service — the request and response shapes
this module needs are identical, and the API image already installs ``openai``.
"""

from __future__ import annotations

from api_svc.config import settings

# Provider -> (api-key setting name, default model, base_url or None).
# Order is the auto-detect precedence when API_LLM_PROVIDER is unset: anthropic
# first, preserving the behaviour every one of these call sites had before
# mistral existed.
_PROVIDERS: dict[str, tuple[str, str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001", None),
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini", None),
    "mistral": ("MISTRAL_API_KEY", "mistral-small-latest", "https://api.mistral.ai/v1"),
}

SUPPORTED_PROVIDERS = tuple(_PROVIDERS)


def _key_for(provider: str) -> str:
    setting, _, _ = _PROVIDERS[provider]
    return getattr(settings, setting, "") or ""


def resolve_provider() -> str | None:
    """The provider these features should use, or None when no key is set.

    An explicit API_LLM_PROVIDER wins if its key is present. An explicit value
    whose key is missing returns None rather than quietly falling through to a
    different vendor — the same no-silent-fallback rule the semantic worker
    applies, and for the same data-residency reason.
    """
    pinned = (settings.API_LLM_PROVIDER or "").strip().lower()
    if pinned:
        if pinned not in _PROVIDERS:
            return None
        return pinned if _key_for(pinned) else None
    for provider in _PROVIDERS:
        if _key_for(provider):
            return provider
    return None


def llm_configured() -> bool:
    return resolve_provider() is not None


def provider_config(provider: str) -> tuple[str, str, str | None]:
    """(api_key, default_model, base_url) for a resolved provider."""
    setting, model, base_url = _PROVIDERS[provider]
    return getattr(settings, setting, "") or "", model, base_url


def missing_key_message() -> str:
    pinned = (settings.API_LLM_PROVIDER or "").strip().lower()
    if pinned and pinned in _PROVIDERS:
        return (
            f"API_LLM_PROVIDER={pinned} but {_PROVIDERS[pinned][0]} is not set. "
            f"Add it to your .env."
        )
    if pinned:
        return (
            f"API_LLM_PROVIDER={pinned!r} is not supported. "
            f"Expected one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return (
        "No LLM API key configured. Add ANTHROPIC_API_KEY, OPENAI_API_KEY or "
        "MISTRAL_API_KEY to your .env."
    )


async def complete(system: str, user: str, *, max_tokens: int) -> str:
    """One chat completion, returned as raw text.

    The single place any of the four features talks to a provider, so adding a
    fourth provider is a row in _PROVIDERS rather than another dispatch block.
    """
    provider = resolve_provider()
    if provider is None:
        raise ValueError(missing_key_message())
    api_key, model, base_url = provider_config(provider)

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install shape
            raise ImportError("anthropic package required: pip install anthropic") from exc

        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    try:
        import openai
    except ImportError as exc:  # pragma: no cover - depends on install shape
        raise ImportError("openai package required: pip install openai") from exc

    # base_url is None for openai itself, which leaves the SDK's own default.
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content
