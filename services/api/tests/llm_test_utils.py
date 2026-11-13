"""
Declare an LLM provider for a test, explicitly.

`conftest.py` blanks the ambient provider credentials so every test starts from
the CI state — no LLM configured. A test that exercises a path *behind* that
gate opts in here, rather than relying on a `.env` that may or may not exist on
the machine running it.

Patches `api_svc.llm_provider.settings` because that is the single place
provider selection now reads: patching a caller module's own `settings` no
longer affects the gate, which is exactly the trap this module exists to close.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch


def provider_settings(*, anthropic="", openai="", mistral="", pinned=""):
    """A settings stand-in carrying only the credentials a test asks for."""
    s = MagicMock()
    s.ANTHROPIC_API_KEY = anthropic
    s.OPENAI_API_KEY = openai
    s.MISTRAL_API_KEY = mistral
    s.API_LLM_PROVIDER = pinned
    return s


@contextmanager
def configured_llm(provider: str = "anthropic", *, completion: str | None = None):
    """Run the block with `provider` configured.

    `completion` short-circuits `llm_provider.complete` to that text, so the
    test never touches a provider SDK or the network. Omit it to leave the real
    dispatch in place (for tests that patch the vendor client themselves).
    """
    keys = {"anthropic": "sk-ant-test", "openai": "sk-oai-test", "mistral": "mistral-test"}
    if provider not in keys:
        raise ValueError(f"unknown provider {provider!r}")

    stack = [
        patch("api_svc.llm_provider.settings", provider_settings(**{provider: keys[provider]}))
    ]
    if completion is not None:
        stack.append(patch("api_svc.llm_provider.complete", AsyncMock(return_value=completion)))

    entered = [cm.__enter__() for cm in stack]
    try:
        yield entered[-1] if completion is not None else entered[0]
    finally:
        for cm in reversed(stack):
            cm.__exit__(None, None, None)


@contextmanager
def no_llm():
    """Explicitly assert the no-provider path. Redundant with the conftest
    default, but makes the intent visible in tests that are *about* that path."""
    with patch("api_svc.llm_provider.resolve_provider", return_value=None):
        yield
