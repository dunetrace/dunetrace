"""
Make the API suite independent of whoever's `.env` happens to be on disk.

`api_svc/config.py` calls `_load_dotenv()` at import, so on a developer machine
`settings.ANTHROPIC_API_KEY` is populated from the repo's real `.env` while in
CI it is empty. Provider selection moved behind `api_svc.llm_provider`, which
reads those settings — so any test that patched only its *own* module's
`settings` silently kept working locally off a real key and failed in CI. That
divergence hid a stale assertion and then broke the build twice.

The fixture below blanks the provider credentials for every test, so the
baseline everywhere is "no LLM configured" — the CI state. A test that needs a
configured provider says so explicitly with `configured_llm()` from
`llm_test_utils`, which is both deterministic and self-documenting.

Autouse fixtures do apply to `unittest.TestCase` subclasses, which is what most
of this suite uses.
"""

from __future__ import annotations

import pytest

from api_svc.config import settings

_PROVIDER_SETTINGS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "MISTRAL_API_KEY",
    "API_LLM_PROVIDER",
)


@pytest.fixture(autouse=True)
def _no_ambient_llm_credentials():
    """Blank any provider credential the developer's .env supplied.

    Restored afterwards so nothing leaks between tests or out of the suite.
    """
    saved = {name: getattr(settings, name, "") for name in _PROVIDER_SETTINGS}
    for name in _PROVIDER_SETTINGS:
        setattr(settings, name, "")
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(settings, name, value)
