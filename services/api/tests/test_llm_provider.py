"""
Provider selection for the Customer API's own LLM features.

Run:
    PYTHONPATH=packages/sdk-py:services/explainer:services/api \
      python -m pytest services/api/tests/test_llm_provider.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_svc import llm_provider


def _settings(*, anthropic="", openai="", mistral="", pinned=""):
    s = MagicMock()
    s.ANTHROPIC_API_KEY = anthropic
    s.OPENAI_API_KEY = openai
    s.MISTRAL_API_KEY = mistral
    s.API_LLM_PROVIDER = pinned
    return s


class TestResolveProvider(unittest.TestCase):
    def _resolve(self, **kw):
        with patch("api_svc.llm_provider.settings", _settings(**kw)):
            return llm_provider.resolve_provider()

    def test_none_when_no_key_is_configured(self):
        self.assertIsNone(self._resolve())

    def test_anthropic_still_wins_by_default(self):
        """The precedence every call site had before mistral existed."""
        self.assertEqual(
            self._resolve(anthropic="sk-ant", openai="sk-oai", mistral="m"), "anthropic"
        )

    def test_falls_through_to_openai_then_mistral(self):
        self.assertEqual(self._resolve(openai="sk-oai", mistral="m"), "openai")
        self.assertEqual(self._resolve(mistral="m"), "mistral")

    def test_explicit_provider_wins_over_precedence(self):
        self.assertEqual(
            self._resolve(anthropic="sk-ant", mistral="m", pinned="mistral"), "mistral"
        )

    def test_explicit_provider_is_case_and_space_insensitive(self):
        self.assertEqual(self._resolve(mistral="m", pinned="  Mistral "), "mistral")

    def test_pinned_provider_without_its_key_does_not_fall_back(self):
        """A deployment that pinned mistral for residency must not silently get
        Anthropic because MISTRAL_API_KEY was missing."""
        self.assertIsNone(self._resolve(anthropic="sk-ant", pinned="mistral"))

    def test_unknown_pinned_provider_is_not_treated_as_openai(self):
        self.assertIsNone(self._resolve(openai="sk-oai", pinned="gpt5-turbo-max"))
        with patch("api_svc.llm_provider.settings", _settings(pinned="nonsense")):
            self.assertIn("nonsense", llm_provider.missing_key_message())


class TestComplete(unittest.IsolatedAsyncioTestCase):
    async def test_raises_rather_than_calling_anything_when_unconfigured(self):
        with patch("api_svc.llm_provider.settings", _settings()):
            with self.assertRaises(ValueError):
                await llm_provider.complete("sys", "user", max_tokens=10)

    async def test_mistral_goes_to_mistrals_endpoint_via_the_openai_client(self):
        """Mistral's chat API is OpenAI-compatible, so it reuses that client
        with a different base_url rather than adding a dependency. The base_url
        is what keeps the request inside the European provider — assert it."""
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="hello"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        ctor = MagicMock(return_value=client)

        with (
            patch("api_svc.llm_provider.settings", _settings(mistral="m-key", pinned="mistral")),
            patch("openai.AsyncOpenAI", ctor),
        ):
            out = await llm_provider.complete("sys", "user", max_tokens=64)

        self.assertEqual(out, "hello")
        self.assertEqual(ctor.call_args.kwargs["base_url"], "https://api.mistral.ai/v1")
        self.assertEqual(ctor.call_args.kwargs["api_key"], "m-key")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"], "mistral-small-latest"
        )

    async def test_openai_keeps_the_sdk_default_base_url(self):
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="hi"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        ctor = MagicMock(return_value=client)

        with (
            patch("api_svc.llm_provider.settings", _settings(openai="sk-oai")),
            patch("openai.AsyncOpenAI", ctor),
        ):
            await llm_provider.complete("sys", "user", max_tokens=64)

        self.assertIsNone(ctor.call_args.kwargs["base_url"])

    async def test_anthropic_uses_the_anthropic_client(self):
        msg = MagicMock()
        msg.content = [MagicMock(text="answer")]
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=msg)

        with (
            patch("api_svc.llm_provider.settings", _settings(anthropic="sk-ant")),
            patch("anthropic.AsyncAnthropic", MagicMock(return_value=client)),
        ):
            out = await llm_provider.complete("sys", "user", max_tokens=64)

        self.assertEqual(out, "answer")


if __name__ == "__main__":
    unittest.main()
