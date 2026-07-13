from __future__ import annotations

import unittest
from unittest.mock import patch

from deepeval.models import AnthropicModel, GPTModel

from semantic_svc.evaluators.models import (
    _DEFAULT_MODEL_BY_PROVIDER,
    build_deepeval_model,
    resolve_model_name,
)


class TestResolveModelName(unittest.TestCase):
    def test_explicit_name_is_used_as_is(self):
        self.assertEqual(resolve_model_name("openai", "gpt-4o"), "gpt-4o")

    def test_falls_back_to_provider_default_when_none(self):
        self.assertEqual(resolve_model_name("openai", None), _DEFAULT_MODEL_BY_PROVIDER["openai"])
        self.assertEqual(
            resolve_model_name("anthropic", None), _DEFAULT_MODEL_BY_PROVIDER["anthropic"]
        )

    def test_falls_back_to_provider_default_for_empty_string(self):
        self.assertEqual(resolve_model_name("openai", ""), _DEFAULT_MODEL_BY_PROVIDER["openai"])


class TestBuildDeepevalModel(unittest.TestCase):
    def _fake_keys(self):
        return patch(
            "semantic_svc.evaluators.models.settings",
            OPENAI_API_KEY="sk-fake",
            ANTHROPIC_API_KEY="sk-ant-fake",
        )

    def test_openai_provider_returns_gptmodel(self):
        with self._fake_keys():
            model = build_deepeval_model("openai", "gpt-4o-mini")
        self.assertIsInstance(model, GPTModel)

    def test_anthropic_provider_returns_anthropicmodel(self):
        with self._fake_keys():
            model = build_deepeval_model("anthropic", "claude-haiku-4-5")
        self.assertIsInstance(model, AnthropicModel)

    def test_default_model_used_when_none_given(self):
        with self._fake_keys():
            model = build_deepeval_model("openai", None)
        self.assertEqual(model.get_model_name(), _DEFAULT_MODEL_BY_PROVIDER["openai"])

    def test_unknown_provider_falls_back_to_openai(self):
        with self._fake_keys():
            model = build_deepeval_model("some-other-provider", "gpt-4o-mini")
        self.assertIsInstance(model, GPTModel)


if __name__ == "__main__":
    unittest.main()
