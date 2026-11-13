"""
Tests for the Mistral evaluator provider.

No network. A fake mistralai module stands in for the real SDK so the actual
MistralModel code path runs, load_model() included.

Run:
    PYTHONPATH=services/explainer:services/semantic \
      python -m pytest services/semantic/tests/test_mistral_model.py -v
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from deepeval.models import AnthropicModel, GPTModel
from deepeval.models.utils import EvaluationCost
from pydantic import BaseModel


def _install_fake_mistralai(content='{"score": 7, "reason": "looks fine"}'):
    """Fake the mistralai v2 layout: mistralai.client.Mistral, with *_async on
    the same class rather than a separate async client."""
    mod = types.ModuleType("mistralai")
    client_mod = types.ModuleType("mistralai.client")

    class _Chat:
        def complete(self, *, model=None, messages=None, **kwargs):
            return client_mod._response()

        async def complete_async(self, *, model=None, messages=None, **kwargs):
            return client_mod._response()

    class Mistral:
        def __init__(self, api_key=None, retry_config=None, **kwargs):
            self.api_key = api_key
            self.retry_config = retry_config
            self.chat = _Chat()
            client_mod.last_client = self

    # mistralai.client.utils, where the real package keeps its retry types.
    # Signatures mirror mistralai 2.9.1: RetryConfig(strategy, backoff,
    # retry_connection_errors) and BackoffStrategy(initial_interval,
    # max_interval, exponent, max_elapsed_time).
    utils_mod = types.ModuleType("mistralai.client.utils")

    class BackoffStrategy:
        def __init__(self, initial_interval, max_interval, exponent, max_elapsed_time):
            self.initial_interval = initial_interval
            self.max_interval = max_interval
            self.exponent = exponent
            self.max_elapsed_time = max_elapsed_time

    class RetryConfig:
        def __init__(self, strategy, backoff, retry_connection_errors):
            self.strategy = strategy
            self.backoff = backoff
            self.retry_connection_errors = retry_connection_errors

    utils_mod.BackoffStrategy = BackoffStrategy
    utils_mod.RetryConfig = RetryConfig
    client_mod.utils = utils_mod

    def _default_response():
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        usage = types.SimpleNamespace(prompt_tokens=31, completion_tokens=12, total_tokens=43)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    client_mod.Mistral = Mistral
    client_mod._response = _default_response
    client_mod.last_client = None
    mod.client = client_mod
    sys.modules["mistralai"] = mod
    sys.modules["mistralai.client"] = client_mod
    sys.modules["mistralai.client.utils"] = utils_mod
    return client_mod


def _uninstall_fake_mistralai():
    for name in ("mistralai.client.utils", "mistralai.client", "mistralai"):
        sys.modules.pop(name, None)


class _Schema(BaseModel):
    score: int
    reason: str


class TestMistralModelBasics(unittest.TestCase):
    def setUp(self):
        self._client_mod = _install_fake_mistralai()

    def tearDown(self):
        _uninstall_fake_mistralai()

    def _model(self, **kw):
        from semantic_svc.evaluators.mistral_model import MistralModel

        kw.setdefault("api_key", "key-fake")
        return MistralModel(**kw)

    def test_default_model_name(self):
        from semantic_svc.evaluators.mistral_model import DEFAULT_MISTRAL_EVAL_MODEL

        self.assertEqual(self._model().get_model_name(), DEFAULT_MISTRAL_EVAL_MODEL)

    def test_explicit_model_name(self):
        self.assertEqual(
            self._model(model="mistral-large-latest").get_model_name(),
            "mistral-large-latest",
        )

    def test_litellm_style_provider_prefix_is_stripped(self):
        """DeepEval's own parse_model_name has its stripping commented out in
        4.0.9, so a "mistral/" prefix would otherwise reach the API verbatim
        while the price table resolved it fine."""
        self.assertEqual(
            self._model(model="mistral/mistral-large-latest").get_model_name(),
            "mistral-large-latest",
        )

    def test_stripped_name_matches_what_the_price_table_resolves(self):
        from explainer_svc.cost import estimate_cost

        name = self._model(model="mistral/mistral-large-latest").get_model_name()
        self.assertAlmostEqual(
            estimate_cost(name, 1_000_000, 1_000_000),
            estimate_cost("mistral/mistral-large-latest", 1_000_000, 1_000_000),
        )

    def test_api_key_reaches_the_client(self):
        self._model(api_key="key-abc")
        self.assertEqual(self._client_mod.last_client.api_key, "key-abc")

    def test_missing_api_key_is_an_actionable_error(self):
        from semantic_svc.evaluators.mistral_model import MistralModel

        with self.assertRaises(ValueError) as ctx:
            MistralModel(api_key=None)
        self.assertIn("MISTRAL_API_KEY", str(ctx.exception))

    def test_missing_sdk_is_an_actionable_error(self):
        _uninstall_fake_mistralai()
        sys.modules["mistralai"] = None  # force ImportError
        try:
            from semantic_svc.evaluators.mistral_model import MistralModel

            with self.assertRaises(ImportError) as ctx:
                MistralModel(api_key="k")
            self.assertIn("mistralai", str(ctx.exception))
        finally:
            del sys.modules["mistralai"]
            self._client_mod = _install_fake_mistralai()


class TestMistralModelGeneration(unittest.TestCase):
    def setUp(self):
        self._client_mod = _install_fake_mistralai()

    def tearDown(self):
        _uninstall_fake_mistralai()

    def _model(self):
        from semantic_svc.evaluators.mistral_model import MistralModel

        return MistralModel(model="mistral-small-latest", api_key="k")

    def test_generate_returns_text_and_cost(self):
        text, cost = self._model().generate("hi")
        self.assertEqual(text, '{"score": 7, "reason": "looks fine"}')
        self.assertIsInstance(cost, EvaluationCost)

    def test_cost_carries_token_counts(self):
        """This is what keeps cost tracking alive for a non-native model:
        accrue_token_usage() only accrues when cost is an EvaluationCost."""
        _, cost = self._model().generate("hi")
        self.assertEqual(cost.input_tokens, 31)
        self.assertEqual(cost.output_tokens, 12)

    def test_generate_with_schema_validates(self):
        result, _ = self._model().generate("hi", schema=_Schema)
        self.assertIsInstance(result, _Schema)
        self.assertEqual(result.score, 7)
        self.assertEqual(result.reason, "looks fine")

    def test_schema_parsing_tolerates_surrounding_prose(self):
        """trim_and_load_json slices first { to last }, which is what makes the
        no-response_format approach survive a chatty model."""
        _uninstall_fake_mistralai()
        self._client_mod = _install_fake_mistralai(
            'Sure! Here is my evaluation:\n{"score": 3, "reason": "weak"}\nHope that helps.'
        )
        result, _ = self._model().generate("hi", schema=_Schema)
        self.assertEqual(result.score, 3)

    def test_list_shaped_content_is_joined(self):
        _uninstall_fake_mistralai()
        self._client_mod = _install_fake_mistralai()

        def _chunked():
            chunks = [
                types.SimpleNamespace(text='{"score": 5,'),
                types.SimpleNamespace(text=' "reason": "ok"}'),
            ]
            message = types.SimpleNamespace(content=chunks)
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1)
            return types.SimpleNamespace(choices=[choice], usage=usage)

        self._client_mod._response = _chunked
        result, _ = self._model().generate("hi", schema=_Schema)
        self.assertEqual(result.score, 5)

    def test_a_generate(self):
        text, cost = asyncio.run(self._model().a_generate("hi"))
        self.assertIn("score", text)
        self.assertEqual(cost.input_tokens, 31)

    def test_no_raw_response_methods(self):
        """GEval treats a missing (a_)generate_raw_response as the signal to
        fall back to the schema path instead of asking for logprobs Mistral
        does not support. Adding these methods would break five of the seven
        evaluators, so their absence is a contract, not an omission."""
        model = self._model()
        self.assertFalse(hasattr(model, "generate_raw_response"))
        self.assertFalse(hasattr(model, "a_generate_raw_response"))


class TestGEvalTokenAccountingWithMistral(unittest.TestCase):
    """The load-bearing claim of the design: a model class defined outside
    DeepEval can never pass is_native_model(), but token accounting still works
    because a_generate_with_schema_and_extract unpacks a (result, cost) tuple
    from a non-native model and accrues it anyway."""

    def setUp(self):
        self._client_mod = _install_fake_mistralai('{"score": 8, "reason": "meets the criteria"}')

    def tearDown(self):
        _uninstall_fake_mistralai()

    def test_mistral_model_is_not_native(self):
        from deepeval.metrics.utils import is_native_model

        from semantic_svc.evaluators.mistral_model import MistralModel

        self.assertFalse(is_native_model(MistralModel(api_key="k")))

    def test_geval_populates_token_counts_despite_not_being_native(self):
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        from semantic_svc.evaluators.mistral_model import MistralModel

        metric = GEval(
            name="Helpfulness",
            evaluation_steps=["Decide whether the answer addresses the question."],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=MistralModel(api_key="k"),
        )
        metric.measure(LLMTestCase(input="2+2?", actual_output="4"))

        self.assertIsNotNone(metric.score)
        self.assertTrue(metric.input_tokens, "input_tokens was not accrued")
        self.assertTrue(metric.output_tokens, "output_tokens was not accrued")

    def test_cost_is_priced_from_dunetrace_table_not_deepeval(self):
        from explainer_svc.cost import estimate_cost

        # 31 in / 12 out at mistral-small's 0.15 / 0.60 per 1M.
        expected = (31 * 0.15 + 12 * 0.60) / 1_000_000
        self.assertAlmostEqual(estimate_cost("mistral-small-latest", 31, 12), expected)


class TestBuildDeepevalModelWithMistral(unittest.TestCase):
    def setUp(self):
        self._client_mod = _install_fake_mistralai()

    def tearDown(self):
        _uninstall_fake_mistralai()

    def _fake_keys(self):
        return patch(
            "semantic_svc.evaluators.models.settings",
            OPENAI_API_KEY="sk-fake",
            ANTHROPIC_API_KEY="sk-ant-fake",
            MISTRAL_API_KEY="mistral-fake",
        )

    def test_mistral_provider_returns_mistral_model(self):
        from semantic_svc.evaluators.mistral_model import MistralModel
        from semantic_svc.evaluators.models import build_deepeval_model

        with self._fake_keys():
            model = build_deepeval_model("mistral", "mistral-large-latest")
        self.assertIsInstance(model, MistralModel)
        self.assertEqual(model.get_model_name(), "mistral-large-latest")

    def test_mistral_default_model(self):
        from semantic_svc.evaluators.models import (
            _DEFAULT_MODEL_BY_PROVIDER,
            build_deepeval_model,
            resolve_model_name,
        )

        self.assertEqual(_DEFAULT_MODEL_BY_PROVIDER["mistral"], "mistral-small-latest")
        self.assertEqual(resolve_model_name("mistral", None), "mistral-small-latest")
        with self._fake_keys():
            model = build_deepeval_model("mistral", None)
        self.assertEqual(model.get_model_name(), "mistral-small-latest")

    def test_client_is_built_with_a_retry_config(self):
        """mistralai defaults retry_config to Unset(), i.e. no retries at all on
        429/5xx, while both other providers carry DeepEval retry decorators. A
        single rate-limit response would otherwise end the evaluation — and the
        run's already-completed evaluators are billable, so the retry pays for
        them twice."""
        from semantic_svc.evaluators.models import build_deepeval_model

        with self._fake_keys():
            build_deepeval_model("mistral", "mistral-small-latest")
        retry = self._client_mod.last_client.retry_config
        self.assertIsNotNone(retry)
        self.assertEqual(retry.strategy, "backoff")
        self.assertTrue(retry.retry_connection_errors)
        # Bounded, so a sustained outage fails rather than pinning the thread.
        self.assertGreater(retry.backoff.max_elapsed_time, 0)

    def test_byok_key_is_passed_through(self):
        from semantic_svc.evaluators.models import build_deepeval_model

        with self._fake_keys():
            build_deepeval_model("mistral", "mistral-small-latest")
        self.assertEqual(self._client_mod.last_client.api_key, "mistral-fake")

    def test_no_silent_fallback_to_openai(self):
        """Selecting mistral must never quietly become an OpenAI call. That
        would ship the run's text to a US API for a customer who chose a
        European provider precisely to avoid it."""
        from semantic_svc.evaluators.models import build_deepeval_model

        with self._fake_keys():
            model = build_deepeval_model("mistral", "mistral-small-latest")
        self.assertNotIsInstance(model, GPTModel)
        self.assertNotIsInstance(model, AnthropicModel)

    def test_mistral_failure_propagates_rather_than_falling_back(self):
        from semantic_svc.evaluators.models import build_deepeval_model

        with patch(
            "semantic_svc.evaluators.models.settings",
            OPENAI_API_KEY="sk-fake",
            ANTHROPIC_API_KEY="sk-ant-fake",
            MISTRAL_API_KEY="",
        ):
            with self.assertRaises(ValueError):
                build_deepeval_model("mistral", "mistral-small-latest")

    def test_existing_providers_are_unchanged(self):
        from semantic_svc.evaluators.models import build_deepeval_model

        with self._fake_keys():
            self.assertIsInstance(build_deepeval_model("openai", "gpt-4o-mini"), GPTModel)
            self.assertIsInstance(
                build_deepeval_model("anthropic", "claude-haiku-4-5"), AnthropicModel
            )
            # An unknown provider is now an error rather than a silent OpenAI
            # call — same reasoning as mistral's own no-fallback rule.
            with self.assertRaises(ValueError):
                build_deepeval_model("nonsense", "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
