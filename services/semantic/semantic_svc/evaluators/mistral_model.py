"""
A DeepEval model class for Mistral's La Plateforme API.

DeepEval 4.0.9 ships no Mistral class (models/llms/ has 14 files, none of them
mistral), so this fills the gap. It mirrors DeepEval's own AnthropicModel
(models/llms/anthropic_model.py) rather than inventing a shape: same
(result, cost) return contract, same "parse the text, don't send the schema to
the API" approach to structured output.

Two DeepEval internals decide this design, both verified against the installed
4.0.9 rather than the docs.

1. GEval asks for token logprobs, and Mistral has none. metrics/g_eval/g_eval.py
   wraps its logprob path in `try: ... except AttributeError:` (:332, :401) and
   treats a missing a_generate_raw_response as the signal to fall back to the
   plain schema path. no_log_prob_support() (metrics/g_eval/utils.py:248) only
   returns True for GPTModel/AzureOpenAIModel, so that AttributeError is the
   only available escape hatch. This class therefore deliberately does NOT
   implement generate_raw_response/a_generate_raw_response. AnthropicModel
   doesn't either, which is why the existing anthropic provider works. Five of
   the seven evaluators are GEval or ConversationalGEval, so getting this wrong
   would break most of them.

   The routing via LiteLLMModel would have looked simpler, and LiteLLMModel is
   in the native whitelist, but it *does* implement generate_raw_response and
   sends logprobs=True (models/llms/litellm_model.py:153). Mistral rejects that,
   and a provider BadRequestError is not an AttributeError, so the fallback
   above would not catch it. litellm also isn't installed here.

2. Token accounting survives a non-native model. is_native_model()
   (metrics/utils.py:706) is a hardcoded isinstance check over DeepEval's own 13
   classes, so a class defined here can never be "native". That does not zero
   token tracking: a_generate_with_schema_and_extract (metrics/utils.py:543-549)
   unpacks a (result, cost) tuple from a non-native model and calls
   accrue_token_usage() on it anyway, as long as cost is an EvaluationCost
   carrying input_tokens/output_tokens. So the evaluators' metric.input_tokens /
   metric.output_tokens reads keep working.

Pricing is not passed to DeepEval and metric.evaluation_cost is ignored, exactly
as for the other two providers. Evaluators read the token counts and call
explainer_svc.cost.estimate_cost() themselves. See evaluators/models.py's
docstring for why DeepEval's bundled prices can't be trusted here.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import EvaluationCost
from pydantic import BaseModel

# Cheapest current chat model that still follows JSON instructions reliably.
# Matches the cost-conscious defaults the other two providers use.
DEFAULT_MISTRAL_EVAL_MODEL = "mistral-small-latest"

_MISSING_SDK_MESSAGE = (
    "SEMANTIC_LLM_PROVIDER=mistral needs the 'mistralai' package (v2 or newer). "
    "Install it with: pip install 'mistralai>=2.0'"
)


class MistralModel(DeepEvalBaseLLM):
    """Evaluator LLM backed by Mistral La Plateforme.

    Deliberately has no generate_raw_response/a_generate_raw_response. See the
    module docstring: their absence is what routes GEval down the schema path
    instead of asking Mistral for logprobs it does not support.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        self._api_key = api_key or None
        self.temperature = temperature
        # DeepEval's parse_model_name() advertises "<provider>/<model>"
        # stripping, but the implementation is commented out in 4.0.9 and it
        # returns the name unchanged. A LiteLLM-style "mistral/mistral-large-
        # latest" would then reach the API verbatim and be rejected, while
        # Dunetrace's own price table resolves it fine (explainer_svc/cost.py
        # strips the same prefix in _normalise). Strip it here so the model
        # actually called and the model priced are the same one.
        name = model or DEFAULT_MISTRAL_EVAL_MODEL
        if name.startswith("mistral/"):
            name = name[len("mistral/") :]
        # Base __init__ calls load_model(), so the key has to be set before it.
        super().__init__(model=name)

    def load_model(self, *args, **kwargs):
        """One client serves both sync and async: mistralai v2 puts *_async
        methods on the same class rather than shipping a separate async client."""
        try:
            from mistralai.client import Mistral
        except ImportError as exc:  # pragma: no cover - depends on install shape
            raise ImportError(_MISSING_SDK_MESSAGE) from exc

        if not self._api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. The semantic worker needs it to run "
                "evaluators with SEMANTIC_LLM_PROVIDER=mistral."
            )
        # mistralai defaults retry_config to Unset(), which means zero SDK-level
        # retries on 429/5xx (client/sdkconfiguration.py; chat.py only builds a
        # policy for a real RetryConfig). Both other providers retry — DeepEval's
        # GPTModel and AnthropicModel each carry a create_retry_decorator — so
        # without this a single rate-limit response ends the evaluation. Mistral's
        # per-minute limit is easy to hit when a poll batch evaluates
        # concurrently, and a semantic evaluation is a billable call: failing it
        # means paying for the run's other evaluators again on the next poll.
        return Mistral(api_key=self._api_key, retry_config=self._retry_config())

    @staticmethod
    def _retry_config():
        """Exponential backoff on the connection and the retryable status codes.
        Bounded by max_elapsed_time so a sustained outage fails the evaluation
        rather than pinning a worker thread indefinitely."""
        from mistralai.client.utils import BackoffStrategy, RetryConfig

        return RetryConfig(
            "backoff",
            BackoffStrategy(
                initial_interval=500,
                max_interval=8_000,
                exponent=1.5,
                max_elapsed_time=30_000,
            ),
            retry_connection_errors=True,
        )

    def get_model_name(self, *args, **kwargs) -> str:
        return self.name

    # ── generation ────────────────────────────────────────────────────────────

    def _request_kwargs(self, prompt: str) -> dict:
        request = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        return request

    @staticmethod
    def _cost(usage) -> EvaluationCost:
        """Token counts ride along on the returned cost. The dollar value stays
        0.0 on purpose: evaluators price Mistral from Dunetrace's own table via
        estimate_cost(), and a second number here would just be a fork of it."""
        return EvaluationCost(
            0.0,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    @staticmethod
    def _text(response) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multimodal and reasoning replies arrive as a chunk list.
            return "".join(str(getattr(chunk, "text", "") or "") for chunk in content)
        return ""

    def _finish(self, response, schema):
        text = self._text(response)
        cost = self._cost(getattr(response, "usage", None))
        if schema is None:
            return text, cost
        # The schema is never sent to the API, matching AnthropicModel: the
        # evaluator prompt already asks for JSON and this parses what comes
        # back. Phase 5 measures how reliable that is for Mistral specifically.
        return schema.model_validate(trim_and_load_json(text)), cost

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], EvaluationCost]:
        response = self.model.chat.complete(**self._request_kwargs(prompt))
        return self._finish(response, schema)

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], EvaluationCost]:
        response = await self.model.chat.complete_async(**self._request_kwargs(prompt))
        return self._finish(response, schema)
