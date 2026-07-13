"""
Builds a DeepEval-native model instance for a given provider + model name.

Deliberately instantiates GPTModel/AnthropicModel explicitly rather than using
DeepEval's own should_use_anthropic_model()/CLI-keyfile provider switch — that
switch is global process state (a CLI-managed settings file or process-wide
env flag), unsafe in a worker that may evaluate different orgs/agents with
different provider configs concurrently. Passing an explicit instance makes
DeepEval's own is_native_model() check succeed via isinstance(), which is what
turns on DeepEval's native per-call token tracking (metric.input_tokens/
output_tokens after measure()).

Token pricing is NOT sourced from these constructors' cost_per_input_token/
cost_per_output_token args, on purpose: verified against DeepEval 4.0.9's
require_costs() (models/utils.py) that when a model is already in DeepEval's
own bundled OPENAI_MODELS_DATA/ANTHROPIC_MODELS_DATA table, DeepEval's own
price silently wins over any override passed at construction — e.g. its
claude-haiku-4-5 entry is $1.00/$5.00 per 1M tokens, ours
(explainer_svc/cost.py) says $0.80/$4.00. Trusting the override would mean our
own price table is authoritative only for models DeepEval doesn't already
know, silently forked for every model it does. Instead: evaluators read
metric.input_tokens/output_tokens (accurate — reported by the real API
response) and call explainer_svc.cost.estimate_cost() themselves for the
dollar figure, ignoring metric.evaluation_cost entirely. See evaluators/
hallucination.py and task_completion.py.
"""

from __future__ import annotations

from semantic_svc.config import settings

_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


def resolve_model_name(provider: str, model_name: str | None) -> str:
    return model_name or _DEFAULT_MODEL_BY_PROVIDER.get(provider, "gpt-4o-mini")


def build_deepeval_model(provider: str, model_name: str | None = None):
    """provider: "openai" | "anthropic". model_name falls back to a
    cost-conscious per-provider default when empty/None."""
    resolved = resolve_model_name(provider, model_name)

    if provider == "anthropic":
        from deepeval.models import AnthropicModel

        return AnthropicModel(model=resolved, api_key=settings.ANTHROPIC_API_KEY or None)

    from deepeval.models import GPTModel

    return GPTModel(model=resolved, api_key=settings.OPENAI_API_KEY or None)
