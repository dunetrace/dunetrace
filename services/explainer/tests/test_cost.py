"""
Tests for model pricing and cost estimation.

Run:
    cd services/explainer
    python -m unittest tests.test_cost -v
"""

from __future__ import annotations

import os
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location


cost_path = os.path.join(os.path.dirname(__file__), "..", "explainer_svc", "cost.py")
spec = spec_from_file_location("explainer_cost", cost_path)
assert spec and spec.loader
cost = module_from_spec(spec)
sys.modules[spec.name] = cost
spec.loader.exec_module(cost)
estimate_cost = cost.estimate_cost


class TestCostEstimation(unittest.TestCase):
    def test_new_model_price_lookups(self):
        cases = [
            ("o3", 10.00),
            ("meta-llama/Llama-3.3-70B-Instruct-Turbo", 1.76),
            ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", 7.00),
            ("deepseek-ai/DeepSeek-R1", 10.00),
            ("deepseek-ai/DeepSeek-V3", 2.50),
            ("Qwen/Qwen2.5-72B-Instruct-Turbo", 2.40),
        ]

        for model, expected in cases:
            with self.subTest(model=model):
                self.assertAlmostEqual(
                    estimate_cost(model, 1_000_000, 1_000_000),
                    expected,
                )

    def test_mistral_price_lookups(self):
        # USD for 1M input + 1M output, verified against
        # https://mistral.ai/pricing/api on 2026-08-08. The old table priced
        # mistral-large at (3.00, 9.00), which was 6x the current Large 3 rate.
        cases = [
            ("mistral-large-latest", 2.00),
            ("mistral-large-2512", 2.00),
            ("mistral-medium-latest", 9.00),
            ("mistral-medium-3-5", 9.00),
            ("mistral-small-latest", 0.75),
            ("mistral-small-2603", 0.75),
            ("ministral-3b-latest", 0.20),
            ("ministral-8b-2512", 0.30),
            ("ministral-14b-latest", 0.40),
            ("codestral-2508", 1.20),
            # Vendor prefix is stripped by _normalise before lookup.
            ("mistral/mistral-large-latest", 2.00),
        ]

        for model, expected in cases:
            with self.subTest(model=model):
                self.assertAlmostEqual(estimate_cost(model, 1_000_000, 1_000_000), expected)

    def test_mistral_embeddings_bill_input_only(self):
        self.assertAlmostEqual(estimate_cost("mistral-embed", 1_000_000, 0), 0.10)
        self.assertAlmostEqual(estimate_cost("codestral-embed", 1_000_000, 0), 0.15)

    def test_codestral_embed_does_not_take_the_chat_rate(self):
        # _normalise sorts keys longest-first, so codestral-embed must not
        # normalise down to codestral (0.30 in, 0.90 out).
        self.assertAlmostEqual(estimate_cost("codestral-embed", 1_000_000, 0), 0.15)

    def test_ministral_does_not_collide_with_mistral_small(self):
        self.assertAlmostEqual(estimate_cost("ministral-8b-latest", 1_000_000, 0), 0.15)

    def test_reasoning_tokens_are_billed_as_output_tokens(self):
        # reasoning_tokens is a subset of completion_tokens; no double-billing
        self.assertAlmostEqual(
            estimate_cost("o4-mini", 1_000_000, 1_000_000, reasoning_tokens=500_000),
            5.50,
        )

    def test_only_reasoning_tokens_are_billable(self):
        # Pure reasoning run: completion_tokens == reasoning_tokens
        self.assertAlmostEqual(
            estimate_cost("o3", 0, 1_000_000, reasoning_tokens=1_000_000),
            8.00,
        )


class TestPriceTableParityWithSdk(unittest.TestCase):
    """This table and the SDK's `_MODEL_PRICES` price the same models in two
    processes: the SDK evaluates the `cost_usd` policy trigger in-process before
    a run finishes, this one produces every server-side cost rollup. They are
    separate because the SDK must keep zero dependencies, so nothing but a test
    stops them drifting — and they had drifted, with `gpt-4o` priced at double
    here for months.

    Units differ on purpose: the SDK stores USD per token, this stores USD per
    1M tokens.
    """

    @staticmethod
    def _sdk_prices():
        from dunetrace.policies import _MODEL_PRICES

        return {
            name: (round(p["input"] * 1e6, 6), round(p["output"] * 1e6, 6))
            for name, p in _MODEL_PRICES.items()
        }

    def test_shared_models_are_priced_identically(self):
        service = {k: (float(i), float(o)) for k, (i, o) in cost._PRICES.items()}
        sdk = self._sdk_prices()
        shared = sorted(set(sdk) & set(service))
        self.assertTrue(shared, "the two tables share no model keys at all")
        mismatched = {k: (sdk[k], service[k]) for k in shared if sdk[k] != service[k]}
        self.assertEqual(
            mismatched,
            {},
            "price tables disagree (sdk, service) — update both or neither",
        )

    def test_every_mistral_model_is_in_both_tables(self):
        """The two tables are allowed to carry different model *sets* — the SDK
        ships a smaller one. Mistral is the exception: it was added to both in
        one change, so a key missing from either is a mistake rather than a
        deliberate omission."""
        sdk = {k for k in self._sdk_prices() if "mistral" in k or "codestral" in k}
        service = {k for k in cost._PRICES if "mistral" in k or "codestral" in k}
        self.assertEqual(sdk, service)


if __name__ == "__main__":
    unittest.main()
