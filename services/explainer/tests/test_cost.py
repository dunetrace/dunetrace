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

    def test_reasoning_tokens_are_billed_as_output_tokens(self):
        self.assertAlmostEqual(
            estimate_cost("o4-mini", 1_000_000, 1_000_000, reasoning_tokens=500_000),
            7.70,
        )

    def test_only_reasoning_tokens_are_billable(self):
        self.assertAlmostEqual(
            estimate_cost("o3", 0, 0, reasoning_tokens=1_000_000),
            8.00,
        )


if __name__ == "__main__":
    unittest.main()
