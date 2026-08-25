"""
Cost aggregation for the per-agent performance-trends chart.

The bug: `agent_performance_trends` summed `prompt_tokens` across BOTH
`llm.called` and `llm.responded` in one pass. The same call reports that field
twice — llm.called carries the SDK's chars//4 estimate, llm.responded the exact
count once the provider reports it — so the chart plotted roughly the estimate
plus the exact count. Measured against the live dev database, agent "agent"
(12,342 runs) came out 152% overstated, while the run-detail page — which
already overrides (queries.py::get_run_detail) — showed the correct figure for
the very same runs.

Two sibling endpoints had already been fixed for this; performance-trends was
missed. These tests pin the rule so it cannot regress a third time.

No DB and no credentials: the arithmetic lives in the pure layer per that
module's own docstring, and estimate_cost is injected.

Run from services/api/ with:
  PYTHONPATH=../../packages/sdk-py:../explainer:. \
    python -m pytest tests/test_performance_trends_cost.py -v
"""

import unittest

from api_svc.performance_trends import compute_cost_by_day, run_prompt_tokens


def _row(day="2026-08-25", model="gpt-4o", responded=0, called=0, completion=0):
    return {
        "day": day,
        "model": model,
        "prompt_tokens_responded": responded,
        "prompt_tokens_called": called,
        "completion_tokens": completion,
    }


class TestRunPromptTokens(unittest.TestCase):
    def test_overrides_rather_than_adds(self):
        """The regression, in one line: 1 + 12 must be 12, not 13."""
        self.assertEqual(run_prompt_tokens(responded=12, called=1), 12)

    def test_real_shape_from_the_dev_database(self):
        # agent "agent", run 00042bc0…: llm.called estimated 1, llm.responded
        # reported 12. The old SUM produced 13.
        self.assertEqual(run_prompt_tokens(12, 1), 12)
        self.assertEqual(run_prompt_tokens(20, 1), 20)
        self.assertEqual(run_prompt_tokens(17, 3), 17)

    def test_falls_back_to_called_when_responded_is_absent(self):
        """Direct-SDK traces report tokens only on llm.called. Taking the
        responded value unconditionally would zero their cost entirely."""
        self.assertEqual(run_prompt_tokens(responded=0, called=6612), 6612)
        self.assertEqual(run_prompt_tokens(responded=None, called=6612), 6612)

    def test_uses_responded_when_called_is_absent(self):
        """LangChain-shaped traces report tokens only on llm.responded."""
        self.assertEqual(run_prompt_tokens(responded=23455, called=0), 23455)

    def test_both_absent_is_zero(self):
        self.assertEqual(run_prompt_tokens(None, None), 0)
        self.assertEqual(run_prompt_tokens(0, 0), 0)


class TestComputeCostByDay(unittest.TestCase):
    @staticmethod
    def _price(model, prompt, completion):
        # Stand-in for estimate_cost: $1 per prompt token, $2 per completion.
        return prompt * 1.0 + completion * 2.0

    def test_does_not_double_count_prompt_tokens(self):
        out = compute_cost_by_day([_row(responded=12, called=1, completion=5)], self._price)
        self.assertEqual(out, {"2026-08-25": 12 * 1.0 + 5 * 2.0})

    def test_old_summing_behaviour_would_have_been_higher(self):
        """Guards the direction of the fix, not just its value."""
        out = compute_cost_by_day([_row(responded=12, called=1)], self._price)
        summed = compute_cost_by_day([_row(responded=13, called=0)], self._price)
        self.assertLess(out["2026-08-25"], summed["2026-08-25"])

    def test_sums_across_runs_within_a_day(self):
        rows = [_row(responded=10), _row(responded=5), _row(responded=1)]
        self.assertEqual(compute_cost_by_day(rows, self._price), {"2026-08-25": 16.0})

    def test_keeps_days_separate(self):
        rows = [_row(day="2026-08-24", responded=3), _row(day="2026-08-25", responded=7)]
        self.assertEqual(
            compute_cost_by_day(rows, self._price), {"2026-08-24": 3.0, "2026-08-25": 7.0}
        )

    def test_missing_model_is_priced_as_unknown_not_dropped(self):
        seen = {}

        def price(model, prompt, completion):
            seen["model"] = model
            return 0.0

        compute_cost_by_day([_row(model=None, responded=5)], price)
        self.assertEqual(seen["model"], "unknown")

    def test_no_rows_is_an_empty_mapping(self):
        self.assertEqual(compute_cost_by_day([], self._price), {})

    def test_zero_token_run_contributes_zero(self):
        """The seeded agent the user hit: llm.responded with no token counts at
        all. Cost is honestly 0 — the chart's job is to say so, not to guess."""
        out = compute_cost_by_day([_row(model=None, responded=0, called=0)], self._price)
        self.assertEqual(out, {"2026-08-25": 0.0})


class TestAgainstTheRealPriceTable(unittest.TestCase):
    """One end-to-end case through the real estimate_cost, so the wiring is
    covered and not just the arithmetic."""

    def test_real_estimate_cost_is_applied_once_per_run(self):
        from explainer_svc.cost import estimate_cost

        rows = [_row(model="gpt-4o", responded=1_000_000, called=400_000, completion=0)]
        out = compute_cost_by_day(rows, estimate_cost)
        # gpt-4o input is $2.50/1M — the 400k estimate must not be added on top.
        self.assertAlmostEqual(out["2026-08-25"], 2.50, places=6)


if __name__ == "__main__":
    unittest.main()
