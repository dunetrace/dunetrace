"""
Tests for the pure adaptive sampling engine — no DB, no mocking needed.
"""

from __future__ import annotations

import unittest

from semantic_svc.sampling import (
    MIN_CONVERSATION_RUNS,
    SamplingRates,
    decide_conversation_sampling,
    decide_sampling,
    stable_bucket,
    with_overrides,
)


class TestStableBucket(unittest.TestCase):
    def test_bucket_in_range(self):
        for i in range(500):
            b = stable_bucket(f"run-{i}")
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, 100)

    def test_deterministic_across_calls(self):
        for i in range(50):
            key = f"run-{i}"
            self.assertEqual(stable_bucket(key), stable_bucket(key))

    def test_different_keys_generally_differ(self):
        buckets = {stable_bucket(f"run-{i}") for i in range(200)}
        # 200 distinct keys into 100 buckets — collisions expected, but not everything
        # colliding into one bucket.
        self.assertGreater(len(buckets), 20)


class TestDecideSamplingPrecedence(unittest.TestCase):
    def test_structural_signal_wins_over_everything(self):
        agent_config = {"sample_rate": 0.0, "semantic_critical": False}
        sampled, reason, rate = decide_sampling(
            "run-x", has_structural_signal=True, has_retrieval_event=True, agent_config=agent_config
        )
        self.assertTrue(sampled)
        self.assertEqual(reason, "structural_signal")
        self.assertEqual(rate, 1.0)

    def test_semantic_critical_wins_over_retrieval_and_override(self):
        agent_config = {"sample_rate": 0.0, "semantic_critical": True}
        sampled, reason, rate = decide_sampling(
            "run-x",
            has_structural_signal=False,
            has_retrieval_event=True,
            agent_config=agent_config,
        )
        self.assertTrue(sampled)
        self.assertEqual(reason, "semantic_critical")
        self.assertEqual(rate, 1.0)

    def test_agent_override_wins_over_retrieval(self):
        agent_config = {"sample_rate": 0.0, "semantic_critical": False}
        sampled, reason, rate = decide_sampling(
            "run-x",
            has_structural_signal=False,
            has_retrieval_event=True,
            agent_config=agent_config,
        )
        self.assertFalse(sampled)
        self.assertEqual(reason, "agent_override")
        self.assertEqual(rate, 0.0)

    def test_retrieval_used_when_no_override_and_no_gates(self):
        sampled, reason, rate = decide_sampling(
            "run-x", has_structural_signal=False, has_retrieval_event=True, agent_config=None
        )
        self.assertEqual(reason, "retrieval")
        self.assertEqual(rate, SamplingRates().retrieval_rate)

    def test_baseline_used_when_nothing_else_applies(self):
        sampled, reason, rate = decide_sampling(
            "run-x", has_structural_signal=False, has_retrieval_event=False, agent_config=None
        )
        self.assertEqual(reason, "baseline")
        self.assertEqual(rate, SamplingRates().baseline_rate)

    def test_agent_config_without_sample_rate_key_falls_through(self):
        # e.g. an agent_config row that only sets budget_monthly, no sample_rate override.
        agent_config = {"sample_rate": None, "semantic_critical": False}
        sampled, reason, rate = decide_sampling(
            "run-x",
            has_structural_signal=False,
            has_retrieval_event=False,
            agent_config=agent_config,
        )
        self.assertEqual(reason, "baseline")

    def test_rate_zero_never_samples(self):
        for i in range(200):
            sampled, reason, rate = decide_sampling(
                f"run-{i}",
                has_structural_signal=False,
                has_retrieval_event=False,
                agent_config={"sample_rate": 0.0, "semantic_critical": False},
            )
            self.assertFalse(sampled)

    def test_rate_one_always_samples(self):
        for i in range(200):
            sampled, reason, rate = decide_sampling(
                f"run-{i}",
                has_structural_signal=False,
                has_retrieval_event=False,
                agent_config={"sample_rate": 1.0, "semantic_critical": False},
            )
            self.assertTrue(sampled)


class TestDeterminism(unittest.TestCase):
    def test_same_run_id_same_decision_across_repeated_calls(self):
        for i in range(100):
            run_id = f"run-determinism-{i}"
            first = decide_sampling(
                run_id, has_structural_signal=False, has_retrieval_event=False, agent_config=None
            )
            second = decide_sampling(
                run_id, has_structural_signal=False, has_retrieval_event=False, agent_config=None
            )
            self.assertEqual(first, second)


class TestSamplingRatesMatchSpecOverLargeSamples(unittest.TestCase):
    """Statistical check: over a large number of distinct run_ids, the fraction
    sampled should land close to the configured rate. sha256-derived buckets
    are effectively uniform, so tolerance can be tight without flaking."""

    N = 20_000
    TOLERANCE = 0.01  # 1 percentage point

    def _observed_rate(self, **kwargs) -> float:
        sampled_count = sum(
            1 for i in range(self.N) if decide_sampling(f"stat-run-{i}", **kwargs)[0]
        )
        return sampled_count / self.N

    def test_baseline_rate_matches_spec(self):
        rate = self._observed_rate(
            has_structural_signal=False, has_retrieval_event=False, agent_config=None
        )
        self.assertAlmostEqual(rate, SamplingRates().baseline_rate, delta=self.TOLERANCE)

    def test_retrieval_rate_matches_spec(self):
        rate = self._observed_rate(
            has_structural_signal=False, has_retrieval_event=True, agent_config=None
        )
        self.assertAlmostEqual(rate, SamplingRates().retrieval_rate, delta=self.TOLERANCE)

    def test_structural_signal_rate_is_100_percent(self):
        rate = self._observed_rate(
            has_structural_signal=True, has_retrieval_event=False, agent_config=None
        )
        self.assertEqual(rate, 1.0)

    def test_semantic_critical_rate_is_100_percent(self):
        rate = self._observed_rate(
            has_structural_signal=False,
            has_retrieval_event=False,
            agent_config={"sample_rate": None, "semantic_critical": True},
        )
        self.assertEqual(rate, 1.0)

    def test_custom_rates_from_config_are_respected(self):
        custom = SamplingRates(baseline_rate=0.30)
        sampled_count = sum(
            1
            for i in range(self.N)
            if decide_sampling(
                f"custom-run-{i}",
                has_structural_signal=False,
                has_retrieval_event=False,
                agent_config=None,
                rates=custom,
            )[0]
        )
        self.assertAlmostEqual(sampled_count / self.N, 0.30, delta=self.TOLERANCE)


class TestDecideConversationSampling(unittest.TestCase):
    def test_below_min_runs_never_sampled_regardless_of_rate(self):
        rates = SamplingRates(conversation_evaluation_rate=1.0)
        for run_count in range(MIN_CONVERSATION_RUNS):
            sampled, reason = decide_conversation_sampling("conv-1", run_count, rates=rates)
            self.assertFalse(sampled)
            self.assertEqual(reason, "too_few_runs")

    def test_rate_zero_never_samples_even_above_min_runs(self):
        rates = SamplingRates(conversation_evaluation_rate=0.0)
        sampled, reason = decide_conversation_sampling("conv-1", MIN_CONVERSATION_RUNS, rates=rates)
        self.assertFalse(sampled)
        self.assertEqual(reason, "not_sampled")

    def test_rate_one_always_samples_once_above_min_runs(self):
        rates = SamplingRates(conversation_evaluation_rate=1.0)
        for i in range(50):
            sampled, reason = decide_conversation_sampling(
                f"conv-{i}", MIN_CONVERSATION_RUNS, rates=rates
            )
            self.assertTrue(sampled)
            self.assertEqual(reason, "conversation_sample_rate")

    def test_same_conversation_id_same_decision_across_repeated_calls(self):
        for i in range(50):
            conv_id = f"conv-determinism-{i}"
            first = decide_conversation_sampling(conv_id, MIN_CONVERSATION_RUNS)
            second = decide_conversation_sampling(conv_id, MIN_CONVERSATION_RUNS)
            self.assertEqual(first, second)

    def test_rate_matches_spec_over_large_sample(self):
        rates = SamplingRates(conversation_evaluation_rate=0.30)
        n = 20_000
        sampled_count = sum(
            1
            for i in range(n)
            if decide_conversation_sampling(f"conv-stat-{i}", MIN_CONVERSATION_RUNS, rates=rates)[0]
        )
        self.assertAlmostEqual(sampled_count / n, 0.30, delta=0.01)


class TestWithOverrides(unittest.TestCase):
    def test_applies_only_given_keys(self):
        base = SamplingRates()
        updated = with_overrides(base, baseline_rate=0.5)
        self.assertEqual(updated.baseline_rate, 0.5)
        self.assertEqual(updated.retrieval_rate, base.retrieval_rate)
        self.assertEqual(updated.structural_signal_rate, base.structural_signal_rate)

    def test_no_overrides_returns_equivalent_rates(self):
        base = SamplingRates()
        self.assertEqual(with_overrides(base), base)


if __name__ == "__main__":
    unittest.main()
