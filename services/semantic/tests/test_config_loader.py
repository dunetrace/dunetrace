from __future__ import annotations

import os
import tempfile
import unittest

from semantic_svc.config_loader import load_evaluator_config, load_sampling_rates
from semantic_svc.sampling import SamplingRates


class TestLoadSamplingRates(unittest.TestCase):
    def _write(self, contents: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, "w") as f:
            f.write(contents)
        self.addCleanup(os.remove, path)
        return path

    def test_missing_file_returns_defaults(self):
        rates = load_sampling_rates(config_path="/nonexistent/path/semantic-sampling.yml")
        self.assertEqual(rates, SamplingRates())

    def test_valid_overrides_applied(self):
        path = self._write(
            "structural_signal_rate: 1.0\n"
            "semantic_critical_rate: 1.0\n"
            "retrieval_rate: 0.5\n"
            "baseline_rate: 0.1\n"
        )
        rates = load_sampling_rates(config_path=path)
        self.assertEqual(rates.retrieval_rate, 0.5)
        self.assertEqual(rates.baseline_rate, 0.1)

    def test_partial_file_keeps_remaining_defaults(self):
        path = self._write("baseline_rate: 0.2\n")
        rates = load_sampling_rates(config_path=path)
        self.assertEqual(rates.baseline_rate, 0.2)
        self.assertEqual(rates.retrieval_rate, SamplingRates().retrieval_rate)

    def test_conversation_evaluation_rate_override_applied(self):
        path = self._write("conversation_evaluation_rate: 0.75\n")
        rates = load_sampling_rates(config_path=path)
        self.assertEqual(rates.conversation_evaluation_rate, 0.75)

    def test_out_of_range_value_falls_back_to_default(self):
        path = self._write("baseline_rate: 1.5\n")
        rates = load_sampling_rates(config_path=path)
        self.assertEqual(rates.baseline_rate, SamplingRates().baseline_rate)

    def test_non_numeric_value_falls_back_to_default(self):
        path = self._write("baseline_rate: not_a_number\n")
        rates = load_sampling_rates(config_path=path)
        self.assertEqual(rates.baseline_rate, SamplingRates().baseline_rate)

    def test_empty_file_returns_defaults(self):
        path = self._write("")
        rates = load_sampling_rates(config_path=path)
        self.assertEqual(rates, SamplingRates())

    def test_unrelated_keys_ignored(self):
        path = self._write("some_other_key: 42\nbaseline_rate: 0.15\n")
        rates = load_sampling_rates(config_path=path)
        self.assertEqual(rates.baseline_rate, 0.15)


class TestLoadEvaluatorConfig(unittest.TestCase):
    def _write(self, contents: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, "w") as f:
            f.write(contents)
        self.addCleanup(os.remove, path)
        return path

    def test_missing_file_returns_empty(self):
        config = load_evaluator_config(config_path="/nonexistent/semantic-evaluators.yml")
        self.assertEqual(config, {})

    def test_require_second_opinion_true_parsed(self):
        path = self._write("HALLUCINATION:\n  require_second_opinion: true\n")
        config = load_evaluator_config(config_path=path)
        self.assertTrue(config["HALLUCINATION"]["require_second_opinion"])

    def test_missing_require_second_opinion_key_defaults_false(self):
        path = self._write("HALLUCINATION:\n  alert_confidence_floor: 0.6\n")
        config = load_evaluator_config(config_path=path)
        self.assertFalse(config["HALLUCINATION"]["require_second_opinion"])

    def test_second_opinion_provider_and_model_parsed(self):
        path = self._write(
            "HALLUCINATION:\n"
            "  require_second_opinion: true\n"
            "  second_opinion_provider: anthropic\n"
            "  second_opinion_model: claude-sonnet-4-6\n"
        )
        config = load_evaluator_config(config_path=path)
        self.assertEqual(config["HALLUCINATION"]["second_opinion_provider"], "anthropic")
        self.assertEqual(config["HALLUCINATION"]["second_opinion_model"], "claude-sonnet-4-6")

    def test_provider_and_model_default_to_none_when_omitted(self):
        path = self._write("HALLUCINATION:\n  require_second_opinion: true\n")
        config = load_evaluator_config(config_path=path)
        self.assertIsNone(config["HALLUCINATION"]["second_opinion_provider"])
        self.assertIsNone(config["HALLUCINATION"]["second_opinion_model"])

    def test_evaluator_name_uppercased(self):
        path = self._write("hallucination:\n  require_second_opinion: true\n")
        config = load_evaluator_config(config_path=path)
        self.assertIn("HALLUCINATION", config)

    def test_empty_file_returns_empty_dict(self):
        path = self._write("")
        self.assertEqual(load_evaluator_config(config_path=path), {})

    def test_multiple_evaluators_parsed_independently(self):
        path = self._write(
            "HALLUCINATION:\n"
            "  require_second_opinion: true\n"
            "TASK_COMPLETION:\n"
            "  require_second_opinion: false\n"
        )
        config = load_evaluator_config(config_path=path)
        self.assertTrue(config["HALLUCINATION"]["require_second_opinion"])
        self.assertFalse(config["TASK_COMPLETION"]["require_second_opinion"])


if __name__ == "__main__":
    unittest.main()
