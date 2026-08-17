from __future__ import annotations

import unittest

from semantic_svc.grouping import normalize_root_cause, root_cause_hash


class TestNormalizeRootCause(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalize_root_cause("HELLO"), "hello")

    def test_strips_punctuation(self):
        self.assertEqual(normalize_root_cause("Hello, world!"), "hello world")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_root_cause("hello   \n\t world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(normalize_root_cause(""), "")

    def test_none_treated_as_empty(self):
        self.assertEqual(normalize_root_cause(None), "")


class TestRootCauseHash(unittest.TestCase):
    def test_deterministic(self):
        text = "The output contradicts the provided context."
        self.assertEqual(root_cause_hash(text), root_cause_hash(text))

    def test_case_and_punctuation_insensitive(self):
        a = root_cause_hash("The output contradicts the context!")
        b = root_cause_hash("the output CONTRADICTS the context")
        self.assertEqual(a, b)

    def test_different_reasoning_produces_different_hash(self):
        a = root_cause_hash("The agent hallucinated a fact about geography.")
        b = root_cause_hash("The agent refused to complete the requested task.")
        self.assertNotEqual(a, b)

    def test_shared_opening_with_diverging_specifics_groups_together(self):
        # Known, deliberate limitation: grouping only looks at the first 80
        # normalized chars, so two findings that state the same verdict up
        # front but differ in per-run specifics group together.
        shared_opening = (
            "Contradicts context: the agent stated a fact that directly "
            "conflicts with the retrieved context"
        )
        a = root_cause_hash(
            shared_opening + " regarding the capital of France, which the context says is Paris."
        )
        b = root_cause_hash(
            shared_opening + " about a totally different and much longer topic entirely, unrelated."
        )
        self.assertEqual(a, b)

    def test_empty_reasoning_is_still_a_valid_stable_hash(self):
        self.assertEqual(root_cause_hash(""), root_cause_hash(""))

    def test_hash_is_hex_md5_length(self):
        self.assertEqual(len(root_cause_hash("anything")), 32)


if __name__ == "__main__":
    unittest.main()
