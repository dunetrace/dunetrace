"""
Tests for the condition-expression evaluator (dunetrace.policies.evaluator).

Every operator with true and false cases, missing-field handling, type
coercion, composition (AND/OR short-circuit + trace completeness), the
observability trace, regex safety, determinism, and a micro-benchmark that
asserts the <100µs target on a representative policy.
"""

import time
import unittest

from dunetrace.policies import parse_match_block
from dunetrace.policies.evaluator import (
    ComparisonTrace,
    EvaluationContext,
    evaluate,
)


def ev(block, *, args=None, run=None, agent=None, org=None, event=None, trace=None):
    expr = parse_match_block(block, policy_name="t")
    ctx = EvaluationContext(
        args=args or {}, run=run or {}, agent=agent or {}, org=org or {}, event=event or {}
    )
    return evaluate(expr, ctx, trace=trace)


class TestComparisonOperators(unittest.TestCase):
    def test_eq(self):
        self.assertTrue(ev({"args.x": {"eq": 5}}, args={"x": 5}))
        self.assertFalse(ev({"args.x": {"eq": 5}}, args={"x": 6}))

    def test_ne(self):
        self.assertTrue(ev({"args.x": {"ne": 5}}, args={"x": 6}))
        self.assertFalse(ev({"args.x": {"ne": 5}}, args={"x": 5}))

    def test_gt_gte(self):
        self.assertTrue(ev({"args.x": {"gt": 10}}, args={"x": 11}))
        self.assertFalse(ev({"args.x": {"gt": 10}}, args={"x": 10}))
        self.assertTrue(ev({"args.x": {"gte": 10}}, args={"x": 10}))

    def test_lt_lte(self):
        self.assertTrue(ev({"args.x": {"lt": 10}}, args={"x": 9}))
        self.assertFalse(ev({"args.x": {"lt": 10}}, args={"x": 10}))
        self.assertTrue(ev({"args.x": {"lte": 10}}, args={"x": 10}))

    def test_in_not_in(self):
        self.assertTrue(ev({"org.plan": {"in": ["free", "starter"]}}, org={"plan": "free"}))
        self.assertFalse(ev({"org.plan": {"in": ["free", "starter"]}}, org={"plan": "pro"}))
        self.assertTrue(ev({"org.plan": {"not_in": ["free"]}}, org={"plan": "pro"}))
        self.assertFalse(ev({"org.plan": {"not_in": ["free"]}}, org={"plan": "free"}))

    def test_contains_string(self):
        self.assertTrue(ev({"args.name": {"contains": "cat"}}, args={"name": "concatenate"}))
        self.assertFalse(ev({"args.name": {"contains": "dog"}}, args={"name": "concatenate"}))

    def test_contains_list(self):
        self.assertTrue(ev({"args.tags": {"contains": "urgent"}}, args={"tags": ["a", "urgent"]}))
        self.assertFalse(ev({"args.tags": {"contains": "urgent"}}, args={"tags": ["a", "b"]}))

    def test_starts_ends_with(self):
        self.assertTrue(
            ev({"args.email": {"ends_with": "@acme.com"}}, args={"email": "a@acme.com"})
        )
        self.assertFalse(ev({"args.email": {"ends_with": "@acme.com"}}, args={"email": "a@x.com"}))
        self.assertTrue(ev({"args.id": {"starts_with": "cus_"}}, args={"id": "cus_123"}))
        self.assertFalse(ev({"args.id": {"starts_with": "cus_"}}, args={"id": "acct_1"}))

    def test_matches(self):
        self.assertTrue(ev({"args.id": {"matches": r"^ord_\d+$"}}, args={"id": "ord_42"}))
        self.assertFalse(ev({"args.id": {"matches": r"^ord_\d+$"}}, args={"id": "ord_x"}))

    def test_matches_non_string_is_false(self):
        self.assertFalse(ev({"args.id": {"matches": r"\d+"}}, args={"id": 123}))

    def test_exists_not_exists(self):
        self.assertTrue(ev({"args.x": {"exists": True}}, args={"x": None}))  # present-but-null
        self.assertFalse(ev({"args.x": {"exists": True}}, args={}))
        self.assertTrue(ev({"args.x": {"not_exists": True}}, args={}))
        self.assertFalse(ev({"args.x": {"not_exists": True}}, args={"x": 1}))


class TestMissingFields(unittest.TestCase):
    def test_missing_field_ordered_is_false(self):
        self.assertFalse(ev({"args.amount": {"gt": 10}}, args={}))

    def test_missing_field_eq_is_false(self):
        self.assertFalse(ev({"args.x": {"eq": 5}}, args={}))

    def test_missing_field_not_in_is_false(self):
        # not_in against an absent field is False, NOT True (presence semantics).
        self.assertFalse(ev({"args.x": {"not_in": ["a"]}}, args={}))

    def test_missing_field_ne_is_false(self):
        self.assertFalse(ev({"args.x": {"ne": 5}}, args={}))

    def test_unpopulated_namespace_is_absent(self):
        # agent.* / org.* have no source yet — always absent.
        self.assertFalse(ev({"agent.tier": {"eq": "trial"}}))
        self.assertTrue(ev({"agent.tier": {"not_exists": True}}))

    def test_nested_path_partial_miss(self):
        self.assertFalse(ev({"args.customer.email": {"exists": True}}, args={"customer": 5}))
        self.assertTrue(
            ev({"args.customer.email": {"eq": "a@b.com"}}, args={"customer": {"email": "a@b.com"}})
        )


class TestTypeCoercion(unittest.TestCase):
    def test_numeric_string_vs_number_ordered(self):
        self.assertTrue(ev({"args.amount": {"gt": 10000}}, args={"amount": "10500"}))
        self.assertFalse(ev({"args.amount": {"gt": 10000}}, args={"amount": "9000"}))

    def test_numeric_string_vs_number_eq(self):
        self.assertTrue(ev({"args.amount": {"eq": 10000}}, args={"amount": "10000"}))

    def test_two_strings_compare_lexicographically(self):
        self.assertTrue(ev({"args.v": {"gt": "1.2.0"}}, args={"v": "1.3.0"}))

    def test_non_numeric_string_vs_number_is_false(self):
        self.assertFalse(ev({"args.x": {"gt": 10}}, args={"x": "abc"}))

    def test_bool_only_equals_bool(self):
        self.assertTrue(ev({"args.flag": {"eq": True}}, args={"flag": True}))
        self.assertFalse(ev({"args.flag": {"eq": True}}, args={"flag": 1}))  # 1 is not True here
        self.assertFalse(ev({"args.n": {"eq": 1}}, args={"n": True}))  # True is not 1 here

    def test_list_value_eq_matches_actual_list(self):
        # Parser normalizes the value list to a tuple; eq must still match a list.
        self.assertTrue(ev({"args.tags": {"eq": ["a", "b"]}}, args={"tags": ["a", "b"]}))

    def test_in_applies_numeric_coercion(self):
        self.assertTrue(ev({"args.code": {"in": [200, 404]}}, args={"code": "404"}))


class TestComposition(unittest.TestCase):
    def test_and_all_true(self):
        block = {"args.amount": {"gt": 10}, "args.currency": {"eq": "USD"}}
        self.assertTrue(ev(block, args={"amount": 20, "currency": "USD"}))
        self.assertFalse(ev(block, args={"amount": 20, "currency": "EUR"}))

    def test_or_any_true(self):
        block = {"or": [{"agent.tier": {"eq": "trial"}}, {"org.plan": {"in": ["free"]}}]}
        self.assertTrue(ev(block, org={"plan": "free"}))
        self.assertFalse(ev(block, org={"plan": "pro"}, agent={"tier": "paid"}))

    def test_high_value_refund_headline_case(self):
        # amount > 10000 AND (agent.tier == trial OR org.plan in [free, starter])
        block = {
            "args.amount": {"gt": 10000},
            "or": [{"agent.tier": {"eq": "trial"}}, {"org.plan": {"in": ["free", "starter"]}}],
        }
        self.assertTrue(ev(block, args={"amount": 15000}, org={"plan": "free"}))
        self.assertTrue(ev(block, args={"amount": 15000}, agent={"tier": "trial"}))
        self.assertFalse(ev(block, args={"amount": 15000}, org={"plan": "pro"}))  # OR fails
        self.assertFalse(ev(block, args={"amount": 500}, org={"plan": "free"}))  # AND fails

    def test_range_via_multiple_operators(self):
        block = {"args.amount": {"gt": 10, "lt": 100}}
        self.assertTrue(ev(block, args={"amount": 50}))
        self.assertFalse(ev(block, args={"amount": 5}))
        self.assertFalse(ev(block, args={"amount": 500}))


class TestObservabilityTrace(unittest.TestCase):
    def test_trace_records_each_comparison(self):
        trace = []
        ev({"args.amount": {"gt": 10000}}, args={"amount": 15000}, trace=trace)
        self.assertEqual(len(trace), 1)
        t = trace[0]
        self.assertIsInstance(t, ComparisonTrace)
        self.assertEqual(
            (t.field_path, t.operator, t.expected, t.actual, t.result),
            (
                "args.amount",
                "gt",
                10000,
                15000,
                True,
            ),
        )
        self.assertTrue(t.present)

    def test_trace_marks_absent_field(self):
        trace = []
        ev({"args.amount": {"gt": 10000}}, args={}, trace=trace)
        self.assertEqual(trace[0].actual, "<absent>")
        self.assertFalse(trace[0].present)
        self.assertFalse(trace[0].result)

    def test_trace_is_complete_no_shortcircuit(self):
        # With a trace, both AND children must be recorded even though the first fails.
        trace = []
        ev({"args.a": {"eq": 1}, "args.b": {"eq": 2}}, args={"a": 99, "b": 2}, trace=trace)
        fields = {t.field_path for t in trace}
        self.assertEqual(fields, {"args.a", "args.b"})

    def test_trace_records_expected_none_for_presence_ops(self):
        trace = []
        ev({"args.x": {"exists": True}}, args={"x": 1}, trace=trace)
        self.assertIsNone(trace[0].expected)


class TestSafetyAndDeterminism(unittest.TestCase):
    def test_deterministic_repeated_eval(self):
        block = {
            "args.amount": {"gt": 10000},
            "or": [{"org.plan": {"in": ["free"]}}, {"args.id": {"matches": r"^ord_\d+$"}}],
        }
        ctx_args = dict(args={"amount": 15000, "id": "ord_9"}, org={"plan": "pro"})
        results = {ev(block, **ctx_args) for _ in range(100)}
        self.assertEqual(results, {True})

    def test_redos_pattern_does_not_hang(self):
        # Pathological pattern + input; must return quickly (timeout → False), not hang.
        start = time.perf_counter()
        result = ev(
            {"args.s": {"matches": r"(a+)+$"}},
            args={"s": "a" * 50 + "!"},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertIsInstance(result, bool)
        self.assertLess(elapsed_ms, 100, "regex evaluation should be bounded")

    def test_no_crash_on_weird_actual_types(self):
        # dict / None actuals must never raise, just not match.
        self.assertFalse(ev({"args.x": {"gt": 1}}, args={"x": {"nested": 1}}))
        self.assertFalse(ev({"args.x": {"starts_with": "a"}}, args={"x": None}))
        self.assertTrue(ev({"args.x": {"eq": None}}, args={"x": None}))


class TestBenchmark(unittest.TestCase):
    def test_representative_eval_under_100us(self):
        # Representative production policy: a value threshold AND a 2-way OR of a
        # membership test and an equality test.
        expr = parse_match_block(
            {
                "args.amount": {"gt": 10000},
                "or": [
                    {"agent.tier": {"eq": "trial"}},
                    {"org.plan": {"in": ["free", "starter"]}},
                ],
            },
            policy_name="high-value-refund",
        )
        ctx = EvaluationContext(
            args={"amount": 15000}, org={"plan": "free"}, agent={"tier": "paid"}
        )
        n = 20000
        # Warm up.
        for _ in range(1000):
            evaluate(expr, ctx)
        start = time.perf_counter()
        for _ in range(n):
            evaluate(expr, ctx)
        per_call_us = (time.perf_counter() - start) / n * 1e6
        print(f"\n[bench] representative eval: {per_call_us:.2f} µs/call over {n} iters")
        # Generous CI-safe bound (the standalone bench script reports real p50/p95/p99).
        self.assertLess(per_call_us, 100.0, f"{per_call_us:.2f}µs exceeds 100µs target")


if __name__ == "__main__":
    unittest.main()
