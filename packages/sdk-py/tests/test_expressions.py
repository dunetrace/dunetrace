"""
Tests for the condition-expression parser (dunetrace.policies.expressions).

Parse-and-validate only — no evaluation here (see test_evaluator.py, Phase 2).
Covers: every operator parses, every field prefix parses, composition (AND/OR,
explicit and implicit, nesting), the neq→ne alias, and the full rejection
matrix with clear-error assertions. Also verifies the tree is immutable,
hashable, comparable, and serializable.
"""

import unittest

from dunetrace.policies import (
    And,
    Comparison,
    ConditionExpression,
    EXPRESSION_OPERATORS,
    ExpressionError,
    FIELD_PREFIXES,
    MAX_DEPTH,
    Or,
    parse_condition,
    parse_match_block,
)


def P(block, name="test-policy"):
    return parse_match_block(block, policy_name=name)


class TestParseValidComparisons(unittest.TestCase):
    def test_single_comparison(self):
        expr = P({"args.amount": {"gt": 10000}})
        self.assertIsInstance(expr, Comparison)
        self.assertEqual(expr.field_path, "args.amount")
        self.assertEqual(expr.prefix, "args")
        self.assertEqual(expr.path, ("amount",))
        self.assertEqual(expr.operator, "gt")
        self.assertEqual(expr.value, 10000)

    def test_nested_field_path(self):
        expr = P({"args.customer.email": {"ends_with": "@acme.com"}})
        self.assertIsInstance(expr, Comparison)
        self.assertEqual(expr.prefix, "args")
        self.assertEqual(expr.path, ("customer", "email"))

    def test_all_operators_parse(self):
        samples = {
            "eq": 1,
            "ne": 1,
            "gt": 1,
            "gte": 1,
            "lt": 1,
            "lte": 1,
            "in": [1, 2],
            "not_in": [1, 2],
            "contains": "x",
            "starts_with": "x",
            "ends_with": "x",
            "matches": r"^x\d+$",
            "exists": True,
            "not_exists": True,
        }
        # Every whitelisted operator must have a sample here.
        self.assertEqual(set(samples), set(EXPRESSION_OPERATORS))
        for op, val in samples.items():
            expr = P({"args.field": {op: val}})
            self.assertIsInstance(expr, Comparison, op)
            self.assertEqual(expr.operator, op)

    def test_all_field_prefixes_parse(self):
        for prefix in FIELD_PREFIXES:
            expr = P({f"{prefix}.some_field": {"exists": True}})
            self.assertEqual(expr.prefix, prefix)

    def test_neq_alias_normalizes_to_ne(self):
        expr = P({"args.status": {"neq": "done"}})
        self.assertEqual(expr.operator, "ne")

    def test_exists_ignores_value(self):
        a = P({"args.x": {"exists": True}})
        b = P({"args.x": {"exists": "anything"}})
        self.assertIsNone(a.value)
        self.assertIsNone(b.value)
        self.assertEqual(a, b)  # value is dropped, so these are identical

    def test_in_value_normalized_to_tuple(self):
        expr = P({"org.plan": {"in": ["free", "starter"]}})
        self.assertEqual(expr.value, ("free", "starter"))

    def test_multiple_operators_on_one_field_and_together(self):
        # Range check: gt 10 AND lt 100 on the same field.
        expr = P({"args.amount": {"gt": 10, "lt": 100}})
        self.assertIsInstance(expr, And)
        ops = sorted(c.operator for c in expr.children)
        self.assertEqual(ops, ["gt", "lt"])


class TestParseComposition(unittest.TestCase):
    def test_multiple_fields_and_by_default(self):
        expr = P({"args.amount": {"gt": 10}, "args.currency": {"eq": "USD"}})
        self.assertIsInstance(expr, And)
        self.assertEqual(len(expr.children), 2)

    def test_explicit_or_block(self):
        expr = P(
            {
                "args.amount": {"gt": 10000},
                "or": [
                    {"agent.tier": {"eq": "trial"}},
                    {"org.plan": {"in": ["free", "starter"]}},
                ],
            }
        )
        # amount>10000 AND (tier==trial OR plan in [...])
        self.assertIsInstance(expr, And)
        kinds = sorted(type(c).__name__ for c in expr.children)
        self.assertEqual(kinds, ["Comparison", "Or"])
        or_node = next(c for c in expr.children if isinstance(c, Or))
        self.assertEqual(len(or_node.children), 2)

    def test_explicit_and_block(self):
        expr = P({"and": [{"args.a": {"eq": 1}}, {"args.b": {"eq": 2}}]})
        self.assertIsInstance(expr, And)
        self.assertEqual(len(expr.children), 2)

    def test_nested_or_within_or_at_max_depth(self):
        # depth 3: root(1) -> or(2) -> or(3)
        expr = P(
            {
                "or": [
                    {"args.a": {"eq": 1}},
                    {"or": [{"args.b": {"eq": 2}}, {"args.c": {"eq": 3}}]},
                ]
            }
        )
        self.assertIsInstance(expr, Or)

    def test_field_paths_collects_all_references(self):
        expr = P(
            {
                "args.amount": {"gt": 10000},
                "or": [{"agent.tier": {"eq": "trial"}}, {"org.plan": {"eq": "free"}}],
            }
        )
        self.assertEqual(
            expr.field_paths(),
            frozenset({"args.amount", "agent.tier", "org.plan"}),
        )


class TestParseCondition(unittest.TestCase):
    def test_no_match_key_returns_none(self):
        # Legacy flat condition — no expression present.
        self.assertIsNone(
            parse_condition({"trigger": "tool_call_count", "operator": "gt", "value": 5})
        )

    def test_match_key_parsed(self):
        expr = parse_condition(
            {"trigger": "before_tool_call", "value": "refund", "match": {"args.amount": {"gt": 1}}},
            policy_name="p",
        )
        self.assertIsInstance(expr, Comparison)

    def test_non_dict_condition_returns_none(self):
        self.assertIsNone(parse_condition("nope"))  # type: ignore[arg-type]


class TestRejections(unittest.TestCase):
    def _err(self, block):
        with self.assertRaises(ExpressionError) as ctx:
            parse_match_block(block, policy_name="require-approval-refund")
        return str(ctx.exception)

    def test_unknown_operator_suggests_correction(self):
        msg = self._err({"args.amount": {"greaterthan": 10000}})
        self.assertIn("greaterthan", msg)
        self.assertIn("require-approval-refund", msg)
        self.assertIn("Did you mean 'gt'?", msg)

    def test_unknown_operator_difflib_suggestion(self):
        msg = self._err({"args.amount": {"conatins": "x"}})  # typo of contains
        self.assertIn("Did you mean 'contains'?", msg)

    def test_unknown_field_prefix_rejected(self):
        msg = self._err({"argz.amount": {"gt": 1}})
        self.assertIn("argz", msg)
        self.assertIn("Did you mean 'args", msg)

    def test_field_path_without_dot_rejected(self):
        msg = self._err({"amount": {"gt": 1}})
        self.assertIn("must be dotted", msg)

    def test_field_path_empty_subpath_rejected(self):
        self.assertIn("missing a sub-path", self._err({"args.": {"gt": 1}}))
        self.assertIn("empty segment", self._err({"args.a..b": {"gt": 1}}))

    def test_empty_match_block_rejected(self):
        self.assertIn("Empty condition block", self._err({}))

    def test_match_not_a_mapping_rejected(self):
        self.assertIn("must be a mapping", self._err(["args.a"]))  # type: ignore[arg-type]

    def test_field_maps_to_non_dict_rejected(self):
        self.assertIn("operator mapping", self._err({"args.amount": 10000}))

    def test_field_maps_to_empty_dict_rejected(self):
        self.assertIn("no operators", self._err({"args.amount": {}}))

    def test_or_not_a_list_rejected(self):
        self.assertIn("must map to a list", self._err({"or": {"args.a": {"eq": 1}}}))

    def test_empty_or_list_rejected(self):
        self.assertIn("Empty 'or' list", self._err({"or": []}))

    def test_in_requires_list_value(self):
        self.assertIn("requires a list", self._err({"args.x": {"in": "notalist"}}))

    def test_starts_with_requires_string(self):
        self.assertIn("requires a string", self._err({"args.x": {"starts_with": 123}}))

    def test_matches_requires_string(self):
        self.assertIn("string regex", self._err({"args.x": {"matches": 5}}))

    def test_matches_invalid_regex_rejected(self):
        msg = self._err({"args.x": {"matches": "("}})  # unbalanced paren
        self.assertIn("Invalid regex", msg)

    def test_depth_over_limit_rejected(self):
        # depth 4: root(1) -> or(2) -> or(3) -> or(4)
        block = {
            "or": [
                {"or": [{"or": [{"args.a": {"eq": 1}}]}]},
            ]
        }
        msg = self._err(block)
        self.assertIn(f"maximum depth of {MAX_DEPTH}", msg)

    def test_depth_exactly_at_limit_ok(self):
        # depth 3 must NOT raise.
        block = {"or": [{"or": [{"args.a": {"eq": 1}}]}]}
        self.assertIsInstance(parse_match_block(block, policy_name="p"), Or)


class TestTreeProperties(unittest.TestCase):
    def test_comparison_is_frozen(self):
        expr = P({"args.amount": {"gt": 10}})
        with self.assertRaises(Exception):
            expr.value = 20  # type: ignore[misc]

    def test_expression_is_hashable(self):
        a = P({"args.amount": {"gt": 10}})
        b = P({"args.amount": {"gt": 10}})
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)  # dedup in a set

    def test_equal_trees_compare_equal(self):
        a = P({"args.a": {"eq": 1}, "or": [{"args.b": {"eq": 2}}, {"args.c": {"eq": 3}}]})
        b = P({"args.a": {"eq": 1}, "or": [{"args.b": {"eq": 2}}, {"args.c": {"eq": 3}}]})
        self.assertEqual(a, b)

    def test_list_value_makes_comparison_hashable(self):
        expr = P({"org.plan": {"in": ["free", "starter"]}})
        hash(expr)  # must not raise (value normalized to tuple)

    def test_to_dict_roundtrip_shape(self):
        expr = P(
            {
                "args.amount": {"gt": 10000},
                "or": [{"agent.tier": {"eq": "trial"}}, {"org.plan": {"in": ["free"]}}],
            }
        )
        d = expr.to_dict()
        self.assertEqual(d["and"][0], {"field": "args.amount", "op": "gt", "value": 10000})
        or_dict = d["and"][1]["or"]
        self.assertEqual(or_dict[1], {"field": "org.plan", "op": "in", "value": ["free"]})

    def test_to_dict_omits_value_for_presence_ops(self):
        expr = P({"args.x": {"exists": True}})
        self.assertEqual(expr.to_dict(), {"field": "args.x", "op": "exists"})

    def test_describe_is_readable(self):
        expr = P({"args.amount": {"gt": 10000}})
        self.assertEqual(expr.describe(), "args.amount gt 10000")
        expr2 = P({"args.a": {"eq": 1}, "args.b": {"eq": 2}})
        self.assertIn(" AND ", expr2.describe())


if __name__ == "__main__":
    unittest.main()
