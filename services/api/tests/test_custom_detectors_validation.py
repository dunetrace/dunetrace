"""
Tests for routers/custom_detectors.py's _validate_config — the 422 gate a
custom detector config must pass before being saved. Covers the pre-existing
metric-condition path (no test file existed for this before content conditions
were added) and the new content-condition path.

Run:
    PYTHONPATH=packages/sdk-py:services/explainer:services/api pytest services/api/tests/test_custom_detectors_validation.py -v
"""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from api_svc.routers.custom_detectors import _validate_config


def _base(**overrides) -> dict:
    config = {
        "detector_name": "CUSTOM_TEST",
        "conditions": [{"metric": "tool_call_count", "operator": ">=", "threshold": 3}],
        "severity": "HIGH",
        "requires_content": False,
    }
    config.update(overrides)
    return config


class TestDetectorNameValidation(unittest.TestCase):
    def test_valid_name_passes(self):
        _validate_config(_base())  # must not raise

    def test_missing_name_rejected(self):
        with self.assertRaises(HTTPException) as cm:
            _validate_config(_base(detector_name=""))
        self.assertEqual(cm.exception.status_code, 422)

    def test_name_without_custom_prefix_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(_base(detector_name="TOOL_LOOP"))


class TestRequiresContentGate(unittest.TestCase):
    def test_requires_content_true_is_rejected(self):
        with self.assertRaises(HTTPException) as cm:
            _validate_config(_base(requires_content=True))
        self.assertEqual(cm.exception.status_code, 422)


class TestMetricConditionValidation(unittest.TestCase):
    def test_unsupported_metric_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(
                _base(conditions=[{"metric": "vibes", "operator": ">=", "threshold": 1}])
            )

    def test_invalid_operator_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(
                _base(conditions=[{"metric": "tool_call_count", "operator": "~=", "threshold": 1}])
            )

    def test_non_numeric_threshold_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(
                _base(
                    conditions=[
                        {"metric": "tool_call_count", "operator": ">=", "threshold": "a lot"}
                    ]
                )
            )

    def test_missing_conditions_field_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(_base(conditions=[{"metric": "tool_call_count", "operator": ">="}]))

    def test_empty_conditions_list_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(_base(conditions=[]))


class TestContentConditionValidation(unittest.TestCase):
    def test_valid_content_condition_passes(self):
        _validate_config(
            _base(conditions=[{"field": "tool_args", "operator": "contains", "value": "error"}])
        )  # must not raise

    def test_unsupported_field_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(
                _base(conditions=[{"field": "system_prompt", "operator": "contains", "value": "x"}])
            )

    def test_unsupported_content_operator_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(
                _base(conditions=[{"field": "tool_args", "operator": "fuzzy_match", "value": "x"}])
            )

    def test_length_gt_requires_numeric_value(self):
        with self.assertRaises(HTTPException):
            _validate_config(
                _base(conditions=[{"field": "tool_args", "operator": "length_gt", "value": "big"}])
            )

    def test_length_gt_accepts_numeric_value(self):
        _validate_config(
            _base(conditions=[{"field": "tool_args", "operator": "length_gt", "value": 100}])
        )  # must not raise

    def test_contains_does_not_require_numeric_value(self):
        _validate_config(
            _base(
                conditions=[{"field": "tool_args", "operator": "contains", "value": "not a number"}]
            )
        )  # must not raise

    def test_case_sensitive_must_be_boolean(self):
        with self.assertRaises(HTTPException):
            _validate_config(
                _base(
                    conditions=[
                        {
                            "field": "tool_args",
                            "operator": "contains",
                            "value": "x",
                            "case_sensitive": "yes",
                        }
                    ]
                )
            )

    def test_missing_value_field_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(_base(conditions=[{"field": "tool_args", "operator": "contains"}]))


class TestMixedAndMalformedConditions(unittest.TestCase):
    def test_mix_of_metric_and_content_conditions_passes(self):
        _validate_config(
            _base(
                conditions=[
                    {"metric": "tool_call_count", "operator": ">=", "threshold": 3},
                    {"field": "tool_args", "operator": "contains", "value": "error"},
                ]
            )
        )  # must not raise

    def test_condition_with_neither_metric_nor_field_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(_base(conditions=[{"operator": ">=", "threshold": 3}]))

    def test_non_dict_condition_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_config(_base(conditions=["not a dict"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
