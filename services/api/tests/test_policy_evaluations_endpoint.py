"""
GET /v1/policies/{id}/evaluations endpoint + row mapping (Phase 5).

The handler is exercised directly (async) with the two query functions patched,
so no database is needed. Covers 404 on unknown policy, org-scoped passthrough,
limit clamping, and the _policy_eval_row transformation.

Run from services/api/ with:
  PYTHONPATH=../../packages/sdk-py:../explainer:. \
    python -m unittest discover -s tests -p "test_policy_evaluations_endpoint.py"
"""

import asyncio
import datetime
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.db.queries import _policy_eval_row
from api_svc.routers import policies as pol


class TestRowMapping(unittest.TestCase):
    def test_maps_columns_and_renames_trigger(self):
        row = {
            "id": 5,
            "policy_id": 7,
            "policy_name": "refund-guard",
            "agent_id": "billing",
            "run_id": "r1",
            "trigger_name": "before_tool_call",
            "trigger_matched": True,
            "fired": False,
            "sampled": False,
            "reason": "did not fire: args.amount gt 10000 — actual 500",
            "conditions": '[{"field_path": "args.amount", "result": false}]',
            "evaluated_at": datetime.datetime(2026, 7, 13, 12, 0, 0),
        }
        out = _policy_eval_row(row)
        self.assertEqual(out["trigger"], "before_tool_call")  # renamed from trigger_name
        self.assertNotIn("trigger_name", out)
        self.assertEqual(out["conditions"][0]["field_path"], "args.amount")  # JSON parsed
        self.assertFalse(out["fired"])
        self.assertIsInstance(out["evaluated_at"], float)  # timestamp epoch

    def test_null_conditions_becomes_empty_list(self):
        row = {
            "id": 1,
            "policy_id": 1,
            "policy_name": "p",
            "agent_id": "a",
            "run_id": None,
            "trigger_name": None,
            "trigger_matched": None,
            "fired": None,
            "sampled": False,
            "reason": None,
            "conditions": None,
            "evaluated_at": 0,
        }
        self.assertEqual(_policy_eval_row(row)["conditions"], [])


class TestEndpoint(unittest.TestCase):
    def _call(self, policy_id=7, limit=100):
        return asyncio.run(pol.get_evaluations(policy_id, limit=limit, org_id="org-1"))

    def test_404_when_policy_not_found(self):
        with patch.object(pol, "get_policy_by_id", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as e:
                self._call()
            self.assertEqual(e.exception.status_code, 404)

    def test_returns_evaluations_for_existing_policy(self):
        sample = [{"policy_id": 7, "fired": True, "reason": "fired"}]
        with (
            patch.object(pol, "get_policy_by_id", AsyncMock(return_value={"id": 7})),
            patch.object(pol, "fetch_policy_evaluations", AsyncMock(return_value=sample)) as fetch,
        ):
            result = self._call()
        self.assertEqual(result, sample)
        # org-scoped, policy-scoped call
        fetch.assert_awaited_once()
        self.assertEqual(fetch.await_args.args[0], "org-1")
        self.assertEqual(fetch.await_args.args[1], 7)

    def test_limit_clamped_high(self):
        with (
            patch.object(pol, "get_policy_by_id", AsyncMock(return_value={"id": 7})),
            patch.object(pol, "fetch_policy_evaluations", AsyncMock(return_value=[])) as fetch,
        ):
            self._call(limit=99999)
        self.assertEqual(fetch.await_args.kwargs["limit"], 500)

    def test_limit_clamped_low(self):
        with (
            patch.object(pol, "get_policy_by_id", AsyncMock(return_value={"id": 7})),
            patch.object(pol, "fetch_policy_evaluations", AsyncMock(return_value=[])) as fetch,
        ):
            self._call(limit=0)
        self.assertEqual(fetch.await_args.kwargs["limit"], 1)


if __name__ == "__main__":
    unittest.main()
