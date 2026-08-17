"""
Tests for Phase 4.1's Linear delivery path: alerts_svc/linear_client.py
(issue creation), sender.py::send_linear, formatters/linear.py, and
worker.py's per-org Slack/Linear destination resolution. No DB, no real
HTTP calls.

Run:
    cd services/alerts
    PYTHONPATH=packages/sdk-py:services/explainer:services/alerts \
        python -m pytest tests/test_linear_integration.py -v
"""

from __future__ import annotations

import sys
import os
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explainer_svc.models import Explanation, CodeFix
from alerts_svc.linear_client import create_issue
from alerts_svc.sender import send_linear
from alerts_svc.formatters.linear import format_linear_issue
import alerts_svc.db as db_module
import alerts_svc.worker as worker_module


def make_explanation() -> Explanation:
    return Explanation(
        failure_type="TOOL_LOOP",
        severity="HIGH",
        run_id="run-lin-001",
        agent_id="test-agent",
        agent_version="abc00001",
        confidence=0.9,
        step_index=3,
        detected_at=time.time(),
        evidence={"tool": "search", "count": 5},
        title="Tool loop detected: `search` called 5x",
        what="The agent called search repeatedly without progress.",
        why_it_matters="Loops waste tokens.",
        evidence_summary="search called 5 times. Confidence: 90%.",
        suggested_fixes=[
            CodeFix(description="Limit tool calls", language="python", code="if count > 3: break")
        ],
    )


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_body or {}
    return resp


class TestCreateIssue(unittest.TestCase):
    def test_successful_creation_returns_issue_id(self):
        resp = _mock_response(
            json_body={"data": {"issueCreate": {"success": True, "issue": {"id": "issue-42"}}}}
        )
        with patch("httpx.post", return_value=resp) as mock_post:
            issue_id = create_issue("lin_key", "team-1", "Title", "Description")

        self.assertEqual(issue_id, "issue-42")
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "lin_key")  # no Bearer prefix

    def test_project_id_included_when_given(self):
        resp = _mock_response(
            json_body={"data": {"issueCreate": {"success": True, "issue": {"id": "i1"}}}}
        )
        with patch("httpx.post", return_value=resp) as mock_post:
            create_issue("lin_key", "team-1", "Title", "Desc", project_id="proj-1")

        variables = mock_post.call_args.kwargs["json"]["variables"]
        self.assertEqual(variables["input"]["projectId"], "proj-1")

    def test_project_id_omitted_when_not_given(self):
        resp = _mock_response(
            json_body={"data": {"issueCreate": {"success": True, "issue": {"id": "i1"}}}}
        )
        with patch("httpx.post", return_value=resp) as mock_post:
            create_issue("lin_key", "team-1", "Title", "Desc")

        variables = mock_post.call_args.kwargs["json"]["variables"]
        self.assertNotIn("projectId", variables["input"])

    def test_graphql_errors_return_none_not_raise(self):
        resp = _mock_response(
            json_body={"errors": [{"message": "Field 'projectId' doesn't exist"}]}
        )
        with patch("httpx.post", return_value=resp):
            issue_id = create_issue("lin_key", "team-1", "Title", "Desc", project_id="proj-1")
        self.assertIsNone(issue_id)

    def test_success_false_returns_none(self):
        resp = _mock_response(json_body={"data": {"issueCreate": {"success": False}}})
        with patch("httpx.post", return_value=resp):
            issue_id = create_issue("lin_key", "team-1", "Title", "Desc")
        self.assertIsNone(issue_id)

    def test_network_failure_returns_none_not_raise(self):
        with patch("httpx.post", side_effect=ConnectionError("boom")):
            issue_id = create_issue("lin_key", "team-1", "Title", "Desc")
        self.assertIsNone(issue_id)


class TestSendLinear(unittest.TestCase):
    def test_success_returns_metadata_with_issue_id(self):
        with patch("alerts_svc.linear_client.create_issue", return_value="issue-99"):
            result = send_linear("lin_key", "team-1", "proj-1", "Title", "Desc")

        self.assertTrue(result.success)
        self.assertEqual(result.destination, "linear")
        self.assertEqual(result.metadata["linear_issue_id"], "issue-99")

    def test_failure_returns_unsuccessful_result(self):
        with patch("alerts_svc.linear_client.create_issue", return_value=None):
            result = send_linear("lin_key", "team-1", None, "Title", "Desc")

        self.assertFalse(result.success)
        self.assertIsNone(result.metadata)


class TestFormatLinearIssue(unittest.TestCase):
    def setUp(self):
        self.exp = make_explanation()

    def test_returns_title_and_description(self):
        title, description = format_linear_issue(self.exp)
        self.assertEqual(title, self.exp.title)
        self.assertIsInstance(description, str)

    def test_description_contains_what_and_why(self):
        _, description = format_linear_issue(self.exp)
        self.assertIn("What happened", description)
        self.assertIn("Why it matters", description)
        self.assertIn(self.exp.what, description)
        self.assertIn(self.exp.why_it_matters, description)

    def test_description_contains_run_link(self):
        _, description = format_linear_issue(self.exp)
        self.assertIn(self.exp.run_id, description)

    def test_description_contains_suggested_fix(self):
        _, description = format_linear_issue(self.exp)
        self.assertIn("Limit tool calls", description)


class TestFetchOrgAlertIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            result = await db_module.fetch_org_alert_integration("org-1", "slack")
        self.assertIsNone(result)

    async def test_record_linear_issue_mapping_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            await db_module.record_linear_issue_mapping("org-1", 1, "issue-1")  # must not raise


class TestResolveSlackDestination(unittest.IsolatedAsyncioTestCase):
    async def test_org_configured_uses_org_webhook(self):
        with (
            patch(
                "alerts_svc.worker.fetch_org_alert_integration",
                AsyncMock(
                    return_value={"encrypted_credentials": "tok", "config": {"channel": "#x"}}
                ),
            ),
            patch(
                "alerts_svc.worker.decrypt_credentials",
                return_value={"webhook_url": "https://hooks.slack.com/org-specific"},
            ),
        ):
            url = await worker_module._resolve_slack_destination("org-1")

        self.assertEqual(url, "https://hooks.slack.com/org-specific")

    async def test_no_org_config_returns_none(self):
        with patch("alerts_svc.worker.fetch_org_alert_integration", AsyncMock(return_value=None)):
            url = await worker_module._resolve_slack_destination("org-1")

        self.assertIsNone(url)

    async def test_decrypt_failure_returns_none(self):
        with (
            patch(
                "alerts_svc.worker.fetch_org_alert_integration",
                AsyncMock(return_value={"encrypted_credentials": "corrupt", "config": {}}),
            ),
            patch("alerts_svc.worker.decrypt_credentials", side_effect=RuntimeError("bad token")),
        ):
            url = await worker_module._resolve_slack_destination("org-1")

        self.assertIsNone(url)


class TestResolveLinearConfig(unittest.IsolatedAsyncioTestCase):
    async def test_returns_full_config_when_configured(self):
        with (
            patch(
                "alerts_svc.worker.fetch_org_alert_integration",
                AsyncMock(
                    return_value={
                        "encrypted_credentials": "tok",
                        "config": {"team_id": "team-1", "project_id": "proj-1"},
                    }
                ),
            ),
            patch(
                "alerts_svc.worker.decrypt_credentials",
                return_value={"api_key": "lin_key", "webhook_secret": "whsec"},
            ),
        ):
            config = await worker_module._resolve_linear_config("org-1")

        self.assertEqual(config["api_key"], "lin_key")
        self.assertEqual(config["team_id"], "team-1")
        self.assertEqual(config["project_id"], "proj-1")

    async def test_no_org_config_returns_none(self):
        with patch("alerts_svc.worker.fetch_org_alert_integration", AsyncMock(return_value=None)):
            config = await worker_module._resolve_linear_config("org-1")

        self.assertIsNone(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
