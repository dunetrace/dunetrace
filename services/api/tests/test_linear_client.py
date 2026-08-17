"""
Tests for the api_svc Linear GraphQL client (team/project picker + the
workflow-state lookup used by the webhook receiver). Mocks httpx. No
network required.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_svc.linear_client import (
    LinearApiError,
    fetch_projects,
    fetch_teams,
    fetch_workflow_state_type,
)


def _mock_response(data=None, errors=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": data, "errors": errors}
    return resp


class TestFetchTeams(unittest.IsolatedAsyncioTestCase):
    async def test_returns_team_nodes(self):
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_mock_response({"teams": {"nodes": [{"id": "t1", "name": "Eng"}]}})
        )
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client):
            teams = await fetch_teams("lin_key")

        self.assertEqual(teams, [{"id": "t1", "name": "Eng"}])

    async def test_uses_api_key_without_bearer_prefix(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({"teams": {"nodes": []}}))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client):
            await fetch_teams("lin_key_123")

        headers = client.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "lin_key_123")

    async def test_graphql_errors_raise_linear_api_error(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(errors=[{"message": "unauthorized"}]))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client):
            with self.assertRaises(LinearApiError):
                await fetch_teams("bad_key")


class TestFetchProjects(unittest.IsolatedAsyncioTestCase):
    async def test_returns_project_nodes_for_team(self):
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_mock_response(
                {"team": {"projects": {"nodes": [{"id": "p1", "name": "Backend"}]}}}
            )
        )
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client):
            projects = await fetch_projects("lin_key", "team-1")

        self.assertEqual(projects, [{"id": "p1", "name": "Backend"}])


class TestFetchWorkflowStateType(unittest.IsolatedAsyncioTestCase):
    async def test_returns_state_type(self):
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_mock_response({"workflowState": {"type": "completed"}})
        )
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client):
            state_type = await fetch_workflow_state_type("lin_key", "state-1")

        self.assertEqual(state_type, "completed")

    async def test_returns_none_when_state_not_found(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({"workflowState": None}))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client):
            state_type = await fetch_workflow_state_type("lin_key", "missing-state")

        self.assertIsNone(state_type)


if __name__ == "__main__":
    unittest.main()
