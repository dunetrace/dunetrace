"""
Endpoint-level tests for Phase 4.3's tier-1 explicit source mapping
(api_svc/routers/agents.py's source-config endpoints). Calls route
functions directly (this codebase's established pattern), mocked DB
calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.agents import (
    SourceConfigRequest,
    get_source_config,
    remove_source_config,
    set_source_config,
)


class TestSetSourceConfig(unittest.IsolatedAsyncioTestCase):
    async def test_repo_and_file_path_stored(self):
        with patch("api_svc.routers.agents.upsert_agent_source_config", AsyncMock()) as upsert_mock:
            result = await set_source_config(
                "agent-1",
                SourceConfigRequest(repo="acme/bot", file_path="src/bot.py"),
                org_id="org-1",
            )

        upsert_mock.assert_awaited_once_with("org-1", "agent-1", "acme/bot", "src/bot.py")
        self.assertTrue(result.configured)
        self.assertEqual(result.repo, "acme/bot")
        self.assertEqual(result.file_path, "src/bot.py")

    async def test_repo_only_file_path_none_still_accepted(self):
        with patch("api_svc.routers.agents.upsert_agent_source_config", AsyncMock()) as upsert_mock:
            result = await set_source_config(
                "agent-1", SourceConfigRequest(repo="acme/bot"), org_id="org-1"
            )

        upsert_mock.assert_awaited_once_with("org-1", "agent-1", "acme/bot", None)
        self.assertTrue(result.configured)
        self.assertIsNone(result.file_path)


class TestGetSourceConfig(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_returns_configured_false(self):
        with patch("api_svc.routers.agents.get_agent_source_config", AsyncMock(return_value=None)):
            result = await get_source_config("agent-1", org_id="org-1")
        self.assertFalse(result.configured)
        self.assertIsNone(result.repo)

    async def test_configured_returns_stored_values(self):
        with patch(
            "api_svc.routers.agents.get_agent_source_config",
            AsyncMock(return_value={"repo": "acme/bot", "file_path": "src/bot.py"}),
        ):
            result = await get_source_config("agent-1", org_id="org-1")
        self.assertTrue(result.configured)
        self.assertEqual(result.repo, "acme/bot")
        self.assertEqual(result.file_path, "src/bot.py")


class TestRemoveSourceConfig(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.agents.delete_agent_source_config", AsyncMock(return_value=False)
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_source_config("agent-1", org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_found_deletes_without_raising(self):
        with patch(
            "api_svc.routers.agents.delete_agent_source_config", AsyncMock(return_value=True)
        ):
            await remove_source_config("agent-1", org_id="org-1")  # must not raise


if __name__ == "__main__":
    unittest.main()
