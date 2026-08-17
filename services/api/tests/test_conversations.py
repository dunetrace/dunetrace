"""
Endpoint-level tests for Phase 3.3's conversation detail + cross-conversation
search (api_svc/routers/conversations.py). Calls route functions directly
(this codebase's established pattern), mocked DB calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.conversations import get_conversation, search


def _conversation_detail(**overrides):
    fields = {
        "id": 42,
        "agent_id": "support-bot",
        "user_id": None,
        "external_id": "conv_8f3a1c",
        "first_run_at": 1_752_000_000.0,
        "last_run_at": 1_752_003_600.0,
        "run_count": 2,
        "runs": [
            {"run_id": "run-1", "agent_version": "v1", "started_at": 1_752_000_000.0},
            {"run_id": "run-2", "agent_version": "v1", "started_at": 1_752_003_600.0},
        ],
        "signals": [
            {
                "id": 99,
                "failure_type": "USER_FRUSTRATION",
                "severity": "HIGH",
                "confidence": 0.82,
                "detected_at": 1_752_003_600.0,
                "evidence": {
                    "conversation_id": "conv_8f3a1c",
                    "run_ids_considered": ["run-1", "run-2"],
                },
            }
        ],
    }
    fields.update(overrides)
    return fields


class TestGetConversation(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.conversations.get_conversation_detail",
            AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await get_conversation(999, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_returns_full_detail_with_runs_and_signals(self):
        with patch(
            "api_svc.routers.conversations.get_conversation_detail",
            AsyncMock(return_value=_conversation_detail()),
        ):
            result = await get_conversation(42, org_id="org-1")

        self.assertEqual(result.id, 42)
        self.assertEqual(result.agent_id, "support-bot")
        self.assertEqual(result.external_id, "conv_8f3a1c")
        self.assertEqual(len(result.runs), 2)
        self.assertEqual(result.runs[0].run_id, "run-1")
        self.assertEqual(len(result.signals), 1)
        self.assertEqual(result.signals[0].failure_type, "USER_FRUSTRATION")
        self.assertEqual(result.signals[0].evidence["conversation_id"], "conv_8f3a1c")

    async def test_scoped_to_caller_org(self):
        with patch(
            "api_svc.routers.conversations.get_conversation_detail",
            AsyncMock(return_value=_conversation_detail()),
        ) as mock:
            await get_conversation(42, org_id="org-7")
        mock.assert_called_once_with("org-7", 42)


class TestSearchConversations(unittest.IsolatedAsyncioTestCase):
    async def test_returns_empty_when_no_matches(self):
        with patch(
            "api_svc.routers.conversations.search_conversations",
            AsyncMock(return_value=([], 0)),
        ):
            result = await search(offset=0, limit=20, org_id="org-1")

        self.assertEqual(result.conversations, [])
        self.assertEqual(result.page.total, 0)

    async def test_filters_passed_through_to_query(self):
        with patch(
            "api_svc.routers.conversations.search_conversations",
            AsyncMock(return_value=([], 0)),
        ) as mock:
            await search(
                agent_id="support-bot",
                user_id="user-42",
                has_frustration_signal=True,
                offset=10,
                limit=20,
                org_id="org-1",
            )

        mock.assert_called_once_with("org-1", "support-bot", "user-42", True, 10, 20)

    async def test_maps_result_rows_into_summaries(self):
        rows = [
            {
                "id": 1,
                "agent_id": "support-bot",
                "user_id": None,
                "external_id": "conv_a",
                "last_run_at": 1_752_000_000.0,
                "run_count": 3,
                "has_frustration_signal": True,
            },
            {
                "id": 2,
                "agent_id": "billing-bot",
                "user_id": None,
                "external_id": "conv_b",
                "last_run_at": 1_751_999_000.0,
                "run_count": 1,
                "has_frustration_signal": False,
            },
        ]
        with patch(
            "api_svc.routers.conversations.search_conversations",
            AsyncMock(return_value=(rows, 2)),
        ):
            result = await search(offset=0, limit=20, org_id="org-1")

        self.assertEqual(len(result.conversations), 2)
        self.assertEqual(result.conversations[0].external_id, "conv_a")
        self.assertTrue(result.conversations[0].has_frustration_signal)
        self.assertFalse(result.conversations[1].has_frustration_signal)
        self.assertEqual(result.page.total, 2)

    async def test_page_has_more_computed_correctly(self):
        with patch(
            "api_svc.routers.conversations.search_conversations",
            AsyncMock(return_value=([], 25)),
        ):
            result = await search(offset=0, limit=10, org_id="org-1")

        self.assertTrue(result.page.has_more)


if __name__ == "__main__":
    unittest.main()
