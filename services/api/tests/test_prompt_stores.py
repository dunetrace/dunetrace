"""
Tests for prompt_stores.py — the external-prompt-store abstraction apply-fix
uses for customer_code fixes.

No network — LangfuseExternalPromptStore.push_fix delegation is tested with
a mocked apply_langfuse_fix; the real Langfuse HTTP logic is exercised by
test_langfuse_client.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_svc.prompt_stores import (
    ExternalPromptStore,
    LangfuseExternalPromptStore,
    get_connected_prompt_store,
)


class TestLangfuseExternalPromptStore(unittest.IsolatedAsyncioTestCase):
    async def test_push_fix_delegates_to_apply_langfuse_fix(self):
        store = LangfuseExternalPromptStore()
        expected = {
            "new_version": 3,
            "prompt_url": "https://example.com/prompts/my-prompt",
            "old_text": "old",
            "new_text": "old\n\nnew",
        }
        with patch(
            "api_svc.langfuse_client.apply_langfuse_fix", AsyncMock(return_value=expected)
        ) as mock_apply:
            result = await store.push_fix("my-prompt", "new")
        mock_apply.assert_awaited_once_with("my-prompt", "new")
        self.assertEqual(result, expected)


class TestGetConnectedPromptStore(unittest.TestCase):
    def test_returns_langfuse_store_when_configured(self):
        s = MagicMock()
        s.langfuse_configured = True
        with patch("api_svc.config.settings", s):
            store = get_connected_prompt_store()
        self.assertIsInstance(store, LangfuseExternalPromptStore)

    def test_returns_none_when_nothing_configured(self):
        s = MagicMock()
        s.langfuse_configured = False
        with patch("api_svc.config.settings", s):
            store = get_connected_prompt_store()
        self.assertIsNone(store)

    def test_external_prompt_store_is_abstract(self):
        with self.assertRaises(TypeError):
            ExternalPromptStore()


if __name__ == "__main__":
    unittest.main(verbosity=2)
