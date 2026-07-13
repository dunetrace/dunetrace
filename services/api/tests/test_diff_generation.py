"""
Tests for Phase 4.3's real diff generation (api_svc/diff_generation.py) —
asks the LLM for corrected full file content (not a diff to mechanically
apply), then computes the diff ourselves via difflib. Mocks the LLM client;
compute_unified_diff itself uses real difflib against known strings, so
that part is genuinely exercised, not mocked.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_svc.diff_generation import compute_unified_diff, generate_real_file_content


class TestGenerateRealFileContent(unittest.IsolatedAsyncioTestCase):
    async def test_returns_none_when_no_llm_key_configured(self):
        with patch("api_svc.diff_generation.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = ""
            mock_settings.OPENAI_API_KEY = ""
            result = await generate_real_file_content("root cause", "fix it", "a.py", "old content")
        self.assertIsNone(result)

    async def test_returns_new_content_from_anthropic(self):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="def fixed():\n    pass\n")]
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_msg)

        with (
            patch("api_svc.diff_generation.settings") as mock_settings,
            patch("anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            mock_settings.ANTHROPIC_API_KEY = "sk-ant-x"
            mock_settings.OPENAI_API_KEY = ""
            result = await generate_real_file_content(
                "loop bug", "add a limit", "a.py", "def broken():\n    pass\n"
            )

        self.assertEqual(result, "def fixed():\n    pass")

    async def test_returns_none_when_llm_call_fails(self):
        with (
            patch("api_svc.diff_generation.settings") as mock_settings,
            patch("anthropic.AsyncAnthropic", side_effect=RuntimeError("API down")),
        ):
            mock_settings.ANTHROPIC_API_KEY = "sk-ant-x"
            mock_settings.OPENAI_API_KEY = ""
            result = await generate_real_file_content("rc", "fix", "a.py", "old")

        self.assertIsNone(result)

    async def test_returns_none_when_content_unchanged(self):
        """The model declining to apply a fix (returning the same content
        verbatim, per its own instructions) must be treated as 'nothing to
        apply,' not a successful no-op fix."""
        unchanged = "def same():\n    pass\n"
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=unchanged)]
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_msg)

        with (
            patch("api_svc.diff_generation.settings") as mock_settings,
            patch("anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            mock_settings.ANTHROPIC_API_KEY = "sk-ant-x"
            mock_settings.OPENAI_API_KEY = ""
            result = await generate_real_file_content("rc", "fix", "a.py", unchanged)

        self.assertIsNone(result)

    async def test_returns_none_when_content_empty(self):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="")]
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_msg)

        with (
            patch("api_svc.diff_generation.settings") as mock_settings,
            patch("anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            mock_settings.ANTHROPIC_API_KEY = "sk-ant-x"
            mock_settings.OPENAI_API_KEY = ""
            result = await generate_real_file_content("rc", "fix", "a.py", "old")

        self.assertIsNone(result)

    async def test_strips_markdown_fences(self):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="```python\ndef fixed():\n    pass\n```")]
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_msg)

        with (
            patch("api_svc.diff_generation.settings") as mock_settings,
            patch("anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            mock_settings.ANTHROPIC_API_KEY = "sk-ant-x"
            mock_settings.OPENAI_API_KEY = ""
            result = await generate_real_file_content("rc", "fix", "a.py", "def broken(): pass")

        self.assertEqual(result, "def fixed():\n    pass")

    async def test_falls_back_to_openai_when_no_anthropic_key(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "def fixed():\n    pass\n"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with (
            patch("api_svc.diff_generation.settings") as mock_settings,
            patch("openai.AsyncOpenAI", return_value=mock_client),
        ):
            mock_settings.ANTHROPIC_API_KEY = ""
            mock_settings.OPENAI_API_KEY = "sk-oai-x"
            result = await generate_real_file_content("rc", "fix", "a.py", "def broken(): pass")

        self.assertEqual(result, "def fixed():\n    pass")


class TestComputeUnifiedDiff(unittest.TestCase):
    def test_produces_well_formed_diff(self):
        old = "def broken():\n    pass\n"
        new = "def fixed():\n    return True\n"
        diff = compute_unified_diff("a.py", old, new)

        self.assertIn("--- a/a.py", diff)
        self.assertIn("+++ b/a.py", diff)
        self.assertIn("-def broken():", diff)
        self.assertIn("+def fixed():", diff)

    def test_identical_content_produces_empty_diff(self):
        content = "def same():\n    pass\n"
        diff = compute_unified_diff("a.py", content, content)
        self.assertEqual(diff, "")

    def test_diff_is_always_parseable_unified_format(self):
        """Since this is computed by us from two known strings (never an
        LLM-authored diff), it must always have well-formed @@ hunks."""
        old = "line1\nline2\nline3\n"
        new = "line1\nCHANGED\nline3\nline4\n"
        diff = compute_unified_diff("f.txt", old, new)
        self.assertIn("@@", diff)


if __name__ == "__main__":
    unittest.main()
