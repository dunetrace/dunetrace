"""
Tests for custom_detector_translator.py: prompt rendering (must not KeyError on
the new content_fields_list placeholder) and the LLM call path with a mocked
Anthropic response. Fully offline — no real LLM call, no credentials needed.

Both `httpx` and `settings` are imported at module level in
custom_detector_translator.py, so the standard patch target here is the
importing module's own name, not api_svc.config.settings.

Run:
    PYTHONPATH=packages/sdk-py:services/explainer:services/api pytest services/api/tests/test_custom_detector_translator.py -v
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_svc.custom_detector_translator import (
    CONTENT_FIELDS,
    CONTENT_OPERATORS,
    SUPPORTED_METRICS,
    _SYSTEM_PROMPT,
    translate_description,
)
from llm_test_utils import configured_llm, no_llm


class TestSystemPromptRendering(unittest.TestCase):
    def test_prompt_renders_without_keyerror(self):
        metrics_list = "\n".join(f"  - {k}: {v}" for k, v in SUPPORTED_METRICS.items())
        content_fields_list = "\n".join(f"  - {k}: {v}" for k, v in CONTENT_FIELDS.items())
        rendered = _SYSTEM_PROMPT.format(
            metrics_list=metrics_list, content_fields_list=content_fields_list
        )
        for metric in SUPPORTED_METRICS:
            self.assertIn(metric, rendered)
        for field in CONTENT_FIELDS:
            self.assertIn(field, rendered)
        for op in CONTENT_OPERATORS:
            self.assertIn(op, rendered)


class TestTranslateDescription(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _llm_returning(body: dict):
        """Patch the one seam every provider now goes through.

        Deliberately not a mocked httpx/settings: api_svc.config loads the
        repo's .env at import, so patching this module's `settings` alone left
        llm_provider reading a developer's real ANTHROPIC_API_KEY — and the
        test only stayed offline because a patched httpx.AsyncClient happened
        to sit underneath the Anthropic SDK.
        """
        # configured_llm patches the GATE as well as the seam. Patching only
        # `complete` left llm_configured() reading real settings, which passed
        # on a machine with a .env and failed in CI.
        return configured_llm(completion=json.dumps(body))

    async def test_no_llm_key_raises_value_error(self):
        with no_llm():
            with self.assertRaises(ValueError):
                await translate_description("fires when tool_call_count > 3")

    async def test_metric_config_returned_from_anthropic(self):
        body = {
            "detector_name": "CUSTOM_TOO_MANY_CALLS",
            "conditions": [{"metric": "tool_call_count", "operator": ">=", "threshold": 5}],
            "severity": "HIGH",
            "evidence_template": "Too many tool calls",
            "fix_template": "Add a step limit",
            "requires_content": False,
        }
        with self._llm_returning(body):
            result = await translate_description("fires when more than 5 tool calls happen")
        self.assertEqual(result["detector_name"], "CUSTOM_TOO_MANY_CALLS")
        self.assertFalse(result["requires_content"])

    async def test_content_config_returned_from_anthropic(self):
        body = {
            "detector_name": "CUSTOM_ERROR_IN_OUTPUT",
            "conditions": [
                {
                    "field": "tool_error",
                    "operator": "contains",
                    "value": "timeout",
                    "case_sensitive": False,
                }
            ],
            "severity": "MEDIUM",
            "evidence_template": "Tool error mentioned a timeout",
            "fix_template": "Add retry logic",
            "requires_content": False,
        }
        with self._llm_returning(body):
            result = await translate_description("fires when a tool error mentions a timeout")
        self.assertEqual(result["conditions"][0]["field"], "tool_error")

    async def test_declined_response_passed_through_unchanged(self):
        body = {"requires_content": True, "reason": "needs semantic judgment of tone"}
        with self._llm_returning(body):
            result = await translate_description("fires when the agent sounds frustrated")
        self.assertTrue(result["requires_content"])
        self.assertIn("reason", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
