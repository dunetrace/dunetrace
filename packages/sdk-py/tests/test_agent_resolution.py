"""
Tests for dunetrace.integrations._agent_resolution.resolve_agent_id() — the
shared tiered agent_id resolution used by LangChain/CrewAI auto-instrumentation
when they need to open a new run (tiers 2-4; tier 1, an already-active
dt.run(), is checked by each caller before this function is ever invoked).

No network, no framework packages required.
"""

import unittest

from dunetrace.integrations._agent_resolution import FALLBACK_AGENT_ID, resolve_agent_id


class TestTierPrecedence(unittest.TestCase):
    def test_per_call_wins_over_everything(self):
        result = resolve_agent_id(
            per_call_agent_id="from-call",
            framework_native_agent_id="from-framework",
            default_agent_id="from-default",
        )
        self.assertEqual(result, "from-call")

    def test_framework_native_wins_over_default(self):
        result = resolve_agent_id(
            per_call_agent_id=None,
            framework_native_agent_id="from-framework",
            default_agent_id="from-default",
        )
        self.assertEqual(result, "from-framework")

    def test_default_used_when_nothing_more_specific(self):
        result = resolve_agent_id(default_agent_id="from-default")
        self.assertEqual(result, "from-default")

    def test_empty_string_per_call_falls_through(self):
        """An empty string is treated the same as None — not a real override."""
        result = resolve_agent_id(per_call_agent_id="", default_agent_id="from-default")
        self.assertEqual(result, "from-default")

    def test_empty_string_framework_native_falls_through(self):
        result = resolve_agent_id(framework_native_agent_id="", default_agent_id="from-default")
        self.assertEqual(result, "from-default")


class TestLoudFallback(unittest.TestCase):
    def test_fallback_used_when_all_tiers_empty(self):
        result = resolve_agent_id()
        self.assertEqual(result, FALLBACK_AGENT_ID)

    def test_fallback_logs_a_warning_naming_the_integration(self):
        with self.assertLogs("dunetrace.auto", level="WARNING") as cm:
            resolve_agent_id(integration="langchain")
        self.assertTrue(any("langchain" in m for m in cm.output))
        self.assertTrue(any("could not determine an agent_id" in m for m in cm.output))

    def test_no_warning_logged_when_a_tier_resolves(self):
        import logging

        logger = logging.getLogger("dunetrace.auto")
        handled = []

        class _Counter(logging.Handler):
            def emit(self, record):
                handled.append(record)

        h = _Counter()
        logger.addHandler(h)
        try:
            resolve_agent_id(default_agent_id="from-default")
        finally:
            logger.removeHandler(h)
        self.assertEqual(len(handled), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
