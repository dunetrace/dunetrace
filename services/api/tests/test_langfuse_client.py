"""
Tests for langfuse_client — no DB, no HTTP server, no Langfuse credentials needed.

Run:
    cd services/api
    python -m unittest tests.test_langfuse_client -v
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── helpers ────────────────────────────────────────────────────────────────────


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_generation(messages: list, output: str = "ok", name: str = "llm") -> dict:
    return {
        "type": "GENERATION",
        "name": name,
        "input": {"messages": messages},
        "output": output,
        "level": "DEFAULT",
        "metadata": {},
    }


def _make_span(name: str, input_val: str = "x", output_val: str = "y") -> dict:
    return {
        "type": "SPAN",
        "name": name,
        "input": input_val,
        "output": output_val,
        "level": "DEFAULT",
        "metadata": {},
    }


def _make_signal(**kw) -> dict:
    return {
        "failure_type": kw.get("failure_type", "TOOL_LOOP"),
        "severity": kw.get("severity", "HIGH"),
        "confidence": kw.get("confidence", 0.92),
        "run_id": kw.get("run_id", "run-abc"),
        "what": kw.get("what", "Agent called search 7 times"),
        "evidence_summary": kw.get("evidence_summary", "Steps 2-6 repeated identical calls"),
        "evidence": kw.get("evidence", {"first_step": 2, "last_step": 6}),
    }


# ── _extract_system_prompt ─────────────────────────────────────────────────────


class TestExtractSystemPrompt(unittest.TestCase):
    def setUp(self):
        from api_svc.langfuse_client import _extract_system_prompt

        self.fn = _extract_system_prompt

    def test_standard_role(self):
        obs_input = {
            "messages": [
                {"role": "system", "content": "You are a research assistant."},
                {"role": "human", "content": "Tell me about climate"},
            ]
        }
        self.assertEqual(self.fn(obs_input), "You are a research assistant.")

    def test_type_field_fallback(self):
        """LangChain sometimes uses 'type' instead of 'role'."""
        obs_input = {
            "messages": [
                {"type": "system", "content": "Use tools carefully."},
                {"type": "human", "content": "Go"},
            ]
        }
        self.assertEqual(self.fn(obs_input), "Use tools carefully.")

    def test_returns_first_system_message(self):
        obs_input = {
            "messages": [
                {"role": "system", "content": "First"},
                {"role": "system", "content": "Second"},
            ]
        }
        self.assertEqual(self.fn(obs_input), "First")

    def test_no_system_message_returns_none(self):
        obs_input = {
            "messages": [
                {"role": "human", "content": "Hello"},
                {"role": "ai", "content": "Hi"},
            ]
        }
        self.assertIsNone(self.fn(obs_input))

    def test_non_dict_input_returns_none(self):
        self.assertIsNone(self.fn("raw string"))
        self.assertIsNone(self.fn(None))
        self.assertIsNone(self.fn(["list"]))

    def test_empty_messages_returns_none(self):
        self.assertIsNone(self.fn({"messages": []}))

    def test_truncates_at_800_chars(self):
        long_content = "x" * 1000
        obs_input = {"messages": [{"role": "system", "content": long_content}]}
        result = self.fn(obs_input)
        self.assertEqual(len(result), 800)

    def test_no_messages_key_returns_none(self):
        self.assertIsNone(self.fn({"input": "something"}))


# ── _format_observations ───────────────────────────────────────────────────────


class TestFormatObservations(unittest.TestCase):
    def setUp(self):
        from api_svc.langfuse_client import _format_observations

        self.fn = _format_observations

    def test_empty_list(self):
        text, sys_prompt = self.fn([], 0, 0)
        self.assertIn("no observations", text)
        self.assertIsNone(sys_prompt)

    def test_extracts_system_prompt_from_generation(self):
        obs = [
            _make_generation(
                [
                    {"role": "system", "content": "You are a helpful agent."},
                    {"role": "human", "content": "Do something"},
                ]
            )
        ]
        _, sys_prompt = self.fn(obs, 0, 0)
        self.assertEqual(sys_prompt, "You are a helpful agent.")

    def test_no_system_prompt_returns_none(self):
        obs = [_make_span("search_tool")]
        _, sys_prompt = self.fn(obs, 0, 0)
        self.assertIsNone(sys_prompt)

    def test_skips_event_type(self):
        obs = [
            {
                "type": "EVENT",
                "name": "some_event",
                "input": "x",
                "output": "y",
                "level": "DEFAULT",
                "metadata": {},
            },
            _make_span("real_tool"),
        ]
        text, _ = self.fn(obs, 0, 0)
        self.assertNotIn("some_event", text)
        self.assertIn("real_tool", text)

    def test_skips_score_type(self):
        obs = [
            {
                "type": "SCORE",
                "name": "quality",
                "input": "x",
                "output": "y",
                "level": "DEFAULT",
                "metadata": {},
            },
        ]
        text, _ = self.fn(obs, 0, 0)
        self.assertIn("no observations", text)

    def test_skips_langchain_chain_names(self):
        obs = [
            {
                "type": "SPAN",
                "name": "langchain_on_chain_start",
                "input": "x",
                "output": "y",
                "level": "DEFAULT",
                "metadata": {},
            },
            {
                "type": "SPAN",
                "name": "langchain_on_chain_end",
                "input": "x",
                "output": "y",
                "level": "DEFAULT",
                "metadata": {},
            },
            _make_span("real_tool"),
        ]
        text, _ = self.fn(obs, 0, 0)
        self.assertNotIn("langchain_on_chain_start", text)
        self.assertIn("real_tool", text)

    def test_focus_steps_use_higher_limit(self):
        long_input = "a" * 400
        # obs at index 0 — will be in focus_steps {0}
        obs = [_make_span("focused_tool", input_val=long_input)]
        text, _ = self.fn(obs, 0, 0, signal_steps=[0])
        # 400 chars fits within the 600-char focus limit
        self.assertIn("a" * 400, text)

    def test_non_focus_steps_truncated_at_150(self):
        long_input = "b" * 400
        obs = [_make_span("background_tool", input_val=long_input)]
        # signal_steps doesn't include 0, so limit is 150
        text, _ = self.fn(obs, 5, 10, signal_steps=[5, 6, 7, 8, 9, 10])
        self.assertNotIn("b" * 400, text)
        self.assertIn("b" * 150, text)

    def test_metadata_step_used_for_focus(self):
        obs = [
            {
                "type": "SPAN",
                "name": "important_tool",
                "input": "c" * 400,
                "output": "result",
                "level": "DEFAULT",
                "metadata": {"step": 3},
            }
        ]
        text, _ = self.fn(obs, 3, 3, signal_steps=[3])
        self.assertIn("c" * 400, text)

    def test_level_tag_shown_for_errors(self):
        obs = [
            {
                "type": "SPAN",
                "name": "failing_tool",
                "input": "x",
                "output": "err",
                "level": "ERROR",
                "metadata": {},
            }
        ]
        text, _ = self.fn(obs, 0, 0)
        self.assertIn("[ERROR]", text)

    def test_default_level_not_shown(self):
        obs = [_make_span("tool")]
        text, _ = self.fn(obs, 0, 0)
        self.assertNotIn("[DEFAULT]", text)


# ── fetch_langfuse_trace ───────────────────────────────────────────────────────


class TestFetchLangfuseTrace(unittest.IsolatedAsyncioTestCase):
    def _make_settings(self):
        s = MagicMock()
        s.langfuse_configured = True
        s.LANGFUSE_PUBLIC_KEY = "pk-test"
        s.LANGFUSE_SECRET_KEY = "sk-test"
        s.LANGFUSE_HOST = "https://cloud.langfuse.com"
        return s

    def _make_response(self, status: int, body: dict) -> MagicMock:
        r = MagicMock()
        r.status_code = status
        r.json.return_value = body
        r.raise_for_status = MagicMock()
        if status >= 400:
            import httpx

            r.raise_for_status.side_effect = httpx.HTTPStatusError(
                "err", request=MagicMock(), response=r
            )
        return r

    def _patch_client(self, mock_get: AsyncMock):
        """Return a context manager that injects a mock httpx.AsyncClient."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get
        return patch("httpx.AsyncClient", return_value=mock_client)

    async def test_returns_trace_with_few_observations(self):
        trace_body = {"id": "abc", "observations": [{"type": "SPAN"}] * 5}
        mock_get = AsyncMock(return_value=self._make_response(200, trace_body))

        with self._patch_client(mock_get), patch("api_svc.config.settings", self._make_settings()):
            from api_svc.langfuse_client import fetch_langfuse_trace

            result = await fetch_langfuse_trace("abc")

        self.assertEqual(result["id"], "abc")
        # Only one GET call — no pagination fallback needed
        self.assertEqual(mock_get.call_count, 1)

    async def test_fetches_observations_separately_when_paginated(self):
        # 10 embedded observations triggers the fallback
        trace_body = {"id": "abc", "observations": [{"type": "SPAN"}] * 10}
        full_obs = [{"type": "SPAN", "name": f"step_{i}"} for i in range(20)]
        obs_body = {"data": full_obs}

        mock_get = AsyncMock(
            side_effect=[
                self._make_response(200, trace_body),
                self._make_response(200, obs_body),
            ]
        )

        with self._patch_client(mock_get), patch("api_svc.config.settings", self._make_settings()):
            from api_svc.langfuse_client import fetch_langfuse_trace

            result = await fetch_langfuse_trace("abc")

        self.assertEqual(len(result["observations"]), 20)
        self.assertEqual(mock_get.call_count, 2)
        second_call_url = mock_get.call_args_list[1][0][0]
        self.assertIn("/api/public/observations", second_call_url)

    async def test_raises_lookup_error_on_404(self):
        mock_get = AsyncMock(return_value=self._make_response(404, {}))

        with (
            self._patch_client(mock_get),
            patch("api_svc.config.settings", self._make_settings()),
            patch("api_svc.langfuse_client.asyncio.sleep", new_callable=AsyncMock),
        ):
            from api_svc.langfuse_client import fetch_langfuse_trace

            with self.assertRaises(LookupError):
                await fetch_langfuse_trace("missing")

    async def test_raises_value_error_when_not_configured(self):
        s = MagicMock()
        s.langfuse_configured = False

        with patch("api_svc.config.settings", s):
            from api_svc.langfuse_client import fetch_langfuse_trace

            with self.assertRaises(ValueError):
                await fetch_langfuse_trace("any")

    async def test_strips_dashes_from_trace_id(self):
        trace_body = {"id": "abc123", "observations": []}
        mock_get = AsyncMock(return_value=self._make_response(200, trace_body))

        with self._patch_client(mock_get), patch("api_svc.config.settings", self._make_settings()):
            from api_svc.langfuse_client import fetch_langfuse_trace

            await fetch_langfuse_trace("abc-123-def")

        called_url = mock_get.call_args_list[0][0][0]
        self.assertNotIn("-", called_url.split("/")[-1])


# ── build_explain_prompt ───────────────────────────────────────────────────────


class TestBuildExplainPrompt(unittest.IsolatedAsyncioTestCase):
    def _make_trace(self, observations=None, user_input="Research climate"):
        return {
            "input": user_input,
            "observations": observations or [],
        }

    async def test_prompt_leads_with_signal_info(self):
        signal = _make_signal()
        trace = self._make_trace()

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("TOOL_LOOP", prompt)
        self.assertIn("92%", prompt)
        self.assertIn("HIGH", prompt)
        self.assertIn("Steps 2-6", prompt)  # evidence_summary text

    async def test_failing_steps_range_in_prompt(self):
        signal = _make_signal(evidence={"first_step": 3, "last_step": 7})
        trace = self._make_trace()

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("3", prompt)
        self.assertIn("7", prompt)

    async def test_system_prompt_surfaced_when_present(self):
        signal = _make_signal()
        obs = [
            _make_generation(
                [
                    {
                        "role": "system",
                        "content": "Search the web for relevant sources.",
                    },
                    {"role": "human", "content": "Find climate data"},
                ]
            )
        ]
        trace = self._make_trace(observations=obs)

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("Search the web for relevant sources.", prompt)
        self.assertIn("System prompt in use:", prompt)

    async def test_system_prompt_section_shows_not_found(self):
        signal = _make_signal()
        trace = self._make_trace(observations=[_make_span("tool")])

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("System prompt: (not found in trace)", prompt)

    async def test_trace_input_in_prompt(self):
        signal = _make_signal()
        trace = self._make_trace(user_input="Research climate policy trends")

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("Research climate policy trends", prompt)

    async def test_what_field_included(self):
        signal = _make_signal(what="Agent called search 7 times without progress")
        trace = self._make_trace()

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("Agent called search 7 times without progress", prompt)

    async def test_dict_trace_input_converted(self):
        signal = _make_signal()
        trace = {"input": {"query": "test", "context": "ctx"}, "observations": []}

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("query", prompt)

    async def test_handles_missing_evidence_gracefully(self):
        signal = _make_signal(evidence={})
        trace = self._make_trace()

        from api_svc.langfuse_client import build_explain_prompt

        # Should not raise
        prompt = await build_explain_prompt(signal, trace)
        self.assertIsInstance(prompt, str)

    async def test_observations_present_in_prompt(self):
        signal = _make_signal()
        obs = [_make_span("web_search", input_val="climate data", output_val="found 5 results")]
        trace = self._make_trace(observations=obs)

        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)

        self.assertIn("web_search", prompt)


# ── _get_step_range ────────────────────────────────────────────────────────────


class TestGetStepRange(unittest.TestCase):
    def setUp(self):
        from api_svc.langfuse_client import _get_step_range

        self.fn = _get_step_range

    def test_tool_loop_uses_first_last_step(self):
        ev = {"first_step": 2, "last_step": 7}
        self.assertEqual(self.fn(ev, "TOOL_LOOP", 0), (2, 7))

    def test_llm_truncation_loop_uses_truncation_fields(self):
        ev = {"first_truncation_step": 3, "last_truncation_step": 6}
        self.assertEqual(self.fn(ev, "LLM_TRUNCATION_LOOP", 0), (3, 6))

    def test_context_bloat_uses_call_step_fields(self):
        ev = {"first_call_step": 1, "last_call_step": 9}
        self.assertEqual(self.fn(ev, "CONTEXT_BLOAT", 0), (1, 9))

    def test_retry_storm_single_step_from_first_fail(self):
        ev = {"first_fail_step": 4}
        self.assertEqual(self.fn(ev, "RETRY_STORM", 0), (4, 4))

    def test_cascading_tool_failure_uses_first_fail_step(self):
        ev = {"first_fail_step": 5}
        self.assertEqual(self.fn(ev, "CASCADING_TOOL_FAILURE", 0), (5, 5))

    def test_goal_abandonment_maps_tool_step_to_current(self):
        ev = {"last_tool_step": 3, "current_step": 8}
        self.assertEqual(self.fn(ev, "GOAL_ABANDONMENT", 0), (3, 8))

    def test_first_step_failure_uses_failed_step(self):
        ev = {"failed_step": 1}
        self.assertEqual(self.fn(ev, "FIRST_STEP_FAILURE", 0), (1, 1))

    def test_slow_step_uses_step_index(self):
        ev = {"step_index": 6}
        self.assertEqual(self.fn(ev, "SLOW_STEP", 0), (6, 6))

    def test_empty_llm_response_uses_first_step(self):
        ev = {"first_step": 4}
        self.assertEqual(self.fn(ev, "EMPTY_LLM_RESPONSE", 0), (4, 4))

    def test_no_step_range_detectors_fall_back_to_signal_step(self):
        # TOOL_THRASHING, TOOL_AVOIDANCE, REASONING_STALL, etc. have no step range
        for ft in (
            "TOOL_THRASHING",
            "TOOL_AVOIDANCE",
            "REASONING_STALL",
            "STEP_COUNT_INFLATION",
            "RAG_EMPTY_RETRIEVAL",
            "PROMPT_INJECTION_SIGNAL",
        ):
            with self.subTest(failure_type=ft):
                self.assertEqual(self.fn({}, ft, 5), (5, 5))

    def test_missing_evidence_fields_fall_back_to_fallback(self):
        self.assertEqual(self.fn({}, "TOOL_LOOP", 3), (3, 3))

    def test_bad_values_fall_back_to_fallback(self):
        ev = {"first_step": "bad", "last_step": None}
        self.assertEqual(self.fn(ev, "TOOL_LOOP", 7), (7, 7))


# ── build_explain_prompt uses correct step range per detector ──────────────────


class TestBuildExplainPromptStepRange(unittest.IsolatedAsyncioTestCase):
    def _make_signal(self, failure_type, evidence, step_index=5):
        return {
            "failure_type": failure_type,
            "severity": "HIGH",
            "confidence": 0.9,
            "run_id": "run-abc",
            "step_index": step_index,
            "what": "",
            "evidence_summary": "",
            "evidence": evidence,
        }

    async def test_llm_truncation_loop_step_range_correct(self):
        signal = self._make_signal(
            "LLM_TRUNCATION_LOOP",
            {"first_truncation_step": 3, "last_truncation_step": 6},
        )
        trace = {"input": "test", "observations": []}
        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)
        self.assertIn("3", prompt)
        self.assertIn("6", prompt)

    async def test_retry_storm_step_range_uses_first_fail(self):
        signal = self._make_signal("RETRY_STORM", {"first_fail_step": 4})
        trace = {"input": "test", "observations": []}
        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)
        self.assertIn("4", prompt)

    async def test_no_step_range_detector_uses_signal_step_index(self):
        signal = self._make_signal(
            "TOOL_THRASHING", {"tool_a": "search", "tool_b": "calc"}, step_index=8
        )
        trace = {"input": "test", "observations": []}
        from api_svc.langfuse_client import build_explain_prompt

        prompt = await build_explain_prompt(signal, trace)
        self.assertIn("8", prompt)


if __name__ == "__main__":
    unittest.main()
