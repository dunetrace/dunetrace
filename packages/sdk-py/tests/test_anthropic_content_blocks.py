"""
_anthropic_content — an Anthropic reply is a LIST of content blocks.

The bug: the extractor read `resp.content[0]` only. With extended thinking
enabled block 0 is the thinking block, whose `text` is empty, so every response
from a reasoning model recorded output=""/output_length=0 even though the model
had answered normally — the same class of silent under-reporting as the
LegacyAPIResponse bug in test_openai_raw_response_unwrap.py, reached from a
different direction. Any multi-block reply (text + tool_use, or several text
blocks) was likewise truncated to its first block.

The three-way return contract, shared with _openai_content and
_bedrock_converse_text and asserted below:
    text  — the shape was read and there was text
    ""    — the shape was read and there was legitimately no text
            (a tool-only turn), which must NOT look like an instrumentation
            failure
    None  — the shape could not be read at all

Tests construct the shapes directly — no network, no API key.

Run: python -m pytest tests/test_anthropic_content_blocks.py -v
"""

from __future__ import annotations

import unittest

from dunetrace import Dunetrace
from dunetrace.auto import _anthropic_content, _emit_anthropic_response
from dunetrace.models import EventType


# ── Block shapes ─────────────────────────────────────────────────────────────
#
# Real blocks are pydantic models, not dicts: attributes, no .get(). A thinking
# or tool_use block has no `text` attribute at all.


def _text_block(text):
    return type("TextBlock", (), {"type": "text", "text": text})()


def _thinking_block(thinking="Let me work through this..."):
    return type("ThinkingBlock", (), {"type": "thinking", "thinking": thinking})()


def _tool_use_block(name="get_weather"):
    return type("ToolUseBlock", (), {"type": "tool_use", "name": name, "input": {}})()


def _message(content, stop_reason="end_turn", input_tokens=12, output_tokens=30):
    usage = type("Usage", (), {"input_tokens": input_tokens, "output_tokens": output_tokens})()
    return type("Message", (), {"content": content, "stop_reason": stop_reason, "usage": usage})()


class TestAnthropicContentBlocks(unittest.TestCase):
    # ── returns text ─────────────────────────────────────────────────────────

    def test_single_text_block(self):
        self.assertEqual(
            _anthropic_content(_message([_text_block("The capital of France is Paris.")])),
            "The capital of France is Paris.",
        )

    def test_thinking_block_first_does_not_swallow_the_answer(self):
        """The regression. Extended thinking puts the thinking block at index 0,
        so reading only block 0 reported an empty response for every call."""
        resp = _message([_thinking_block(), _text_block("The answer is 42.")])
        self.assertEqual(_anthropic_content(resp), "The answer is 42.")

    def test_multiple_text_blocks_are_joined_not_truncated(self):
        resp = _message([_text_block("Part one. "), _text_block("Part two.")])
        self.assertEqual(_anthropic_content(resp), "Part one. Part two.")

    def test_text_before_tool_use_is_kept(self):
        resp = _message([_text_block("Let me check the weather."), _tool_use_block()])
        self.assertEqual(_anthropic_content(resp), "Let me check the weather.")

    # ── returns "" ───────────────────────────────────────────────────────────

    def test_tool_only_turn_is_empty_string_not_none(self):
        """No text-bearing block: the shape was read fine and the model
        genuinely produced no text. None here would mark a normal tool call as
        an instrumentation failure."""
        self.assertEqual(_anthropic_content(_message([_tool_use_block()])), "")

    def test_thinking_only_turn_is_empty_string(self):
        self.assertEqual(_anthropic_content(_message([_thinking_block()])), "")

    def test_text_block_with_empty_text_is_empty_string(self):
        self.assertEqual(_anthropic_content(_message([_text_block("")])), "")

    def test_empty_content_list_is_empty_string(self):
        """A readable envelope carrying no blocks is a genuinely empty response,
        not an unreadable one — same as _bedrock_converse_text's empty list."""
        self.assertEqual(_anthropic_content(_message([])), "")

    # ── returns None ─────────────────────────────────────────────────────────

    def test_no_content_attribute_is_none(self):
        self.assertIsNone(_anthropic_content(object()))

    def test_content_none_is_none(self):
        self.assertIsNone(_anthropic_content(_message(None)))

    def test_non_list_content_is_none(self):
        """A raw envelope or a version skew, not a Message."""
        self.assertIsNone(_anthropic_content(_message("just a string")))
        self.assertIsNone(_anthropic_content(_message({"text": "a dict"})))
        self.assertIsNone(_anthropic_content(_message(42)))

    def test_none_response_is_none(self):
        self.assertIsNone(_anthropic_content(None))


class TestAnthropicResponseEmission(unittest.TestCase):
    """End-to-end through _emit_anthropic_response: the joined text and its
    length reach the event, and the degraded marker follows the same three-way
    split."""

    def _client(self):
        c = Dunetrace(api_key="k")
        c._ship = lambda batch: None
        return c

    def _emit(self, resp):
        dt = self._client()
        with dt.run("agent") as run:
            run.llm_called("claude-sonnet-4-6")
            _emit_anthropic_response(run, resp, 0.0)
            lc = run.state.llm_calls[-1]
            payload = [e for e in run.state.events if e.event_type == EventType.LLM_RESPONDED][
                -1
            ].payload
        dt.shutdown(timeout=1)
        return lc, payload

    def test_reasoning_model_reports_real_output_length(self):
        resp = _message([_thinking_block(), _text_block("The answer is 42.")])
        lc, payload = self._emit(resp)

        self.assertEqual(payload["output"], "The answer is 42.")
        self.assertEqual(lc.output_length, len("The answer is 42."))
        self.assertIsNone(lc.instrumentation_degraded)

    def test_tool_only_turn_is_not_marked_degraded(self):
        lc, payload = self._emit(_message([_tool_use_block()]))

        self.assertEqual(payload["output"], "")
        self.assertEqual(lc.output_length, 0)
        self.assertIsNone(lc.instrumentation_degraded)

    def test_unreadable_shape_is_marked_degraded(self):
        lc, _payload = self._emit(_message("not a block list", stop_reason=None))

        self.assertIsNotNone(lc.instrumentation_degraded)


if __name__ == "__main__":
    unittest.main()
