"""
Tests for DunetraceModelClient and DunetraceAutoGenObserver (AutoGen 0.4+).

Stubs out autogen_core so no autogen package is required.
Run:
    cd packages/sdk-py
    python -m unittest tests.test_integrations.test_autogen -v
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Stub autogen_core so we can import the integration without the real package
# ---------------------------------------------------------------------------
def _stub_autogen() -> None:
    if "autogen_core" in sys.modules:
        return

    autogen_core = types.ModuleType("autogen_core")
    autogen_core_models = types.ModuleType("autogen_core.models")

    class ChatCompletionClient:
        pass

    class RequestUsage:
        def __init__(self, prompt_tokens=0, completion_tokens=0):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class CreateResult:
        def __init__(self, content="", finish_reason="stop", usage=None):
            self.content = content
            self.finish_reason = finish_reason
            self.usage = usage or RequestUsage()

    class LLMMessage:
        pass

    class CancellationToken:
        pass

    autogen_core_models.ChatCompletionClient = ChatCompletionClient
    autogen_core_models.CreateResult = CreateResult
    autogen_core_models.LLMMessage = LLMMessage
    autogen_core_models.RequestUsage = RequestUsage
    autogen_core.models = autogen_core_models
    autogen_core.CancellationToken = CancellationToken

    sys.modules["autogen_core"] = autogen_core
    sys.modules["autogen_core.models"] = autogen_core_models


_stub_autogen()

from dunetrace import Dunetrace  # noqa: E402
from dunetrace.integrations.autogen import DunetraceModelClient, DunetraceAutoGenObserver  # noqa: E402
from dunetrace.models import EventType  # noqa: E402


def _make_dt() -> tuple[Dunetrace, list]:
    from dunetrace.policies import PolicyEngine

    dt = Dunetrace.__new__(Dunetrace)
    dt._ingest_url = "http://localhost:8001/v1/ingest"
    dt._api_key = ""
    dt._otel_exporter = None
    dt._policy_engine = PolicyEngine()
    emitted = []
    dt._emit = lambda event: emitted.append(event)
    dt.flush = lambda: None
    return dt, emitted


def _make_inner_client(content="Paris", prompt_tokens=100, completion_tokens=40):
    """Return a mock inner AutoGen client."""
    from autogen_core.models import CreateResult, RequestUsage

    result = CreateResult(
        content=content,
        finish_reason="stop",
        usage=RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    inner = MagicMock()
    inner.create = AsyncMock(return_value=result)
    inner.model_info = {"model": "gpt-4o-mini"}
    return inner, result


class TestDunetraceModelClientCreate(unittest.IsolatedAsyncioTestCase):
    async def test_emits_llm_called_before_inner_create(self):
        dt, emitted = _make_dt()
        inner, _ = _make_inner_client()
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test") as run:
            await client.create([MagicMock(content="hello")])

        called = [e for e in emitted if e.event_type == EventType.LLM_CALLED]
        self.assertEqual(len(called), 1)

    async def test_emits_llm_responded_after_inner_create(self):
        dt, emitted = _make_dt()
        inner, _ = _make_inner_client()
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test") as run:
            await client.create([MagicMock(content="hello")])

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)

    async def test_completion_tokens_from_usage(self):
        dt, emitted = _make_dt()
        inner, _ = _make_inner_client(completion_tokens=55)
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test") as run:
            await client.create([MagicMock(content="q")])

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED][0]
        self.assertEqual(responded.payload.get("completion_tokens"), 55)

    async def test_no_prompt_tokens_in_llm_responded(self):
        """AutoGen uses llm.called for prompt_tokens — llm.responded must not double-count."""
        dt, emitted = _make_dt()
        inner, _ = _make_inner_client(prompt_tokens=200, completion_tokens=40)
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test") as run:
            await client.create([MagicMock(content="msg")])

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED][0]
        self.assertNotIn("prompt_tokens", responded.payload)

    async def test_prompt_tokens_estimated_in_llm_called(self):
        """Prompt tokens are estimated from message text and placed in llm.called."""
        dt, emitted = _make_dt()
        inner, _ = _make_inner_client()
        msg = MagicMock()
        msg.content = "a" * 400  # 400 chars → ~100 tokens (÷4)
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test") as run:
            await client.create([msg])

        called = [e for e in emitted if e.event_type == EventType.LLM_CALLED][0]
        self.assertGreater(called.payload.get("prompt_tokens", 0), 0)

    async def test_usage_none_does_not_raise(self):
        """Regression: if usage is None (malformed client), create() must not raise."""
        from autogen_core.models import CreateResult

        result = CreateResult(content="ok", finish_reason="stop", usage=None)
        inner = MagicMock()
        inner.create = AsyncMock(return_value=result)
        inner.model_info = {"model": "gpt-4o"}
        dt, emitted = _make_dt()
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test"):
            result = await client.create([MagicMock(content="q")])

        self.assertEqual(result.content, "ok")
        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)
        self.assertEqual(responded[0].payload.get("completion_tokens"), 0)

    async def test_llm_error_emits_responded_with_error_finish_reason(self):
        dt, emitted = _make_dt()
        inner = MagicMock()
        inner.create = AsyncMock(side_effect=RuntimeError("timeout"))
        inner.model_info = {"model": "gpt-4o"}
        client = DunetraceModelClient(inner)

        with self.assertRaises(RuntimeError):
            with dt.run("autogen-agent", user_input="test"):
                await client.create([MagicMock(content="q")])

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)
        self.assertEqual(responded[0].payload.get("finish_reason"), "error")

    async def test_no_run_context_no_emit(self):
        """create() outside a dt.run() context must not crash and must not emit."""
        dt, emitted = _make_dt()
        inner, _ = _make_inner_client()
        client = DunetraceModelClient(inner)

        result = await client.create([MagicMock(content="q")])
        self.assertEqual(result.content, "Paris")
        self.assertEqual(emitted, [])

    async def test_delegates_to_inner_create(self):
        dt, _ = _make_dt()
        inner, expected = _make_inner_client(content="The answer is 42")
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test"):
            result = await client.create([MagicMock(content="q")])

        self.assertEqual(result.content, "The answer is 42")

    async def test_finish_reason_passed_through(self):
        from autogen_core.models import CreateResult, RequestUsage

        result = CreateResult(content="", finish_reason="length", usage=RequestUsage(10, 5))
        inner = MagicMock()
        inner.create = AsyncMock(return_value=result)
        inner.model_info = {"model": "gpt-4o"}
        dt, emitted = _make_dt()
        client = DunetraceModelClient(inner)

        with dt.run("autogen-agent", user_input="test"):
            await client.create([MagicMock(content="q")])

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED][0]
        self.assertEqual(responded.payload.get("finish_reason"), "length")


class TestDunetraceModelClientDelegation(unittest.TestCase):
    """Non-create methods must delegate to the inner client."""

    def _make_client(self) -> tuple[DunetraceModelClient, MagicMock]:
        dt, _ = _make_dt()
        inner, _ = _make_inner_client()
        inner.count_tokens = MagicMock(return_value=42)
        inner.remaining_tokens = MagicMock(return_value=999)
        inner.actual_usage = MagicMock(return_value="usage")
        inner.total_usage = MagicMock(return_value="total")
        inner.close = AsyncMock()
        return DunetraceModelClient(inner), inner

    def test_count_tokens_delegates(self):
        client, inner = self._make_client()
        self.assertEqual(client.count_tokens("msg"), 42)
        inner.count_tokens.assert_called_once()

    def test_remaining_tokens_delegates(self):
        client, inner = self._make_client()
        self.assertEqual(client.remaining_tokens(), 999)

    def test_model_info_property(self):
        client, _ = self._make_client()
        self.assertEqual(client.model_info, {"model": "gpt-4o-mini"})

    def test_actual_usage_delegates(self):
        client, inner = self._make_client()
        self.assertEqual(client.actual_usage(), "usage")

    def test_total_usage_delegates(self):
        client, inner = self._make_client()
        self.assertEqual(client.total_usage(), "total")

    def test_model_name_from_info_dict(self):
        client, _ = self._make_client()
        self.assertEqual(client._model_name(), "gpt-4o-mini")

    def test_model_name_falls_back_to_unknown(self):
        dt, _ = _make_dt()
        inner, _ = _make_inner_client()
        inner.model_info = None
        client = DunetraceModelClient(inner)
        self.assertEqual(client._model_name(), "unknown")


class TestEstimateTokens(unittest.TestCase):
    def test_empty_returns_zero(self):
        self.assertEqual(DunetraceModelClient._estimate_tokens([]), 0)

    def test_none_returns_zero(self):
        self.assertEqual(DunetraceModelClient._estimate_tokens(None), 0)

    def test_400_chars_approx_100_tokens(self):
        msg = MagicMock()
        msg.content = "a" * 400
        self.assertEqual(DunetraceModelClient._estimate_tokens([msg]), 100)

    def test_multiple_messages_summed(self):
        msgs = [MagicMock(content="a" * 200), MagicMock(content="b" * 200)]
        self.assertEqual(DunetraceModelClient._estimate_tokens(msgs), 100)

    def test_message_without_content_attr(self):
        msg = MagicMock(spec=[])  # no content attribute
        self.assertGreaterEqual(DunetraceModelClient._estimate_tokens([msg]), 0)


class TestAutoGenObserver(unittest.IsolatedAsyncioTestCase):
    async def test_run_context_emits_run_started(self):
        dt, emitted = _make_dt()
        observer = DunetraceAutoGenObserver(dt, agent_id="test-agent")

        async with observer.run(user_input="Hello"):
            pass

        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED]
        self.assertEqual(len(started), 1)

    async def test_run_context_emits_run_completed(self):
        dt, emitted = _make_dt()
        observer = DunetraceAutoGenObserver(dt, agent_id="test-agent")

        async with observer.run(user_input="Hello"):
            pass

        completed = [e for e in emitted if e.event_type == EventType.RUN_COMPLETED]
        self.assertEqual(len(completed), 1)

    async def test_wrap_client_returns_dt_client(self):
        dt, _ = _make_dt()
        observer = DunetraceAutoGenObserver(dt)
        inner, _ = _make_inner_client()
        wrapped = observer.wrap_client(inner)
        self.assertIsInstance(wrapped, DunetraceModelClient)

    async def test_full_pipeline_run_events(self):
        """Full pipeline: run_started → llm_called → llm_responded → run_completed."""
        dt, emitted = _make_dt()
        observer = DunetraceAutoGenObserver(dt, agent_id="pipeline-agent")
        inner, _ = _make_inner_client(content="answer", completion_tokens=30)
        dt_client = observer.wrap_client(inner)

        async with observer.run(user_input="What is 2+2?"):
            await dt_client.create([MagicMock(content="What is 2+2?")])

        types_seen = [e.event_type for e in emitted]
        self.assertIn(EventType.RUN_STARTED, types_seen)
        self.assertIn(EventType.LLM_CALLED, types_seen)
        self.assertIn(EventType.LLM_RESPONDED, types_seen)
        self.assertIn(EventType.RUN_COMPLETED, types_seen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
