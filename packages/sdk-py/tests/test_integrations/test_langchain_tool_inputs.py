"""LangChain tool recording must preserve privacy and approval behavior."""

import asyncio
import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

try:
    from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager

    HAVE_LANGCHAIN = True
except ImportError:
    HAVE_LANGCHAIN = False

from dunetrace.context import _current_run
from dunetrace.detectors import MalformedToolCallDetector
from dunetrace.integrations.langchain import DunetraceCallbackHandler
from dunetrace.models import EventType
from dunetrace.policies import Policy, PolicyEngine
from dunetrace.run_builder import build_run_state
from dunetrace.run_context import RunContext


@unittest.skipUnless(HAVE_LANGCHAIN, "requires langchain-core")
class TestLangChainToolInputs(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.client = SimpleNamespace(
            _policy_engine=PolicyEngine(), _emit=self.events.append, flush=lambda: None
        )
        self.handler = DunetraceCallbackHandler(self.client, agent_id="test")
        self.run = RunContext(self.client, "test", "v1", available_tools=[], input_text="q")
        token = _current_run.set(self.run)
        self.addCleanup(_current_run.reset, token)

    def test_sync_structured_input_uses_original_dictionary(self):
        inputs = {"query": "refunds", "token": "private"}
        CallbackManager(handlers=[self.handler]).on_tool_start(
            {"name": "search"}, str(inputs), inputs=inputs
        )
        args = json.loads(self.run.state.tool_calls[0].args)
        self.assertEqual(args, {"query": "refunds", "token": "[REDACTED]"})
        self.assertIsNone(MalformedToolCallDetector().on_run_completion(self.run.state))

    def test_async_structured_input_uses_original_dictionary(self):
        async def call():
            await AsyncCallbackManager(handlers=[self.handler]).on_tool_start(
                {"name": "search"}, "{'query': 'refunds'}", inputs={"query": "refunds"}
            )

        asyncio.run(call())
        self.assertEqual(json.loads(self.run.state.tool_calls[0].args), {"query": "refunds"})
        self.assertIsNone(MalformedToolCallDetector().on_run_completion(self.run.state))

    def test_string_inputs_still_use_configured_redaction(self):
        seen = []

        def redact(args):
            seen.append(args)
            return {"input": args["input"].replace("private", "[REDACTED]")}

        self.run._redact_hook = redact
        text = '{"secret_note": "private",}'
        CallbackManager(handlers=[self.handler]).on_tool_start({"name": "code"}, text)
        self.assertEqual(seen, [{"input": text}])
        self.assertNotIn("private", self.run.state.tool_calls[0].args)
        called = next(e for e in self.events if e.event_type == EventType.TOOL_CALLED)
        self.assertNotIn("private", called.payload["args"])
        self.assertIsNotNone(MalformedToolCallDetector().on_run_completion(self.run.state))

    def test_approval_policies_keep_the_raw_input_view(self):
        for value, match in [
            ("private", {"args.input": {"eq": "private"}}),
            ({"amount": 500}, {"args.input.amount": {"gt": 100}}),
        ]:
            for callback in ("tool_start", "agent_action"):
                with self.subTest(value=value, callback=callback):
                    self.client._policy_engine = PolicyEngine()
                    self.client._policy_engine.add(
                        Policy(
                            name="approval",
                            condition={
                                "trigger": "before_tool_call",
                                "operator": "eq",
                                "value": "send",
                                "match": match,
                            },
                            action={"type": "require_approval"},
                        )
                    )
                    self.run.request_approval = MagicMock()
                    if callback == "tool_start":
                        CallbackManager(handlers=[self.handler]).on_tool_start(
                            {"name": "send"},
                            str(value),
                            inputs=value if isinstance(value, dict) else None,
                        )
                    else:
                        self.handler.on_agent_action(SimpleNamespace(tool="send", tool_input=value))
                    self.run.request_approval.assert_called_once_with(
                        "send", {"input": value}, timeout_s=300
                    )

    def test_ambient_retriever_callbacks_share_indexes(self):
        manager = CallbackManager(handlers=[self.handler])
        first = manager.on_retriever_start({"name": "A"}, "q")
        second = manager.on_retriever_start({"name": "B"}, "q")
        first.on_retriever_end([])
        second.on_retriever_error(ValueError("down"))
        state = build_run_state([asdict(e) for e in self.events])
        self.assertEqual([r.index_name for r in state.retrievals], ["A", "B"])
        self.assertEqual(self.handler._ctx({}).child_retrieval_indexes, {})
