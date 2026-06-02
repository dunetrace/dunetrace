"""
Dunetrace integration for Microsoft AutoGen (autogen-agentchat 0.4+).

Provides a ``DunetraceModelClient`` wrapper that tracks every LLM call as
``llm.called`` / ``llm.responded`` events, and a ``DunetraceAutoGenObserver``
convenience class for wrapping team/agent runs with a Dunetrace run context.

Usage::

    import asyncio
    import os
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from dunetrace import Dunetrace
    from dunetrace.integrations.autogen import DunetraceAutoGenObserver

    dt = Dunetrace()
    observer = DunetraceAutoGenObserver(dt, agent_id="my-autogen", model="gpt-4o-mini")

    base_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    async def main():
        # Wrap the model client so every LLM call is tracked
        dt_client = observer.wrap_client(base_client)

        assistant = AssistantAgent("assistant", model_client=dt_client,
                                    system_message="You are helpful.")
        team = RoundRobinGroupChat([assistant],
                                    termination_condition=MaxMessageTermination(5))

        async with observer.run(user_input="What is the capital of France?"):
            result = await team.run(task="What is the capital of France?")

        await base_client.close()
        dt.shutdown()

    asyncio.run(main())

Compatible with autogen-agentchat >= 0.4 (tested with 0.7.x).
For older pyautogen 0.2.x, use the manual dt.run() + run.llm_called() approach instead.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence

try:
    from autogen_core.models import (
        ChatCompletionClient,
        CreateResult,
        LLMMessage,
    )
    from autogen_core import CancellationToken
    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False
    ChatCompletionClient = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.autogen")


class DunetraceModelClient(ChatCompletionClient):  # type: ignore[misc]
    """
    Transparent wrapper around any AutoGen ChatCompletionClient that emits
    Dunetrace ``llm.called`` / ``llm.responded`` events for every ``create()``
    call.

    Delegates all other methods (``count_tokens``, ``model_info``, etc.) to the
    wrapped client, so it is a drop-in replacement.
    """

    def __init__(self, inner: Any) -> None:
        if not _AUTOGEN_AVAILABLE:
            raise ImportError(
                "autogen-agentchat and autogen-core are not installed. "
                "Run: pip install autogen-agentchat autogen-ext"
            )
        self._inner = inner

    async def create(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[Any] = None,
    ) -> "CreateResult":
        from dunetrace.context import _current_run
        from dunetrace.models import hash_content

        run = _current_run.get(None)
        model = self._model_name()

        if run:
            prompt_tokens = self._estimate_tokens(messages)
            run.llm_called(model, prompt_tokens=prompt_tokens)

        t0 = time.time()
        try:
            result: CreateResult = await self._inner.create(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )
            if run:
                latency     = int((time.time() - t0) * 1000)
                output_text = result.content if isinstance(result.content, str) else ""
                run.llm_responded(
                    completion_tokens=result.usage.completion_tokens,
                    latency_ms=latency,
                    finish_reason=str(result.finish_reason),
                    output_hash=hash_content(output_text),
                    output_length=len(output_text),
                )
            return result
        except Exception:
            if run:
                run.llm_responded(
                    latency_ms=int((time.time() - t0) * 1000),
                    finish_reason="error",
                )
            raise

    async def create_stream(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.create_stream(*args, **kwargs)

    def count_tokens(self, *args: Any, **kwargs: Any) -> int:
        return self._inner.count_tokens(*args, **kwargs)

    def remaining_tokens(self, *args: Any, **kwargs: Any) -> int:
        return self._inner.remaining_tokens(*args, **kwargs)

    @property
    def model_info(self) -> Any:
        return self._inner.model_info

    @property
    def capabilities(self) -> Any:
        return self._inner.capabilities

    def actual_usage(self) -> Any:
        return self._inner.actual_usage()

    def total_usage(self) -> Any:
        return self._inner.total_usage()

    async def close(self) -> None:
        await self._inner.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _model_name(self) -> str:
        try:
            info = self._inner.model_info
            return info.get("model", "unknown") if isinstance(info, dict) else str(info)
        except Exception:
            return "unknown"

    @staticmethod
    def _estimate_tokens(messages: Any) -> int:
        if not messages:
            return 0
        try:
            total = sum(
                len(str(getattr(m, "content", m) or ""))
                for m in messages
            )
            return total // 4
        except Exception:
            return 0

    # ── Component protocol (required by autogen_core) ─────────────────────────
    # These are forwarded so dump_component / load_component work if needed.
    def dump_component(self) -> Any:
        return self._inner.dump_component()

    @classmethod
    def load_component(cls, model: Any) -> Any:
        return cls._inner.load_component(model)


class DunetraceAutoGenObserver:
    """
    Convenience wrapper for monitoring an autogen-agentchat run.

    1. Call ``wrap_client(base_client)`` to get a Dunetrace-instrumented model
       client to pass to your agents.
    2. Use ``async with observer.run(user_input=...):`` to open a Dunetrace run
       for the duration of the conversation.
    """

    def __init__(
        self,
        client:   "Dunetrace",
        agent_id: str = "autogen-agent",
        model:    str = "unknown",
        tools:    Optional[List[str]] = None,
    ) -> None:
        if not _AUTOGEN_AVAILABLE:
            raise ImportError(
                "autogen-agentchat and autogen-core are not installed. "
                "Run: pip install autogen-agentchat autogen-ext"
            )
        self._client   = client
        self._agent_id = agent_id
        self._model    = model
        self._tools    = tools or []

    def wrap_client(self, inner_client: Any) -> DunetraceModelClient:
        """
        Return a DunetraceModelClient wrapping ``inner_client``.
        Pass the result to AssistantAgent(model_client=...) instead of the original.
        """
        return DunetraceModelClient(inner_client)

    @asynccontextmanager
    async def run(self, user_input: str = "", **run_opts: Any):
        """
        Async context manager that opens a Dunetrace run for the block.

        Usage::

            async with observer.run(user_input=query):
                result = await team.run(task=query)
        """
        with self._client.run(
            self._agent_id,
            user_input=user_input,
            model=self._model,
            tools=self._tools,
            **run_opts,
        ) as ctx:
            yield ctx
            ctx.final_answer()
