"""
The object returned by `dt.run(...)`. Provides emit helpers like tool_called
and llm_called, and builds up a RunState for local detection.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from dunetrace.models import (
    AgentEvent,
    EventType,
    ExternalSignal,
    LlmCall,
    RetrievalResult,
    RunState,
    ToolCall,
    hash_content,
)
import logging
from dunetrace.policies import PolicyViolation, build_metrics

logger = logging.getLogger("dunetrace.run")

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace


class RunContext:
    """Thin wrapper around a single agent run."""

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str,
        agent_version: str,
        available_tools: list,
        input_text_hash: str,
        parent_run_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._client = client
        self.run_id = run_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.step = 0
        self.exit_reason: Optional[str] = None
        self._parent_run_id = parent_run_id

        self.state = RunState(
            run_id=self.run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            available_tools=available_tools,
            input_text_hash=input_text_hash,
        )

        # Policy enforcement state
        self.model_override: str | None = None  # set by switch_model action
        self.prompt_additions: list = []  # appended by inject_prompt action
        self._triggered_policies: set = set()  # policy keys fired in this run

    # ── LLM hooks ─────────────────────────────────────────────────────────────

    def llm_called(self, model: str, prompt_tokens: int = 0) -> None:
        self.state.llm_calls.append(
            LlmCall(
                model=model,
                prompt_tokens=prompt_tokens,
                finish_reason=None,
                latency_ms=None,
                step_index=self.step,
                timestamp=time.time(),
            )
        )
        self._emit(
            EventType.LLM_CALLED,
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
            },
        )

    def llm_responded(
        self,
        completion_tokens: int = 0,
        latency_ms: int = 0,
        finish_reason: str = "stop",
        output_hash: str = "",
        output_length: int = 0,
    ) -> None:
        # Back-fill the most recent LlmCall with response data.
        if self.state.llm_calls:
            lc = self.state.llm_calls[-1]
            lc.finish_reason = finish_reason
            lc.latency_ms = latency_ms
            lc.output_length = output_length
            lc.completion_tokens = completion_tokens or None
        self._emit(
            EventType.LLM_RESPONDED,
            {
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "output_hash": output_hash,
                "output_length": output_length,
            },
            advance=False,
        )

    # ── Tool hooks ────────────────────────────────────────────────────────────

    def tool_called(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> None:
        args_hash = hash_content(str(args or {}))
        self.state.tool_calls.append(
            ToolCall(
                tool_name=tool_name,
                args_hash=args_hash,
                step_index=self.step,
                timestamp=time.time(),
            )
        )
        self._emit(
            EventType.TOOL_CALLED,
            {
                "tool_name": tool_name,
                "args_hash": args_hash,
            },
        )

    def tool_responded(
        self,
        tool_name: str,
        success: bool = True,
        output_length: int = 0,
        latency_ms: int = 0,
        error: Optional[str] = None,
    ) -> None:
        error_hash = hash_content(error) if (not success and error) else None
        # Back-fill success and error_hash on the most recent matching ToolCall
        for tc in reversed(self.state.tool_calls):
            if tc.tool_name == tool_name and tc.success is None:
                tc.success = success
                tc.error_hash = error_hash
                break
        payload: dict = {
            "tool_name": tool_name,
            "success": success,
            "output_length": output_length,
            "latency_ms": latency_ms,
        }
        if error_hash:
            payload["error_hash"] = error_hash
        self._emit(EventType.TOOL_RESPONDED, payload, advance=False)

    # ── Retrieval hooks (RAG) ─────────────────────────────────────────────────

    def retrieval_called(self, index_name: str, query_hash: str = "") -> None:
        self._emit(
            EventType.RETRIEVAL_CALLED,
            {
                "index_name": index_name,
                "query_hash": query_hash,
            },
        )

    def retrieval_responded(
        self,
        index_name: str,
        result_count: int,
        top_score: Optional[float] = None,
        latency_ms: int = 0,
    ) -> None:
        self.state.retrievals.append(
            RetrievalResult(
                index_name=index_name,
                result_count=result_count,
                top_score=top_score,
                step_index=self.step,
            )
        )
        self._emit(
            EventType.RETRIEVAL_RESPONDED,
            {
                "index_name": index_name,
                "result_count": result_count,
                "top_score": top_score,
                "latency_ms": latency_ms,
            },
            advance=False,
        )

    # ── External signal hooks ─────────────────────────────────────────────────

    def external_signal(self, signal_name: str, source: str = "", **meta: Any) -> None:
        """
        Emit an infrastructure context event at the current agent step without advancing the step counter.

        Usage::

            run.external_signal("rate_limit", source="openai")
            run.external_signal("cache_miss", source="redis", key_prefix="emb:")
            run.external_signal("upstream_error", source="serp_api", http_status=503)

        SLOW_STEP and other detectors correlate these signals with failures so evidence reads
        "tool took 100s — coincided with rate_limit from openai" rather than just "tool took 100s".
        """
        ts = time.time()
        self.state.external_signals.append(
            ExternalSignal(
                signal_name=signal_name,
                step_index=self.step,
                timestamp=ts,
                source=source,
                meta=dict(meta),
            )
        )
        payload: dict = {"signal_name": signal_name}
        if source:
            payload["source"] = source
        if meta:
            payload["meta"] = dict(meta)
        # Emit directly — bypass _emit() so step counter does not advance.
        event = AgentEvent(
            event_type=EventType.EXTERNAL_SIGNAL,
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            step_index=self.step,
            timestamp=ts,
            payload=payload,
            parent_run_id=self._parent_run_id,
        )
        self.state.events.append(event)
        self._client._emit(event)

    def pop_prompt_addition(self) -> str | None:
        """Return and remove the oldest injected prompt addition, or None."""
        return self.prompt_additions.pop(0) if self.prompt_additions else None

    def final_answer(self) -> None:
        """Call when the agent produces its final answer."""
        self.exit_reason = "final_answer"
        self.state.exit_reason = "final_answer"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_policies(self) -> None:
        """
        Evaluate registered policies against the current run metrics.

        Called after tool_called, llm_responded, and tool_responded. Raises
        PolicyViolation for 'stop' actions; sets model_override / prompt_additions
        for soft actions; emits POLICY_TRIGGERED for 'log' actions.
        """
        engine = self._client._policy_engine  # type: ignore[attr-defined]
        if not len(engine):
            return

        metrics = build_metrics(self.state, self.step)

        # Augment metrics with signal detection if any policy uses trigger="signal".
        # We do this lazily to avoid running detectors when no signal policies exist.
        with engine._lock:
            needs_signal = any(
                p.condition.get("trigger") == "signal"
                for p in engine._policies
                if p.enabled and p.agent_id in ("*", self.agent_id)
            )
        if needs_signal:
            from dunetrace.detectors import run_detectors

            sigs = run_detectors(self.state)
            metrics["signal"] = [s.failure_type.value for s in sigs] if sigs else []

        result = engine.evaluate(self.agent_id, metrics, self._triggered_policies)
        if result is None:
            return

        policy, action = result
        action_type = action.get("type", "log")
        params = action.get("params") or {}

        # Emit a policy.triggered event for all action types
        self._emit(
            EventType.POLICY_TRIGGERED,
            {
                "policy_name": policy.name,
                "action_type": action_type,
                "trigger": policy.condition.get("trigger"),
                "value": metrics.get(policy.condition.get("trigger", "")),
            },
            advance=False,
        )

        if action_type == "stop":
            self._triggered_policies.add(policy.key)
            raise PolicyViolation(
                policy.name,
                action,
                params.get("message", f"Policy '{policy.name}' stopped the run"),
            )

        elif action_type == "switch_model":
            model = params.get("model")
            if model:
                self.model_override = model
                logger.info("Policy '%s': model_override → %s", policy.name, model)
            self._triggered_policies.add(policy.key)

        elif action_type == "inject_prompt":
            prompt = params.get("prompt", "")
            if prompt:
                self.prompt_additions.append(prompt)
                logger.info("Policy '%s': injected prompt (%d chars)", policy.name, len(prompt))
            self._triggered_policies.add(policy.key)

        elif action_type == "log":
            # Log policies fire every time; don't add to triggered_already
            logger.info(
                "Policy '%s' logged: %s=%s",
                policy.name,
                policy.condition.get("trigger"),
                metrics.get(policy.condition.get("trigger", "")),
            )

    def _emit(self, event_type: EventType, payload: dict, *, advance: bool = True) -> None:
        if advance:
            self.step += 1
        event = AgentEvent(
            event_type=event_type,
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            step_index=self.step,
            payload=payload,
            parent_run_id=self._parent_run_id,
        )
        self.state.events.append(event)
        self._client._emit(event)

        # Evaluate policies after events that change tracked metrics.
        # POLICY_TRIGGERED is excluded to prevent recursive evaluation.
        if event_type in (
            EventType.TOOL_CALLED,
            EventType.LLM_RESPONDED,
            EventType.TOOL_RESPONDED,
        ):
            self._check_policies()
