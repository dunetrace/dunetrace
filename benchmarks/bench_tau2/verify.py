"""Lossless check: rebuild each converted run the way the detector does and
compare it field by field with the trajectory.

`build_run_state` is the single production reconstruction (detector worker and
API both use it), so a field that survives it is a field the detectors see.
The comparison allows exactly one transformation, the SDK's own field cap
(`DEFAULT_MAX_FIELD_CHARS`): a capped field is expected to equal the
trajectory's text cut at the cap, and is counted rather than hidden.
"""

from __future__ import annotations

import json
from typing import List

from bench_tau2._paths import install_repo_paths
from bench_tau2.convert import (
    ConvertedRun,
    FileContext,
    _raw_choice,
    _reasoning_tokens,
    pair_tool_calls,
    parse_ts,
)

install_repo_paths()

from dunetrace.redaction import DEFAULT_MAX_FIELD_CHARS  # noqa: E402
from dunetrace.run_builder import build_run_state  # noqa: E402


def _capped(text: str) -> str:
    return (text or "")[:DEFAULT_MAX_FIELD_CHARS]


def verify_run(run: ConvertedRun, ctx: FileContext, sim: dict) -> List[str]:
    """Mismatches between the rebuilt RunState and the trajectory. Empty means lossless."""
    messages = sim["messages"]
    all_calls, by_result = pair_tool_calls(messages)
    state = build_run_state(run.events)
    problems: List[str] = []

    def check(what: str, got, want) -> None:
        if got != want:
            problems.append(f"{what}: got {got!r:.120} want {want!r:.120}")

    user_text = "\n".join(
        messages[i]["content"]
        for i in run.user_indices
        if messages[i].get("role") == "user" and (messages[i].get("content") or "").strip()
    )
    check("input_text", state.input_text, _capped(user_text))
    check("system_prompt", state.system_prompt, _capped(ctx.system_prompt) or None)
    check("available_tools", state.available_tools, ctx.tools)

    assistants = [messages[i] for i in run.agent_indices if messages[i].get("role") == "assistant"]
    check("llm call count", len(state.llm_calls), len(assistants))
    for n, (call, message) in enumerate(zip(state.llm_calls, assistants)):
        usage = message.get("usage") or {}
        check(f"llm[{n}].model", call.model, ctx.model)
        check(f"llm[{n}].output_text", call.output_text, _capped(message.get("content")) or None)
        check(f"llm[{n}].output_length", call.output_length, len(message.get("content") or ""))
        check(f"llm[{n}].prompt_tokens", call.prompt_tokens, int(usage.get("prompt_tokens") or 0))
        check(
            f"llm[{n}].completion_tokens",
            call.completion_tokens,
            int(usage.get("completion_tokens") or 0),
        )
        check(f"llm[{n}].reasoning_tokens", call.reasoning_tokens or 0, _reasoning_tokens(message))
        check(
            f"llm[{n}].finish_reason", call.finish_reason, _raw_choice(message).get("finish_reason")
        )

    agent_set = set(run.agent_indices)
    expected_tools = []
    for index in run.agent_indices:
        call = by_result.get(index)
        if messages[index].get("role") == "tool" and call is not None and call.owner == "agent":
            expected_tools.append((call, messages[index]))
    answered = {id(c) for c, _ in expected_tools}
    unanswered = [
        c
        for c in all_calls
        if c.owner == "agent" and c.message_index in agent_set and id(c) not in answered
    ]
    check("tool call count", len(state.tool_calls), len(expected_tools) + len(unanswered))
    for n, (tc, (call, message)) in enumerate(zip(state.tool_calls, expected_tools)):
        content = message.get("content") or ""
        failed = bool(message.get("error"))
        check(f"tool[{n}].name", tc.tool_name, call.name)
        if call.raw_arguments is not None:
            check(f"tool[{n}].args", tc.args, _capped(call.raw_arguments))
        elif tc.args_length is None:
            check(f"tool[{n}].args(json)", json.loads(tc.args), call.arguments)
        check(f"tool[{n}].success", tc.success, not failed)
        check(f"tool[{n}].error", tc.error, (content or "error") if failed else None)
        check(f"tool[{n}].output", tc.output, _capped(content) or None)
    for n, tc in enumerate(state.tool_calls[len(expected_tools) :]):
        check(f"unanswered tool[{n}].success", tc.success, None)

    final = run.turn_index == run.n_turns - 1
    want_exit = (
        "error"
        if final and run.termination_reason not in ("user_stop", "agent_stop")
        else (
            "final_answer"
            if messages[run.agent_indices[-1]].get("role") == "assistant"
            else "completed"
        )
    )
    check("exit_reason", state.exit_reason, want_exit)

    stamps = [e["timestamp"] for e in run.events]
    check("timestamps non-decreasing", stamps, sorted(stamps))
    want_start = parse_ts(messages[run.user_indices[-1]].get("timestamp"))
    if want_start is not None and not run.stats.get("synthesized_timestamps"):
        check("run.started timestamp", run.events[0]["timestamp"], want_start)
    return problems
