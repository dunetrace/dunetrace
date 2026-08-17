"""
Reconstructs an EvaluationInput from raw event dicts — the same flat DB rows
detector_svc.db.fetch_run_events returns, read independently here since
semantic_svc has no dependency on detector_svc.
"""

from __future__ import annotations

from semantic_svc.evaluators.base import EvaluationInput, ToolCallData


def _as_input_parameters(args) -> dict | None:
    if isinstance(args, dict):
        return args
    if args:
        return {"raw": str(args)}
    return None


def build_evaluation_input(events: list[dict]) -> EvaluationInput | None:
    """Returns None when there isn't enough to evaluate — no input_text, or no
    output text at all (an evaluator has nothing meaningful to score against).
    """
    input_text: str | None = None
    actual_output: str | None = None
    context: list[str] = []
    tools_called: list[ToolCallData] = []

    for e in events:
        event_type = e.get("event_type")
        payload = e.get("payload") or {}

        if event_type == "run.started":
            input_text = payload.get("input_text") or input_text

        elif event_type in ("llm.responded", "tool.responded"):
            output = payload.get("output")
            if output:
                actual_output = output
            if event_type == "tool.responded":
                # Matches by tool_name against the most recent unmatched call —
                # same LIFO convention as detector_svc.run_builder, and the
                # same limitation: two calls to the same tool fired back-to-back
                # before either responds would pair up in reverse order. Correct
                # for the common call-then-respond-before-next-call pattern.
                tool_name = payload.get("tool_name", "")
                for i in range(len(tools_called) - 1, -1, -1):
                    if tools_called[i].name == tool_name and tools_called[i].output is None:
                        tools_called[i] = ToolCallData(
                            name=tool_name,
                            input_parameters=tools_called[i].input_parameters,
                            output=output,
                        )
                        break

        elif event_type == "tool.called":
            tools_called.append(
                ToolCallData(
                    name=payload.get("tool_name", "unknown"),
                    input_parameters=_as_input_parameters(payload.get("args")),
                )
            )

        elif event_type == "retrieval.responded":
            content = payload.get("content")
            if content:
                context.append(content)

    if not input_text or not actual_output:
        return None

    return EvaluationInput(
        input_text=input_text,
        actual_output=actual_output,
        context=context,
        tools_called=tools_called,
    )
