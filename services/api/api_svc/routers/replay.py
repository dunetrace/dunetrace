"""
Replay a flagged run with modified events to simulate counterfactual outcomes.

No LLM calls — purely deterministic: take stored events, mutate them, re-run
the detector pipeline in-memory, diff the results against the original signals.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_svc.auth import require_org

logger = logging.getLogger("dunetrace.api.replay")
router = APIRouter(tags=["Replay"])

_VALID_MODS = frozenset(
    {
        "break_tool_loop",
        "fix_truncation",
        "fix_tool_failures",
        "reduce_context",
        "fix_rag_retrieval",
        "add_final_answer",
        "truncate_at_step",
    }
)

# Which signals each modification is designed to resolve.
MOD_TARGETS: dict[str, list[str]] = {
    "break_tool_loop": ["TOOL_LOOP", "TOOL_THRASHING"],
    "fix_truncation": ["LLM_TRUNCATION_LOOP"],
    "fix_tool_failures": ["RETRY_STORM", "CASCADING_TOOL_FAILURE"],
    "reduce_context": ["CONTEXT_BLOAT"],
    "fix_rag_retrieval": ["RAG_EMPTY_RETRIEVAL"],
    "add_final_answer": ["GOAL_ABANDONMENT"],
    "truncate_at_step": [],
}


# ── Event mutations ───────────────────────────────────────────────────────────


def _apply_modifications(events: list[dict], mods: list) -> list[dict]:
    events = copy.deepcopy(events)

    for mod in mods:
        if isinstance(mod, str):
            mod_type = mod
            mod_params: dict = {}
        elif isinstance(mod, dict):
            mod_type = mod.get("type", "")
            mod_params = mod
        else:
            continue

        if mod_type == "break_tool_loop":
            # Simulate the agent not looping: keep at most THRESHOLD-1 calls of any
            # one tool. TOOL_LOOP counts by tool_name (not by args), so an
            # arg-varying loop has to be thinned by name — deduping (name, args)
            # left arg-varying loops firing. Drop each surplus tool.called and its
            # paired tool.responded (same step) so build_run_state sees no orphans.
            from dunetrace.detectors import ToolLoopDetector  # noqa: PLC0415

            cap = max(1, ToolLoopDetector.THRESHOLD - 1)
            per_tool: dict[str, int] = {}
            dropped_steps: set[int] = set()
            filtered: list[dict] = []
            for e in events:
                if e["event_type"] == "tool.called":
                    name = (e.get("payload") or {}).get("tool_name", "")
                    per_tool[name] = per_tool.get(name, 0) + 1
                    if per_tool[name] > cap:
                        dropped_steps.add(e.get("step_index", -1))
                        continue
                elif e["event_type"] == "tool.responded":
                    if e.get("step_index", -1) in dropped_steps:
                        continue
                filtered.append(e)
            events = filtered

        elif mod_type == "fix_truncation":
            for e in events:
                if e["event_type"] == "llm.responded":
                    p = e.get("payload") or {}
                    if p.get("finish_reason") == "length":
                        p["finish_reason"] = "stop"

        elif mod_type == "fix_tool_failures":
            for e in events:
                if e["event_type"] == "tool.responded":
                    p = e.get("payload") or {}
                    if p.get("success") is False:
                        p["success"] = True
                        p.pop("error", None)

        elif mod_type == "reduce_context":
            # Simulate effective context compression by capping prompt-token GROWTH,
            # not by uniformly scaling every call. CONTEXT_BLOAT fires on the
            # last/first ratio, which is scale-invariant — multiplying every call by
            # the same factor (the old behavior) never changed the ratio, so the
            # signal never resolved. Cap each later call so growth stays under the
            # detector's threshold.
            from dunetrace.detectors import ContextBloatDetector  # noqa: PLC0415

            factor = max(0.1, min(1.0, float(mod_params.get("factor", 0.5))))
            # Allowed growth headroom scaled by factor: factor=0.5 -> ~2x (< 3x).
            cap_ratio = 1.0 + (ContextBloatDetector.GROWTH_FACTOR - 1.0) * factor
            # prompt_tokens can live on llm.called (direct SDK) or llm.responded
            # (e.g. LangChain, where llm.called carries 0). build_run_state uses
            # whichever is non-zero, so cap both to actually flatten the effective
            # growth the detector reads.
            first_pt: int | None = None
            for e in events:
                if e["event_type"] in ("llm.called", "llm.responded"):
                    p = e.get("payload") or {}
                    pt = p.get("prompt_tokens")
                    if not pt:
                        continue
                    if first_pt is None:
                        first_pt = pt
                        continue
                    cap = max(1, int(first_pt * cap_ratio))
                    if pt > cap:
                        p["prompt_tokens"] = cap

        elif mod_type == "fix_rag_retrieval":
            for e in events:
                if e["event_type"] == "retrieval.responded":
                    p = e.get("payload") or {}
                    if (p.get("result_count") or 0) == 0:
                        p["result_count"] = 5
                        if p.get("top_score") is None:
                            p["top_score"] = 0.85

        elif mod_type == "add_final_answer":
            has_terminal = any(e["event_type"] in ("run.completed", "run.errored") for e in events)
            if not has_terminal and events:
                last = events[-1]
                events.append(
                    {
                        "event_type": "run.completed",
                        "run_id": last.get("run_id", ""),
                        "agent_id": last.get("agent_id", ""),
                        "agent_version": last.get("agent_version", ""),
                        "step_index": last.get("step_index", 0) + 1,
                        "timestamp": (last.get("timestamp") or 0.0) + 0.001,
                        "payload": {"exit_reason": "completed"},
                        "parent_run_id": None,
                    }
                )

        elif mod_type == "truncate_at_step":
            step = int(mod_params.get("step", 0))
            events = [e for e in events if (e.get("step_index") or 0) <= step]
            if events:
                last = events[-1]
                events.append(
                    {
                        "event_type": "run.completed",
                        "run_id": last.get("run_id", ""),
                        "agent_id": last.get("agent_id", ""),
                        "agent_version": last.get("agent_version", ""),
                        "step_index": step + 1,
                        "timestamp": (last.get("timestamp") or 0.0) + 0.001,
                        "payload": {"exit_reason": "completed"},
                        "parent_run_id": None,
                    }
                )

    return events


# ── DB helpers ────────────────────────────────────────────────────────────────


async def _fetch_events(org_id: str, run_id: str) -> list[dict]:
    from api_svc.db import queries as _q  # noqa: PLC0415

    pool = _q._pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_type, run_id, agent_id, agent_version,
                   step_index, timestamp, payload, parent_run_id
            FROM events
            WHERE run_id = $1 AND org_id = $2
            ORDER BY step_index ASC, timestamp ASC
            """,
            run_id,
            org_id,
        )
    return [
        {
            **dict(r),
            "payload": (
                json.loads(r["payload"])
                if isinstance(r["payload"], str)
                else dict(r["payload"] or {})
            ),
        }
        for r in rows
    ]


async def _fetch_signals(org_id: str, run_id: str) -> list[dict]:
    from api_svc.db import queries as _q  # noqa: PLC0415

    pool = _q._pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT failure_type, severity, step_index, confidence
            FROM failure_signals
            WHERE run_id = $1 AND org_id = $2 AND shadow = FALSE
            ORDER BY step_index ASC
            """,
            run_id,
            org_id,
        )
    return [dict(r) for r in rows]


# ── Endpoint ──────────────────────────────────────────────────────────────────


class ReplayRequest(BaseModel):
    modifications: List[Union[str, Dict[str, Any]]] = []


@router.post(
    "/v1/runs/{run_id}/replay",
    summary="Simulate run replay with modified events; shows which signals resolve",
    response_model=Dict[str, Any],
)
async def replay_run(
    run_id: str,
    body: ReplayRequest,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    for mod in body.modifications:
        mod_type = (
            mod if isinstance(mod, str) else (mod.get("type", "") if isinstance(mod, dict) else "")
        )
        if mod_type not in _VALID_MODS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown modification {mod_type!r}. Valid: {sorted(_VALID_MODS)}",
            )

    events = await _fetch_events(org_id, run_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found or has no events.")

    original_signals = await _fetch_signals(org_id, run_id)
    modified_events = _apply_modifications(events, body.modifications)

    try:
        from api_svc.run_builder import build_run_state  # noqa: PLC0415
        from dunetrace.detectors import run_detectors  # noqa: PLC0415

        state = build_run_state(modified_events)
        replay_sigs = run_detectors(state)

        # Preserve PROMPT_INJECTION_SIGNAL if the modified events still have injection evidence
        has_injection = any(
            (e.get("payload") or {}).get("injection_signal")
            for e in modified_events
            if e["event_type"] == "run.started"
        )
    except Exception as exc:
        logger.warning("Replay detection failed for run_id=%s: %s", run_id, exc)
        raise HTTPException(status_code=502, detail="Replay simulation failed.")

    replay_signal_list = [
        {
            "failure_type": s.failure_type.value,
            "severity": s.severity.value,
            "confidence": round(s.confidence, 4),
            "step_index": s.step_index,
        }
        for s in replay_sigs
    ]

    if has_injection and not any(
        s["failure_type"] == "PROMPT_INJECTION_SIGNAL" for s in replay_signal_list
    ):
        orig_inj = next(
            (s for s in original_signals if s["failure_type"] == "PROMPT_INJECTION_SIGNAL"), None
        )
        if orig_inj:
            replay_signal_list.append(dict(orig_inj))

    orig_types = {s["failure_type"] for s in original_signals}
    replay_types = {s["failure_type"] for s in replay_signal_list}

    return {
        "run_id": run_id,
        "original_signals": original_signals,
        "replay_signals": replay_signal_list,
        "resolved": sorted(orig_types - replay_types),
        "new": sorted(replay_types - orig_types),
        "unchanged": sorted(orig_types & replay_types),
    }
