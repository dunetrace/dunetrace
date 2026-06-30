"""Signals endpoints — list, filter, explain, and apply fixes for detected failure signals."""

from __future__ import annotations
import csv
import io
import json as _json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api_svc.auth import require_customer
from api_svc.config import settings
from api_svc.db.queries import (
    get_signal_by_id,
    list_signals,
    export_signals,
    record_fix,
    get_signal_fix_status,
)
from api_svc.schemas import SignalDetail, SignalListResponse, Page

logger = logging.getLogger("dunetrace.api.signals")

router = APIRouter(tags=["Signals"])

# Detectors where the right fix is a code/infra change, not a system prompt addition.
_CODE_CHANGE_TYPES = frozenset(
    {
        "CONTEXT_BLOAT",
        "RAG_EMPTY_RETRIEVAL",
        "SLOW_STEP",
        "CASCADING_TOOL_FAILURE",
        "LLM_TRUNCATION_LOOP",
        "FIRST_STEP_FAILURE",
        "COST_SPIKE",
        "SESSION_LATENCY",
    }
)

# Detectors that must never have auto-apply enabled — the signal itself indicates
# untrusted input in the trace, so LLM output cannot be safely acted on automatically.
_NO_AUTO_APPLY_TYPES = frozenset({"PROMPT_INJECTION_SIGNAL"})

_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_VALID_FAILURE_TYPES = {
    "TOOL_LOOP",
    "TOOL_THRASHING",
    "TOOL_AVOIDANCE",
    "RETRY_STORM",
    "CASCADING_TOOL_FAILURE",
    "LLM_TRUNCATION_LOOP",
    "CONTEXT_BLOAT",
    "EMPTY_LLM_RESPONSE",
    "GOAL_ABANDONMENT",
    "REASONING_STALL",
    "RAG_EMPTY_RETRIEVAL",
    "SLOW_STEP",
    "FIRST_STEP_FAILURE",
    "STEP_COUNT_INFLATION",
    "PROMPT_INJECTION_SIGNAL",
    "COST_SPIKE",
    "SESSION_LATENCY",
}


@router.get(
    "/v1/agents/{agent_id}/signals",
    response_model=SignalListResponse,
    summary="List failure signals for an agent",
)
async def get_signals(
    agent_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    severity: Optional[str] = Query(
        None, description="Filter by severity: LOW | MEDIUM | HIGH | CRITICAL"
    ),
    failure_type: Optional[str] = Query(None, description="Filter by failure type e.g. TOOL_LOOP"),
    include_shadow: bool = Query(
        False, description="Include shadow signals (stored but not alerted) in results"
    ),
    _customer: str = Depends(require_customer),
) -> SignalListResponse:
    if severity and severity.upper() not in _VALID_SEVERITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid severity {severity!r}. Valid: {sorted(_VALID_SEVERITIES)}",
        )
    if failure_type and failure_type.upper() not in _VALID_FAILURE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid failure_type {failure_type!r}. Valid: {sorted(_VALID_FAILURE_TYPES)}",
        )
    rows, total = await list_signals(
        agent_id, offset, limit, severity, failure_type, include_shadow
    )

    def _ts(v):
        if v is None:
            return None
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    signals = [SignalDetail(**{**r, "detected_at": _ts(r["detected_at"])}) for r in rows]
    return SignalListResponse(
        signals=signals,
        page=Page(total=total, offset=offset, limit=limit, has_more=(offset + limit) < total),
    )


_EXPORT_COLUMNS = [
    "id",
    "failure_type",
    "severity",
    "run_id",
    "agent_id",
    "agent_version",
    "step_index",
    "confidence",
    "detected_at",
    "evidence",
]


@router.get(
    "/v1/agents/{agent_id}/signals/export",
    summary="Export filtered signals as CSV or NDJSON",
    response_class=StreamingResponse,
)
async def export_signals_endpoint(
    agent_id: str,
    format: str = Query("csv", pattern="^(csv|ndjson)$", description="csv or ndjson"),
    severity: Optional[str] = Query(None, description="LOW | MEDIUM | HIGH | CRITICAL"),
    failure_type: Optional[str] = Query(None, description="e.g. TOOL_LOOP"),
    from_: Optional[str] = Query(None, alias="from", description="ISO-8601 start datetime (UTC)"),
    to_: Optional[str] = Query(None, alias="to", description="ISO-8601 end datetime (UTC)"),
    include_shadow: bool = Query(False),
    customer_id: str = Depends(require_customer),
) -> StreamingResponse:
    if severity and severity.upper() not in _VALID_SEVERITIES:
        raise HTTPException(
            422, f"Invalid severity {severity!r}. Valid: {sorted(_VALID_SEVERITIES)}"
        )
    if failure_type and failure_type.upper() not in _VALID_FAILURE_TYPES:
        raise HTTPException(
            422,
            f"Invalid failure_type {failure_type!r}. Valid: {sorted(_VALID_FAILURE_TYPES)}",
        )

    from_ts: Optional[float] = None
    to_ts: Optional[float] = None
    try:
        if from_:
            from datetime import datetime, timezone

            from_ts = (
                datetime.fromisoformat(from_.rstrip("Z")).replace(tzinfo=timezone.utc).timestamp()
            )
        if to_:
            from datetime import datetime, timezone

            to_ts = datetime.fromisoformat(to_.rstrip("Z")).replace(tzinfo=timezone.utc).timestamp()
    except ValueError as exc:
        raise HTTPException(422, f"Invalid datetime: {exc}")

    gen = export_signals(
        agent_id,
        severity=severity,
        failure_type=failure_type,
        include_shadow=include_shadow,
        from_ts=from_ts,
        to_ts=to_ts,
    )

    filename = f"signals-{agent_id}.{format}"

    if format == "ndjson":

        async def _ndjson_stream(batches: AsyncGenerator) -> AsyncGenerator[str, None]:
            async for batch in batches:
                for row in batch:
                    # evidence is already a dict — serialize the whole row in one pass
                    yield _json.dumps(row, separators=(",", ":")) + "\n"

        return StreamingResponse(
            _ndjson_stream(gen),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    async def _csv_stream(batches: AsyncGenerator) -> AsyncGenerator[str, None]:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        yield buf.getvalue()
        async for batch in batches:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=_EXPORT_COLUMNS, extrasaction="ignore")
            for row in batch:
                row["evidence"] = _json.dumps(row["evidence"])
                writer.writerow(row)
            yield buf.getvalue()

    return StreamingResponse(
        _csv_stream(gen),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class ExplainRequest(BaseModel):
    langfuse_trace_id: Optional[str] = None


@router.post(
    "/v1/signals/{signal_id}/explain",
    summary="Explain a signal using Langfuse trace data",
    response_model=Dict[str, Any],
)
async def explain_signal(
    signal_id: int,
    body: ExplainRequest = ExplainRequest(),
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    if not settings.langfuse_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "Langfuse is not configured. "
                "Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to your .env."
            ),
        )
    if not (settings.ANTHROPIC_API_KEY or settings.OPENAI_API_KEY):
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM API key configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env."
            ),
        )

    signal = await get_signal_by_id(signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    from api_svc.langfuse_client import (
        fetch_langfuse_trace,
        build_explain_prompt,
        _detect_langfuse_prompt,
    )
    import time as _time

    trace_lookup_id = body.langfuse_trace_id or signal["run_id"]
    # Only retry on 404 for fresh signals — Langfuse ingests asynchronously so a
    # trace flushed seconds ago may not be queryable yet. For anything older than
    # 5 minutes the trace is either there or not; retrying just adds 18s of latency.
    signal_age_s = _time.time() - (signal.get("detected_at") or 0)
    max_retries = 3 if signal_age_s < 300 else 0

    trace = {}
    source = "signal_only"
    try:
        trace = await fetch_langfuse_trace(trace_lookup_id, max_retries=max_retries)
        source = "langfuse"
    except LookupError:
        logger.info(
            "Trace %s not in Langfuse; falling back to signal-only analysis",
            trace_lookup_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.warning("Langfuse fetch failed for run %s: %s", signal["run_id"], exc)
        # Non-fatal: degrade to signal-only rather than failing the whole request

    failure_type = signal["failure_type"]
    user_prompt = await build_explain_prompt(signal, trace)
    prompt_name, prompt_version = _detect_langfuse_prompt(trace.get("observations", []))

    try:
        llm_result = await _call_llm(user_prompt, failure_type)
    except Exception as exc:
        logger.warning("LLM explain call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Analysis unavailable. Try again.")

    if failure_type in _CODE_CHANGE_TYPES:
        fix_type = "code_change"
    elif failure_type in _NO_AUTO_APPLY_TYPES:
        fix_type = "no_auto_apply"
    else:
        fix_type = "prompt_addition"

    apply_blocked = fix_type != "prompt_addition"

    return {
        "signal_id": signal_id,
        "source": source,
        "root_cause": llm_result["root_cause"],
        "fix_content": llm_result["fix_content"],
        "fix_patch": llm_result["fix_patch"],
        "fix_type": fix_type,
        "apply_blocked": apply_blocked,
        "langfuse_prompt_name": prompt_name,
        "langfuse_prompt_version": prompt_version,
    }


class ApplyFixRequest(BaseModel):
    fix_content: str
    langfuse_prompt_name: str
    applied_via: str = "langfuse"


@router.post(
    "/v1/signals/{signal_id}/apply-fix",
    summary="Apply a fix to the Langfuse-managed system prompt for this signal",
    response_model=Dict[str, Any],
)
async def apply_fix(
    signal_id: int,
    body: ApplyFixRequest,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    if not settings.langfuse_configured:
        raise HTTPException(status_code=503, detail="Langfuse is not configured.")

    signal = await get_signal_by_id(signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    if signal["failure_type"] in _NO_AUTO_APPLY_TYPES:
        raise HTTPException(
            status_code=403,
            detail=(
                "Auto-apply is blocked for PROMPT_INJECTION_SIGNAL. "
                "The trace contains untrusted input — review the fix manually "
                "before applying it to your prompt."
            ),
        )

    from api_svc.langfuse_client import apply_langfuse_fix

    try:
        result = await apply_langfuse_fix(body.langfuse_prompt_name, body.fix_content)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.warning("apply_langfuse_fix failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not apply fix via Langfuse.")

    fix_id = None
    try:
        fix_id = await record_fix(
            signal_id=signal_id,
            run_id=signal["run_id"],
            fix_content=body.fix_content,
            applied_via=body.applied_via,
            langfuse_prompt_name=body.langfuse_prompt_name,
            langfuse_version=result.get("new_version"),
        )
    except Exception as exc:
        logger.warning("record_fix failed (non-fatal): %s", exc)

    return {
        "fix_id": fix_id,
        "signal_id": signal_id,
        "new_version": result["new_version"],
        "prompt_url": result["prompt_url"],
        "old_text": result["old_text"],
        "new_text": result["new_text"],
    }


@router.post(
    "/v1/signals/{signal_id}/record-copy",
    summary="Record that the fix was copied to clipboard (clipboard path)",
    response_model=Dict[str, Any],
    include_in_schema=False,
)
async def record_copy(
    signal_id: int,
    body: ApplyFixRequest,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    """Thin endpoint so clipboard-path fixes are also tracked in the fixes table."""
    signal = await get_signal_by_id(signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    fix_id = None
    try:
        fix_id = await record_fix(
            signal_id=signal_id,
            run_id=signal["run_id"],
            fix_content=body.fix_content,
            applied_via="clipboard",
        )
    except Exception as exc:
        logger.warning("record_fix (clipboard) failed: %s", exc)

    return {"fix_id": fix_id, "signal_id": signal_id}


class OpenPRRequest(BaseModel):
    root_cause: str
    fix_content: str
    fix_patch: str


@router.post(
    "/v1/signals/{signal_id}/open-pr",
    summary="Create a GitHub draft PR with the code-change fix suggestion",
    response_model=Dict[str, Any],
)
async def open_pr(
    signal_id: int,
    body: OpenPRRequest,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    if not settings.github_configured:
        raise HTTPException(
            status_code=503,
            detail="GitHub not configured. Add GITHUB_TOKEN and GITHUB_REPO to .env.",
        )

    signal = await get_signal_by_id(signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    from api_svc.github_client import create_fix_pr

    try:
        result = await create_fix_pr(
            signal_id=signal_id,
            agent_id=signal["agent_id"],
            failure_type=signal["failure_type"],
            root_cause=body.root_cause,
            fix_content=body.fix_content,
            fix_patch=body.fix_patch,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.warning("create_fix_pr failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Could not create GitHub PR. Check GITHUB_TOKEN and GITHUB_REPO.",
        )

    try:
        await record_fix(
            signal_id=signal_id,
            run_id=signal["run_id"],
            fix_content=body.fix_content,
            applied_via="github_pr",
            langfuse_version=result.get("pr_number"),
        )
    except Exception as exc:
        logger.warning("record_fix (github_pr) failed (non-fatal): %s", exc)

    return result


@router.get(
    "/v1/signals/{signal_id}/fix-status",
    summary="Check whether a previously applied fix has reduced recurrence",
    response_model=Dict[str, Any],
)
async def fix_status(
    signal_id: int,
    _customer: str = Depends(require_customer),
) -> Dict[str, Any]:
    signal = await get_signal_by_id(signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    try:
        status = await get_signal_fix_status(
            agent_id=signal["agent_id"],
            failure_type=signal["failure_type"],
            signal_id=signal_id,
        )
    except Exception as exc:
        logger.warning("get_signal_fix_status failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not retrieve fix status.")

    return status


async def _call_llm(user_prompt: str, failure_type: str = "") -> Dict[str, str]:
    if failure_type in _CODE_CHANGE_TYPES:
        fix_instruction = (
            "fix_content: one sentence (under 120 chars) describing the specific "
            "code or infrastructure change needed.\n"
            "fix_patch: a unified diff (max 30 lines) showing the exact change. "
            "Use '--- a/path/to/file.py' and '+++ b/path/to/file.py' headers — "
            "infer the file path from the tool name or error pattern in the trace; "
            "if unknown use '--- a/agent.py'. Include @@ line numbers. "
            "Lines starting with '-' are removed, '+' added, ' ' are context."
        )
    elif failure_type in _NO_AUTO_APPLY_TYPES:
        fix_instruction = (
            "fix_content: one sentence (under 120 chars) describing a defensive guard to add.\n"
            "fix_patch: a concrete code snippet (max 10 lines) implementing the "
            "guard — e.g. an input validation function or a regex check on user input."
        )
    else:
        fix_instruction = (
            "fix_content: one sentence (under 100 chars) to add verbatim to the agent's system prompt.\n"
            "fix_patch: a short unified diff (5-8 lines) showing where to insert "
            "fix_content in the system prompt. Use the actual system prompt text shown above. "
            'Format: lines starting with " " are context, "+" is the new line. '
            "If the system prompt was not found, just show the single + line."
        )

    system = (
        "You are analyzing an AI agent behavioral failure. "
        "Given a detection signal and the agent's execution trace, identify the "
        "specific root cause and the single most important fix.\n"
        "Respond ONLY in JSON with exactly three fields:\n"
        '{"root_cause": "...", "fix_content": "...", "fix_patch": "..."}\n'
        "root_cause: max 120 words explaining specifically WHY this happened, "
        "quoting relevant prompt text or tool output.\n"
        f"{fix_instruction}"
    )

    raw = ""
    if settings.ANTHROPIC_API_KEY:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package required: pip install anthropic") from exc

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=900,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text
    else:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=900,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content

    # Strip markdown code fences (model sometimes wraps JSON in ```json...```)
    raw_clean = raw.strip()
    if raw_clean.startswith("```"):
        lines = raw_clean.splitlines()
        raw_clean = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else raw_clean

    try:
        parsed = _json.loads(raw_clean)
        if isinstance(parsed, dict) and "root_cause" in parsed:
            return {
                "root_cause": str(parsed.get("root_cause", "")),
                "fix_content": str(parsed.get("fix_content", "")),
                "fix_patch": str(parsed.get("fix_patch", "")),
            }
    except (_json.JSONDecodeError, TypeError):
        pass

    return {"root_cause": raw, "fix_content": "", "fix_patch": ""}
