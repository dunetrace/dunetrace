"""Signals endpoints — list, filter, and explain detected failure signals."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from api_svc.auth import require_customer
from api_svc.config import settings
from api_svc.db.queries import get_signal_by_id, list_signals
from api_svc.schemas import SignalDetail, SignalListResponse, Page

logger = logging.getLogger("dunetrace.api.signals")

router = APIRouter(tags=["Signals"])

_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_VALID_FAILURE_TYPES = {
    "TOOL_LOOP", "TOOL_THRASHING", "TOOL_AVOIDANCE", "RETRY_STORM",
    "CASCADING_TOOL_FAILURE", "LLM_TRUNCATION_LOOP", "CONTEXT_BLOAT",
    "EMPTY_LLM_RESPONSE", "GOAL_ABANDONMENT", "REASONING_STALL",
    "RAG_EMPTY_RETRIEVAL", "SLOW_STEP", "FIRST_STEP_FAILURE",
    "STEP_COUNT_INFLATION", "PROMPT_INJECTION_SIGNAL",
}


@router.get(
    "/v1/agents/{agent_id}/signals",
    response_model=SignalListResponse,
    summary="List failure signals for an agent",
)
async def get_signals(
    agent_id:       str,
    offset:         int           = Query(0, ge=0),
    limit:          int           = Query(settings.PAGE_SIZE_DEFAULT, ge=1,
                                          le=settings.PAGE_SIZE_MAX),
    severity:       Optional[str] = Query(None,
        description="Filter by severity: LOW | MEDIUM | HIGH | CRITICAL"),
    failure_type:   Optional[str] = Query(None,
        description="Filter by failure type e.g. TOOL_LOOP"),
    include_shadow: bool          = Query(False,
        description="Include shadow signals (stored but not alerted) in results"),
    _customer:      str           = Depends(require_customer),
) -> SignalListResponse:
    if severity and severity.upper() not in _VALID_SEVERITIES:
        raise HTTPException(status_code=422,
            detail=f"Invalid severity {severity!r}. Valid: {sorted(_VALID_SEVERITIES)}")
    if failure_type and failure_type.upper() not in _VALID_FAILURE_TYPES:
        raise HTTPException(status_code=422,
            detail=f"Invalid failure_type {failure_type!r}. Valid: {sorted(_VALID_FAILURE_TYPES)}")
    rows, total = await list_signals(agent_id, offset, limit, severity, failure_type, include_shadow)

    def _ts(v):
        if v is None:
            return None
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    signals = [SignalDetail(**{**r, "detected_at": _ts(r["detected_at"])}) for r in rows]
    return SignalListResponse(
        signals=signals,
        page=Page(total=total, offset=offset, limit=limit,
                  has_more=(offset + limit) < total),
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
                "No LLM API key configured. "
                "Add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env."
            ),
        )

    signal = await get_signal_by_id(signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    from api_svc.langfuse_client import fetch_langfuse_trace, build_explain_prompt
    trace_lookup_id = body.langfuse_trace_id or signal["run_id"]
    try:
        trace = await fetch_langfuse_trace(trace_lookup_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.warning("Langfuse fetch failed for run %s: %s", signal["run_id"], exc)
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Langfuse. Check your API key and host.",
        )

    user_prompt = await build_explain_prompt(signal, trace)

    try:
        explanation = await _call_llm(user_prompt)
    except Exception as exc:
        logger.warning("LLM explain call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Analysis unavailable. Try again.")

    return {"explanation": explanation, "source": "langfuse", "signal_id": signal_id}


async def _call_llm(user_prompt: str) -> str:
    system = (
        "You are analyzing an AI agent behavioral failure. "
        "Given a detection signal and the agent's execution trace, identify the "
        "specific root cause and suggest one concrete fix. "
        "Be direct and specific. Max 150 words."
    )

    if settings.ANTHROPIC_API_KEY:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package required: pip install anthropic") from exc

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    # OpenAI fallback
    try:
        import openai
    except ImportError as exc:
        raise ImportError("openai package required: pip install openai") from exc

    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
