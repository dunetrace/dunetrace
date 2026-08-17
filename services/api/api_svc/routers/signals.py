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
from api_svc.auth import require_org
from api_svc import llm_provider
from api_svc.config import settings
from api_svc.failure_types import (
    invalid_failure_type_detail,
    is_valid_failure_type,
)
from api_svc.db.queries import (
    get_signal_by_id,
    list_signals,
    export_signals,
    record_fix,
    get_signal_fix_status,
    get_run_detail,
    get_organization_semantic_feedback,
    record_signal_feedback,
    get_org_github_integration,
)
from api_svc.fix_classification import build_suggested_policy, classify_fix
from api_svc.schemas import SignalDetail, SignalListResponse, Page

logger = logging.getLogger("dunetrace.api.signals")

router = APIRouter(tags=["Signals"])

# Detectors where the right fix is a code/infra change, not a system prompt addition.
# Applies only to customer_code signals — CASCADING_TOOL_FAILURE was here before
# fix_classification.classify_fix() existed; it's now dunetrace_native (a policy
# fixes it directly), so this set no longer needs to cover it.
_CODE_CHANGE_TYPES = frozenset(
    {
        "CONTEXT_BLOAT",
        "RAG_EMPTY_RETRIEVAL",
        "SLOW_STEP",
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
# Validation lives in api_svc.failure_types — a hand-maintained literal here
# froze at 17 entries and 422'd types that exist in the data. See that module.


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
    org_id: str = Depends(require_org),
) -> SignalListResponse:
    if severity and severity.upper() not in _VALID_SEVERITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid severity {severity!r}. Valid: {sorted(_VALID_SEVERITIES)}",
        )
    if failure_type and not is_valid_failure_type(failure_type):
        raise HTTPException(
            status_code=422,
            detail=invalid_failure_type_detail(failure_type),
        )
    rows, total = await list_signals(
        org_id, agent_id, offset, limit, severity, failure_type, include_shadow
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
    org_id: str = Depends(require_org),
) -> StreamingResponse:
    if severity and severity.upper() not in _VALID_SEVERITIES:
        raise HTTPException(
            422, f"Invalid severity {severity!r}. Valid: {sorted(_VALID_SEVERITIES)}"
        )
    if failure_type and not is_valid_failure_type(failure_type):
        raise HTTPException(
            422,
            invalid_failure_type_detail(failure_type),
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
        org_id,
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


@router.post(
    "/v1/signals/{signal_id}/explain",
    summary="Explain a signal using Dunetrace's own native event data",
    response_model=Dict[str, Any],
)
async def explain_signal(
    signal_id: int,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    """Root-cause analysis is fully native — Dunetrace analyzes its own stored
    events (including the system prompt, when the caller passes one to
    dt.run()) and always produces a root cause and a fix. No external tracing
    system is consulted at all.
    """
    if not llm_provider.llm_configured():
        raise HTTPException(
            status_code=503,
            detail=(llm_provider.missing_key_message()),
        )

    signal = await get_signal_by_id(org_id, signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    from api_svc.native_explain import build_native_explain_prompt

    run_detail = await get_run_detail(org_id, signal["run_id"], include_shadow=True)
    events = run_detail["events"] if run_detail else []
    source = "native" if events else "signal_only"
    user_prompt = await build_native_explain_prompt(signal, events)

    failure_type = signal["failure_type"]

    try:
        llm_result = await _call_llm(user_prompt, failure_type)
    except Exception as exc:
        logger.warning("LLM explain call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Analysis unavailable. Try again.")

    fix_category = classify_fix(signal)

    if fix_category == "dunetrace_native":
        suggested_policy = build_suggested_policy(signal)
        return {
            "signal_id": signal_id,
            "source": source,
            "root_cause": llm_result["root_cause"],
            "fix_category": fix_category,
            "suggested_policy": suggested_policy,
            "fix_type": "policy",
            "apply_blocked": False,  # a Policy is Dunetrace's own config — always directly applicable
        }

    if failure_type in _CODE_CHANGE_TYPES:
        fix_type = "code_change"
    elif failure_type in _NO_AUTO_APPLY_TYPES:
        fix_type = "no_auto_apply"
    else:
        fix_type = "prompt_addition"

    if fix_type == "code_change":
        # code_change has a real one-click path — POST /v1/signals/{id}/open-pr —
        # gated on GitHub being configured, either per-org (Phase 4.3's
        # GitHub App) or the legacy global PAT fallback.
        github_integration = await get_org_github_integration(org_id)
        apply_blocked = not (
            (github_integration and github_integration.get("repos")) or settings.github_configured
        )
    elif fix_type == "no_auto_apply":
        apply_blocked = True  # security signal — never auto-apply, review manually
    else:
        # prompt_addition touches the customer's system prompt — Dunetrace has
        # no write access to wherever that actually lives (the customer's own
        # code), so this is always a diff to copy in manually.
        apply_blocked = True

    return {
        "signal_id": signal_id,
        "source": source,
        "root_cause": llm_result["root_cause"],
        "fix_category": fix_category,
        "fix_content": llm_result["fix_content"],
        "fix_patch": llm_result["fix_patch"],
        "fix_type": fix_type,
        "apply_blocked": apply_blocked,
    }


class RecordCopyRequest(BaseModel):
    fix_content: str


@router.post(
    "/v1/signals/{signal_id}/record-copy",
    summary="Record that the fix was copied to clipboard (clipboard path)",
    response_model=Dict[str, Any],
    include_in_schema=False,
)
async def record_copy(
    signal_id: int,
    body: RecordCopyRequest,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    """Thin endpoint so clipboard-path fixes are also tracked in the fixes table."""
    signal = await get_signal_by_id(org_id, signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    fix_id = None
    try:
        fix_id = await record_fix(
            org_id=org_id,
            signal_id=signal_id,
            run_id=signal["run_id"],
            fix_content=body.fix_content,
            applied_via="clipboard",
        )
    except Exception as exc:
        logger.warning("record_fix (clipboard) failed: %s", exc)

    return {"fix_id": fix_id, "signal_id": signal_id}


_VALID_FEEDBACK_VERDICTS = frozenset({"false_positive"})


class SignalFeedbackRequest(BaseModel):
    verdict: str
    notes: Optional[str] = None


@router.post(
    "/v1/signals/{signal_id}/feedback",
    summary='Record feedback on a semantic signal (e.g. "not a real issue")',
    response_model=Dict[str, Any],
    status_code=201,
)
async def submit_signal_feedback(
    signal_id: int,
    body: SignalFeedbackRequest,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    """Phase 1.4.3. Only applies to semantic signals — structural detector
    false-positives already have their own mechanism (the Slack "Mark false
    positive" button -> agent_detector_overrides), which this does not
    replace or touch. Opt-in per org: the org must have enabled the feedback
    loop (POST/GET /v1/orgs/semantic-feedback) before this does anything.
    """
    if body.verdict not in _VALID_FEEDBACK_VERDICTS:
        raise HTTPException(
            status_code=422,
            detail=f"verdict must be one of {sorted(_VALID_FEEDBACK_VERDICTS)}.",
        )

    signal = await get_signal_by_id(org_id, signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    if signal["source"] != "semantic":
        raise HTTPException(
            status_code=400,
            detail=(
                "Feedback is only supported for semantic signals. Structural "
                "detector false positives are marked via the Slack integration."
            ),
        )

    org_settings = await get_organization_semantic_feedback(org_id)
    if not org_settings or not org_settings["enabled"]:
        raise HTTPException(
            status_code=403,
            detail=(
                "The semantic feedback loop is not enabled for this org. "
                "Enable it via PATCH /v1/orgs/semantic-feedback."
            ),
        )

    feedback_id = await record_signal_feedback(signal_id, org_id, body.verdict, body.notes)
    return {"feedback_id": feedback_id, "signal_id": signal_id, "verdict": body.verdict}


class OpenPRRequest(BaseModel):
    root_cause: str
    fix_content: str
    fix_patch: str


async def _resolve_github_auth(org_id: str) -> Optional[Dict[str, Any]]:
    """Per-org GitHub App installation first (Phase 4.3), else the legacy
    global PAT (pre-4.3 self-hosted single-tenant path) — same
    org-config-first-then-global-fallback precedent Phase 4.1 established
    for Slack. Returns None if neither is configured."""
    from api_svc.github_app_auth import get_installation_token

    github_integration = await get_org_github_integration(org_id)
    if github_integration and github_integration.get("repos"):
        token = await get_installation_token(github_integration["installation_id"])
        return {
            "token": token,
            "repos": github_integration["repos"],
            "reviewers": github_integration.get("reviewers") or [],
        }

    if settings.github_configured:
        return {
            "token": settings.GITHUB_TOKEN,
            "repos": [{"repo": settings.GITHUB_REPO, "base_branch": settings.GITHUB_BASE_BRANCH}],
            "reviewers": [],
        }

    return None


async def _attempt_real_diff(
    org_id: str,
    agent_id: str,
    token: str,
    repos: list,
    root_cause: str,
    fix_content: str,
) -> Optional[Dict[str, Any]]:
    """Returns {"repo", "file_path", "base_branch", "new_content"} if source
    mapping resolves a file AND the LLM produces a usable corrected version
    AND it passes the security guardrail — else None, meaning "fall back to
    the safe markdown-summary-file PR." Never raises — any failure along
    this path degrades to that fallback, logged but not surfaced as an error
    to the caller (opening *some* PR still succeeds)."""
    from api_svc.source_resolution import resolve_source
    from api_svc.github_client import fetch_file_content
    from api_svc.diff_generation import generate_real_file_content
    from api_svc.fix_security import validate_target_path

    try:
        resolved = await resolve_source(org_id, agent_id)
        if not resolved:
            return None

        repo_cfg = next((r for r in repos if r.get("repo") == resolved["repo"]), None)
        if repo_cfg is None:
            # Source mapping resolved a repo this org hasn't connected the
            # GitHub App to — can't write there regardless of how confident
            # the resolution was.
            return None
        base_branch = repo_cfg.get("base_branch") or "main"

        ok, reason = validate_target_path(resolved["file_path"], resolved["file_path"])
        if not ok:
            logger.warning(
                "Real diff blocked by security guardrail for org=%s agent=%s: %s",
                org_id,
                agent_id,
                reason,
            )
            return None

        current_content = await fetch_file_content(
            token, resolved["repo"], resolved["file_path"], base_branch
        )
        if current_content is None:
            return None

        new_content = await generate_real_file_content(
            root_cause, fix_content, resolved["file_path"], current_content
        )
        if new_content is None:
            return None

        return {
            "repo": resolved["repo"],
            "file_path": resolved["file_path"],
            "base_branch": base_branch,
            "old_content": current_content,
            "new_content": new_content,
        }
    except Exception as exc:
        logger.warning(
            "Real diff attempt failed for org=%s agent=%s (falling back to summary PR): %s",
            org_id,
            agent_id,
            exc,
        )
        return None


@router.post(
    "/v1/signals/{signal_id}/open-pr",
    summary="Create a GitHub draft PR with the code-change fix suggestion",
    response_model=Dict[str, Any],
)
async def open_pr(
    signal_id: int,
    body: OpenPRRequest,
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    # GitHub-config check stays first, matching pre-4.3 behavior/tests —
    # this is a config-gating error independent of whether the signal_id
    # itself is valid, so it should surface regardless of signal lookup.
    auth = await _resolve_github_auth(org_id)
    if auth is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "GitHub not configured. Install the Dunetrace GitHub App "
                "(POST /v1/orgs/integrations/github) or set GITHUB_TOKEN/"
                "GITHUB_REPO for a single-tenant self-hosted install."
            ),
        )

    signal = await get_signal_by_id(org_id, signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    if signal["failure_type"] in _NO_AUTO_APPLY_TYPES:
        raise HTTPException(
            status_code=403,
            detail=(
                "Auto-apply is blocked for PROMPT_INJECTION_SIGNAL. "
                "The trace contains untrusted input — review the fix manually "
                "before opening a PR."
            ),
        )

    real_file = await _attempt_real_diff(
        org_id, signal["agent_id"], auth["token"], auth["repos"], body.root_cause, body.fix_content
    )

    from api_svc.github_client import create_fix_pr
    from api_svc.diff_generation import compute_unified_diff

    if real_file:
        fix_patch = compute_unified_diff(
            real_file["file_path"], real_file["old_content"], real_file["new_content"]
        )
        target_repo = real_file["repo"]
        base_branch = real_file["base_branch"]
        real_file_arg = {
            "file_path": real_file["file_path"],
            "new_content": real_file["new_content"],
        }
    else:
        fix_patch = body.fix_patch
        first_repo = auth["repos"][0]
        target_repo = first_repo.get("repo")
        base_branch = first_repo.get("base_branch") or "main"
        real_file_arg = None

    if not target_repo:
        raise HTTPException(
            status_code=503,
            detail="No repo configured for this GitHub integration.",
        )

    try:
        result = await create_fix_pr(
            token=auth["token"],
            repo=target_repo,
            base_branch=base_branch,
            signal_id=signal_id,
            agent_id=signal["agent_id"],
            failure_type=signal["failure_type"],
            root_cause=body.root_cause,
            fix_content=body.fix_content,
            fix_patch=fix_patch,
            reviewers=auth["reviewers"],
            real_file=real_file_arg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.warning("create_fix_pr failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Could not create GitHub PR. Check your GitHub App installation or GITHUB_TOKEN/GITHUB_REPO.",
        )

    try:
        await record_fix(
            org_id=org_id,
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
    org_id: str = Depends(require_org),
) -> Dict[str, Any]:
    signal = await get_signal_by_id(org_id, signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    try:
        status = await get_signal_fix_status(
            org_id=org_id,
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
        "You are analyzing an AI agent structural failure. "
        "Given a detection signal and the agent's execution trace, identify the "
        "specific root cause and the single most important fix.\n"
        "Respond ONLY in JSON with exactly three fields:\n"
        '{"root_cause": "...", "fix_content": "...", "fix_patch": "..."}\n'
        "root_cause: max 120 words explaining specifically WHY this happened, "
        "quoting relevant prompt text or tool output.\n"
        f"{fix_instruction}"
    )

    raw = await llm_provider.complete(system, user_prompt, max_tokens=900)

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
