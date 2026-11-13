"""
Generates a real, applicable code change once source mapping resolves a
file (Phase 4.3) — asks the LLM for the corrected FULL file content
(grounded in the real current content, not guessed from trace context
alone the way today's fix_patch generation works), then computes the diff
ourselves via difflib rather than trusting an LLM-authored unified diff to
be mechanically appliable.

This deliberately sidesteps implementing a general patch-apply algorithm
(context-line mismatches, multi-hunk diffs, fuzzy matching) — since the
diff is computed from two full strings we already have (the real old
content and the LLM's new content), it is always well-formed. See
BACKLOG.md's Phase 4.3 entry for why this was chosen over mechanically
applying an LLM-authored unified diff.
"""

from __future__ import annotations

import difflib
import logging

from api_svc import llm_provider
from api_svc.config import settings

logger = logging.getLogger("dunetrace.api.diff_generation")

_MAX_TOKENS = 4096


async def generate_real_file_content(
    root_cause: str, fix_content: str, file_path: str, current_content: str
) -> str | None:
    """Returns the corrected full file content, or None if the LLM call
    fails, isn't configured, or returns something unusable (empty, or
    unchanged from the input — nothing to apply). Reuses the root_cause/
    fix_content already produced by explain_signal rather than re-deriving
    them — this call's only job is "apply this known fix to this real
    file," not root-cause analysis again.
    """
    if not llm_provider.llm_configured():
        return None

    system = (
        "You are applying a known, already-diagnosed fix to a real source "
        "file. You are given the root cause of a bug, a one-sentence "
        "description of the fix, and the file's current full content. "
        "Return ONLY the complete, corrected file content — no "
        "explanation, no markdown code fences, no diff syntax. If you "
        "cannot confidently apply this fix to this specific file, return "
        "the current content completely unchanged, verbatim."
    )
    user_prompt = (
        f"Root cause: {root_cause}\n\n"
        f"Fix to apply: {fix_content}\n\n"
        f"File: {file_path}\n"
        f"Current content:\n{current_content}"
    )

    raw = ""
    try:
        raw = await llm_provider.complete(system, user_prompt, max_tokens=_MAX_TOKENS)
    except Exception as exc:
        logger.warning("generate_real_file_content LLM call failed: %s", exc)
        return None

    new_content = (raw or "").strip()
    if new_content.startswith("```"):
        lines = new_content.splitlines()
        if len(lines) > 2:
            new_content = "\n".join(lines[1:-1])

    if not new_content or new_content == current_content.strip():
        return None

    return new_content


def compute_unified_diff(file_path: str, old_content: str, new_content: str) -> str:
    """Always well-formed — generated from two known strings, never an
    LLM-authored diff that would need mechanical application."""
    diff_lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff_lines)
