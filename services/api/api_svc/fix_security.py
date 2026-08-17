"""
Security guardrails for real diff application (Phase 4.3). Pure,
deterministic checks — no LLM judgment involved in the check itself, so
this is fully unit-testable and never asks a model whether something is
safe. Run BEFORE any GitHub write call; any failure here means "fall back
to the safe markdown-summary-file PR," never a partial or corrupted write.
"""

from __future__ import annotations

import re

# Path fragments/patterns a resolved source file must never match, even if
# source mapping (tier-1 config or tier-2 SDK detection) somehow points
# there — defense in depth, independent of whatever the LLM's diff claims.
_SENSITIVE_PATH_PATTERNS = [
    r"^\.github/workflows/",
    r"^\.github/actions/",
    r"\.gitlab-ci\.yml$",
    r"^\.circleci/",
    r"^Jenkinsfile$",
    r"(^|/)\.env(\.|$)",
    r"\.pem$",
    r"\.key$",
    r"(^|/)id_rsa",
    r"(secret|credential|password)",  # case-insensitive, checked separately below
    r"(^|/)Dockerfile$",
    r"docker-compose.*\.ya?ml$",
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATH_PATTERNS), re.IGNORECASE)


def is_sensitive_path(path: str) -> bool:
    """True if this path must never be touched by an automated fix,
    regardless of what source mapping or the LLM's diff claims."""
    normalized = path.lstrip("/")
    return bool(_SENSITIVE_RE.search(normalized))


def validate_target_path(resolved_path: str, diff_target_path: str | None) -> tuple[bool, str]:
    """The core guardrail: the file we're about to write to must be the
    resolved source path from source mapping — not whatever an LLM's diff
    happens to mention (which could be a hallucinated or unrelated file).

    Returns (ok, reason). ok=False for any of:
      - resolved_path itself is a sensitive path
      - diff_target_path is given and doesn't match resolved_path exactly
    """
    if is_sensitive_path(resolved_path):
        return False, f"resolved path {resolved_path!r} matches a sensitive-path pattern"

    if diff_target_path is not None:
        norm_resolved = resolved_path.lstrip("/")
        norm_diff = diff_target_path.lstrip("/")
        if norm_resolved != norm_diff:
            return False, (
                f"diff targets {diff_target_path!r} but source mapping resolved "
                f"{resolved_path!r} — refusing to apply to an unverified path"
            )

    return True, ""
