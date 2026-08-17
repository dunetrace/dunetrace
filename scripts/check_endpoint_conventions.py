#!/usr/bin/env python3
"""
Greps every FastAPI router file for org-scoped endpoints that declare
org_id as a URL path parameter (e.g. "/v1/orgs/{org_id}/packs") instead of
deriving it from the caller's own API key via Depends(require_org).

Why this matters: a path-param org_id lets any authenticated caller read or
write a DIFFERENT org's config just by editing the URL — this exact bug was
caught and fixed by hand in api_svc/routers/github_integration.py (four
endpoints initially used org_id: str = Query(...) instead of
Depends(require_org)) and flagged again during Phase 1.0's pack-activation
design (services/api/api_svc/routers/packs.py). Every org-scoped write
endpoint in this codebase derives org_id from require_org() instead — see
that function's docstring in api_svc/auth.py.

This is a static, regex-based scan (not a full AST/route-graph analysis) —
it flags anything that LOOKS like the anti-pattern for a human to check, not
a certified absence-of-bugs proof.

Legitimate exceptions exist: a third-party webhook (e.g.
api_svc/routers/linear_webhook.py's POST /v1/webhooks/linear/{org_id}) can
carry no Dunetrace API key at all — org_id in the path is how the per-org
webhook secret gets looked up *before* the payload can be verified, the same
shape as github_integration.py's /callback (which uses a query param
`state` instead, so it doesn't even match this regex). Mark a genuine
exception by adding "# org-id-path-ok: <reason>" on the line immediately
before the @router decorator — this script skips any match preceded by
that marker rather than either special-casing filenames here (which the
next legitimate exception wouldn't be covered by) or silently allowing every
{org_id} path (which would defeat the point).

Usage:
    python scripts/check_endpoint_conventions.py
    python scripts/check_endpoint_conventions.py --dir services/api

Exit code 0: no violations found.
Exit code 1: at least one violation found (suitable for a CI check).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches @router.get("...")/@router.post("...")/etc., capturing the path
# string. Multi-line decorators (path on its own line, common in this
# codebase's style) are handled by DOTALL + a non-greedy match up to the
# first closing paren of the decorator call.
_ROUTE_DECORATOR_RE = re.compile(
    r'@router\.(get|post|put|patch|delete)\(\s*["\']([^"\']+)["\']', re.MULTILINE
)

# A path segment literally named {org_id} — the exact anti-pattern.
_ORG_ID_PATH_PARAM_RE = re.compile(r"\{org_id\}")


def find_router_files(root: Path) -> list[Path]:
    return sorted(root.glob("services/*/*/routers/*.py"))


_SUPPRESSION_MARKER = "org-id-path-ok:"


def _has_suppression_comment_directly_above(lines: list[str], decorator_line_no: int) -> bool:
    """Walks backwards from the line directly above the decorator (1-indexed
    decorator_line_no), through a CONTIGUOUS run of comment/blank lines only
    — stopping at the first line that is neither, since that's necessarily
    part of a different, preceding statement (another endpoint's body, a
    decorator, an import, etc.) and must not be treated as this endpoint's
    own explanatory comment. Without this boundary, a suppression comment on
    one endpoint can silently swallow the very next endpoint below it when
    two endpoints are only a few lines apart (the common case)."""
    for i in range(decorator_line_no - 2, -1, -1):
        stripped = lines[i].strip()
        if stripped == "":
            continue
        if not stripped.startswith("#"):
            return False
        if _SUPPRESSION_MARKER in stripped:
            return True
    return False


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Returns a list of (line_number, method, path) violations."""
    violations = []
    text = path.read_text()
    lines = text.split("\n")
    for match in _ROUTE_DECORATOR_RE.finditer(text):
        method, route_path = match.group(1), match.group(2)
        if not _ORG_ID_PATH_PARAM_RE.search(route_path):
            continue
        line_no = text.count("\n", 0, match.start()) + 1
        if _has_suppression_comment_directly_above(lines, line_no):
            continue
        violations.append((line_no, method.upper(), route_path))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=".",
        help="Repo root to scan from (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    router_files = find_router_files(root)

    if not router_files:
        print(f"No router files found under {root}/services/*/*/routers/*.py")
        return 0

    total_violations = 0
    for path in router_files:
        for line_no, method, route_path in check_file(path):
            total_violations += 1
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            print(
                f"{rel}:{line_no}: {method} {route_path!r} declares org_id as a "
                f"path param — derive it from Depends(require_org) instead."
            )

    if total_violations:
        print(
            f"\n{total_violations} violation(s) found. See api_svc/auth.py's "
            f"require_org() docstring and api_svc/routers/packs.py for the "
            f"correct pattern."
        )
        return 1

    print(f"Checked {len(router_files)} router file(s). No org_id-in-path-param violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
