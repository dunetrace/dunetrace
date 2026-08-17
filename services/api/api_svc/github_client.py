"""GitHub API client — creates fix branches and draft PRs for code-change signals.

Phase 4.3 changed this from reading settings.GITHUB_TOKEN/GITHUB_REPO
directly to taking an explicit token/repo/base_branch — the caller
(routers/signals.py::open_pr) resolves whether to use a per-org GitHub App
installation token or the legacy global PAT fallback, then passes the
result in here. This module itself no longer knows or cares which auth
model produced the token.

Two write paths:
  - real_file given: commits the actual corrected file content directly
    (Phase 4.3's real diff application, once source mapping resolves a
    file and diff_generation.py produces new content) — the PR's actual
    file change, not a summary.
  - real_file omitted: the original, always-safe fallback — write a new
    dunetrace-fixes/signal-{id}.md file documenting the suggested fix
    (with fix_patch shown as a code block for humans to read), never
    touching real source. Used whenever source mapping doesn't resolve,
    diff generation fails, or the security guardrail rejects the target.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("dunetrace.api.github")

_GITHUB_API = "https://api.github.com"


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def fetch_file_content(token: str, repo: str, file_path: str, ref: str) -> Optional[str]:
    """Returns the file's current text content at `ref`, or None if it
    doesn't exist there. Decodes GitHub's base64-wrapped contents response."""
    async with httpx.AsyncClient(
        base_url=_GITHUB_API, headers=_headers(token), timeout=20
    ) as client:
        r = await client.get(f"/repos/{repo}/contents/{file_path}", params={"ref": ref})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if data.get("encoding") != "base64":
            return None
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")


async def create_fix_pr(
    token: str,
    repo: str,
    base_branch: str,
    signal_id: int,
    agent_id: str,
    failure_type: str,
    root_cause: str,
    fix_content: str,
    fix_patch: str,
    reviewers: Optional[List[str]] = None,
    real_file: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a branch, commit the fix (real file or markdown summary),
    and open a draft PR.

    real_file: {"file_path": ..., "new_content": ...} — when given, this
    exact file is written with this exact content, real code, in the real
    repo. Callers are responsible for having already run this path's
    content through fix_security.validate_target_path.

    Returns {"pr_url": str, "pr_number": int, "branch": str, "applied_to_real_file": bool}.
    Raises httpx.HTTPStatusError for API failures.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", failure_type.lower()).strip("-")
    branch = f"dunetrace/signal-{signal_id}-{slug}"

    async with httpx.AsyncClient(
        base_url=_GITHUB_API, headers=_headers(token), timeout=20
    ) as client:
        base_sha = await _get_branch_sha(client, repo, base_branch)
        await _create_branch(client, repo, branch, base_sha)

        if real_file:
            await _commit_real_file(
                client, repo, branch, real_file["file_path"], real_file["new_content"], signal_id
            )
        else:
            await _upsert_summary_file(
                client,
                repo,
                branch,
                signal_id,
                agent_id,
                failure_type,
                root_cause,
                fix_content,
                fix_patch,
            )

        result = await _open_pr(
            client,
            repo,
            branch,
            base_branch,
            signal_id,
            agent_id,
            failure_type,
            root_cause,
            fix_content,
            fix_patch,
            reviewers=reviewers,
            real_file_path=real_file["file_path"] if real_file else None,
        )
        result["applied_to_real_file"] = bool(real_file)
        return result


async def _get_branch_sha(client: httpx.AsyncClient, repo: str, branch: str) -> str:
    r = await client.get(f"/repos/{repo}/git/ref/heads/{branch}")
    r.raise_for_status()
    return r.json()["object"]["sha"]


async def _create_branch(client: httpx.AsyncClient, repo: str, branch: str, sha: str) -> None:
    r = await client.post(
        f"/repos/{repo}/git/refs",
        json={
            "ref": f"refs/heads/{branch}",
            "sha": sha,
        },
    )
    if r.status_code == 422:
        logger.debug("Branch %s already exists — reusing", branch)
    else:
        r.raise_for_status()


async def _commit_real_file(
    client: httpx.AsyncClient,
    repo: str,
    branch: str,
    file_path: str,
    new_content: str,
    signal_id: int,
) -> None:
    encoded = base64.b64encode(new_content.encode()).decode()

    existing_sha = None
    r = await client.get(f"/repos/{repo}/contents/{file_path}", params={"ref": branch})
    if r.status_code == 200:
        existing_sha = r.json().get("sha")

    body: Dict[str, Any] = {
        "message": f"dunetrace: fix suggestion for signal #{signal_id}",
        "content": encoded,
        "branch": branch,
    }
    if existing_sha:
        body["sha"] = existing_sha

    r = await client.put(f"/repos/{repo}/contents/{file_path}", json=body)
    r.raise_for_status()


async def _upsert_summary_file(
    client: httpx.AsyncClient,
    repo: str,
    branch: str,
    signal_id: int,
    agent_id: str,
    failure_type: str,
    root_cause: str,
    fix_content: str,
    fix_patch: str,
) -> None:
    path = f"dunetrace-fixes/signal-{signal_id}.md"
    content = _build_fix_file(signal_id, agent_id, failure_type, root_cause, fix_content, fix_patch)
    encoded = base64.b64encode(content.encode()).decode()

    existing_sha = None
    r = await client.get(f"/repos/{repo}/contents/{path}", params={"ref": branch})
    if r.status_code == 200:
        existing_sha = r.json().get("sha")

    body: Dict[str, Any] = {
        "message": f"dunetrace: fix suggestion for {failure_type} (signal #{signal_id})",
        "content": encoded,
        "branch": branch,
    }
    if existing_sha:
        body["sha"] = existing_sha

    r = await client.put(f"/repos/{repo}/contents/{path}", json=body)
    r.raise_for_status()


async def _open_pr(
    client: httpx.AsyncClient,
    repo: str,
    branch: str,
    base_branch: str,
    signal_id: int,
    agent_id: str,
    failure_type: str,
    root_cause: str,
    fix_content: str,
    fix_patch: str,
    reviewers: Optional[List[str]] = None,
    real_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    title = (
        f"[DuneTrace] Fix {failure_type.replace('_', ' ').title()} "
        f"in {agent_id} (signal #{signal_id})"
    )
    pr_body = _build_pr_body(
        signal_id, agent_id, failure_type, root_cause, fix_content, fix_patch, real_file_path
    )
    r = await client.post(
        f"/repos/{repo}/pulls",
        json={
            "title": title,
            "body": pr_body,
            "head": branch,
            "base": base_branch,
            "draft": True,
        },
    )
    if r.status_code == 422:
        # PR already exists for this branch — find and return it
        owner = repo.split("/")[0]
        r2 = await client.get(
            f"/repos/{repo}/pulls",
            params={"head": f"{owner}:{branch}", "state": "open"},
        )
        prs = r2.json()
        if prs:
            return {
                "pr_url": prs[0]["html_url"],
                "pr_number": prs[0]["number"],
                "branch": branch,
            }
    r.raise_for_status()
    pr = r.json()

    if reviewers:
        try:
            await client.post(
                f"/repos/{repo}/pulls/{pr['number']}/requested_reviewers",
                json={"reviewers": reviewers},
            )
        except Exception as exc:
            # Reviewer assignment failure (e.g. a username no longer has
            # repo access) must not fail PR creation itself — the PR
            # already exists and is the thing that matters.
            logger.warning(
                "Failed to request reviewers %s on PR #%d: %s", reviewers, pr["number"], exc
            )

    return {"pr_url": pr["html_url"], "pr_number": pr["number"], "branch": branch}


def _build_fix_file(
    signal_id: int,
    agent_id: str,
    failure_type: str,
    root_cause: str,
    fix_content: str,
    fix_patch: str,
) -> str:
    lines = [
        f"# DuneTrace Fix Suggestion — Signal #{signal_id}",
        "",
        f"**Agent:** `{agent_id}`  ",
        f"**Failure type:** `{failure_type}`",
        "",
        "## Root Cause",
        "",
        root_cause,
        "",
        "## Suggested Fix",
        "",
        fix_content,
        "",
    ]
    if fix_patch:
        lines += [
            "## Code Diff",
            "",
            "```diff",
            fix_patch,
            "```",
            "",
        ]
    lines += [
        "---",
        "*Generated by [DuneTrace](https://dunetrace.io) — review before merging.*",
    ]
    return "\n".join(lines)


def _build_pr_body(
    signal_id: int,
    agent_id: str,
    failure_type: str,
    root_cause: str,
    fix_content: str,
    fix_patch: str,
    real_file_path: Optional[str] = None,
) -> str:
    applied_note = (
        f"\n> This PR edits `{real_file_path}` directly — the diff below is exactly "
        "what changed, computed from the file's real before/after content.\n"
        if real_file_path
        else "\n> This PR documents the fix in a new file — Dunetrace could not "
        "confidently resolve which source file to edit directly. See "
        "`dunetrace-fixes/` for the details.\n"
    )
    body = f"""\
## DuneTrace Auto-Detected Fix
{applied_note}
**Signal:** #{signal_id}
**Agent:** `{agent_id}`
**Failure type:** `{failure_type.replace("_", " ").title()}`

### Root Cause

{root_cause}

### Suggested Fix

{fix_content}
"""
    if fix_patch:
        body += f"""
### Code Diff

```diff
{fix_patch}
```
"""
    body += """
---
> Generated by [DuneTrace](https://dunetrace.io) — review carefully before merging.
"""
    return body
