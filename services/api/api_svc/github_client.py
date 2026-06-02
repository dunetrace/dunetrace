"""GitHub API client — creates fix branches and draft PRs for code-change signals."""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict

import httpx

from api_svc.config import settings

logger = logging.getLogger("dunetrace.api.github")

_GITHUB_API = "https://api.github.com"


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def create_fix_pr(
    signal_id: int,
    agent_id: str,
    failure_type: str,
    root_cause: str,
    fix_content: str,
    fix_patch: str,
) -> Dict[str, Any]:
    """Create a branch, commit a fix-suggestion file, and open a draft PR.

    Returns {"pr_url": str, "pr_number": int, "branch": str}.
    Raises ValueError for config errors, httpx.HTTPStatusError for API failures.
    """
    repo = settings.GITHUB_REPO
    base_branch = settings.GITHUB_BASE_BRANCH

    slug = re.sub(r"[^a-z0-9]+", "-", failure_type.lower()).strip("-")
    branch = f"dunetrace/signal-{signal_id}-{slug}"

    async with httpx.AsyncClient(base_url=_GITHUB_API, headers=_headers(), timeout=20) as client:
        base_sha = await _get_branch_sha(client, repo, base_branch)
        await _create_branch(client, repo, branch, base_sha)
        await _upsert_file(
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
        return await _open_pr(
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
        )


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


async def _upsert_file(
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
) -> Dict[str, Any]:
    title = (
        f"[DuneTrace] Fix {failure_type.replace('_', ' ').title()} "
        f"in {agent_id} (signal #{signal_id})"
    )
    pr_body = _build_pr_body(signal_id, agent_id, failure_type, root_cause, fix_content, fix_patch)
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
) -> str:
    body = f"""\
## DuneTrace Auto-Detected Fix

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
