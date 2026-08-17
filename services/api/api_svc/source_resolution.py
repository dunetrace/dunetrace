"""
Two-tier source mapping (Phase 4.3) — resolves "which repo/file does this
agent's code live in" so a real fix can be applied to real code, instead of
the LLM guessing a file path from trace context alone (which today's
fix_patch generation does, and gets wrong often — see BACKLOG.md).

Tier 1 (explicit config, agent_source_config table): a customer declares
repo + file_path directly via POST /v1/agents/{agent_id}/source-config.
Highest confidence — used as-is when both fields are set.

Tier 2 (SDK auto-detection): the SDK captures the calling file's path via
Python's inspect module at dt.run() time (packages/sdk-py/dunetrace/
run_context.py), stored in run.started's payload — a local, often-absolute
filesystem path, NOT a repo-relative path GitHub's API needs. Resolving it
requires knowing which repo to check it against, and a repo-relative path
lookup:

  - If tier-1 declares a repo but no file_path, tier-2's path is
    suffix-matched against THAT repo's real file tree.
  - If no tier-1 config exists at all, but the org has exactly one
    connected GitHub repo, tier-2's path is suffix-matched against that
    repo (still unambiguous).
  - Suffix-matching against a repo neither tier declares, or an org with
    multiple connected repos and no tier-1 hint, is not attempted — that
    would be guessing, not resolving.

A suffix match must be exact-and-unique: zero or multiple matching files
in the repo's tree means no resolution at all (never picked ambiguously).
"""

from __future__ import annotations

import logging

import httpx

from api_svc.config import settings
from api_svc.db.queries import (
    get_agent_source_config,
    get_latest_run_started_payload,
    get_org_github_integration,
)
from api_svc.github_app_auth import get_installation_token

logger = logging.getLogger("dunetrace.api.source_resolution")

_GITHUB_API = "https://api.github.com"


def _detected_source_file(run_started_payload: dict | None) -> str | None:
    if not run_started_payload:
        return None
    return run_started_payload.get("source_file") or None


async def _fetch_repo_tree_paths(installation_id: int, repo: str) -> list[str]:
    """All file paths in the repo's default branch, via the git trees API
    (recursive). Used only for suffix-matching a tier-2 detected path —
    never for anything write-related."""
    token = await get_installation_token(installation_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(base_url=_GITHUB_API, headers=headers, timeout=20) as client:
        repo_resp = await client.get(f"/repos/{repo}")
        repo_resp.raise_for_status()
        default_branch = repo_resp.json()["default_branch"]

        tree_resp = await client.get(
            f"/repos/{repo}/git/trees/{default_branch}", params={"recursive": "1"}
        )
        tree_resp.raise_for_status()
        tree = tree_resp.json()

    if tree.get("truncated"):
        logger.warning("Repo %s's tree is truncated — suffix-match may miss files", repo)

    return [item["path"] for item in tree.get("tree", []) if item.get("type") == "blob"]


def _suffix_match(detected_path: str, tree_paths: list[str]) -> str | None:
    """Returns the unique repo-relative path whose trailing segments match
    detected_path, or None if there's zero or more than one match."""
    normalized = detected_path.replace("\\", "/").lstrip("/")
    matches = [
        p
        for p in tree_paths
        if normalized == p
        or normalized.endswith("/" + p)
        or p.endswith("/" + normalized)
        or p == normalized
    ]
    # The above is intentionally permissive on direction (either could be
    # the "longer" absolute-ish path) — but still requires a real, exact
    # tail alignment, not a loose substring match.
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


async def resolve_source(org_id: str, agent_id: str) -> dict | None:
    """Returns {"repo": ..., "file_path": ...} or None if no confident
    resolution exists. Never guesses — see module docstring."""
    tier1 = await get_agent_source_config(org_id, agent_id)
    github_integration = await get_org_github_integration(org_id)

    if tier1 and tier1.get("repo") and tier1.get("file_path"):
        return {"repo": tier1["repo"], "file_path": tier1["file_path"]}

    # From here on we need a detected file path (tier-2) to combine with
    # either tier-1's repo-only declaration or an unambiguous single repo.
    run_started = await get_latest_run_started_payload(org_id, agent_id)
    detected_path = _detected_source_file(run_started)
    if not detected_path:
        return None

    candidate_repo = None
    if tier1 and tier1.get("repo"):
        candidate_repo = tier1["repo"]
    elif github_integration and len(github_integration.get("repos", [])) == 1:
        candidate_repo = github_integration["repos"][0].get("repo")

    if not candidate_repo or not github_integration:
        return None

    try:
        tree_paths = await _fetch_repo_tree_paths(
            github_integration["installation_id"], candidate_repo
        )
    except Exception as exc:
        logger.warning(
            "Failed to fetch repo tree for %s (org=%s agent=%s): %s",
            candidate_repo,
            org_id,
            agent_id,
            exc,
        )
        return None

    matched_path = _suffix_match(detected_path, tree_paths)
    if matched_path is None:
        return None

    return {"repo": candidate_repo, "file_path": matched_path}
