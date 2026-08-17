# GitHub App Integration (Phase 4.3)

Dunetrace can open a **draft PR** with an automated fix for `customer_code` /
`code_change` signals (`CONTEXT_BLOAT`, `SLOW_STEP`, and other detectors whose
fix is a code/infra change rather than a prompt edit or a runtime policy —
see `fix_classification.py`). This replaces the original global-`GITHUB_TOKEN`
single-repo setup with a per-org **GitHub App** installation, while keeping
the old env vars working as a fallback for existing self-hosted installs.

## Two auth paths

| | Per-org GitHub App (recommended) | Legacy global PAT |
|---|---|---|
| Setup | Customer installs Dunetrace's GitHub App on their own repos | Operator sets `GITHUB_TOKEN`/`GITHUB_REPO` in `.env` |
| Scope | Per-org, per-installation — each org only grants access to repos it chooses | Single repo, shared across every org on the deployment |
| Token lifetime | Short-lived installation tokens (~1hr), minted on demand | Long-lived PAT, whatever expiry you set on GitHub |
| Multi-tenant safe | Yes | No — only appropriate for single-tenant self-hosted installs |

`_resolve_github_auth()` in `services/api/api_svc/routers/signals.py` checks
the per-org installation first; if the org hasn't installed the App (or has
installed it but configured zero repos yet), it falls back to the global PAT.
If neither is configured, `POST /v1/signals/{id}/open-pr` returns 503.

## Setting up the GitHub App (operator, one-time)

1. Create a GitHub App at `https://github.com/settings/apps/new` (or your
   org's equivalent). Permissions needed:
   - **Contents**: Read & write (to read the current file and commit the fix)
   - **Pull requests**: Read & write (to open the draft PR and request reviewers)
   - **Issues**: Read & write (reserved for future issue-linking; not yet used)
   - Disable webhooks — Dunetrace doesn't consume GitHub webhook events for
     this integration.
2. Generate a private key for the App and copy the App ID.
3. Set in `.env`:
   ```bash
   GITHUB_APP_ID=123456
   GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
   GITHUB_APP_SLUG=your-app-slug   # from the App's settings page URL
   ```
   `GITHUB_APP_PRIVATE_KEY` uses `\n` for newlines in `.env` (single-line)
   — `config.py` converts them back to real newlines before use.
4. Restart the API service: `docker compose up --build api -d`.

## Connecting an org (customer-facing flow)

1. `GET /v1/orgs/integrations/github/install-url` returns a URL to GitHub's
   App installation page, with the org's id embedded in `state` (round-tripped
   back on install — the same CSRF-safe pattern used elsewhere for
   OAuth-style integrations, even though the App itself needs no token
   exchange).
2. Customer picks which repos to install the App on. GitHub redirects to
   `GET /v1/orgs/integrations/github/callback?installation_id=...&state=...`
   — this endpoint has no Dunetrace auth (it's a browser redirect GitHub
   controls, not an authenticated API call), so `state` is the only source of
   `org_id` here.
3. `POST /v1/orgs/integrations/github` sets which of the installed repos
   Dunetrace should actually use, plus target branch per repo and PR
   reviewers:
   ```json
   {
     "repos": [{"repo": "acme/agent-service", "base_branch": "main"}],
     "reviewers": ["octocat"]
   }
   ```
   Returns 404 if the org hasn't completed step 2 yet.
4. `GET`/`DELETE /v1/orgs/integrations/github` to check status or disconnect.

There's no encrypted credential to store for this integration —
`installation_id` isn't a secret (GitHub itself treats it as a plain
identifier), only the App's own private key is sensitive, and that's a single
operator-level secret in `.env`, not a per-org one.

## Source mapping: which repo/file does a signal correspond to?

A signal only carries `agent_id`/`run_id` — resolving it to a specific
`repo` + `file_path` for a real code edit uses two tiers, checked in order,
implemented in `services/api/api_svc/source_resolution.py`:

1. **Tier 1 — explicit config.** `POST /v1/agents/{agent_id}/source-config`
   lets a customer declare `{"repo": "...", "file_path": "..."}` directly.
   If both are set, this is used as-is — highest confidence, no guessing.
2. **Tier 2 — SDK auto-detection.** The SDK captures the calling file's
   absolute path via `dt.run()` (`packages/sdk-py/dunetrace/client.py`'s
   `_capture_caller_source_file()`, using raw `sys._getframe()` walking, not
   `inspect.stack()`, to keep per-event overhead in the microseconds) and
   sends it in `run.started`'s payload. This is a local filesystem path, not
   a repo-relative one GitHub's API needs, so it's resolved by suffix-matching
   it against the target repo's real file tree (`git/trees` API) — combined
   with tier 1's repo (if declared without a `file_path`) or, absent any tier-1
   config, the org's one connected repo (only when there's exactly one — an
   org with multiple connected repos and no tier-1 hint gets no resolution;
   picking one would be guessing).

A suffix match must be **exact and unique** — if zero or more than one file in
the repo's tree matches, resolution returns nothing rather than picking
ambiguously. If source mapping doesn't resolve for any reason, `open-pr` falls
back to the always-safe behavior below rather than failing the request.

## What actually gets committed

- **Source resolved + diff generation succeeds**: the PR edits the real file.
  `diff_generation.py::generate_real_file_content()` fetches the file's
  current content, asks the LLM for the corrected *full file content*
  (grounded in what's actually there — not asked to author a diff freehand),
  and `compute_unified_diff()` computes the diff ourselves via Python's
  `difflib` against the two known strings. This sidesteps ever needing to
  mechanically apply an LLM-authored diff (fragile against context-line
  mismatches) — the diff shown in the PR body is always well-formed because
  it's derived from real before/after content, not generated by the model.
- **Anything in that path fails** (no source resolution, file not found in
  the repo, LLM declines/produces no change, or the security guardrail
  rejects the target): falls back to the original, always-safe behavior —
  a new `dunetrace-fixes/signal-{id}.md` file documenting the suggested fix,
  never touching real source. `_attempt_real_diff()` never raises; any
  exception anywhere in this path is logged and treated as "fall back,"
  so opening *some* PR always still succeeds.

Either way, the PR is **always opened as a draft** — the customer must
promote it to ready for review themselves.

## Security guardrails

Before any real file is committed, `fix_security.py::validate_target_path()`
enforces two checks, both deterministic (no LLM judgment in the check itself):

1. **Sensitive path denylist** (`is_sensitive_path()`) — regex-based, blocks
   CI configs (`.github/workflows/`, `.gitlab-ci.yml`, `.circleci/`,
   `Jenkinsfile`), secrets (`.env*`, `*.pem`, `*.key`, `id_rsa*`, and any path
   containing `secret`/`credential`/`password`), and container/infra files
   (`Dockerfile`, `docker-compose*.yml`).
2. **Exact-path enforcement** — the file actually being written must exactly
   match what source resolution resolved; there's no path where the LLM's own
   output can redirect the write target.

Both checks run before `fetch_file_content`/`generate_real_file_content` are
even called for the real-diff path — a rejection here means the same safe
markdown-summary fallback as any other resolution failure.

## Not done / known limits

- No live end-to-end smoke test yet against a real GitHub App installation
  (mirrors `scripts/test_linear_integration.py`'s pattern but doesn't exist
  yet for GitHub — needs real App credentials to write and run).
- Multi-repo orgs with no tier-1 source-config hint get no tier-2 resolution
  at all — there's no "guess the most likely repo" heuristic, by design.
- `_fetch_repo_tree_paths()` warns but does not paginate around GitHub's
  truncated-tree response for very large repos (the tree API caps at ~100k
  entries) — suffix-matching may silently miss files in such repos.
