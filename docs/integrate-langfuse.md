# Integrating Langfuse with Dunetrace

Root-cause analysis (click **Explain +** on any alert) is native — Dunetrace analyzes the run's own stored events, no Langfuse required. Connecting Langfuse adds one thing on top: a one-click **Apply** button that pushes a prompt fix straight to a Langfuse-managed prompt, instead of just showing you a diff to copy in yourself.

---

## What you get

- **Root-cause analysis** — works with or without Langfuse connected; see [docs/migrations/native-root-cause-v0.6.0.md](migrations/native-root-cause-v0.6.0.md) for how fixes are classified
- **One-click prompt apply** — for fixes that just need a sentence added to the system prompt, publish a new Langfuse prompt version directly from the dashboard instead of copy/pasting
- **Fix tracking** — the dashboard shows whether recurrence dropped after a fix was applied

Structural fixes (tool loops, retry storms, runaway step counts) don't need Langfuse at all — Dunetrace applies those as a runtime policy directly. Code/infra fixes are always a diff you apply yourself; there's no GitHub PR integration yet (tracked in `BACKLOG.md`).

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Langfuse account (cloud or self-hosted) with a project and API keys
- One LLM API key for the analysis call (`ANTHROPIC_API_KEY` preferred, `OPENAI_API_KEY` accepted as fallback)

---

## Step 1: Install

```bash
pip install 'dunetrace[langchain,langfuse]'
```

---

## Step 2: Add credentials to `.env`

```bash
# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # omit for cloud; set for self-hosted

# LLM for explain endpoint (Anthropic preferred, OpenAI fallback)
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

Restart the API container after editing `.env`:

```bash
docker compose up -d api
```

---

## Step 3: Run both callbacks together

Pass `DunetraceCallbackHandler` and `LangfuseCallbackHandler` in the same `callbacks` list. They are fully independent — no coupling required.

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # v4+

dt    = Dunetrace(endpoint="http://localhost:8001")
dt_cb = DunetraceCallbackHandler(dt, agent_id="my-agent", model="gpt-4o-mini", tools=["web_search"])
lf_cb = LangfuseCallbackHandler()   # reads LANGFUSE_* from env

result = agent.invoke(
    {"messages": [("human", query)]},
    config={"callbacks": [dt_cb, lf_cb]},
)

dt.shutdown(timeout=5)

import langfuse as lf_module
lf_module.get_client().flush()   # ensure trace is uploaded before querying

# IDs for the join:
dt_run_id   = dt_cb.last_run_id     # e.g. "b5ed23be-e4f0-43bc-..."
lf_trace_id = lf_cb.last_trace_id   # e.g. "b5ed23bee4f043bc..."  (same UUID, no dashes)
```

> **Langfuse v4 trace ID format:** Langfuse v4 uses OTel-style 32-character hex IDs (no dashes). The Dunetrace API strips dashes automatically when querying Langfuse — both IDs represent the same run.

---

## Step 4: Call the explain endpoint

```bash
POST /v1/signals/{signal_id}/explain
Content-Type: application/json
Authorization: Bearer <your-key>

{
  "langfuse_trace_id": "b5ed23bee4f043bc8625914223875508"
}
```

If `langfuse_trace_id` is omitted, the endpoint falls back to the signal's own `run_id`.

Response:

```json
{
  "signal_id": 344,
  "source": "langfuse",
  "root_cause": "The agent re-issued the same search query because the system prompt contains no instruction to track previous queries...",
  "fix_content": "Do not repeat a search query you have already executed in this run.",
  "fix_type": "prompt_addition",
  "apply_blocked": false,
  "langfuse_prompt_name": "research-agent-prompt",
  "langfuse_prompt_version": 3
}
```

**`fix_type` values:**

| `fix_type` | Meaning | Dashboard action |
|---|---|---|
| `prompt_addition` | One sentence to append to the system prompt | **Apply via Langfuse** — pushes new prompt version |
| `code_change` | Code or infra fix (CONTEXT_BLOAT, SLOW_STEP, CASCADING_TOOL_FAILURE, etc.) | **Open PR on GitHub ↗** — creates a draft PR with unified diff |
| `no_auto_apply` | Security signal (PROMPT_INJECTION_SIGNAL) — never auto-apply | No apply action — review manually |

`langfuse_prompt_name` is `null` when the trace's system prompt was a hardcoded string rather than a Langfuse-managed prompt. The apply button only appears when this field is non-null and `apply_blocked` is false.

---

## Step 5a: Apply a prompt fix via Langfuse

When `fix_type` is `prompt_addition` and `langfuse_prompt_name` is returned:

```bash
POST /v1/signals/{signal_id}/apply-fix
Content-Type: application/json
Authorization: Bearer <your-key>

{
  "fix_content": "Do not repeat a search query you have already executed in this run.",
  "langfuse_prompt_name": "research-agent-prompt"
}
```

Response:

```json
{
  "fix_id": 12,
  "signal_id": 344,
  "new_version": 4,
  "prompt_url": "https://cloud.langfuse.com/prompts/research-agent-prompt",
  "old_text": "You are a research assistant...",
  "new_text": "You are a research assistant...\n\nDo not repeat a search query..."
}
```

The fix is appended to the current prompt and published as a new Langfuse version. The dashboard shows "Applied as v4 in Langfuse ↗" with a link.

---

## Step 5b: Open a GitHub PR for code-change fixes

When `fix_type` is `code_change`:

```bash
POST /v1/signals/{signal_id}/open-pr
Content-Type: application/json
Authorization: Bearer <your-key>

{
  "root_cause":  "Context window growing unboundedly because...",
  "fix_content": "Add a sliding window that drops the oldest messages when token count exceeds 80% of the model limit.",
  "fix_patch":   "--- a/agent.py\n+++ b/agent.py\n@@ -42,6 +42,10 @@\n ..."
}
```

Response:

```json
{
  "pr_url":    "https://github.com/owner/repo/pull/17",
  "pr_number": 17,
  "branch":    "dunetrace/signal-42-context-bloat"
}
```

The PR is created as a draft on branch `dunetrace/signal-{id}-{failure-type}`.

**Prerequisites:** set `GITHUB_TOKEN` (needs `repo` scope) and `GITHUB_REPO` (`owner/repo`) in `.env`:

```bash
GITHUB_TOKEN=ghp_...
GITHUB_REPO=owner/repo
GITHUB_BASE_BRANCH=main   # optional, default: main
```

Then `docker compose up -d api`.

---

## Step 6: Track fix effectiveness

```bash
GET /v1/signals/{signal_id}/fix-status
Authorization: Bearer <your-key>
```

Response:

```json
{
  "fix_applied": true,
  "applied_at": 1745000000.0,
  "applied_via": "langfuse",
  "langfuse_prompt_name": "research-agent-prompt",
  "langfuse_version": 4,
  "runs_after_fix": 23,
  "recurrences_after_fix": 0,
  "verdict": "verified"
}
```

Verdicts: `verified` (≥10 runs, 0 recurrences), `likely_fixed` (≥5 runs, 0 recurrences), `still_occurring`, `insufficient_data`.

---

## Full working example

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-... \
LANGFUSE_SECRET_KEY=sk-lf-... \
ANTHROPIC_API_KEY=sk-ant-... \
SCENARIO=tool_loop python packages/sdk-py/examples/langfuse_agent.py
```

See [`packages/sdk-py/examples/langfuse_agent.py`](../packages/sdk-py/examples/langfuse_agent.py) for the complete runnable script.

---

## How the trace lookup works

1. Signal fires → `run_id` stored in Postgres
2. Dashboard calls `POST /v1/signals/{id}/explain` with optional `langfuse_trace_id`
3. API fetches `GET /api/public/traces/{traceId}` from Langfuse (retries up to 4× for ingestion lag; fetches full observation list separately when ≥10 observations are returned paginated)
4. Extracts system prompt from GENERATION observation `messages[]` arrays
5. Normalises step range from evidence using detector-specific field names (e.g. `first_truncation_step` for LLM_TRUNCATION_LOOP)
6. Builds a prompt: signal type + evidence + system prompt + relevant span inputs/outputs (600-char limit for failing steps, 150-char for others)
7. Calls Anthropic Haiku (or GPT-4o-mini fallback) — max 900 tokens — asking for `{"root_cause": "...", "fix_content": "...", "fix_patch": "..."}` JSON
8. Returns structured response with fix type classification

The Langfuse trace is never stored — fetched, analysed, discarded.
