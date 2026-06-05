# Dunetrace MCP Server

Query agent signals, run details, and health scores directly from Claude Code, Cursor, Codex, or any MCP-compatible client — without leaving your editor.

---

## What it is

The MCP server wraps the Dunetrace Customer API in the [Model Context Protocol](https://modelcontextprotocol.io). Your editor (or any LLM) can call it as a tool and ask things like:

- *"Is my `langchain-example-agent` healthy?"*
- *"What failed in the last 24 hours?"*
- *"Show me signal #518 — what happened and how do I fix it?"*
- *"Is the TOOL_LOOP I'm seeing systemic or a one-off?"*
- *"Walk me through run `019e2314-6b7` step by step."*

All data is read-only. Only hashed metadata is exposed — no raw prompts, tool arguments, or model outputs ever leave your process.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- The Customer API accessible at `http://localhost:8002` (or set `DUNETRACE_API_URL`)

---

## Install

```bash
pip install dunetrace-mcp
```

Or install from source (for development):

```bash
cd packages/mcp-server
pip install -e .
```

---

## Client setup

### Claude Code

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "dunetrace": {
      "command": "dunetrace-mcp",
      "env": {
        "DUNETRACE_API_URL": "http://localhost:8002",
        "DUNETRACE_API_KEY": "dt_dev_test"
      }
    }
  }
}
```

Restart Claude Code. The `dunetrace` server will appear in the MCP tools list.

### Cursor

Create `.cursor/mcp.json` in your project root (or global `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "dunetrace": {
      "command": "dunetrace-mcp",
      "env": {
        "DUNETRACE_API_URL": "http://localhost:8002",
        "DUNETRACE_API_KEY": "dt_dev_test"
      }
    }
  }
}
```

### Codex / SSE clients

Run the server in SSE mode (listens on `:8000` by default):

```bash
dunetrace-mcp --sse
dunetrace-mcp --sse --port 9000   # custom port
```

Point your client's tool endpoint at `http://localhost:8000/sse`.

### Manual test (stdio)

```bash
dunetrace-mcp
```

The server speaks MCP over stdin/stdout. You can pipe JSON-RPC messages manually or use the MCP Inspector.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DUNETRACE_API_URL` | `http://localhost:8002` | Customer API base URL |
| `DUNETRACE_API_KEY` | `dt_dev_test` | Bearer token (auth header) |

For production, set `DUNETRACE_API_KEY` to your real API key.

---

## Tools

### `list_agents`

List all monitored agents with their run counts, signal counts, and failure type breakdown.

**No arguments.**

**Example output:**
```
AGENT                                RUNS  SIGS CRIT HIGH  LAST SEEN
───────────────────────────────────────────────────────────────────────────
langfuse-example-agent                 52    48    0   47  12h ago
                                      FIRST_STEP_FAILURE×1, TOOL_LOOP×47
langfuse-ts-example-agent               4     3    0    3  21h ago
                                      TOOL_LOOP×3
langchain-example-agent               134    57    0   48  5d ago
                                      FIRST_STEP_FAILURE×1, TOOL_LOOP×48, STEP_COUNT_INFLATION×8
crewai-example-crew                     5     4    0    0  5d ago
                                      TOOL_AVOIDANCE×4
```

---

### `summarize_agent`

One-shot diagnosis of an agent. Combines health score, failure breakdown, recent signals with their fixes, and health component bars. Start here before diving deeper.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `agent_id` | string | Agent ID (from `list_agents`) |

**Example — `langchain-example-agent`:**
```
═══ Agent summary: langchain-example-agent ═══

Health score:  🔴 48/100
Total runs:    134
Total signals: 57
Last seen:     5d ago

Failure breakdown:
  TOOL_LOOP                             48 signals  (36% of runs)
  STEP_COUNT_INFLATION                   8 signals  (6% of runs)
  FIRST_STEP_FAILURE                     1 signals  (1% of runs)

Most recent signals:
  🟠 TOOL_LOOP  conf=90%  5d ago  run=019e2314…
     The agent called `web_search` 6 times in steps 2–7 with identical
     arguments every time (same args_hash across all calls). It is not
     tracking which queries it has already tried.
     Impact: Looping agents burn tokens and cost money without producing
     value. A 5-step loop at typical gpt-4o pricing costs roughly
     $0.15–$0.30 — with nothing to show for it.
     Fix: Deduplicate `web_search` calls — identical args hash seen 6×

  🟠 TOOL_LOOP  conf=90%  5d ago  run=019e230c…
     [same pattern — web_search called 6× with identical args]

Health components:
  failure_rate         █████░░░░░░░░░░░░░░░  11/40
  loop_avoidance       █████░░░░░░░░░░░░░░░  7/25
  token_efficiency     ███████████████░░░░░  15/20
  latency              ████████████████████  15/15
```

---

### `get_agent_health`

Health score (0–100) and per-component breakdown for an agent.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `agent_id` | string | Agent ID |

**Scoring components:**

| Component | Max points | Measures |
|---|---|---|
| `failure_rate` | 40 | % of runs that triggered any signal |
| `loop_avoidance` | 25 | % of runs without a tool loop |
| `token_efficiency` | 20 | Avg prompt tokens vs. per-agent baseline |
| `latency` | 15 | Avg LLM latency vs. per-agent baseline |

Requires ≥3 runs for a score. Token/latency components return neutral (half points) until ≥30 runs accumulate a baseline.

---

### `get_agent_patterns`

Analyze failure patterns: systemic vs. one-off classification, daily signal trend, failure rates by type, and input hashes that consistently trigger failures.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `agent_id` | string | Agent ID |

**Systemic classification:** a failure marked `SYSTEMIC` has appeared in a high proportion of runs over an extended window. `⚠ Occasional` means isolated incidents.

**Input patterns:** when the same input hash reliably triggers a specific failure type, it appears in this section. Only patterns with ≥50% hit rate are shown.

**Example — `langchain-example-agent`:**
```
Failure patterns for: langchain-example-agent

Systemic patterns (recurring across many runs):
  🚨 SYSTEMIC  TOOL_LOOP  12/16 runs (75%)
            first seen 6d ago  last seen 5d ago

Daily signal counts (last 7 days):
  FAILURE TYPE              04-22  04-29  05-03  05-08  05-12  05-13
  ────────────────────────────────────────────────────────────────────
  TOOL_LOOP                     1      1      1      2      5      7

Failure rate by type (worst single-day rate):
  TOOL_LOOP     ████████████████████  100%  (5/5 runs on 2026-05-12)

Input patterns that reliably trigger failures (rate ≥ 50%):
  hash=e47617d3e1fdaa4f  TOOL_LOOP  40/41 runs (98%)
    → This input hash consistently causes this failure.
  hash=3d338680bc62299d  TOOL_LOOP  4/4 runs (100%)
    → This input hash consistently causes this failure.
  hash=3d338680bc62299d  STEP_COUNT_INFLATION  4/4 runs (100%)
    → This input hash consistently causes this failure.
```

The escalating daily trend (1 → 7) and the 98% input pattern hit rate together indicate this is a deterministic bug, not flaky behaviour — the same query always causes the loop.

---

### `get_agent_runs`

List recent runs for an agent with durations and signal status.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |
| `limit` | int | 20 | Max runs to return (max 100) |

**Example — `langchain-example-agent`:**
```
Recent runs for: langchain-example-agent

RUN ID       STARTED                   DUR STEPS SIGS  STATUS
──────────────────────────────────────────────────────────────────────
019e2314-6b7 5d ago                   4.1s     8  🔴 1
019e2314-53a 5d ago                   2.8s     4  ✅  0
019e2314-018 5d ago                   3.3s     4  ✅  0
019e230c-0c6 5d ago                   5.5s     8  🔴 1
019e230b-e1a 5d ago                   4.9s     4  ✅  0

5 of 134 runs shown.
```

The alternating 🔴 / ✅ pattern here is a tell: runs with 8 steps consistently fail, runs with 4 steps are clean. The loop is adding 4 extra steps every time it fires.

---

### `get_agent_signals`

Recent failure signals for a specific agent, with titles, explanations, and fix suggestions.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |
| `limit` | int | 20 | Max signals to return (max 100) |
| `severity` | string | — | Filter: `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` |

---

### `get_signal_detail`

Full detail for a specific signal: complete evidence dict, impact statement, and all suggested fixes with code snippets.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `signal_id` | int | Integer signal ID (visible in `search_signals` output) |
| `agent_id` | string | Optional — omit to search all agents |

**Example — signal #518 (`langchain-example-agent`):**
```
🟠 Signal #518
Type:      TOOL_LOOP
Severity:  HIGH  confidence=90%
Agent:     langchain-example-agent  v1.0
Run:       019e2314-6b7…
Step:      7
Detected:  5d ago

What happened:
  The agent called `web_search` 6 times in steps 2–7 with identical
  arguments every time (same args_hash across all calls). It is not
  tracking which queries it has already tried.

Why it matters:
  Looping agents burn tokens and cost money without producing value.
  A 5-step loop at typical gpt-4o pricing costs roughly $0.15–$0.30
  — with nothing to show for it. Users waiting on a response will
  time out or give up.

Evidence:
  tool: web_search
  count: 6
  window: 6
  args_identical: True
  first_step: 2
  last_step: 7
  args_hashes: ['ffa8f58f', 'ffa8f58f', 'ffa8f58f', 'ffa8f58f', ...]

Suggested fixes:
  1. Deduplicate `web_search` calls — identical args hash seen 6×
     ```python
     seen_queries = set()
     def web_search(query):
         if query in seen_queries:
             return "Already searched. Try rephrasing."
         seen_queries.add(query)
         return _do_search(query)
     ```
  2. Add to system prompt:
     "Do not repeat a search query you have already tried.
      If a search returned no useful results, reformulate
      the query before trying again."
```

> **Privacy note:** `args_hashes` contains SHA-256 hashes of the original tool arguments — raw arguments never leave your agent process.

---

### `search_signals`

Search signals across all agents with combined filters. Useful for cross-agent audits or time-bounded investigations.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `severity` | string | — | Filter: `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` |
| `failure_type` | string | — | Detector name e.g. `TOOL_LOOP`, `COST_SPIKE`, `CONTEXT_BLOAT` |
| `since_hours` | int | — | Only signals from the last N hours |
| `agent_id` | string | — | Restrict to one agent; searches all agents if omitted |
| `limit` | int | 30 | Max signals to return (max 200) |

**Example — all TOOL_LOOP signals for `langchain-example-agent`:**
```
Signals (3 shown, 48 matched):

🟠     5d ago  [HIGH    ]  TOOL_LOOP          agent=langchain-example-agent
   id=518  run=019e2314-6b7…  conf=90%
   Tool loop detected: `web_search` called 6× in steps 2–7

🟠     5d ago  [HIGH    ]  TOOL_LOOP          agent=langchain-example-agent
   id=496  run=019e230c-0c6…  conf=90%
   Tool loop detected: `web_search` called 6× in steps 2–7

🟠     5d ago  [HIGH    ]  TOOL_LOOP          agent=langchain-example-agent
   id=495  run=019e217d-bd2…  conf=90%
   Tool loop detected: `web_search` called 6× in steps 2–7
```

---

### `get_run_detail`

Full event timeline for a specific run.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `run_id` | string | Run UUID |
| `agent_id` | string | Optional |

**Example — run `019e2314-6b7` (`langchain-example-agent`):**
```
Run: 019e2314-6b7…
Agent:    langchain-example-agent  v1.0
Started:  5d ago
Duration: 4.1s
Steps:    8
Exit:     run.completed

Signals (1):
  🟠 TOOL_LOOP  [HIGH]  conf=90%  step=7
     Tool loop detected: `web_search` called 6× in steps 2–7
     Fix: Deduplicate `web_search` calls — identical args hash seen 6×

Event timeline:
  [  0]  +0.0s  run.started
  [  1]  +0.0s  llm.called        model=gpt-4o-mini  512 tokens  →  820ms
  [  2]  +0.8s  tool.called       tool=web_search  ok=True  195ms
  [  3]  +1.0s  llm.called        model=gpt-4o-mini  612 tokens  →  780ms
  [  4]  +1.8s  tool.called       tool=web_search  ok=True  190ms
  [  5]  +2.0s  llm.called        model=gpt-4o-mini  710 tokens  →  810ms
  [  6]  +2.8s  tool.called       tool=web_search  ok=True  188ms
  [  7]  +3.0s  llm.called        model=gpt-4o-mini  805 tokens  →  800ms
  [  8]  +3.8s  run.completed
```

The prompt token growth across LLM calls (512 → 612 → 710 → 805) is a secondary signal: context is inflating with each redundant search result even though the queries are identical.

---

### `get_agent_token_stats`

Per-window token usage and waste breakdown for an agent: total tokens, wasted tokens (on runs with detected failures), and estimated API cost — for 1d / 7d / 30d windows. The 30-day view also breaks waste down by failure type so you can prioritise which failures to fix first.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `agent_id` | string | Agent ID (from `list_agents`) |

**Example output:**
```
═══ Token stats: langchain-example-agent ═══

── Last 24 h ──
  Runs:               12  (4 with failures)
  Total tokens:     62.4k
  Wasted tokens:    18.1k  (29% of total)
  Total cost:       $0.0624
  Wasted cost:      $0.0181  (29% of total)

── Last 7 days ──
  ...

Waste by failure type (30 days):
  TOOL_LOOP                            150.0k tok     $0.15  (30 runs)
  COST_SPIKE                            75.0k tok     $0.08  (15 runs)
```

---

### `get_instrumentation_guide`

Quick-start code snippet for instrumenting an agent with Dunetrace.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `framework` | string | `langchain`, `python`, `typescript`, `tools`, or `otel` |

Aliases: `langgraph`, `lc`, `ts`, `js`, `node`, `otlp`, `opentelemetry`, `langdock`, `dify`, `tool-calls`, `tracking`.

---

## Worked example: diagnosing `langchain-example-agent`

This is a real diagnostic session using actual data from the agent.

**Step 1 — triage**

```
You:   Is my LangChain agent healthy?

Claude: [calls list_agents]
        langchain-example-agent — 134 runs, 57 signals, health 48/100.
        Dominant failure: TOOL_LOOP (48 signals, 36% of runs).
        Let me dig in.

        [calls summarize_agent("langchain-example-agent")]
        Health is 🔴 48/100. TOOL_LOOP is the main issue — 48 signals
        across 36% of runs, confidence 90% on every one. The agent is
        calling web_search 6 times with identical arguments (steps 2–7)
        without tracking which queries it's already tried.
```

**Step 2 — check if it's systemic**

```
You:   Is it getting worse?

Claude: [calls get_agent_patterns("langchain-example-agent")]
        Yes — it's confirmed SYSTEMIC (75% of runs), and escalating:
        TOOL_LOOP signals went 1 → 1 → 1 → 2 → 5 → 7 over the last 7 days.
        Three specific input hashes trigger it at 98–100% rate — these are
        likely your standard test queries. It's a deterministic bug, not flaky.
```

**Step 3 — look at a specific failing run**

```
You:   Show me a failing run.

Claude: [calls get_agent_runs("langchain-example-agent", limit=5)]
        Most recent failing run: 019e2314-6b7 (4.1s, 8 steps, 1 signal).
        Clean runs have 4 steps. Failing runs always have 8 — the loop
        is adding exactly 4 extra steps every time.

        [calls get_run_detail("019e2314-6b7")]
        Event timeline confirms it: web_search is called at steps 2, 4, 6
        with identical args each time. Prompt tokens also growing 512 → 805
        as the redundant results accumulate in context.
```

**Step 4 — get the fix**

```
You:   What's the fix?

Claude: [calls get_signal_detail(518, "langchain-example-agent")]
        Two options:

        1. Code fix — add dedup in your tool wrapper:
           seen_queries = set()
           def web_search(query):
               if query in seen_queries:
                   return "Already searched. Try rephrasing."
               seen_queries.add(query)
               return _do_search(query)

        2. Prompt addition (faster to ship):
           "Do not repeat a search query you have already tried.
            If a search returned no useful results, reformulate
            the query before trying again."

        Option 2 can be applied via Langfuse in one click from the dashboard.
```

---

## Typical workflows

### Investigate a run from a Slack alert

The Slack alert includes a "View Run" button linking to the dashboard. You can also pass the run ID directly:

```
You:   Check run 019e2314-6b7

Claude: [calls get_run_detail("019e2314-6b7")]
        4.1s, 8 steps. TOOL_LOOP at step 7 — web_search called 6×
        with identical args. Context growing 512→805 tokens per call.
        Fix: add a dedup set or prompt instruction.
```

### Cross-agent audit

```
You:   Which agents had issues in the last 24 hours?

Claude: [calls list_agents]
        langfuse-example-agent — 47 HIGH signals (TOOL_LOOP), last seen 12h ago.
        langfuse-ts-example-agent — 3 HIGH signals (TOOL_LOOP), last seen 21h ago.
        Both are looping on web_search. Likely the same root cause — want me
        to compare?
```

### Before a deploy

```
You:   Is langchain-example-agent stable enough to deploy?

Claude: [calls get_agent_patterns("langchain-example-agent")]
        No — TOOL_LOOP is systemic (75% of runs) and escalating daily.
        Three input hashes trigger it at 98–100%. Ship the dedup fix first.
```

### How to instrument a new agent

```
You:   How do I add Dunetrace to my LangChain agent?

Claude: [calls get_instrumentation_guide("langchain")]
        pip install 'dunetrace[langchain]'

        from dunetrace import Dunetrace
        from dunetrace.integrations.langchain import DunetraceCallbackHandler

        dt = Dunetrace(endpoint="http://localhost:8001")
        callback = DunetraceCallbackHandler(dt, agent_id="my-agent",
                                            model="gpt-4o-mini",
                                            tools=["web_search"])
        agent.invoke(input, config={"callbacks": [callback]})
        dt.shutdown()
```

---

## Privacy

All data served by the MCP tools comes from the Dunetrace Customer API, which stores only hashed or structural metadata:

- Tool arguments → SHA-256 hash (shown as `args_hashes`)
- LLM prompts and outputs → SHA-256 hash (never stored)
- Token counts, latency, step counts → stored as plain numbers
- Run and signal metadata → stored as plain text

The `evidence` dict in signal responses contains the hashed fingerprints the detector used — not the original content.

---

## Troubleshooting

### `starlette` conflict with `fastapi`

```
ERROR: fastapi 0.115.x requires starlette<0.47.0, but you have starlette 1.0.0
```

The `mcp` package pulls in `starlette 1.0.0` (released 2025). FastAPI 0.115 and earlier capped starlette below that. FastAPI 0.136+ removed the upper bound and is fully compatible.

**Fix:**
```bash
pip install --upgrade fastapi
```

---

## Tests

```bash
cd packages/mcp-server
python -m pytest tests/ -v
```

105 tests, all offline — no running stack required. Tests cover every tool including edge cases: unknown IDs, empty results, `None` health scores, time-window filters, evidence truncation, code snippet truncation, all instrumentation guide aliases, and resource registration.

---

## Source

`packages/mcp-server/`

```
dunetrace_mcp/
  __init__.py
  client.py      # thin httpx wrapper around the Customer API
  server.py      # FastMCP server with 10 tools + 7 doc resources
tests/
  test_tools.py  # 105 unit tests (all offline)
pyproject.toml
```
