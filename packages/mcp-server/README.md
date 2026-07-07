# Dunetrace MCP Server

Query agent signals, run details, and health scores directly from Claude Code, Cursor, Codex, or any MCP-compatible client — without leaving your editor. Read-only for signals/runs; write operations (create/update/delete) are available for policies and custom detectors.

Setup (install, client config for Claude Code / Cursor / Codex, environment variables) is covered in **[docs/mcp-server.md](../../docs/mcp-server.md)** — the quick version:

```bash
pip install dunetrace-mcp
```

```json
{
  "mcpServers": {
    "dunetrace": {
      "command": "dunetrace-mcp",
      "env": { "DUNETRACE_API_URL": "http://localhost:8002", "DUNETRACE_API_KEY": "dt_dev_test" }
    }
  }
}
```

---

## Tools

### `list_agents`

List all monitored agents with their run counts, signal counts, and failure type breakdown.

**No arguments.**

**Example output:**
```
AGENT                                RUNS  SIGS CRIT HIGH  LAST SEEN
───────────────────────────────────────────────────────────────────────────────
research-agent                        129    55    0   46  6h ago
                                      TOOL_LOOP×46, STEP_COUNT_INFLATION×8
billing-agent                          36    34    0   33  10h ago
                                      TOOL_LOOP×33
```

---

### `get_agent_signals`

Get recent failure signals for a specific agent, with titles, explanations, and top fix suggestion.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID (from `list_agents`) |
| `limit` | int | 20 | Max signals to return (max 100) |
| `severity` | string | — | Filter: `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` |

**Example:**
```
🟠 [HIGH] TOOL_LOOP  conf=90%  step=7  6h ago
   Tool loop detected: `web_search` called 6× in steps 2–7
   What: The agent called web_search 6 times with identical args.
   Fix:  Deduplicate `web_search` calls — identical arguments seen 6×
```

---

### `get_signal_detail`

Full detail for a specific signal: complete evidence dict, impact statement, and all suggested fixes with code snippets.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `signal_id` | int | required | Integer signal ID (visible in `search_signals` output) |
| `agent_id` | string | — | Agent ID (optional — omit to search all agents) |

**Example output:**
```
🟠 Signal #495
Type:      TOOL_LOOP
Severity:  HIGH  confidence=90%
Agent:     research-agent  vabcd1234
Run:       019e217d-bd24-…
Step:      7
Detected:  2026-05-13 13:19 UTC  (6h ago)

What happened:
  The agent called `web_search` 6 times in steps 2–7 with identical
  arguments every time. It is not tracking which queries it has tried.

Why it matters:
  Looping agents burn tokens without producing value. A 5-step loop at
  typical gpt-4o pricing costs $0.15–$0.30 with nothing to show for it.

Evidence:
  tool: web_search
  count: 6
  args_identical: True
  args: ['{"query": "LLM benchmarks"}', '{"query": "LLM benchmarks"}', …+4 more]

Suggested fixes (2):
  1. Deduplicate `web_search` calls — identical arguments seen 6×
     ```python
     seen = set()
     if args not in seen:
         seen.add(args)
         call_tool(args)
     ```
  2. Set a hard step limit as a circuit breaker
```

---

### `get_agent_health`

Health score (0–100) and per-component breakdown for an agent.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |

**Scoring components:**

| Component | Max points | Measures |
|---|---|---|
| `failure_rate` | 40 | % of runs that triggered any signal |
| `loop_avoidance` | 25 | % of runs without a tool loop |
| `token_efficiency` | 20 | Avg prompt tokens vs. per-agent baseline |
| `latency` | 15 | Avg LLM latency vs. per-agent baseline |

Requires ≥3 runs for a score. Token/latency components return neutral (half points) until ≥30 runs accumulate a baseline.

**Example output:**
```
🔴 Health score for research-agent: 41/100
   Sample runs:     24
   Baseline ready:  no (need ≥30 runs for token/latency)

Component breakdown:
  failure_rate          7/40  (current: 83.3 % runs with failures)
  loop_avoidance        4/25  (current: 83.3 % runs with loops)
  token_efficiency     15/20
  latency              15/15  (current: 3005.0 avg LLM latency ms)
```

---

### `get_run_detail`

Full detail for a specific run: metadata, detected signals with fixes, and a step-by-step event timeline.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `run_id` | string | required | Run UUID |
| `agent_id` | string | — | Optional — not used for the lookup, reserved for future use |

**Example output:**
```
Run: 019e217d-bd24-7d72-a8be-4715c2dcf385
Agent:    research-agent  vabcd1234
Started:  2026-05-13 13:19 UTC  (6h ago)
Duration: 5.6s
Steps:    8
Exit:     run.completed

Signals (1):
  🟠 TOOL_LOOP  [HIGH]  conf=90%  step=7
     Tool loop detected: `web_search` called 6× in steps 2–7
     Fix: Deduplicate `web_search` calls — identical arguments seen 6×

Event timeline (18 events):
  [  0]    +0.0s  run.started
  [  1]    +0.0s  llm.called           model=gpt-4o-mini  p=512 c=98  800ms
  [  2]    +2.8s  tool.called          tool=web_search  ok=True  200ms
  [  3]    +2.8s  tool.called          tool=web_search  ok=True  200ms
  …
  [  8]    +3.1s  run.completed        final_answer
```

Event timeline is capped at 40 entries; longer runs show a count of remaining events.

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

**Example:**
```python
# All CRITICAL signals in the past 24 hours
search_signals(severity="CRITICAL", since_hours=24)

# All TOOL_LOOP signals for one agent
search_signals(failure_type="TOOL_LOOP", agent_id="research-agent")
```

**Example output:**
```
Signals (3 shown, 6 matched):

🟠     6h ago  [HIGH    ]  TOOL_LOOP                       agent=research-agent
   id=495  run=019e217d-bd2…  conf=90%
   Tool loop detected: `web_search` called 6× in steps 2–7
```

---

### `get_agent_patterns`

Analyze failure patterns for an agent: systemic vs. one-off classification, daily signal trend, failure rates by type, and input hashes that consistently trigger failures.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |

**Systemic classification:** a failure is marked `SYSTEMIC` when it has appeared in a high proportion of runs over an extended window. A `⚠ Occasional` label means isolated incidents.

**Input patterns:** when the same input hash (a structural fingerprint of the user query) reliably triggers a specific failure type, it appears in the "Input patterns" section. Only patterns with a hit rate ≥50% are shown — lower rates are noise.

**Example output:**
```
Failure patterns for: research-agent

Systemic patterns:
  🚨 SYSTEMIC  TOOL_LOOP  12/12 runs (100%)
            first seen 5d ago  last seen 6h ago

Daily signal counts (last 7 days):
  FAILURE TYPE                    05-07  05-08  05-09  05-12  05-13
  ─────────────────────────────────────────────────────────────────
  TOOL_LOOP                           1      2      1      5      5

Failure rate by type:
  TOOL_LOOP     ████████████████████  100%  (5/5 runs on 2026-05-13)

Input patterns that reliably trigger failures (rate ≥ 50%):
  hash=e47617d3  TOOL_LOOP  38/39 runs (97%)
    → This input hash consistently causes this failure.
```

---

### `summarize_agent`

One-shot diagnosis of an agent. Combines health score, failure breakdown, recent signals with their fixes, and health component bars. Start here before diving deeper.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |

**Example output:**
```
═══ Agent summary: research-agent ═══

Health score:  🔴 41/100
Total runs:    129
Total signals: 55
Last seen:     6h ago

Failure breakdown:
  TOOL_LOOP                             46 signals  (36% of runs)
  STEP_COUNT_INFLATION                   8 signals  (6% of runs)

Most recent signals:
  🟠 TOOL_LOOP  conf=90%  6h ago  run=019e217d…
     The agent called `web_search` 6 times with identical args.
     Impact: Looping agents burn tokens without producing value.
     Fix: Deduplicate `web_search` calls — identical arguments seen 6×

Health components:
  failure_rate         ███░░░░░░░░░░░░░░░░░  7/40
  loop_avoidance       ███░░░░░░░░░░░░░░░░░  4/25
  token_efficiency     ███████████████░░░░░  15/20
  latency              ████████████████████  15/15
```

---

### `get_agent_runs`

List recent runs for an agent with durations and signal status.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |
| `limit` | int | 20 | Max runs to return (max 100) |

**Example output:**
```
Recent runs for: research-agent

RUN ID       STARTED                  DUR  STEPS SIGS  STATUS
──────────────────────────────────────────────────────────────────────
019e217d-bd2 6h ago                   5.6s     8  🔴 1
019e2163-a89 6h ago                   4.7s     8  🔴 1
019e2163-66f 6h ago                   4.8s     4  ✅  0
```

---

### `get_agent_token_stats`

Per-window token usage and waste breakdown for an agent (1d / 7d / 30d). Shows total tokens consumed, wasted tokens (on runs with at least one live failure signal), and estimated API cost for each window. The 30-day view adds a `Waste by failure type` breakdown so you can see which failure types cost the most to leave unfixed.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID (from `list_agents`) |

**Example output:**
```
═══ Token stats: research-agent ═══

── Last 24 h ──
  Runs:               10  (3 with failures)
  Total tokens:     50.0k
  Wasted tokens:    15.0k  (30% of total)
  Total cost:       $0.0500
  Wasted cost:      $0.0150  (30% of total)

── Last 7 days ──
  Runs:               50  (12 with failures)
  Total tokens:    250.0k
  Wasted tokens:    60.0k  (24% of total)
  Total cost:       $0.2500
  Wasted cost:      $0.0600  (24% of total)

Waste by failure type (30 days):
  TOOL_LOOP                            150.0k tok     $0.15  (30 runs)
  COST_SPIKE                            75.0k tok     $0.08  (15 runs)
```

---

### `get_instrumentation_guide`

Get a quick-start code snippet for instrumenting an agent with Dunetrace. Works for Python, LangChain/LangGraph, TypeScript, and plain tool-call tracking.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `framework` | string | required | Framework name: `python`, `langchain`, `langgraph`, `typescript`, or `tools` |

Aliases accepted: `lc`, `lc-graph`, `lc_graph`, `langgraph`, `ts`, `js`, `javascript`, `node`, `tracking`, `tool_calls` (and more).

---

### `get_fix_status`

Check whether a fix applied for a signal reduced recurrence.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `signal_id` | int | required | The signal ID to check |
| `agent_id` | string | — | Optional agent ID |

Verdict values: `verified`, `likely_fixed`, `still_occurring`, `insufficient_data`.

---

### `list_agent_fixes`

List all fixes that have been applied for an agent's signals, with type, delivery method, and verdict.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |

---

### `trigger_explain`

Trigger root-cause analysis for a signal, built natively from the run's own stored events — no external tracing system required. Returns a `fix_category`: `dunetrace_native` (a runtime policy Dunetrace can apply directly) or `customer_code` (a prompt/code diff you apply yourself, or via a GitHub PR for `code_change` fixes).

**Note:** Makes an LLM call — may take 5–15 seconds.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `signal_id` | int | required | Signal ID to analyze |
| `agent_id` | string | — | Optional agent ID |

---

### `list_policies`

List runtime policies configured for agents. Policies trigger actions (stop, switch_model, inject_prompt, log) when a metric threshold is crossed during a live run.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | — | Filter to one agent (optional) |

---

### `create_policy`

Create a runtime policy.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Policy label |
| `agent_id` | string | required | Target agent (`*` for all) |
| `condition` | string | required | JSON: `{"metric": "cost_usd", "operator": "gt", "threshold": 5.0}` |
| `action` | string | required | JSON: `{"type": "stop"}` or `{"type": "switch_model", "model": "gpt-4o-mini"}` |

---

### `toggle_policy`

Enable or disable a runtime policy.

**Arguments:** `policy_id` (string), `enabled` (bool)

---

### `delete_policy`

Delete a runtime policy.

**Arguments:** `policy_id` (string)

---

### `list_custom_detectors`

List custom detectors with status, fire rate, and run counts.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | — | Filter to one agent (optional) |

---

### `create_custom_detector`

Create a custom detector from a plain-English description. Translates the description to a structured config via LLM, then creates the detector in shadow mode.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `description` | string | required | Plain-English description, e.g. "Alert when tool calls exceed 20 in a run" |
| `agent_id` | string | `*` | Target agent |

---

### `activate_custom_detector`

Activate a custom detector so it fires live alerts.

**Arguments:** `detector_id` (string)

---

### `pause_custom_detector`

Pause a custom detector so it stops evaluating.

**Arguments:** `detector_id` (string)

---

### `delete_custom_detector`

Delete a custom detector permanently (also deletes historical results).

**Arguments:** `detector_id` (string)

---

### `list_agent_issues`

List open or resolved issues for an agent. Issues are aggregated across runs — the same failure type appearing in multiple runs is tracked as a single issue, and auto-resolves after 5 consecutive clean runs.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |
| `status` | string | `open` | `open`, `resolved`, or `all` |

---

### `get_failure_pattern_detail`

Deep dive into a specific failure type: evidence aggregates, a 14-day trend with ASCII bar chart, co-occurring signals, and top example runs.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `agent_id` | string | required | Agent ID |
| `failure_type` | string | required | Detector name, e.g. `TOOL_LOOP` |

---

### `compare_runs`

Compare two runs side by side — duration, step count, token usage, signals detected, and exit reason. Useful for spotting regressions between a good run and a bad one.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `run_id_1` | string | required | First run UUID |
| `run_id_2` | string | required | Second run UUID |
| `agent_id` | string | — | Optional agent ID |

**Example output:**
```
Run comparison

                       RUN 1                      RUN 2
──────────────────────────────────────────────────────────────────────────
Run ID                 run-aaa-111                run-bbb-222
Agent                  my-agent  v1.0             my-agent  v1.1
Duration               10.0s                      20.0s                  !!
Steps                  5                          12                     !!
Tokens                 1,000                      8,000                  !!
Exit reason            final_answer               error                  !!
Signals                none                       TOOL_LOOP [HIGH]       !!

!! = differs between runs
```

---

## Typical workflows

### Triage an alert

```
You:   I got a Slack alert for TOOL_LOOP on research-agent. What's happening?

Agent: [calls summarize_agent("research-agent")]
       Health is 41/100. TOOL_LOOP is systemic — 46 signals across 36% of
       runs. The fix is to deduplicate web_search calls (identical arguments
       seen 6× per run). Signal #495 is the most recent. Want the code?

You:   Yes, show me signal #495.

Agent: [calls get_signal_detail(495, "research-agent")]
       Here's the evidence and fix code…
```

### Investigate a specific run from Slack

The Slack alert includes a "View Run" link: `http://localhost:3000/runs/<run_id>`. You can also pass the run ID directly:

```
You:   Check run 019e217d-bd24-7d72-a8be-4715c2dcf385

Agent: [calls get_run_detail("019e217d-…")]
       Duration 5.6s, 8 steps. TOOL_LOOP at step 7 — web_search called
       6× with identical args. Fix: add a dedup set.
```

### Cross-agent audit

```
You:   Are there any CRITICAL signals in the last 24 hours?

Agent: [calls search_signals(severity="CRITICAL", since_hours=24)]
       2 CRITICAL signals: PROMPT_INJECTION_SIGNAL on billing-agent (2h ago)
       and COST_SPIKE on data-agent (5h ago). Want details on either?
```

### Before a deploy

```
You:   Is research-agent stable enough to deploy to production?

Agent: [calls get_agent_patterns("research-agent")]
       TOOL_LOOP is systemic — 100% of runs in the last 7 days.
       Recommending you fix the dedup issue before deploying.
```

---

## Data Handling

All data served by the MCP tools comes from the Dunetrace Customer API:

- Tool arguments, LLM prompts and outputs → stored and returned as-is (shown as `args`, `output`, etc. in evidence)
- Token counts, latency, step counts → stored as plain numbers
- Run and signal metadata → stored as plain text

The `evidence` dict in signal responses contains the actual content the detector used, not a hash of it.

---

## Tests

```bash
cd packages/mcp-server
python -m pytest tests/ -v
```

154 tests, all offline — no running stack required.

---

## Source

`packages/mcp-server/`

```
dunetrace_mcp/
  __init__.py
  client.py      # thin httpx wrapper around the Customer API
  server.py      # FastMCP server with 26 tools + 6 doc resources
tests/
  test_tools.py  # 154 unit tests (all offline)
pyproject.toml
README.md
```
