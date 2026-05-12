# Dashboard

The Mission Control dashboard is a live, API-driven single-page app served at **[http://localhost:3000](http://localhost:3000)**. Auto-refreshes every 15 seconds. No build step — static HTML fetching from the Customer API.

---

## Navigation

### Global search
A search bar in the topbar matches run IDs, agent names, and failure types across all loaded data. Results are tagged **RUN** / **AGENT** / **FAILURE** and clicking any result navigates directly to the relevant page or run detail.

### Time range selector
**1D / 7D / 30D / All** buttons appear in the header of the **Alerts**, **Analytics**, and **Heatmap** pages. Selecting a period on any of these pages updates all three simultaneously. Defaults to 7D.

### Back navigation
The run detail panel shows a dynamic back button ("← back to alerts", "← back to agents", etc.) that returns to whichever page opened the detail.

---

## Pages

| Page | What it shows |
|---|---|
| **Overview** | Four stat cards (Critical / High / Signals / Runs) with configurable trend deltas (↑↓ vs last hour / yesterday / last week). Risk Trend 24h bar chart (hourly signal count, colour-coded by intensity). Token Waste Drift panel — 24h sparkline of wasted tokens vs 7-day baseline, dashed baseline line, WARNING ZONE badge. Failure Posture gauge — half-circle SVG with needle at avg confidence, daily signals, avg confidence, false positive rate. Top Failure Drivers — ranked by signal count, shows failure type · estimated wasted tokens · confidence · severity. Clicking any failure type navigates to All Runs filtered to the agent most affected by that pattern. Agent Signal Drift — horizontal bars per agent showing 24h signal rate vs 7-day baseline, red/amber/green. Top failure patterns with ↑↓ trend arrows. Live run feed. |
| **All Runs** | Full run table with composable filters: **Day / Week / Month** time range, free-text search (run ID or agent name), severity dropdown, and agent dropdown. Click any row to open run detail. |
| **Alerts** | Signals grouped by failure type, filtered by the global time range. Each group header shows run count, estimated wasted tokens, and a **dismiss** button that hides the group (persisted to `localStorage`). Dismissed groups show a count and a "restore all" link. Shadow signals rendered below with dashed border + SHADOW badge. |
| **Analytics** | Estimated token cost saved this week (configurable $/1k rate). Cross-agent totals. Top failure patterns with per-type token waste. Per-agent breakdown with estimated wasted tokens. All figures respect the global time range. |
| **Risk Heatmap** | Failure type × agent intensity grid. Counts respect the global time range. |
| **Agents** | Per-agent health cards sortable by **Health score / Fail rate / Last seen**. Each card shows failure rate %, dominant pattern, run / critical / high counts, last seen, ungraduated shadow signal count, and an **Agent Health Score** badge (0–100, colour-coded green/amber/red). Score is `null` until 3+ runs are available. While fewer than 30 runs exist, the card shows *"baseline calibrating (N/30 runs)"* — token and latency components are held at neutral until a baseline can be established. Each card links to a **Health Record** panel showing failure rate per failure type over 30 days with a `SYSTEMIC` badge and a **Deploy Timeline** chart showing detector failure rates with blue dashed deploy markers. Clicking any failure type in the sidebar opens the **Why is this happening?** deep-dive panel. |
| **Compare Runs** | Side-by-side run comparison with colour-coded delta table (new / resolved failure types highlighted). |
| **Detectors** | Threshold sliders and alert level selector. Live review panel recomputes on every change. |
| **Policies** | Create, edit, toggle, and delete runtime guardrails. Policies are fetched automatically by the SDK within 60 seconds. |
| **Patterns** | Signal frequency heatmap — agent × detector over 7 days. `TRENDING ↑` badge when last-3-day avg exceeds first-3-day avg × 1.3. Click any cell to drill into runs for that agent+detector pair. |

---

## Agent Health Score

A 0–100 composite score computed from the last 30 days of run data. Powered by `GET /v1/agents/{id}/health-score`.

| Component | Weight | Active from | How it's scored |
|---|---|---|---|
| **Failure rate** | 40 pts | Run 1 | `(1 − failure_rate) × 40`. Zero failures = 40 pts. |
| **Loop avoidance** | 25 pts | Run 1 | `(1 − loop_rate) × 25`. Loop detectors: `TOOL_LOOP`, `TOOL_THRASHING`, `TOOL_AVOIDANCE`, `STEP_COUNT_INFLATION`, `LLM_TRUNCATION_LOOP`. |
| **Token efficiency** | 20 pts | 30 runs | Scored relative to the agent's own P75 baseline (see below). Neutral (15 pts) until baseline is ready. |
| **Latency** | 15 pts | 30 runs | Scored relative to the agent's own P75 baseline. Neutral (10 pts) until baseline is ready. |

**Per-agent baselines (token efficiency and latency)**

Rather than global absolute thresholds, each agent is measured against its own historical normal. The baseline is the P75 of per-run averages computed from the **30–90 day window** — deliberately excluding the most recent 30 days so the reference is independent of the period being scored. This prevents chronic degradation from inflating the baseline (boiling frog problem).

Scoring zones once baseline is ready:

- `avg ≤ 1.5 × P75` → full score (healthy operating range)
- `1.5 × P75 < avg < 4 × P75` → linear decay
- `avg ≥ 4 × P75` → 0 pts

A research agent whose P75 token usage is 2,000 scores full marks up to 3,000 tokens per call. A support bot with a 400-token P75 scores full up to 600. Same formula, different reference point.

**Baseline readiness**

Token and latency components are held at neutral (15 and 10 pts respectively) only when no LLM event data has been recorded at all (i.e. `avg_tokens` / `avg_latency` is `null`). Until the agent has 30 runs, the card shows: *"Health score based on detector signals only — baseline calibrating (N/30 runs)."*

When an agent has LLM data but no pre-window baseline — either because it has fewer than 30 runs (young agent) or because the 30–90 day window is empty — the global absolute thresholds are used instead of neutral: token < 500 = full score, > 4,000 = 0; latency < 1,000ms = full score, > 8,000ms = 0. This ensures a new agent running at 7,000ms latency is flagged rather than silently scored neutral.

The per-agent P75 baseline additionally requires at least 20 runs with LLM event data in the 30–90 day window before it is used. Once available, it supersedes the global absolutes.

---

## Why is this happening?

Clicking any failure type in the **Signal Breakdown** or **Systemic Patterns** sidebar opens a cross-run deep-dive panel inline.

| Section | What it answers |
|---|---|
| **Overview** | Affected runs / total runs, rate, avg confidence, severity breakdown, first and last seen |
| **Fires at step** | P25 / P50 / P75 / avg step index — "does this happen early or late in runs?" |
| **Evidence patterns** | Aggregated evidence fields from detector output |
| **Co-occurs with** | Other failure types firing in the same runs, ranked by co-occurrence rate |
| **14-day trend** | Daily sparkline of affected_runs / rate |
| **Highest confidence runs** | Five example runs, each clickable to open run detail |

Powered by `GET /v1/agents/{agent_id}/failure-patterns/{failure_type}`.

---

## Run detail

Click any run row to open the detail panel. Three tabs:

- **Analysis** — execution timeline (one node per step, loop detection), signal score cards with confidence bars, plain-English explanation + suggested fix. Failure type labels show a plain-English tooltip on hover. When multiple signals fire, each card shows an amber **"confidence boosted · N co-occurring signals"** badge. An "**Explain with Langfuse**" button appears when `LANGFUSE_PUBLIC_KEY` is configured. After explaining:
  - **Prompt-fix signals** (TOOL_LOOP, GOAL_ABANDONMENT, etc.) — **Apply via Langfuse** button pushes a new prompt version directly.
  - **Code-change signals** (CONTEXT_BLOAT, SLOW_STEP, etc.) — **Open PR on GitHub ↗** button creates a draft PR in your repo containing the LLM-generated unified diff. Requires `GITHUB_TOKEN` and `GITHUB_REPO` in `.env`.
- **Run graph** — SVG node graph: green = LLM call, orange = tool call (ok), red = looping tool call, blue = start/end.
- **Event log** — every event in chronological order, expandable to show full payload. A filter input narrows by step index or event type (e.g. `tool.called`). A **copy ID** button in the header copies the full run ID to the clipboard.

---

## Failure type tooltips

Every failure type label in the dashboard (Alerts, All Runs, run detail, heatmap, agents) shows a plain-English description on hover:

| Type | Description |
|---|---|
| `TOOL_LOOP` | Same tool called repeatedly with identical args — agent stuck in a loop |
| `CONTEXT_BLOAT` | Token count growing unsustainably — context window near limit |
| `RETRY_STORM` | Tool failure followed by repeated retries — hammering a broken tool |
| `CASCADING_TOOL_FAILURE` | Multiple tools failing in sequence — systemic tool layer issue |
| `GOAL_ABANDONMENT` | Agent stopped using tools before task completion |
| `PROMPT_INJECTION_SIGNAL` | Input matched injection patterns — potential adversarial content |
| `LLM_TRUNCATION_LOOP` | LLM output repeatedly truncated — hitting output length limits |
| `EMPTY_LLM_RESPONSE` | LLM returned empty output — model response failure |
| `SLOW_STEP` | Step latency exceeds threshold — performance degradation |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned no results — knowledge base gap |
| `STEP_COUNT_INFLATION` | Far more steps than baseline — inefficiency or runaway loop |
| `FIRST_STEP_FAILURE` | Failed on the very first step — likely config or setup issue |

---

## Signal acknowledgment

Each alert group has a **dismiss** button. Dismissed groups are hidden from the Alerts page and stored in `localStorage` so they survive a page refresh. To restore all dismissed groups, click **restore all** in the Alerts header. Dismissal is local to the browser — it does not affect the database or other users.

---

## Stat card info buttons

Each stat card has an `ⓘ` button explaining the threshold for that severity level:

| Card | Threshold |
|---|---|
| Critical | conf ≥ 0.85, or prompt injection / cascading failure regardless of confidence |
| High | conf ≥ 0.70 — tool loops, retry storms, context bloat |
| Signals | All four levels: CRITICAL ≥ 0.85 · HIGH ≥ 0.70 · MEDIUM ≥ 0.50 · LOW < 0.50 |
| Total runs | Processed runs (clean + signal-bearing) counted within one 5s detector poll |

---

## Token waste estimates

Token waste figures across the dashboard are computed client-side from run `step_count` using a fixed estimate of **250 tokens per step**. Dollar costs use a configurable rate — default $0.010/1k tokens, editable on the Analytics page.

These are approximations. Actual token counts depend on model, prompt length, and output verbosity.

---

## Shadow signals

Shadow signals are stored but not alerted — they are detectors in evaluation mode. The dashboard fetches all signals with `?include_shadow=true` and renders them separately:

- **Alerts page** — shadow signals appear below live alerts with a dashed border, reduced opacity, and a `SHADOW` badge.
- **Agents page** — each agent card shows an ungraduated shadow signal count.

To graduate a detector from shadow to live, add it to `LIVE_DETECTORS` in `services/detector/detector_svc/db.py`. → [docs/detectors.md#shadow-mode](detectors.md#shadow-mode)

---

## Data sources

All data is computed client-side from the Customer API. No server-side rendering.

| Page | API calls |
|---|---|
| Overview, Alerts, Analytics, Heatmap, Agents | `GET /v1/agents` + per-agent `/runs` + `/signals?include_shadow=true` |
| All Runs, Compare Runs | Same cached data, no extra calls |
| Run detail | `GET /v1/runs/{id}` (events + signals) |
| Agent view (health record + deploy timeline + runs) | `GET /v1/agents/{id}/runs` + `/signals` + `/insights` (includes `deploy_events`) + `/health-score` |
| Why is this happening? panel | `GET /v1/agents/{id}/failure-patterns/{failure_type}` |
| Detectors | Static — edits require updating `detectors.yml` and restarting the detector service |
| Policies | `GET /v1/policies` + `POST` + `PUT /{id}` + `DELETE /{id}` + `PATCH /{id}/toggle` |
| Patterns | `GET /v1/patterns` |
