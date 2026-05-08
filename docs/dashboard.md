# Dashboard

The Mission Control dashboard is a live, API-driven single-page app served at **[http://localhost:3000](http://localhost:3000)**. Auto-refreshes every 15 seconds. No build step — static HTML fetching from the Customer API.

---

## Navigation

### Global search
A search bar in the topbar matches run IDs, agent names, and failure types across all loaded data. Results are tagged **RUN** / **AGENT** / **FAILURE** and clicking any result navigates directly to the relevant page or run detail.

### Time range selector
Four buttons in the topbar — **1D / 7D / 30D / All** — set a global time window that filters the Alerts, Analytics, and Heatmap pages simultaneously. Defaults to 7D.

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
| **Agents** | Per-agent health cards sortable by **Health score / Fail rate / Last seen**. Each card shows failure rate %, dominant pattern, run / critical / high counts, last seen, ungraduated shadow signal count, and an **Agent Health Score** badge (0–100, colour-coded green/amber/red). Score is `null` until 3+ runs are available. Each card links to a **Health Record** panel showing failure rate per failure type over 30 days with a `SYSTEMIC` badge and a **Deploy Timeline** chart showing detector failure rates with blue dashed deploy markers. Clicking any failure type in the sidebar opens the **Why is this happening?** deep-dive panel. |
| **Compare Runs** | Side-by-side run comparison with colour-coded delta table (new / resolved failure types highlighted). |
| **Detectors** | Threshold sliders and alert level selector. Live review panel recomputes on every change. |
| **Policies** | Create, edit, toggle, and delete runtime guardrails. Policies are fetched automatically by the SDK within 60 seconds. |
| **Patterns** | Signal frequency heatmap — agent × detector over 7 days. `TRENDING ↑` badge when last-3-day avg exceeds first-3-day avg × 1.3. Click any cell to drill into runs for that agent+detector pair. |

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

- **Analysis** — execution timeline (one node per step, loop detection), signal score cards with confidence bars, plain-English explanation + suggested fix. Failure type labels show a plain-English tooltip on hover. When multiple signals fire, each card shows an amber **"confidence boosted · N co-occurring signals"** badge. An "**Explain with Langfuse**" button appears when `LANGFUSE_PUBLIC_KEY` is configured.
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
