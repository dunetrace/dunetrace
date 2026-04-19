# Dashboard

The Mission Control dashboard is a live, API-driven single-page app served at **[http://localhost:3000](http://localhost:3000)**. It auto-refreshes every 15 seconds and requires no build step — static HTML fetching from the Customer API.

---

## Pages

| Page | What it shows |
|---|---|
| **Overview** | Four stat cards (Critical / High / Signals / Runs) with configurable trend deltas (↑↓ vs last hour / yesterday / last week). Risk Trend 24h bar chart (hourly signal count, colour-coded by intensity). Token Waste Drift panel — 24h sparkline of wasted tokens vs 7-day baseline, dashed baseline line, WARNING ZONE badge. Failure Posture gauge — half-circle SVG with needle at avg confidence, daily signals, avg confidence, false positive rate. Top Failure Drivers — ranked by signal count, shows agent · failure type · estimated wasted tokens · confidence · severity. Agent Signal Drift — horizontal bars per agent showing 24h signal rate vs 7-day baseline, red/amber/green. Top failure patterns with ↑↓ trend arrows. Live run feed. |
| **All Runs** | Full run table — agent, signals, severity, failure types, duration, step count. Click any row to open run detail. |
| **Alerts** | Signals grouped by failure type. Each group header shows run count and estimated wasted tokens. Expandable per group with per-run confidence and token estimates. Shadow signals rendered below with dashed border + SHADOW badge. |
| **Analytics** | Estimated token cost saved this week (configurable $/1k rate). Cross-agent totals. Top failure patterns with per-type token waste. Per-agent breakdown with estimated wasted tokens. |
| **Risk Heatmap** | Failure type × agent intensity grid. |
| **Agents** | Per-agent health cards — failure rate %, dominant pattern, run / critical / high counts, last seen, ungraduated shadow signal count. Each agent card links to a **Health Record** panel showing failure rate per failure type over 30 days: a sparkline of daily rate, a `SYSTEMIC` badge when ≥10% of runs in the last 7 days were affected, and a 7-day affected/total count. Powered by `GET /v1/agents/{id}/insights` (`failure_rates` + `systemic_patterns`). Clicking any failure type in the sidebar opens the **Why is this happening?** deep-dive panel (see below). |
| **Compare Runs** | Side-by-side run comparison. Select any two runs from dropdowns — metrics, signals, and max confidence shown in both panels with a colour-coded delta table (new / resolved failure types highlighted). |
| **Detectors** | Threshold sliders and alert level selector. Live review panel: "with current config, N of M past runs would be flagged HIGH or above (N% of runs)" — recomputes on every change. |

---

## Why is this happening?

Clicking any failure type in the **Signal Breakdown** or **Systemic Patterns** sidebar opens a cross-run deep-dive panel inline. Click the same item again or ✕ to dismiss.

The panel shows six sections for the selected failure type:

| Section | What it answers |
|---|---|
| **Overview** | Affected runs / total runs, rate, avg confidence, severity breakdown (CRITICAL / HIGH / MEDIUM / LOW), first and last seen |
| **Fires at step** | P25 / P50 / P75 / avg step index where the failure fires — answers "does this happen early or late in runs?" |
| **Evidence patterns** | Aggregated evidence fields from detector output: which tool loops most, avg call count, same-args rate, token growth factor, avg step duration vs threshold, avg RAG top score, avg stall steps, inflation ratio |
| **Co-occurs with** | Other failure types that fire in the same runs, ranked by co-occurrence rate — surfaces systemic failure clusters |
| **14-day trend** | Daily sparkline of affected_runs / rate — answers "is this getting worse, better, or stable?" |
| **Highest confidence runs** | Five example runs with the highest confidence for this failure type — each row is clickable and opens the run detail panel |

Powered by `GET /v1/agents/{agent_id}/failure-patterns/{failure_type}`.

---

## Run detail

Click any run row to open the detail panel. Three tabs:

- **Analysis** — execution timeline (one node per step, loop detection), signal score cards with confidence bars, plain-English explanation + suggested fix
- **Run graph** — SVG node graph: green = LLM call, orange = tool call (ok), red = looping tool call, blue = start/end. Loop clusters highlighted with a dashed red outline.
- **Event log** — every event in chronological order, expandable to show full payload. Content fields are shown as SHA-256 hashes — no raw text stored.

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

Token waste figures across the dashboard (drift panel, alerts, analytics, top drivers) are computed client-side from run `step_count` using a fixed estimate of **250 tokens per step**. Dollar costs use a configurable rate — default $0.010/1k tokens, editable on the Analytics page.

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
| Agent view (health record + runs) | `GET /v1/agents/{id}/runs` + `/signals` + `/insights` |
| Why is this happening? panel | `GET /v1/agents/{id}/failure-patterns/{failure_type}` |
| Detectors | Static — edits require updating `detectors.yml` and restarting the detector service |
