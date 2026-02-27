# Dashboard

The Dunetrace dashboard is a React application that gives you a live view of your agents, runs, signals, and run timelines.

---

## Views

### Agents

The top-level view. Shows all agents associated with your account with:

- **Live indicator** — green dot if the agent was seen in the last 5 minutes
- **Run count** — total runs processed
- **Signal count** — live (non-shadow) signals detected
- **Critical / High counts** — severity breakdown at a glance

Click any agent to navigate to its runs list.

---

### Runs

Shows the run history for a selected agent. Each row displays:

- Run ID (monospace)
- Time since the run started
- Exit reason (color-coded: green = completed, red = error, yellow = stalled)
- Duration
- Step count
- Signal count — orange `▲ N` if signals exist, `—` otherwise

Use the **With signals** filter to show only runs that triggered detectors.

---

### Signals

Shows all live failure signals across an agent with severity filtering.

Each signal is a collapsible card with:
- Severity badge (color-coded)
- Failure type chip
- Alert status (green dot = delivered to Slack)
- Signal title
- Run ID, step index, confidence, time ago

Click to expand and see:
- **What happened** — plain-English description
- **Why it matters** — business impact
- **Evidence** — formatted block with raw values
- **Suggested fix** — copy-pasteable code snippet

---

### Run Timeline

The run timeline is the main diagnostic view in Dunetrace. It shows every event in a single agent run as a horizontal sequence, with two data strips below it that answer two questions you always have when something goes wrong: *how long did each step take?* and *why are my tokens growing?*

---

## The run selector

At the top you'll see a row of run cards. Each card shows the run ID, its exit status, and whether any signals were detected.

Exit status is color-coded: **green** for `completed`, **red** for `error`, **yellow** for `stalled` or `max_iterations`. The `▲` marker with a signal label (e.g. `▲ TOOL_LOOP`) means at least one live failure was detected in that run. A green `✓ clean` means no signals fired.

Click any card to load that run. The timeline, strips, and metadata header all update together.

---

## The metadata header

The strip of six tiles at the top of the timeline gives you the most important numbers before you look at anything else.

| Tile | What it means |
|---|---|
| **Run ID** | The unique identifier for this run, as emitted by the SDK |
| **Agent** | The `agent_id` passed when the SDK was initialised |
| **Version** | First 8 chars of the agent version hash — changes when your system prompt, model, or tool list changes |
| **Duration** | Wall-clock time from `run.started` to `run.completed` (or `run.errored`) |
| **Token growth** | First → last `prompt_tokens` and the growth multiplier, e.g. `620 → 4,680 (7.5×)`. Turns **orange at ≥ 3×**, matching the CONTEXT_BLOAT detector threshold |
| **Exit** | The exit reason as reported by your agent: `completed`, `error`, `stalled`, `max_iterations` |

If **Token growth** is orange or the **Duration** is longer than you expect, the strips below will show you exactly where the problem is.

---

## The event track

The horizontal track is the spine of the view. Each circle is one event emitted by your agent. Time flows left to right — leftmost circle is `run.started`, rightmost is `run.completed` or `run.errored`.

**Circle colors** tell you what kind of event it is:

- 🟢 Green — run lifecycle (`run.started`, `run.completed`, `run.errored`)
- 🟣 Purple — LLM calls (`llm.called`, `llm.responded`)
- 🟠 Orange — tool calls (`tool.called`, `tool.responded`)

The step number is printed below each circle (e.g. step 0, step 1, step 3). Gaps in the numbering don't happen — every step is shown.

**The connecting line** between circles changes thickness and opacity based on the time gap between those two events. A thin, faint line means near-instant transition. A thick, bright orange line means the agent was waiting — the wider and brighter, the longer it waited. This lets you spot slow steps at a glance even before looking at the duration strip.

### Signal spikes

When a failure detector fires, a vertical spike erupts above the track at the step where the failure was pinpointed.

- The spike is color-coded by severity: **red** = CRITICAL, **orange** = HIGH, **yellow** = MEDIUM
- The failure type is printed in a small badge above the spike (e.g. `TOOL LOOP`, `CONTEXT BLOAT`)
- The circle at that step is replaced by an `!` node with a dashed ring around it

Click the `!` node or the signal badge in the row above the track to open the full explanation popup. The popup shows what happened, why it matters, the raw evidence values, and a copy-pasteable code fix.

### Hovering

Hovering any circle reveals a detail strip below the track showing:

- The event type and step number
- Duration to the next event
- Prompt tokens (if this is an `llm.called` step) and the delta since the prior LLM call
- Biggest source of token growth (tool name + output length in characters)
- Key payload fields: model name, tool name, finish reason, output length, etc.

This is the fastest way to inspect a specific step without scrolling through logs.

---

## The token strip

The token strip sits directly below the event track. It only shows data for `llm.called` events where `prompt_tokens` was reported — one bar per LLM call across the run.

**Bar height** represents the total prompt token count at that LLM call. Taller = more tokens in context.

**Bar color** shows how close you are to trouble:

| Color | Token range | What it means |
|---|---|---|
| Green | < 1,500 | Healthy |
| Yellow | 1,500–4,000 | Growing — watch this |
| Orange | 4,000–10,000 | Getting expensive, approaching common context limits for long reasoning |
| Red | > 10,000 | Very large — risk of truncation on smaller models |

Between consecutive bars you'll see two annotations:

- `+2,120 tok` — how many tokens were added since the prior LLM call. The **largest delta across the run is highlighted** in a colored badge; smaller ones are dim.
- `← database_lookup 8.4kch` — the single tool output most responsible for that jump. The tool name and its `output_length` in characters. If multiple tools ran between two LLM calls, this shows only the biggest contributor; `+2` appended means two more tools also contributed.

This answers "why are my tokens growing?" directly. If you see `← database_lookup 8,400ch` next to your largest delta, that's your fix target — that tool's output needs to be summarised or truncated before being appended to context.

When a CONTEXT_BLOAT signal fires, the bar at the step where it was detected gets a glow ring. Clicking it opens the full explanation with suggested fixes.

**If the strip is empty** (shows a message instead of bars), it means `prompt_tokens` wasn't passed in the `llm.called` payload. Add `prompt_tokens=count_tokens(messages)` to your instrumentation — without it, the CONTEXT_BLOAT detector also cannot fire.

---

## The duration strip

The duration strip sits below the token strip. It shows a bar for every step — not just LLM calls — measuring the time from that event to the next one.

**Bar height** is proportional to the gap. The tallest bar in the run sets the scale; all other bars are shown relative to it.

**Bar color** shows latency at a glance:

| Color | Duration | What it means |
|---|---|---|
| Green | < 2s | Normal |
| Yellow | 2–10s | Slightly slow — may be acceptable for heavy tool calls |
| Orange | 10–30s | Slow — investigate |
| Red | > 30s | Very slow — likely a timeout, hang, or blocked I/O |

The duration label is printed below each bar (e.g. `1.2s`, `42.0s`). Steps with no next event (the final step) have no bar.

When a SLOW_STEP signal fires, the offending bar gets a glow ring and is clickable. The signal popup shows the duration, the threshold that was crossed, the ratio (e.g. `2.8×`), and a code fix — typically adding a timeout to the tool call.

The event track's line thickness and the duration strip bars are driven by the same underlying data, so they tell the same story: use the track for a quick visual scan, use the strip for the exact numbers.

---

## Signal badges

Above the event track, below the metadata header, is a row of signal badges — one per signal detected in this run. Each badge shows the severity, step index, and failure type.

Clicking any badge opens the same explanation popup as clicking the spike on the track. Use these when the run has many steps and the spike is hard to find visually, or when you want to jump straight to the explanation without scrolling the track.

---

## The explanation popup

Clicking any signal — whether the `!` node, the duration strip bar, or the badge — opens the same popup. It has four sections:

**What happened** — a plain-English description of the specific pattern that fired. Names the tool, step, count, or threshold value from the actual evidence.

**Why it matters** — the business impact. Cost, latency, reliability, user experience.

**Evidence** — the raw values the detector used: step index, tool name, count, threshold, duration, token counts. This is what you'd need to reproduce the issue.

**Fix** — a copy-pasteable code snippet addressing the root cause. Not generic advice — the fix is interpolated with the actual tool name, count, or threshold from the evidence.

---

## What the dashboard does not show

A few things the current view intentionally omits:

**Raw prompts and outputs.** Dunetrace never stores them. All content is hashed before it reaches the ingest API. If you need to read the actual conversation, check your own logs — Dunetrace points you to the step and event type so you know where to look.

**Cost in dollars.** Token counts are shown, but Dunetrace doesn't know your model pricing or tier. Multiply `prompt_tokens` by your per-token rate yourself.

**Reasoning quality.** Whether the agent's answer was correct is not something the dashboard can show — that requires reading the output, which Dunetrace deliberately doesn't do. The dashboard shows structural failures: loops, avoidance, truncation, context growth, slow steps. If your agent produces a plausible but wrong answer, that's a content evaluation problem and belongs in your eval pipeline.

---

## Common patterns to look for

**Tool loop** — the event track shows the same tool label repeating in consecutive steps. The connecting lines between them are thin (fast), which is part of the problem: the agent is cycling quickly without pausing to reconsider.

**Context bloat** — the token strip bars grow steadily across the run with no plateau. The delta annotations between the bars show which tools are feeding in the most tokens. If the same tool appears in multiple large deltas, that tool's output truncation is the fix.

**Slow tool call** — one bar in the duration strip is dramatically taller than everything else. The connecting line in the event track at that step will also be thick and bright. If the run otherwise looks healthy and the only problem is one red bar, it's typically a network timeout or an API returning a very large payload.

**Truncation loop** — finish_reason shows `length` on multiple LLM calls. In the event track these appear as `LLM↓` nodes without a subsequent tool call — the model hit its output limit before it could decide to use a tool, so the agent gets stuck calling the LLM, getting a truncated response, and trying again. The token strip will usually show high baseline token counts explaining why the model ran out of output budget.

**Goal abandonment** — the event track ends with several purple LLM nodes in a row (after an orange section) with no final `run.completed` in green. The agent was using tools, then stopped using them, and never resolved.

## Running the Dashboard

The dashboard is a React component (`dashboard/dashboard.jsx`). To run it locally:

### Option A — Vite dev server (recommended)

```bash
# From the repo root
mkdir dashboard-app && cd dashboard-app
npm create vite@latest . -- --template react
npm install

# Replace src/App.jsx with dashboard/dashboard.jsx
cp ../dashboard/dashboard.jsx src/App.jsx

npm run dev
# → http://localhost:5173
```

### Option B — Direct in Claude

Open `dashboard/dashboard.jsx` in this Claude conversation — it renders immediately with mock data.

---

## Connecting to the Real API

The current dashboard uses mock data. To connect to live data, replace the mock constants with API calls in `dashboard.jsx`:

```javascript
// Replace MOCK_DETAIL constant with:
const [runDetail, setRunDetail] = useState(null);

useEffect(() => {
  fetch(`http://localhost:8002/v1/runs/${selectedRun}`, {
    headers: { Authorization: "Bearer dt_dev_test" }
  })
    .then(r => r.json())
    .then(setRunDetail);
}, [selectedRun]);
```

The API response shape is identical to the mock data structure, so no other changes are needed.

---

## Stats Bar

The top bar shows aggregate metrics across all agents:

| Metric | Description |
|---|---|
| Total Runs | Sum of all run counts across agents |
| Live Signals | Total non-shadow signals |
| Critical | Critical-severity signal count |
| High | High-severity signal count |
| Agents | Number of active agents |