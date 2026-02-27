# Detectors

Dunetrace ships six Tier 1 detectors that run against every completed agent run. All are deterministic — no LLM calls, no probabilistic models, no external dependencies.

---

## How Detection Works

After a run completes, the detector worker:

1. Fetches all events for the run from Postgres
2. Replays them in order to build a `RunState`
3. Passes the `RunState` to each detector
4. Writes any resulting `FailureSignal` rows to Postgres

Each detector receives the full `RunState` and returns either `None` (no failure detected) or a `FailureSignal`.

```python
class BaseDetector:
    name: str

    def detect(self, state: RunState) -> Optional[FailureSignal]:
        raise NotImplementedError
```

---

## Tier 1 Detectors

### TOOL_LOOP

**Severity:** HIGH  
**Confidence:** 0.95  

The same tool was called 3 or more times within a 5-step window with identical arguments.

This is the most structurally unambiguous failure mode — there is no legitimate reason for an agent to call the same tool with the same inputs three times in a row.

**Detection logic:**
```
window = last 5 tool calls
for each tool in window:
    if count(tool_name == tool AND args_hash == args_hash) >= 3:
        → TOOL_LOOP
```

**Evidence fields:**
```json
{
  "tool":   "web_search",
  "count":  5,
  "window": 5
}
```

**Common causes:**
- Agent didn't process the tool output before calling again
- Tool returned an error but the agent didn't handle it and retried indefinitely
- Model is stuck in a reasoning loop and keeps deciding to search

---

### TOOL_THRASHING

**Severity:** HIGH  
**Confidence:** 0.90  

The agent oscillated between exactly two tools in an ABABABAB pattern across the last 6 tool calls.

Unlike TOOL_LOOP (same tool), thrashing is the agent switching between two conflicting tools — often because their outputs contradict each other and the agent can't resolve the conflict.

**Detection logic:**
```
window = last 6 tool calls
tool_names = [c.tool_name for c in window]
unique_tools = set(tool_names)
if len(unique_tools) == 2:
    if tool_names alternates between the two tools:
        → TOOL_THRASHING
```

**Evidence fields:**
```json
{
  "tool_a":            "web_search",
  "tool_b":            "database_lookup",
  "oscillation_count": 4
}
```

**Common causes:**
- Two tools return contradictory information (live web vs. internal DB)
- Agent has no resolution strategy when tools disagree
- System prompt doesn't specify which source to trust

---

### TOOL_AVOIDANCE

**Severity:** MEDIUM  
**Confidence:** 0.75  

The agent completed the run without calling any tools, despite having tools available.

Lower confidence than TOOL_LOOP because some queries legitimately don't require tool use (e.g., "what is 2 + 2"). Confidence is 0.75 to account for this.

**Detection logic:**
```
if state.completed AND len(state.tool_calls) == 0 AND len(state.available_tools) > 0:
    → TOOL_AVOIDANCE
```

**Evidence fields:**
```json
{
  "available_tools":  ["web_search", "calculator"],
  "tool_calls_made":  0
}
```

**Common causes:**
- System prompt doesn't explicitly require tool use
- Agent answered from training knowledge instead of retrieving live info
- Model is avoiding tools to reduce latency

---

### GOAL_ABANDONMENT

**Severity:** MEDIUM  
**Confidence:** 0.70  

The agent had 4 or more consecutive LLM calls with no tool calls between them, followed by a final answer — suggesting it gave up on actually completing the task.

**Detection logic:**
```
recent = last 4 events
if all(e.event_type in ["llm.called", "llm.responded"] for e in recent):
    if state.completed:
        stall_steps = steps since last tool call
        → GOAL_ABANDONMENT
```

**Evidence fields:**
```json
{
  "stall_steps":  6,
  "last_tool":    "web_search",
  "last_tool_step": 3
}
```

**Common causes:**
- Task hit an obstacle and the agent silently defaulted to "I can't help with that"
- Max iterations reached and the agent produced a placeholder response
- Agent stopped using tools partway through a multi-step task

---

### PROMPT_INJECTION_SIGNAL

**Severity:** CRITICAL  
**Confidence:** 0.85  
**Status:** ⚠️ Scaffolded — not yet active in the detection pipeline

This detector is the exception to the "metadata only" rule. Injection detection requires reading the raw user input text — you cannot pattern-match a SHA-256 hash against regex patterns. This creates a fundamental tension with the privacy model.

**How it's designed to work:**

Detection must run **client-side in the SDK**, before the raw text is hashed, on your machine. The SDK checks the input against known injection patterns and emits only a structured signal — matched pattern labels and a count — never the raw text:

```python
# Runs locally in SDK before any hashing
matched = PROMPT_INJECTION_DETECTOR.check_input(user_input, state)

# Only this crosses the wire — no raw text
payload = {
    "input_hash":            hash_content(user_input),  # hash only
    "matched_pattern_count": len(matched_patterns),
    "matched_patterns":      ["ignore_instructions", "you_are_now"],
}
```

**Current implementation status:**

The detector class (`PromptInjectionDetector`) and its 18 regex patterns are fully implemented in the SDK. However, the client-side call to `check_input()` before hashing has not yet been wired into the `Dunetrace.run()` context manager or the `DunetraceCallback`. The server-side worker imports the detector but does not call it (because the raw text is not available there).

**In practice:** PROMPT_INJECTION signals are not currently emitted. The detector will become active once the SDK calls `check_input()` at run start, before hashing.

**Detected patterns (18 total):**

`ignore_instructions`, `disregard_instructions`, `forget_instructions`, `you_are_now`, `new_role`, `act_as`, `pretend`, `do_not_follow`, `system_colon`, `system_tag`, `im_start`, `system_pipe`, `hash_system`, `jailbreak`, `dan_mode`, `developer_mode`, `override_safety`, `bypass_safety`

**Evidence fields (when active):**
```json
{
  "matched_pattern_count": 2,
  "matched_patterns":      ["ignore_instructions", "you_are_now"],
  "input_length":          312
}
```

**Common causes:**
- Malicious user attempting to jailbreak the agent
- User testing boundaries (common in customer-facing deployments)
- Content retrieved by a tool containing injected instructions (indirect injection)

---

### RAG_EMPTY_RETRIEVAL

**Severity:** MEDIUM  
**Confidence:** 0.88  

The agent queried a RAG index, got back zero useful results, but then produced a final answer anyway — drawing on LLM training knowledge instead of retrieved context.

**Detection logic:**
```
for each retrieval in state.retrieval_results:
    if result_count == 0 OR top_score < threshold:
        bad_retrievals += 1
if bad_retrievals > 0 AND state.completed:
    → RAG_EMPTY_RETRIEVAL
```

**Evidence fields:**
```json
{
  "index_name":    "knowledge-base",
  "result_count":  0,
  "top_score":     null,
  "bad_retrievals": 1
}
```

**Common causes:**
- Query didn't match anything in the index (vocabulary mismatch)
- Index is empty or hasn't been populated for this topic
- Embedding model changed and index needs to be rebuilt
- Agent didn't check the retrieval result before answering

To use this detector, emit `retrieval.completed` events from your agent:

```python
docs = vectorstore.similarity_search(query, k=4)
run.retrieval_completed(
    index_name="knowledge-base",
    result_count=len(docs),
    top_score=docs[0].metadata.get("score") if docs else None,
)
```

---

## Detector Summary Table

| Detector | Failure Type | Severity | Confidence | Window | Status |
|---|---|---|---|---|---|
| ToolLoopDetector | TOOL_LOOP | HIGH | 95% | 5 steps | ✅ Active |
| ToolThrashingDetector | TOOL_THRASHING | HIGH | 90% | 6 steps | ✅ Active |
| ToolAvoidanceDetector | TOOL_AVOIDANCE | MEDIUM | 75% | full run | ✅ Active |
| GoalAbandonmentDetector | GOAL_ABANDONMENT | MEDIUM | 70% | 4 steps | ✅ Active |
| RagEmptyRetrievalDetector | RAG_EMPTY_RETRIEVAL | MEDIUM | 88% | full run | ✅ Active |
| PromptInjectionDetector | PROMPT_INJECTION_SIGNAL | CRITICAL | 85% | full run | ⚠️ Not yet wired |

---

## Shadow Mode and Graduation

All detectors start in shadow mode. Signals are stored but not alerted until you graduate a detector.

**Graduation checklist:**
1. Run `python scripts/precision_report.py --inspect <DETECTOR_NAME>` to review evidence
2. Spot-check 20 signals manually — mark each TP (true positive) or FP (false positive)
3. If ≥ 80% are TPs → graduate the detector
4. Edit `services/detector/app/db.py`:
   ```python
   LIVE_DETECTORS = {"TOOL_LOOP", "PROMPT_INJECTION_SIGNAL"}
   ```
5. Restart the detector worker

**Recommended graduation order:**

1. `PROMPT_INJECTION_SIGNAL` — pattern-matched, very high precision, security-critical
2. `TOOL_LOOP` — structurally unambiguous, rarely a false positive
3. `TOOL_THRASHING` — high precision but check for legitimate oscillation patterns
4. `RAG_EMPTY_RETRIEVAL` — requires `retrieval.completed` events; check your index health first
5. `TOOL_AVOIDANCE` — most false positives; only graduate after traffic analysis
6. `GOAL_ABANDONMENT` — borderline; depends heavily on your agent's normal behavior

---

## Tier 2 Detectors (Coming Soon)

Tier 2 detectors require LLM calls on the conversation content and are therefore slower and more expensive than Tier 1.

| Detector | Description | Status |
|---|---|---|
| USER_DISSATISFACTION | Detects negative sentiment in user follow-ups | Planned |
| INTENT_MISALIGNMENT | Agent answered a different question than asked | Planned |
| REASONING_STALL | Agent's chain-of-thought loops without progress | Planned |
| CONFIDENT_HALLUCINATION | Agent stated false facts with high confidence | Planned |
