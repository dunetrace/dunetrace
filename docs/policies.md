# Policies

> **Remote policies that change agent behaviour require a verified signature.**
> A remote policy asking for `stop`, `require_approval`, `escalate_to_human`,
> `switch_model`, `inject_prompt`, `inject_recovery_prompt` or
> `stop_current_tts` is **downgraded to log-only** unless a policy secret is
> configured — set `DUNETRACE_POLICY_SECRET` in the agent's environment and
> `POLICY_SIGNING_SECRET` on the server. Without one there is no way to tell the
> policy came from your server, and these actions change what a production agent
> does. Policies you register in-process with `dt.add_policy()` are your own
> code and are never downgraded.
>
> Signing is a shared HMAC secret, so a server that is itself compromised holds
> the signing key. What this protects against is a party who cannot produce a
> correctly signed payload — a spoofed endpoint, a hijacked DNS record, a MITM.
> Closing the compromised-server case needs asymmetric signing, which needs a
> crypto library; the core SDK is deliberately zero-dependency.

Runtime guardrails evaluated mid-run after every `tool_called`, `llm_responded`, and `tool_responded` event. Policies fire at most once per run (except `log` policies, which fire every time).

Policy checks are synchronous and O(1) — running totals for `error_count` and `cost_usd` are maintained incrementally, and signal detector results are cached per step.

---

## Structural signals only

The `trigger="signal"` condition (see [Condition reference](#condition-reference)
below) can only ever match a **structural** detector signal — never a
[semantic evaluation](semantic-evaluation.md) finding. This isn't a policy
choice enforced by a check somewhere; it's true by construction: `trigger="signal"`
runs the SDK's own in-process detector battery
(`dunetrace.detectors.run_detectors`, which defaults to the 29 detectors in
`TIER1_DETECTORS` — the subset of the 32 in [docs/detectors.md](detectors.md)
that can be evaluated in-path) synchronously, inside your
agent's process, before the run has even finished. Semantic evaluation runs
entirely after a run completes, in a separate `semantic_worker` service the
SDK never calls into — there is no code path by which an in-process policy
check could see a semantic finding, because none exist yet at the moment the
check runs.

This also means policies (a **runtime prevention** mechanism) can never
"prevent" a semantic-evaluation finding — by the time semantic evaluation
sees a run, the run is already over. Runtime prevention and semantic
evaluation are complementary, not overlapping: policies stop a bad run
*while it's happening*; semantic evaluation judges quality *after the fact*
for things no in-process check can catch (hallucination, task completion,
cross-turn frustration).

---

## Local policies (no backend required)

```python
from dunetrace import Dunetrace, PolicyViolation

dt = Dunetrace(endpoint="http://localhost:8001")

# Stop the run if tool calls exceed 5
dt.add_policy(
    name="cap tool calls",
    condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
    action={"type": "stop"},
)

# Downgrade model when estimated cost exceeds $0.50
dt.add_policy(
    name="cost cap",
    condition={"trigger": "cost_usd", "operator": "gt", "value": 0.50},
    action={"type": "switch_model", "params": {"model": "gpt-4o-mini"}},
)

# Inject a corrective prompt when a loop is detected mid-run
dt.add_policy(
    name="loop fix",
    condition={"trigger": "signal", "operator": "contains", "value": "TOOL_LOOP"},
    action={"type": "inject_prompt", "params": {
        "prompt": "Stop repeating tool calls. Summarise what you know and answer directly."
    }},
)

# Log without stopping (fires every time the condition is true)
dt.add_policy(
    name="warn slow",
    condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 10000},
    action={"type": "log"},
)
```

---

## Remote policies (dashboard-managed)

When `api_key` and `endpoint` are set, the SDK fetches policies from the backend at run start and caches them for 60 seconds per agent. Policies defined in the dashboard apply automatically — no code changes needed.

```python
dt = Dunetrace(api_key="dt_live_...", endpoint="https://ingest.dunetrace.com")
# Policies defined in the dashboard are pulled at run start.
```

Local policies (added via `add_policy`) take priority over remote ones at the same `priority` level and are never replaced by remote fetches.

**Long-running agents:** the 60-second remote refresh window means a newly pushed policy may not reach an already-running agent until its next run. Signal-trigger policy checks (`trigger="signal"`) cache whether any such policy is active per engine generation — adding a new signal policy via `add_policy()` mid-run takes effect on the next policy-checked event.

---

## Condition reference

| Trigger | Type | What it measures |
|---|---|---|
| `tool_call_count` | int | Total tool calls in the run so far |
| `step_count` | int | Current step index |
| `cost_usd` | float | Accumulated LLM cost in USD (model-aware pricing) |
| `error_count` | int | Failed tool calls (`success=False`) |
| `finish_reason` | str | Latest LLM `finish_reason` (e.g. `"length"`, `"stop"`, `"tool_calls"`) |
| `llm_latency_ms` | int | Latest LLM call latency in milliseconds |
| `signal` | list[str] | Failure types detected so far this run — runs the full detector suite lazily. Pair it with `contains` (e.g. `"TOOL_LOOP"`); `eq` compares against the whole list and never matches |
| `before_tool_call` | str | Name of the tool about to be called — only valid with the `require_approval` action (human-in-the-loop). See [Approvals](approvals.md) |
| `expression` | — | Sentinel for a **pure-expression** policy: the flat fields are a no-op and a [`match` block](#expression-conditions-argument--metadata-values) carries the whole condition |

Supported operators (flat condition): `gt` `gte` `lt` `lte` `eq` `neq` `contains`

### Expression conditions (argument & metadata values)

Beyond the flat trigger, a policy's `condition` may carry an optional **`match`**
block that gates on the *values* inside a run — tool arguments (`args.*`), run
metadata (`run.*`), event fields (`event.*`), and (when available) `agent.*` /
`org.*`. It supports 14 operators, `AND`/`OR` composition, and nesting, and ANDs
with the flat trigger when both are present:

```yaml
condition:
  trigger: before_tool_call
  operator: eq
  value: refund_customer
  match:
    args.amount: {gt: 10000}      # require approval only above $10k
action:
  type: require_approval
```

This is fully additive — policies without a `match` block are unchanged. See the
complete reference (every operator, every field path, composition, coercion,
and debugging) in **[Condition expressions](policies/condition-expressions.md)**.

---

## Action reference

| Action type | Effect | Required params |
|---|---|---|
| `stop` | Raises `PolicyViolation`; run exits with `exit_reason="policy_violation"` | — |
| `switch_model` | Sets `run.model_override` (str) | `model` |
| `inject_prompt` | Appends to `run.prompt_additions` (list) | `prompt` |
| `log` | Emits `policy.triggered` event; no interruption; fires on every matching event | — |
| `require_approval` | Blocks the guarded tool call until a human approves, or raises `ApprovalDenied` on deny/timeout (human-in-the-loop). Pairs only with the `before_tool_call` trigger | — (optional `timeout_s`, default 300) |

### Enforced vs. advisory actions

Two actions actively alter control flow; the rest set an attribute your agent
must read and honor:

| Action | Enforced? | How it takes effect |
|---|---|---|
| `stop` | **Enforced** | Raises `PolicyViolation` inside the tool-call hook, *before* the tool runs — but only if your code lets the exception propagate (don't catch-and-continue). |
| `require_approval` | **Enforced** | Blocks the call until a decision (or timeout), then raises `ApprovalDenied` on deny/timeout. |
| `switch_model` | Advisory | Sets `run.model_override`. The SDK does **not** swap your LLM call — you must read the attribute and use it. |
| `inject_prompt` / `inject_recovery_prompt` | Advisory | Appends to `run.prompt_additions` / sets `run.recovery_prompt`. You must read and prepend/speak it. |
| `escalate_to_human` / `stop_current_tts` | Advisory | Set `run.escalate_to_human` / `run.stop_tts`. Your loop must act on them. |
| `slow_response_pace` | Advisory | Sets `run.response_pace` (default `"slow"`). Your loop must slow the turn (filler token, lower TTS rate). |

If you configure an advisory action but never read the corresponding attribute in
your agent code, the action is a silent no-op. See [switch_model](#switch_model) /
[inject_prompt](#inject_prompt) for the read pattern.

**Conflict footgun:** only the highest-priority matching policy fires per event
(lower `priority` number wins). A high-priority `log`/advisory policy on the same
trigger will therefore **suppress** a lower-priority `stop`/`require_approval`
safety policy. Give safety-critical `stop`/`require_approval` policies the highest
priority (lowest number), and avoid stacking other policies on the same trigger.

### `stop`

`PolicyViolation` propagates up through agent code. `dt.run()` catches it, emits `run.errored` with `exit_reason="policy_violation"` and `policy_name`, then re-raises. Catch it to handle gracefully:

```python
from dunetrace import PolicyViolation

try:
    with dt.run("my-agent", user_input=query, tools=TOOLS) as run:
        for step in agent_loop():
            ...
except PolicyViolation as exc:
    print(f"Stopped by policy: {exc.policy_name}")
```

### `switch_model`

The SDK sets `run.model_override` but does not intercept your LLM calls. Read it between steps:

```python
with dt.run("my-agent", user_input=query) as run:
    for step in agent_loop():
        model = run.model_override or "gpt-4o"
        response = openai_client.chat.completions.create(model=model, ...)
```

### `inject_prompt`

The SDK appends to `run.prompt_additions`. Read with `run.pop_prompt_addition()` and prepend to your next LLM messages:

```python
with dt.run("my-agent", user_input=query) as run:
    messages = [{"role": "system", "content": system_prompt}]
    for step in agent_loop():
        addition = run.pop_prompt_addition()
        if addition:
            messages.insert(0, {"role": "system", "content": addition})
        response = openai_client.chat.completions.create(model="gpt-4o", messages=messages, ...)
```

**Security:** `inject_prompt` policy content is validated against known prompt injection patterns at write time. Content matching patterns such as "ignore previous instructions," role-switching commands, or jailbreak phrases is rejected with HTTP 422. The same detector runs on both creation and updates (including action-only updates). See [Trust boundary](#trust-boundary) below.

### `require_approval`

Blocks a specific tool call until a human approves it. Unlike every other action, this one is evaluated **per tool call** (not fire-once) and **blocks** the run until a decision arrives or the timeout passes (fail-closed: a timeout denies). It's the only action driven by the `before_tool_call` trigger, whose value is the tool name:

```python
dt.add_policy(
    name="approve-wires",
    condition={"trigger": "before_tool_call", "operator": "eq", "value": "wire_money"},
    action={"type": "require_approval", "params": {"timeout_s": 300}},
)
```

Now any call to the `wire_money` tool (via `@dt.tool` or `run.tool_called("wire_money", ...)`) blocks until a human approves it in Slack or the dashboard. A deny or timeout raises `ApprovalDenied` **before the tool runs**. Full walkthrough — delivery channels, the decision flow, sync vs async — in **[Approvals](approvals.md)**.

---

## Conflict resolution

When multiple policies match simultaneously, only the highest-priority one fires per event (lower `priority` number = higher priority). The others are skipped for that event. If the same policy matches on the next event it will fire then, unless it has already been added to `_triggered_policies`.

- A `stop` policy raises immediately — lower-priority policies in the same event never execute.
- Multiple `inject_prompt` policies that fire on different events accumulate additively in `run.prompt_additions`.
- Multiple `switch_model` policies: whichever fires last wins — the override is overwritten silently.

---

## `add_policy` parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | required | Human-readable label shown in `policy.triggered` events |
| `condition` | required | `{trigger, operator, value}` dict; may also carry an optional [`match` expression block](policies/condition-expressions.md) |
| `action` | required | `{type, params?}` dict |
| `agent_id` | `"*"` | `"*"` applies to all agents; pass a specific agent_id to scope |
| `priority` | `100` | Lower numbers fire first |
| `enabled` | `True` | Set to `False` to disable without removing |

---

## Trust boundary

**Who can define policies:** Bearer token authentication is required for all policy CRUD endpoints. In `AUTH_MODE=dev` (local Docker only), auth is skipped — do not expose port 8002 beyond localhost in dev mode. `dt.add_policy()` in-process requires no auth; trust is whoever controls the code.

**Validation at write time:**
1. Trigger, operator, and action type are checked against fixed allowlists — unknown values are rejected with 422.
2. `inject_prompt` prompt content is scanned against the prompt injection pattern detector. Matching content is rejected before it reaches the database.
3. Every policy is signed with HMAC-SHA256 at write time. The signature is stored alongside the policy. The signature covers the whole `condition` — including any [`match` expression block](policies/condition-expressions.md#signing) — so a tampered condition fails verification. Policies using a `match` block are signed under canonical-form version 2 (`sig_version: 2`); legacy policies stay version 1, byte-identical, so existing signatures keep verifying.

**Signature verification:** Configure `POLICY_SIGNING_SECRET` (same value on server and SDK client) to enable end-to-end verification:

```python
dt = Dunetrace(
    api_key="dt_live_...",
    endpoint="https://ingest.dunetrace.com",
    policy_secret="your-shared-secret",
)
```

The SDK verifies each policy's signature before loading it. Policies with a non-matching signature are skipped and logged as warnings — a tampered or replayed policy never reaches the agent.

**Migration note:** Policies created before `POLICY_SIGNING_SECRET` was set will have an empty signature. These are loaded with a warning rather than silently dropped. Re-save each policy in the dashboard to sign it.

**Audit log:** Every create, update, delete, and toggle is written to `policy_audit_log` with the customer ID, timestamp, and full before/after diff. Query it directly in Postgres for forensic review.

---

## Dashboard

Policies can be created, edited, toggled, and deleted from the **Policies** page in the dashboard at `http://localhost:3000`. Changes are fetched by the SDK within the 60-second TTL window.

REST API:

| Endpoint | Description |
|---|---|
| `GET /v1/policies` | List all policies |
| `POST /v1/policies` | Create a policy |
| `GET /v1/policies/{id}` | Get a single policy |
| `GET /v1/policies/{id}/evaluations` | Recent evaluation results — why the policy did/didn't fire (see [Condition expressions › Debugging](policies/condition-expressions.md#debugging-why-did-my-policy-fire)) |
| `PUT /v1/policies/{id}` | Replace a policy |
| `DELETE /v1/policies/{id}` | Delete a policy |
| `PATCH /v1/policies/{id}/toggle` | Enable / disable |

The ingest endpoint also exposes `GET /v1/policies?agent_id=...&api_key=...` (used by the SDK for remote fetch).
