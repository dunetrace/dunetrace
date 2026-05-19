# Policies

Runtime guardrails evaluated mid-run after every `tool_called`, `llm_responded`, and `tool_responded` event. Policies fire at most once per run (except `log` policies, which fire every time).

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
    condition={"trigger": "signal", "operator": "eq", "value": "TOOL_LOOP"},
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
| `signal` | str | Detector signal name — runs the full detector suite lazily (e.g. `"TOOL_LOOP"`) |

Supported operators: `gt` `gte` `lt` `lte` `eq` `neq` `contains`

---

## Action reference

| Action type | Effect | Required params |
|---|---|---|
| `stop` | Raises `PolicyViolation`; run exits with `exit_reason="policy_violation"` | — |
| `switch_model` | Sets `run.model_override` (str) | `model` |
| `inject_prompt` | Appends to `run.prompt_additions` (list) | `prompt` |
| `log` | Emits `policy.triggered` event; no interruption; fires on every matching event | — |

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

---

## `add_policy` parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | required | Human-readable label shown in `policy.triggered` events |
| `condition` | required | `{trigger, operator, value}` dict |
| `action` | required | `{type, params?}` dict |
| `agent_id` | `"*"` | `"*"` applies to all agents; pass a specific agent_id to scope |
| `priority` | `100` | Lower numbers fire first |
| `enabled` | `True` | Set to `False` to disable without removing |

---

## Dashboard

Policies can be created, edited, toggled, and deleted from the **Policies** page in the dashboard at `http://localhost:3000`. Changes are fetched by the SDK within the 60-second TTL window.

REST API:

| Endpoint | Description |
|---|---|
| `GET /v1/policies` | List all policies |
| `POST /v1/policies` | Create a policy |
| `GET /v1/policies/{id}` | Get a single policy |
| `PUT /v1/policies/{id}` | Replace a policy |
| `DELETE /v1/policies/{id}` | Delete a policy |
| `PATCH /v1/policies/{id}/toggle` | Enable / disable |

The ingest endpoint also exposes `GET /v1/policies?agent_id=...&api_key=...` (used by the SDK for remote fetch).
