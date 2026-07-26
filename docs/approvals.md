# Human-in-the-loop approvals

Some tool calls are too consequential to run unattended — wiring money, deleting
data, sending a customer email. An **approval policy** gates a specific tool: the
agent blocks on that call until a human approves it (in Slack or the dashboard),
or the request times out. A timeout is **fail-closed** — nobody decided, so the
tool is blocked, not allowed.

Approvals build on the existing [policy](policies.md) engine — one new trigger
(`before_tool_call`) and one new action (`require_approval`). Nothing else about
your instrumentation changes.

---

## The flow

```
 agent calls guarded tool
        │
        ▼
 require_approval policy matches ──► SDK creates approval, emits approval.requested
        │                                     │
        │                          alerts worker delivers to Slack / webhook
        │                                     │
   SDK blocks, polling ◄───────────  human clicks Approve / Deny
        │                            (Slack button, dashboard, or API)
        ▼
  granted → tool runs
  denied / timeout → raises ApprovalDenied, tool never runs
```

The SDK learns the decision by **polling** the Customer API — there's no inbound
connection to your agent process.

---

## 1. Configure the policy

A `require_approval` action pairs with a `before_tool_call` condition whose value
is the tool name:

```python
dt.add_policy(
    name="approve-wires",
    condition={"trigger": "before_tool_call", "operator": "eq", "value": "wire_money"},
    action={"type": "require_approval", "params": {"timeout_s": 300}},
)
```

`operator` can be `eq` (exact tool name) or `contains` (substring match).
`timeout_s` defaults to 300. Policies can also be created in the dashboard and
pulled by the SDK automatically — same as any other policy.

The approval calls hit the **Customer API**, so the client needs `api_url` (or
`DUNETRACE_API_URL`) and an `api_key`:

```python
dt = Dunetrace(api_key="dt_live_...", api_url="https://your-dunetrace-host")
```

---

## 2. The agent blocks automatically

No new code at the call site. Any guarded tool — decorated or manual — blocks:

```python
from dunetrace import ApprovalDenied

@dt.tool("wire_money")
def wire_money(to: str, amount: int) -> str:
    return payments.transfer(to, amount)

with dt.run("billing-agent", model="gpt-4o") as run:
    try:
        wire_money("acct_123", 5000)   # blocks here until a human approves
    except ApprovalDenied as exc:
        # denied outright, or nobody approved before the timeout (fail-closed)
        print(f"Not wired: {exc.reason}")   # "denied" | "timeout"
```

Manual instrumentation blocks identically:

```python
run.tool_called("wire_money", {"to": "acct_123", "amount": 5000})  # blocks
# ... only reached if approved; run the tool now ...
```

`ApprovalDenied` is raised **before the tool runs**, so a denied action never
executes. `exc.reason` is `"denied"` or `"timeout"`; `exc.tool_name` and
`exc.approval_id` are also available.

### Sync vs async

- **Sync agents** (and manual `run.tool_called()`): the gate blocks the calling
  thread. Nothing to do.
- **Async agents using `@dt.tool`**: the async decorator awaits the gate instead
  of blocking the event loop — automatic, nothing to do.
- **Manual async code** that calls `run.tool_called()` directly (outside
  `@dt.tool`) gets the *sync* gate and will block the event loop. Use `@dt.tool`
  for async agents, or `await run.arequest_approval("tool", args)` explicitly.

Each invocation is approved independently — approval is a **per-call** gate, not
once-per-run. Two calls to a guarded tool require two approvals.

---

## 3. A human decides

The approval request is delivered to whatever the org has configured:

- **Slack** — an interactive message with **Approve** / **Deny** buttons. The
  click is signature-verified and records the decision. (Requires the org's
  Slack integration.)
- **Webhook** — a signed JSON payload (`event: "approval_request"`) for building
  your own UI; call the decision endpoint back to resolve it.
- **Dashboard** — the **Approvals** page lists pending requests with
  Approve / Deny buttons.

> Email and Linear delivery are not implemented — see `BACKLOG.md` (no SMTP
> infrastructure; Linear's issue model has no natural approve/deny path).

A decision that arrives after the SDK has already timed out is rejected — a late
Slack click can't flip a recorded timeout into a grant. Conversely, if a human
approves in the same instant the SDK's deadline passes, the human decision wins.

---

## API reference

| Endpoint | Purpose |
|---|---|
| `POST /v1/approvals` | Create a pending approval (the SDK does this) |
| `GET /v1/approvals` | List this org's approvals (`?status=pending`) |
| `GET /v1/approvals/{id}` | Poll one approval's status (the SDK does this) |
| `POST /v1/approvals/{id}/decision` | Record `granted` / `denied` / `timeout` |

All are scoped to the org from your API key. `approval.requested`,
`approval.granted`, `approval.denied`, and `approval.timeout` events are emitted
on the run so the decision shows up in the run timeline.

---

## See also

- [`examples/approval_agent.py`](../examples/approval_agent.py) — runnable end-to-end
- [Policies](policies.md) — the engine approvals build on
