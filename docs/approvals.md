# Human-in-the-loop approvals

> **The decision endpoint needs a different credential than the agent holds.**
> `POST /v1/approvals/{id}/decision` requires an API key with the `approve`
> scope. Keys issued by `POST /v1/keys` are `ingest`-only by default, which is
> what an SDK/agent needs — so the process being gated cannot grant its own
> approval. Mint an operator key with `{"scopes": ["approve"]}` for whoever
> actually decides, or use the Slack path, which verifies Slack's signature
> rather than a Dunetrace key.

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
                     (carrying the human's note, if they wrote one)
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
        print(f"Not wired: {exc.status}")   # "denied" | "timeout"
        if exc.note:
            print(f"the human said: {exc.note}")
```

Manual instrumentation blocks identically:

```python
run.tool_called("wire_money", {"to": "acct_123", "amount": 5000})  # blocks
# ... only reached if approved; run the tool now ...
```

`ApprovalDenied` is raised **before the tool runs**, so a denied action never
executes. `exc.status` is `"denied"` or `"timeout"`; `exc.note` is what the
deciding human wrote (see [Decision notes](#4-decision-notes) below);
`exc.decided_by`, `exc.tool_name` and `exc.approval_id` are also available.

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
  Approve / Deny buttons and an optional **note** field.

> Email and Linear delivery are not implemented — see `BACKLOG.md` (no SMTP
> infrastructure; Linear's issue model has no natural approve/deny path).

A decision that arrives after the SDK has already timed out is rejected — a late
Slack click can't flip a recorded timeout into a grant. Conversely, if a human
approves in the same instant the SDK's deadline passes, the human decision wins.

---

## 4. Decision notes

A human deciding an approval can attach a **note**, and that note travels back
into the agent process.

> **The philosophy, in one line:** Dunetrace delivers the human's correction into
> the run with provenance; steering remains the agent's job.

Nothing is auto-appended to your prompt. The note arrives on the exception, in
the run's event stream, and on the approval record — what the agent *does* with
it is your code's decision, consistent with every other advisory action.

### The idiom

Catch `ApprovalDenied` as **expected control flow**, not as an error. Read the
note, feed it into the next planning step, retry — **inside the same run**.

```python
from dunetrace import ApprovalDenied, Dunetrace

dt = Dunetrace(api_key="dt_...", api_url="http://localhost:8002")

@dt.tool("delete_customer")
def delete_customer(customer_id: str) -> str:
    return crm.delete(customer_id)

with dt.run("support-agent", user_input="close the account for the Chen family") as run:
    matches = lookup_customer("Chen")          # three Chens come back
    target = pick_one(matches)                 # the agent guesses

    for attempt in range(2):
        try:
            delete_customer(target)            # blocks on a human
            break
        except ApprovalDenied as exc:
            if exc.status == "timeout" or not exc.note:
                run.final_answer()             # nobody to learn from — stop
                break
            # The human corrected us: "wrong Chen — it's Sarah, CUST_8834".
            # Replan with that as input and try again, in THIS run.
            target = replan(matches, correction=exc.note)
```

`exc.note` is `None` on a timeout — nobody decided, so nobody wrote anything.
`exc.decided_by` carries who decided, when the deciding surface attributed it.

Notes are allowed on **grants** too: *"approved, but only this once"* or
*"approved — use CUST_8834 going forward"* lands in the audit trail and in the
`approval.granted` event without raising anything.

### Notes are trusted input

A note is written by someone holding the `approve` scope — a credential the agent
process deliberately does not have. Nothing the model or a tool produces can put
words there, which makes a note the **highest-trust text in a run**. Dunetrace's
own detectors treat it accordingly:

- **`UNGROUNDED_DESTINATION`** counts a note as a grounded surface, so retrying
  with an address the human supplied is not flagged as ungrounded.
- **`UNRESOLVED_AMBIGUITY`** counts a note as a user-authored turn, so a
  selection the human named is warranted.

Without this the feature would defeat itself: the human corrects the agent, the
agent obeys, and our own detectors flag the *corrected* run. A tool call an
approval blocked is likewise never reported as an action the agent took — the
gate stopped it, so nothing happened.

Ordering still applies. A note cannot justify an action that preceded it.

### Limitation: notes are scoped to their own run

**The trust surfaces see a note within the run it was issued in. Retry inside the
run, or the correction won't warrant a later one.**

An agent that ends the run on denial and starts a *new* run for the retry leaves
the note behind in the old run's events — and the corrected behaviour in the new
run reads as ungrounded or unwarranted all over again. That is why the idiom
above retries in-place, and why it is the only pattern documented here.

Lifting notes across runs is a planned extension with the same shape as
`UNGROUNDED_DESTINATION`'s cross-run memory taint. Until it lands, the run
boundary is the boundary.

### Immutability

A note rides the terminal decision write and only that write, so it inherits the
decision's immutability: once recorded it never changes, and there is no
note-update endpoint — an audit trail you can edit afterwards is not an audit
trail. A decision that loses the race against an already-recorded one is
discarded *with its note*. Notes are capped at **2000 characters** and a longer
one is rejected, not truncated; the decision is not applied either, so you never
get half a note attached to a recorded outcome.

---

## API reference

| Endpoint | Purpose |
|---|---|
| `POST /v1/approvals` | Create a pending approval (the SDK does this) |
| `GET /v1/approvals` | List this org's approvals (`?status=pending`) |
| `GET /v1/approvals/{id}` | Poll one approval's status (the SDK does this) |
| `POST /v1/approvals/{id}/decision` | Record `granted` / `denied` / `timeout`, with an optional `note` (≤2000 chars) and `decided_by` |

All are scoped to the org from your API key. `approval.requested`,
`approval.granted`, `approval.denied`, and `approval.timeout` events are emitted
on the run so the decision shows up in the run timeline; the `granted` and
`denied` events carry `note` and `decided_by` when a human supplied them, so the
reasoning lives in the event stream and not only in the approvals table.

---

## See also

- [`examples/approval_agent.py`](../examples/approval_agent.py) — runnable end-to-end
- [Policies](policies.md) — the engine approvals build on
