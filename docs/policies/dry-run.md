# Dry run

A policy in dry run evaluates its condition exactly as an enforcing one does,
records what it would have done, and executes nothing.

The reason to have it is narrow and specific. Most people never arm a policy,
and the reason is not that the language is too limited. It is that arming a
`stop` policy against production traffic is an irreversible-feeling decision
made on a guess. Dry run turns that guess into a number: this policy would have
stopped 23 of your last 412 runs, here are the runs, here is the step.

## Start from a real policy

```python
condition = {
    "trigger": "before_tool_call",
    "operator": "eq",
    "value": "refund_customer",
    "match": {
        "args.amount": {"gt": 10000},
        "or": [
            {"run.error_count": {"gte": 2}},
            {"run.cost_usd": {"gt": 5.0}},
        ],
    },
}
```

Read that as: the agent is about to call `refund_customer`, **and** the amount
is over 10,000, **and** the run has either already failed two tool calls or
already spent more than five dollars.

Everything about the policy engine that matters is in those nine lines. The
flat `{trigger, operator, value}` picks the tool. Keys inside a `match` block
AND together. `or` takes a list of alternatives. `args.*` is the tool's own
arguments, available here because this is a `before_tool_call` gate;
`run.*` is the run's accumulated state.

Now, would you turn that on against live traffic today? You do not know how
often it fires. That is what dry run answers.

## Putting a policy in dry run

```python
dt.add_policy(
    name="high-value-refund-approval",
    condition=condition,
    action={"type": "require_approval", "params": {"timeout_s": 300}},
    mode="dry_run",
)
```

Or set **Mode: dry run** when creating the policy in the dashboard.

New policies are **enforcing** unless you say otherwise. A guardrail that
quietly does nothing by default is the failure this engine is built to avoid,
so dry run is something you opt into, not something you have to opt out of.

## What dry run changes, and what it does not

| | Enforcing | Dry run |
|---|---|---|
| Condition evaluated | yes | yes, identically |
| Verdict recorded | yes | yes |
| Action executed | yes | **no** |
| `policy.triggered` event emitted | yes | no |
| Counted once per run | yes (except `log`) | yes (except `log`) |
| Verdict sampled | yes, 100/min then 1-in-10 | **never** |
| Needs evaluation reporting turned on | yes | no |

Two of those rows are load-bearing.

**Verdicts are never sampled.** Ordinary evaluation records are rate limited so
a hot loop cannot flood the table. A dry-run firing bypasses that, because the
question being asked is "how many times would this have fired" and a sampled
count answers it wrong.

**Counted once per run.** An enforcing `stop` fires once and ends the run. A
dry-run `stop` that recorded a verdict on every later matching step would
report six firings for something that would really have happened once, and the
fire rate would be nonsense. Dry run inherits enforcement's dedupe exactly,
including the exception for `log` actions, which fire on every match in both
modes.

## Reading the verdicts

Each dry-run firing records enough to reconstruct what would have happened
without going back to the policy:

| Field | Meaning |
|---|---|
| `policy_id`, `policy_name` | Which policy |
| `run_id`, `step_index` | Which run, and where in it |
| `trigger` | The flat trigger that matched |
| `matched_branch` | Which part of the condition carried the match |
| `would_action_type` | The action that would have executed |
| `would_action_params` | Its parameters, verbatim |
| `mode` | `dry_run` |
| `evaluated_at` | When |

They live in `policy_evaluations` alongside enforcing evaluations, not in a
separate table. That is deliberate: a dry-run verdict **is** an evaluation, and
one table is what keeps promotion clean. Flipping mode changes nothing about
rows already written, so the history from before enforcement stays readable
next to the history after it.

Every added column is nullable. A row written before dry run existed does not
know its step or its would-be action, and `NULL` is the honest reading of that.

### In the dashboard

Each dry-run policy carries a card under its row in **Policies**:

```
cap-tools                                  STOP  DRY RUN  billing-agent

23 of 412 runs (5.6%) in the last 14 days        [ Promote to enforcing ]
WOULD HAVE   stop — message=too many tools
MATCHED ON   tool_call_count (19) · tool_call_count + run.error_count (4)
RUNS         8f2a1c4b step 3 · 1c7b90de step 5 · 44e0a2f1 step 3
```

The run links open that run with the triggering step in focus, so the verdict
and the tool arguments that produced it are on one screen. A verdict on its own
is a claim; a verdict next to the arguments is evidence.

### Over HTTP

```bash
curl -s "localhost:8002/v1/policies/12/dry-run-summary?days=14"
```

The fire rate is computed server-side, with numerator and denominator drawn
from the same agent and the same window. `would_fire_runs` counts distinct
runs, not rows. `fire_rate` is `null` rather than `0` when there were no runs
to compare against, because "fired on nothing" and "nothing to compare" are
different answers.

## Promoting

Promotion is one operation. In the dashboard, **Promote to enforcing** on the
card. Over the API:

```bash
curl -X PUT localhost:8002/v1/policies/12 \
  -H "Authorization: Bearer $DUNETRACE_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "enforcing"}'
```

The policy keeps its id, so every verdict written while it was in dry run stays
attached to it. You can still read what it would have done before you armed it,
next to what it has done since.

The agent picks the change up on its next policy refresh, within 60 seconds.

**A promoted policy is signed at v3** (see
[Signing](condition-expressions.md#signing)). If your deployment sets
`POLICY_SIGNING_SECRET`, the promotion re-signs it automatically.

## When there is nothing to promote

```
No firings in the last 14 days. Either the condition is not being met, or this
agent has not run. Nothing to promote yet.
```

That is a real answer, not an error. A policy that would never have fired is a
policy you have learned something about. Check the threshold against
[the condition reference](../policies.md#condition-reference), or widen the
window with `?days=`.

## Limits

Dry run previews what a policy would do **from now on**. It does not replay
history: a policy put into dry run today has no verdicts for last week's runs.
Backtesting against historical runs would write into this same verdict
structure, and is not built.

Dry run runs where enforcement runs, in the request path, and answers to the
same latency budget. It adds one branch and one record per firing.

## See also

- [Writing your first policy](first-policy.md) — the whole loop, end to end
- [Policies](../policies.md) — triggers, actions, trust boundary
- [Condition expressions](condition-expressions.md) — the `match` grammar
