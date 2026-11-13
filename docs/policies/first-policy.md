# Writing your first policy

Four steps: write it, watch it in dry run, read what it would have done, arm
it. The whole point of the middle two is that you arm it knowing the number
rather than guessing.

## 1. Pick something you have actually seen go wrong

Do not start from the trigger list. Start from a run that annoyed you.

A good first policy is one where you already know the answer. If your agent has
a tool-loop problem, you have runs that prove it, and a policy capping tool
calls has an obvious right threshold. If you are guessing at both the condition
and the threshold, dry run will tell you, but you will spend a week on it.

```python
dt.add_policy(
    name="cap-tool-calls",
    condition={"trigger": "tool_call_count", "operator": "gt", "value": 8},
    action={"type": "stop", "params": {"message": "Tool call cap reached."}},
    mode="dry_run",
)
```

That is the whole policy. Nine tool calls in one run stops it.

If the condition can never fire, registration fails immediately with an error
that names the fix. That is deliberate: a policy that registers and sits inert
looks identical to a working policy on a quiet day, so the engine refuses
rather than accepting. See
[Condition expressions](condition-expressions.md#field-paths) for the cases.

## 2. Let it run

Give it real traffic. A day is usually enough to tell a threshold that fires
constantly from one that never does; a week is better if your volume is low.

Nothing happens to your runs in the meantime. The policy evaluates and records
and does not act.

## 3. Read what it would have done

**Policies** in the dashboard, under the policy's row:

```
cap-tool-calls                              STOP  DRY RUN  billing-agent

23 of 412 runs (5.6%) in the last 14 days        [ Promote to enforcing ]
WOULD HAVE   stop — message=Tool call cap reached.
MATCHED ON   tool_call_count (23)
RUNS         8f2a1c4b step 9 · 1c7b90de step 11 · 44e0a2f1 step 9
```

Three questions to ask of that number:

**Is the rate plausible?** 5.6% of runs hitting a tool-call cap is a real
signal. 80% means the threshold is wrong, not that your agent is broken 80% of
the time. 0% means it never fires and you have learned the threshold is too
high.

**Are the runs the ones you meant?** Click through. The run opens with the
triggering step in focus and the tool arguments visible. If the runs it caught
were legitimately doing a lot of work, raise the threshold.

**Would stopping them have been right?** This is the one only you can answer,
and it is the reason for looking at the runs rather than only the rate.

Adjust and wait again if the answer is no. Changing the threshold does not
reset anything; new verdicts accumulate alongside the old ones.

## 4. Arm it

**Promote to enforcing** on the card, or:

```bash
curl -X PUT localhost:8002/v1/policies/12 \
  -H "Authorization: Bearer $DUNETRACE_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "enforcing"}'
```

The policy keeps its id and its history. Running agents pick the change up
within 60 seconds.

## Check it is actually enforcing

Two things quietly stop a policy from acting, and both are visible.

**The badge.** A policy showing **DEGRADED** in the policy list is not
enforcing. It requested an enforcing action but is unsigned, so every SDK
downgrades it to log-only. See
[Degraded policies](../policies.md#degraded-policies-not-enforcing) for the
fix.

**The startup log.** An agent whose policies are running degraded says so once
at load:

```
2 of 3 policies are running DEGRADED (downgraded to log-only because their
origin cannot be verified): 'cap-tool-calls' (wanted stop) ...
```

It logs when the set changes, not on every 60-second refresh, so a steady state
is quiet and a policy newly degrading is announced.

## What not to do first

**Do not start with `require_approval`.** It blocks the agent on a human, which
means a person has to be watching. Get a `stop` or a `log` right first.

**Do not put five policies in dry run at once.** You will not read five cards.
One policy, one question.

**Do not skip dry run because the policy looks obvious.** The thresholds that
look most obvious are the ones most often wrong, and the cost of checking is a
day of waiting.

## See also

- [Dry run](dry-run.md) — verdict fields, promotion, limits
- [Policies](../policies.md) — every trigger and action
- [Condition expressions](condition-expressions.md) — AND, OR, `args.*`
- [Approvals](../approvals.md) — once you are ready for a human in the loop
