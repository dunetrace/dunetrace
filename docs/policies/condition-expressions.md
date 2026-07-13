# Condition expressions

Expression conditions let a policy gate on the **values** inside a run — tool
call arguments, run metadata, event fields — not just the trigger type and tool
name. They answer requirements like *"require approval on `refund_customer` only
when `amount > 10000`"* or *"stop trial-tier agents that loop"*.

They are **additive**: a policy without an expression behaves exactly as before.
Everything here lives in a new optional `match` block inside a policy's
`condition`; the existing flat `{trigger, operator, value}` fields are unchanged.

```yaml
condition:
  trigger: before_tool_call     # existing fields — unchanged
  operator: eq
  value: refund_customer
  match:                        # NEW — the expression block
    args.amount: {gt: 10000}
action:
  type: require_approval
```

This reads: gate `refund_customer` **and** only when `args.amount > 10000`.

> **Contents:** [Field paths](#field-paths) · [Operators](#operators) ·
> [Composition](#composition-and-or-nesting) · [Coexistence with the flat
> trigger](#coexistence-with-the-flat-trigger) · [Type coercion](#type-coercion)
> · [Where `args` come from](#where-args-come-from) · [Common
> patterns](#common-patterns) · [Debugging](#debugging-why-did-my-policy-fire) ·
> [Limits & errors](#limits-and-errors) · [Signing](#signing)

---

## Field paths

A field path is a dotted string. The part before the first dot is the
**namespace**; the rest walks into nested keys.

| Prefix | Source | Examples |
|---|---|---|
| `args.*` | The tool call's arguments (only at the `before_tool_call` gate — see [below](#where-args-come-from)) | `args.amount`, `args.customer_id`, `args.customer.email` |
| `run.*` | Run metadata | `run.agent_id`, `run.duration_ms`, `run.event_count`, `run.step_count`, `run.tool_call_count`, `run.error_count`, `run.cost_usd` |
| `event.*` | The current event's fields | `event.type`, `event.tool_name`, `event.hour` (UTC 0–23), `event.timestamp` (unix) |
| `agent.*` | Agent metadata | `agent.tier`, `agent.model` — *not yet populated, see note* |
| `org.*` | Org metadata | `org.plan`, `org.tier` — *not yet populated, see note* |

**Nested paths** use dot notation: `args.customer.email` resolves
`args["customer"]["email"]`. Any missing segment makes the whole path **absent**.

**Missing paths are absent, not false.** A path that doesn't resolve makes every
operator except `exists`/`not_exists` evaluate to `false` (and logs a debug
line). Use `exists` / `not_exists` to reason about presence explicitly.

> **`agent.*` and `org.*` have no source yet.** They are valid to write — a
> policy referencing them loads and validates cleanly — but always evaluate as
> *absent* until a metadata channel lands (tracked in `BACKLOG.md`). Use
> `not_exists` if you want a condition that holds until then, or avoid them for
> now. `args.*`, `run.*`, and `event.*` are fully wired.

No wildcards. `args.*` as a literal path is not supported — name the field.

---

## Operators

Fourteen operators, whitelisted. No others. Each maps `<field> <operator>
<value>`.

| Operator | Meaning | Example | Notes |
|---|---|---|---|
| `eq` | equal | `{args.currency: {eq: "USD"}}` | value equality, with numeric coercion (see [Type coercion](#type-coercion)) |
| `ne` | not equal | `{args.status: {ne: "done"}}` | `neq` is accepted as an alias |
| `gt` | greater than | `{args.amount: {gt: 10000}}` | numeric or lexicographic |
| `gte` | greater than or equal | `{run.error_count: {gte: 3}}` | |
| `lt` | less than | `{args.qty: {lt: 100}}` | |
| `lte` | less than or equal | `{run.cost_usd: {lte: 0.5}}` | |
| `in` | value in list | `{org.plan: {in: ["free", "starter"]}}` | value **must** be a list |
| `not_in` | value not in list | `{args.region: {not_in: ["us", "eu"]}}` | value must be a list |
| `contains` | string/list contains | `{args.tags: {contains: "urgent"}}` | substring (string) or membership (list) |
| `starts_with` | string prefix | `{args.customer_id: {starts_with: "cus_"}}` | value must be a string |
| `ends_with` | string suffix | `{args.email: {ends_with: "@acme.com"}}` | value must be a string |
| `matches` | regex match | `{args.id: {matches: "^ord_\\d+$"}}` | ReDoS-safe engine, 5 ms timeout; string values only |
| `exists` | field is present | `{args.discount: {exists: true}}` | value ignored (present even if `null`) |
| `not_exists` | field is absent | `{args.approval_id: {not_exists: true}}` | value ignored |

**Multiple operators on one field** AND together — the way to express a range:

```yaml
match:
  args.amount: {gt: 10000, lt: 1000000}   # 10000 < amount < 1000000
```

---

## Composition (AND, OR, nesting)

**Multiple fields in one block AND together** (the default):

```yaml
match:
  args.amount: {gt: 10000}
  args.currency: {eq: "USD"}          # amount > 10000 AND currency == USD
```

**OR** via an explicit `or:` list of sub-blocks — any sub-block matching
satisfies it:

```yaml
match:
  args.amount: {gt: 10000}
  or:
    - agent.tier: {eq: "trial"}
    - org.plan: {in: ["free", "starter"]}
```

reads: `amount > 10000 AND (agent.tier == "trial" OR org.plan in ["free",
"starter"])`.

**AND** is also available explicitly as `and:` (useful for grouping ORs), e.g.
`(A OR B) AND (C OR D)`:

```yaml
match:
  and:
    - or: [{org.plan: {eq: "free"}}, {org.plan: {eq: "starter"}}]
    - or: [{args.destructive: {eq: true}}, {args.amount: {gt: 10000}}]
```

**Maximum nesting depth is 3.** The top-level block is level 1; each `or:`/`and:`
descends one level. A 4th level is rejected at load time. If you need deeper
logic, split it into multiple policies.

---

## Coexistence with the flat trigger

The `match` block ANDs with the existing flat condition — **both must hold**.
This is how expression conditions compose with the structural
[`signal` trigger](../policies.md#condition-reference):

```yaml
condition:
  trigger: signal                       # a detector signal fired…
  operator: contains
  value: TOOL_ARGUMENT_FABRICATION
  match:
    args.destructive: {eq: true}        # …AND the call is destructive
action:
  type: stop
```

reads: `signal contains TOOL_ARGUMENT_FABRICATION AND args.destructive == true`.

For a **pure-expression policy** (no metric/signal trigger), set the trigger to
the sentinel `expression`:

```yaml
condition:
  trigger: expression
  match:
    run.error_count: {gte: 3}
action:
  type: stop
```

---

## Type coercion

Deterministic and explicit — the same expression against the same value always
gives the same result. No clock reads, no locale.

- **Ordered** (`gt`/`gte`/`lt`/`lte`): numbers compare numerically. A number vs a
  numeric string coerces the string (`args.amount: {gt: 10000}` matches whether
  the arg is `10500` or `"10500"`). Two strings compare lexicographically
  (`"1.3.0" > "1.2.0"`). Anything incomparable (number vs non-numeric string,
  list vs number) is `false` — never an error.
- **`eq` / `ne`**: exact equality, plus the same number↔numeric-string coercion.
  **Booleans only equal booleans** — `eq: true` never matches the integer `1`,
  and `eq: 1` never matches `true`. Lists and tuples compare element-wise.
- **`in` / `not_in` / `contains`**: membership using the same `eq` semantics
  (so `{args.code: {in: [200, 404]}}` matches the string `"404"`).
- **`matches`**: regex against string values only. A non-string value is
  `false` (never stringified). Patterns run on a ReDoS-safe engine with a 5 ms
  timeout; a timeout is treated as no-match and logged.

---

## Where `args` come from

Raw tool-call arguments are only available at the **`before_tool_call` gate** —
i.e. a `require_approval` policy, evaluated *before* the tool runs. This is by
design: blocking on an argument value only makes sense before the call executes.

In the after-event metric path (the checks that run after `tool_called` /
`llm_responded` / `tool_responded`), `args.*` is **absent** — the tool already
ran, so `run.*` and `event.*` are the useful namespaces there. A policy that
needs `args.*` should use `trigger: before_tool_call` with a `require_approval`
action.

```yaml
# args.amount is live here — evaluated before refund_customer runs
condition:
  trigger: before_tool_call
  operator: eq
  value: refund_customer
  match: {args.amount: {gt: 10000}}
action: {type: require_approval}
```

---

## Common patterns

**High-value approval** — gate an action above a threshold:

```yaml
condition:
  trigger: before_tool_call
  operator: eq
  value: refund_customer
  match: {args.amount: {gt: 10000}}
action: {type: require_approval}
```

**Tier-based gating** — stricter limits for cheaper plans:

```yaml
condition:
  trigger: expression
  match:
    run.tool_call_count: {gt: 8}
    or:
      - agent.tier: {eq: "trial"}
      - org.plan: {in: ["free", "starter"]}
action: {type: stop}
```

**Destructive-argument guard** — pair a detector signal with an argument check:

```yaml
condition:
  trigger: signal
  operator: contains
  value: TOOL_ARGUMENT_FABRICATION
  match: {args.destructive: {eq: true}}
action: {type: stop}
```

**Prefix / format validation** — require approval on malformed identifiers:

```yaml
condition:
  trigger: before_tool_call
  operator: eq
  value: charge_card
  match: {args.customer_id: {starts_with: "cus_"}}
action: {type: require_approval}
```

More runnable examples in [`examples/policies/`](../../examples/policies/).

---

## Debugging: why did my policy fire?

Every evaluation can emit a structured record — the policy, whether the trigger
matched, each condition checked with **the value compared vs. expected**, and the
overall result — through two channels.

**1. Structured logs (always available, local).** Raise the
`dunetrace.policies.evaluation` logger to DEBUG:

```python
import logging
logging.getLogger("dunetrace.policies.evaluation").setLevel(logging.DEBUG)
```

Each evaluation logs one record; the structured payload is on the log record's
`policy_evaluation` attribute (a dict), so a log pipeline can query it. Example
reason: `did not fire: args.amount gt 10000 — actual 500`.

**2. Dashboard endpoint.** Enable reporting on the client:

```python
dt = Dunetrace(api_key="dt_live_...", policy_evaluation_reporting=True)
# or set DUNETRACE_POLICY_EVAL_REPORTING=1
```

Then read recent evaluations:

```
GET /v1/policies/{policy_id}/evaluations
```

returns the most recent records for that policy, each with `trigger_matched`,
`fired`, `reason`, and the per-condition `conditions` array (`field_path`,
`operator`, `expected`, `actual`, `result`, `present`).

Both channels are **rate-limited to 100 evaluations per policy per minute**;
beyond that a deterministic sample is kept and flagged `sampled: true`. Reporting
is **off by default** (it ships records over the network) — structured logging
needs no network and is the zero-config option. Enabling reporting has no effect
until the SDK is at least the version that added it.

---

## Limits and errors

Everything is validated at **load/create time**, never at fire time — a bad
policy is rejected up front, never silently mis-fires:

- **Unknown operator** → rejected with a suggestion: *"Unknown operator
  'greaterthan' … Did you mean 'gt'?"*
- **Unknown field prefix** → rejected (only `args`/`run`/`agent`/`org`/`event`).
- **Nesting past depth 3** → rejected.
- **Empty `match` block or empty `or:` list** → rejected (an empty block would
  match unconditionally).
- **Type mismatches** → `in`/`not_in` require a list; `starts_with`/`ends_with`
  require a string; `matches` requires a valid regex (compile-checked at load).

A malformed remotely-fetched policy is skipped with a warning rather than
breaking the whole policy load — the rest still apply.

---

## Performance

Evaluation is designed for the runtime hot path — `eval`-free, field paths
pre-split at parse time, operators pre-resolved to a whitelist table.

Measured (`scripts/bench_policy_expressions.py`, `scripts/loadtest_policy_conditions.py`;
Apple Silicon, single thread, GC-controlled amortized timing):

| Metric | Result |
|---|---|
| Single expression evaluation (mean) | ~3–12 µs |
| Single expression evaluation (worst-case p99) | ~31 µs — **under the 100 µs target** |
| 1000 policies loaded (RSS) | ~26 MB |
| Memory growth over 15 M evaluations | ~0 (no leak; Python-heap Δ < 1 KB) |
| Throughput, realistic (1000 policies across 50 agents → ~20 evaluated/event) | ~4,000 events/s |
| Throughput, worst case (1000 wildcard policies, all evaluated, none match) | ~140 events/s |

**Scaling note.** Per-*evaluation* cost is well under 100 µs. Per-*event* cost is
that times the number of policies actually evaluated for the event's agent —
policies scoped to other agents are skipped by a ~0.1 µs `agent_id` check before
any expression runs, and evaluation short-circuits at the first match. So the
lever for throughput is **scope policies to agents** (and give safety policies
high priority so they short-circuit early); a single process evaluating 1000
unmatched wildcard policies on every event is the pathological case (~140
events/s/core). In real deployments the SDK also runs distributed across the
customer's many agent processes, so aggregate event rates far above a single
core's are routine.

---

## Signing

Expression conditions are covered by the same HMAC-SHA256 policy signature as
everything else — the `match` block lives inside `condition`, which is fully
hashed, so **a tampered expression fails verification**. Policies using a `match`
block are signed under canonical-form **version 2** (`sig_version: 2`); legacy
policies stay version 1, byte-identical, so existing signed policies keep
verifying unchanged. See [Trust boundary](../policies.md#trust-boundary).
