# UNRESOLVED_AMBIGUITY shadow-mode exit data

Reproduce with `python scripts/shadow_run_unresolved_ambiguity.py` (add
`--verbose` to print every signal). Structural, deterministic — no LLM, no API
key, no database.

This is a **base-rate measurement, not a threshold sweep**, and that is not a
shortcut. Every other detector shipped with a `scripts/calibrate_*.py` because it
had a numeric boundary to place. This one has none: the verdict is a set
operation on tokens, and the single operator-facing knob (`irreversible_tools`)
is a *declaration*, not a tuning parameter. There is no value of it that trades
precision against recall — it decides which tool calls are in scope at all. What
does need measuring before promotion is the rate at which the mechanism fires on
real traffic, which is what this records.

## Result

144 distinct recorded runs across every corpus in `dunetrace-demos`:

| corpus | runs | reached a candidate set | strong | weak |
|---|---:|---:|---:|---:|
| `demo1/backup-traces` | 16 | 0 | 0 | 0 |
| `demo4/runs` | 128 | 0 | 0 | 0 |
| **total** | **144** | **0** | **0** | **0** |

**Signals per 100 runs: 0.00 strong, 0.00 weak.**

Declared irreversible for the measurement (deliberately generous — the list can
only over-report, so a low rate measured this way is a real result and a high one
would be an upper bound): `delete_customer`, `refund_customer`, `issue_refund`,
`send_email`, `cancel_subscription`, `cancel_order`, `update_customer`.
Read-only lookups are excluded; declaring one irreversible would be a category
error, not a conservative choice.

## Why the rate is zero

A zero is only worth anything if you can say what produced it. Each run is
charged to the first stage of the detector it fails to reach:

| runs | stopped at |
|---:|---|
| 106 | no irreversible tool call in the run at all |
| 34 | an irreversible call, but no prior tool result held ≥ 2 records — there was nothing to choose between |
| 4 | an irreversible call, but no tool output captured anywhere in the run |

Nothing is being suppressed by a threshold. **No recorded run in these corpora
ever contains the shape this detector looks for**: an irreversible call
downstream of a lookup that returned more than one candidate.

The two most interesting groups are worth naming, because both are the detector
being *correctly* silent rather than blind:

- **`demo1/backup-traces/fabricated_destructive` (4 runs).** The agent calls
  `delete_customer('CHEN01')` having never run a lookup. The identifier appeared
  in no prior result, so there is no candidate set and no selection to warrant.
  That is `TOOL_ARGUMENT_FABRICATION`'s finding, and this detector must not
  double-report it. These four runs are the closest thing in the corpus to the
  motivating example — same agent, same *"close the account for the Chen
  family"* request — and they miss it in the one way that puts them out of
  scope: the agent skipped the lookup instead of choosing badly from it.
- **`demo4/runs` act2 (30 runs).** `issue_refund(order_id='ORD-4471')` after a
  `search_knowledge_base` that returns exactly one record, which does not contain
  `ORD-4471`. No multi-record set, no anchor.

## Positive control

A zero fire rate and a broken harness print the same number, so the script ends
by pushing the worked example through the *same* path the corpora take — raw
event dicts → `build_run_state()` → the detector:

```
  weak  ambiguous request: MEDIUM conf=0.6 selected=CUST_1183 of 2
        unused=['3000', 'cust_1183', 'emily', 'standard']
strong  request names the sibling: HIGH conf=0.9 selected=CUST_1183 of 2
        unused=['3000', 'cust_1183', 'emily', 'standard'] matched=['sarah']
```

Both tiers fire with the contracted severity and confidence. The zero above is a
statement about the corpora, not about the pipeline.

Mechanism correctness is covered separately and exhaustively by
`packages/sdk-py/tests/test_unresolved_ambiguity.py` — 68 cases built from the
real `customers.py` records (a fidelity test holds those copies to
`customers.py` itself whenever a demos checkout is present), including all five
misselection rows, all seven warranted rows, all seven genuine-ambiguity rows,
and every suppression rule.

## In-path cost

The shipped default declares nothing irreversible, so the in-path instance costs
one attribute check — **~1.7µs p50** — on an unconfigured agent. Once
`irreversible_tools` is declared, cost scales with the size of the candidate set:
a 3-record set completes inside the 1ms in-path budget, and larger sets hit the
`max_scan_ns` abort and **fail open**, exactly like `UNGROUNDED_DESTINATION`. The
server-side instance runs with the 50ms class default and completes a set at the
50-record cap in single-digit milliseconds. Degradation is prevent → detect,
never detect → nothing.

## Review findings folded in

An adversarial review of the first working draft raised 27 findings across four
lenses; two survived independent refutation, and a further nine were reproduced
and fixed before this measurement was taken. The ones worth recording because
they changed behaviour rather than prose:

- **A narrowed lookup returns a bare object, not a one-element list.** The demos'
  own `lookup_customer` returns `{"customers": [...]}` for a name match and a
  bare record `dict` for an id/email match. Only lists were recognised, so the
  spec's canonical narrowing case walked past the narrowed result to the wider
  one and reported ambiguity for a run that had resolved it.
- **`max_depth` truncated a record silently** instead of failing open, which can
  turn a token two records genuinely share into a discriminator of one of them —
  the detector manufacturing the evidence it is weighing, up to a HIGH-severity
  `strong`.
- **Warrant was not ordered against the action.** A user turn recorded *after*
  the irreversible call counted as justification for it.
- **A rejected call (`success = false`) was reported as an irreversible action.**
- **The scan budget was polled only between irreversible calls**, so neither the
  anchor walk nor the per-candidate tokenisation was bounded.

Each has a named regression test in the acceptance suite.

## What this does and does not license

It establishes that the detector **does not fire spuriously**: 144 real runs, a
generous irreversible-tool declaration, zero false alarms. It does **not**
establish a real-traffic rate for either tier, because no run in the corpus
reached the evaluation stage. Zero of zero is not a rate.

## Promotion bar

Recorded here so it is on the record before anyone reaches for `LIVE_DETECTORS`:

- **`strong` is promotable on mechanism-correctness alone.** Its claim — the
  request's tokens identify a *different* record in the same candidate set — is
  provable from the trace with no knowledge of intent. If the mechanism is
  right, the finding is right, and a base rate does not change that.
- **`weak` stays dashboard-only** until its observed rate on real traffic is
  understood. WEAK is correct-but-interrupting by design: it fires on a
  lucky-correct guess, because an agent that guessed right still guessed. Its
  correct actuation is an **approval gate**, not an alert — the human is the only
  party who knows which candidate was meant. Alerting on it before knowing how
  often it fires would train operators to ignore the detector, which is the
  failure mode that matters most for a control whose whole value is that someone
  reads it.

## Corpus limitation, stated plainly

`dunetrace-demos` was built to exercise fabrication, injection, memory poisoning
and delegation loops. It contains no scenario where an agent looks up an
ambiguous name, gets several records back, and acts on one of them — which is
exactly the gap this detector was built for, and exactly why the corpus cannot
measure it. Closing that needs either a new demo scenario or real customer
traffic with `irreversible_tools` declared. Until then the shadow period has
produced a clean negative result and nothing more.
