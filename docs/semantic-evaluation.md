# Semantic Evaluation

Semantic evaluation is Dunetrace's Tier 2 detection layer — LLM-based
judgment of things structural pattern-matching can't catch: did the agent
hallucinate a fact, did it actually complete the task it was asked to do, is
the user showing signs of frustration across a multi-turn conversation.

**This is strictly post-hoc.** Every semantic evaluation runs after a run has
already completed, in a separate `semantic_worker` service, never inside the
SDK's request path. The SDK's sub-500μs per-event overhead claim holds
unconditionally — there is no code path by which a semantic evaluation can
add latency to your agent, because none of it runs until after your agent
has already returned. This is also why [policies](policies.md) (Dunetrace's
runtime-prevention mechanism) can only ever fire on structural signals: there
is nothing left to prevent once a run has finished, and semantic evaluation
by construction only starts once it has. See
[policies.md's "Structural signals only" section](policies.md#structural-signals-only)
for the full explanation.

If you're looking for the structural detectors instead (Tool Loop, Retry
Storm, Cost Spike, and the rest — the ones that can and do fire policies),
see [docs/detectors.md](detectors.md). There are 34 of them; the 31 that can
be evaluated in-path are the ones `trigger="signal"` policies can match.

---

## How a run gets evaluated

1. `semantic_worker` polls the shared `events` table on a 5-second interval
   (same poll-the-shared-table pattern `detector_worker`/`alerts_worker`
   already use — no message broker anywhere in this codebase) for
   completed/errored runs it hasn't seen yet.
2. **Adaptive sampling** decides whether this specific run gets evaluated at
   all — see below.
3. If sampled in, the configured evaluators run against the run's own
   events (never a second copy of your data — same events structural
   detectors already read).
4. Any finding is grouped into its recurring pattern, checked against
   accumulated false-positive feedback, optionally re-checked by a second
   model if it's a HIGH-severity finding from an evaluator configured to
   require one, and written to the same `failure_signals` table structural
   signals use — tagged `source='semantic'` (or the provider name, for
   [externally-sourced evaluations](integrations/external-evaluation.md)).

Disabled by default — set `SEMANTIC_WORKER_ENABLED=true` to turn it on; this
preserves existing OSS behavior for anyone not opting in.

**You also need the `semantic` container running.** It's in both
`docker-compose.yml` and `docker-compose.ghcr.yml`, but with the flag off it
logs one line and exits 0 — so it shows as `Exited (0)`, which is expected, not
a failure. Setting the flag and re-running `docker compose up -d` starts it for
real.

Per-evaluator model overrides are read from `HALLUCINATION_MODEL`,
`TASK_COMPLETION_MODEL`, `TASK_UNDERSTANDING_FAILURE_MODEL`,
`OFF_TOPIC_DRIFT_MODEL`, `USER_FRUSTRATION_MODEL`, `CONFUSION_LOOP_MODEL`, and
`SYCOPHANCY_SIGNAL_MODEL`. Leave one blank to use the provider default chosen by
`SEMANTIC_LLM_PROVIDER` (`openai`, `anthropic`, or `mistral` — defaults
`gpt-4o-mini`, `claude-haiku-4-5`, `mistral-small-latest` respectively). Each
provider reads its own key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`MISTRAL_API_KEY`. There is no cross-provider fallback — a failure surfaces as a
failure, and an unrecognised provider name is a startup error. Mistral also
keeps second opinions in region; see
[integrations/mistral.md](integrations/mistral.md).

---

## Adaptive sampling

Evaluating every run through an LLM is expensive at scale, so most runs are
sampled rather than always evaluated. Rules, in priority order (config in
[`docs/config/semantic-sampling.yml`](config/semantic-sampling.yml)):

| Rule | Default rate | Why |
|---|---|---|
| A structural detector already fired on this run | 100% | Augment a known problem with semantic context — highest value per evaluation |
| Agent flagged `semantic_critical=true` | 100% | Customer explicitly said this agent's correctness matters enough to always check |
| Run had a retrieval event (`retrieval.called`/`retrieval.responded`) | 20% | RAG is the highest hallucination-risk pattern |
| Everything else | 5% | Baseline coverage without evaluating every single run |

Sampling is **deterministic**: `hash(run_id) % 100 < rate` means the same
run always gets the same sampling decision — no flapping between "evaluated"
and "not evaluated" on retries or re-queries.

Per-agent overrides live in the `agent_semantic_config` table (not the YAML
file): `sample_rate` (overrides the retrieval/baseline rates for that
agent's general population — does not override the structural-signal or
`semantic_critical` 100% rules), `budget_monthly` (a per-agent evaluation
cap, independent of the org-wide billing quota below), and `evaluators`
(which evaluators to run when sampled in).

---

## Evaluators (DeepEval-backed)

Seven evaluators ship, all wrapping [DeepEval](https://github.com/confident-ai/deepeval)
metrics behind a Dunetrace-native interface (`services/semantic/semantic_svc/evaluators/base.py`)
so they can be swapped or extended without touching worker code. Four are
run-level; three (below) operate across a conversation:

- **`HALLUCINATION`** — did the agent state something as fact that its own
  context doesn't support.
- **`TASK_COMPLETION`** — did the agent actually do what it was asked, not
  just respond plausibly (thoroughness of the right task).
- **`TASK_UNDERSTANDING_FAILURE`** — did the agent solve the *wrong* problem:
  its response addresses a different task than the user requested. Distinct from
  `TASK_COMPLETION` (which is about how fully the *right* task was done) and from
  `HALLUCINATION` (factuality). A fully-complete answer to the wrong question
  fails this while passing the other two. Calibrated on 108 labeled runs (margin
  0.21 vs 0.60 between wrong- and right-task; 0% false positives at the default
  0.5 threshold, including partial / ambiguous / multi-part hard negatives) —
  see `scripts/calibration/task_understanding_failure_calibration.md`.

Two more operate on conversations rather than single runs — see
[Conversation-level evaluation](#conversation-level-evaluation) below:

- **`USER_FRUSTRATION`** — is the user getting frustrated across the
  conversation (complaints, sarcasm, escalating tone).
- **`CONFUSION_LOOP`** — is the user re-asking the same underlying question
  because the agent's answers aren't resolving their intent. Distinct from
  frustration: it's about repeated *intent*, not emotional tone — a calm user
  politely re-asking the same thing three times is a confusion loop.
  Calibrated on 108 labeled conversations (score margin 0.20 vs 0.83 between
  loops and non-loops; 0% false positives including hard negatives at the
  default 0.5 threshold) — see `scripts/calibration/confusion_loop_calibration.md`.
- **`SYCOPHANCY_SIGNAL`** — did the agent flip-flop its position to agree with
  the user because they pushed back, rather than because they gave it a good
  reason. Correcting a genuine mistake on a user-supplied correct fact is *not*
  sycophancy. Calibrated on 100 conversations; because caving-to-pressure and
  legitimate-correction genuinely overlap, it ships precision-first at threshold
  **0.4** (0% false positives, 88% recall) rather than the standard 0.5 — see
  `scripts/calibration/sycophancy_signal_calibration.md`.
- **`OFF_TOPIC_DRIFT`** (run-level) — did the response start on the user's
  question and then wander off it (unrelated features, marketing, a tangent).
  Distinct from `TASK_UNDERSTANDING_FAILURE`: drift starts on-topic and loses
  the thread, rather than misreading the task from the start. Covering multiple
  related aspects of a broad question is not drift. Calibrated on 100 runs (margin
  0.44 vs 0.82; 0% false positives including broad multi-aspect answers at the
  default 0.5 threshold) — see `scripts/calibration/off_topic_drift_calibration.md`.

LLM provider is configurable per evaluator (`SEMANTIC_LLM_PROVIDER`,
default OpenAI `gpt-4o-mini` for cost; upgradeable to `gpt-4o` or
Anthropic's `claude-sonnet-4-6` on demand). No custom Dunetrace-proprietary
evaluators ship in v1 — only DeepEval's own metrics; a custom evaluator for
a failure mode DeepEval doesn't cover is a deliberate later step, not a v1
scope item.

---

## False positive management

Every semantic finding ships with the tooling to keep false positives from
eroding trust in the feature:

**Confidence scoring** — every finding has a 0.0–1.0 confidence score,
shown in the dashboard. Findings below a per-evaluator floor (default 0.6,
[`docs/config/semantic-evaluators.yml`](config/semantic-evaluators.yml)) are
still stored and visible, but never alerted.

**Grouping and dedup** — findings from the same agent + evaluator +
normalized root-cause pattern (a hash of the evaluator's reasoning text,
`signal_groups`/`signal_group_members` tables) are grouped into one
recurring issue rather than N separate alerts, tracking `first_seen`,
`last_seen`, and affected run count.

**Feedback capture** — a "not a real issue" action in the dashboard records
feedback per signal. Opt-in per org (`PATCH /v1/orgs/semantic-feedback`,
off by default). Once a recurring pattern accumulates 3+ false-positive
marks, future findings matching that pattern get suppressed if
`auto_suppress` is enabled for the org, or otherwise have their confidence
lowered.

**Second-opinion evaluation** — for a HIGH-severity finding from an
evaluator with `require_second_opinion: true` (Hallucination, by default —
the highest-consequence failure mode, worth the added cost), a second
evaluator run using a *different* model must independently agree before the
finding stays at HIGH severity. Disagreement downgrades it to MEDIUM.
Roughly doubles cost for the runs it applies to — configured per evaluator
in `docs/config/semantic-evaluators.yml`, off by default for lower-stakes
evaluators like Task Completion.

---

## Billing and quotas

Two independent quota layers, both enforced in the sampling/evaluation path
(a skipped evaluation writes no signal and costs nothing):

- **Org-wide monthly quota** (`organizations.semantic_evaluation_quota`,
  default 1000/month) — the mandatory plan-tier ceiling across every agent
  in the org combined. `allow_semantic_overage` (default false) stops
  sampling once hit; set true to keep evaluating (and billing) past it.
- **Per-agent budget** (`agent_semantic_config.budget_monthly`, optional,
  `NULL` = unlimited) — a customer-configurable sub-limit for a specific
  agent, independent of the org-wide quota.

`GET /v1/orgs/semantic-usage` returns current usage, remaining quota, and a
linear month-end usage/cost projection
(`api_svc/semantic_usage.py` — deliberately simple extrapolation, not a
forecasting model). Hitting the org quota with overage disabled emits an
internal `SEMANTIC_QUOTA_EXCEEDED` signal (shadow-only, an operational
signal, not a customer-facing failure) rather than failing silently.

Conversation-level evaluation (below) has its own separate quota
(`conversation_evaluation_quota`, default 200/month) — it's deliberately more
expensive per call than single-run evaluation, so it can't silently eat the
per-run budget or vice versa.

Every evaluation call (fired or not) is logged with token counts and cost in
`semantic_evaluation_log` — the honest source for cost reporting, since
`failure_signals` only ever captures cost for evaluations that actually
fired, and most evaluations on healthy runs don't.

---

## Conversation-level evaluation

Semantic evaluation isn't limited to single runs. Pass `conversation_id` to
`dt.run()` to group multiple runs into one conversation:

```python
with dt.run("support-agent", conversation_id="conv_123") as run:
    ...
```

Backward compatible — omitting `conversation_id` behaves exactly as before
(the run simply isn't associated with any conversation). Once a conversation
has accumulated at least 3 runs, it becomes eligible for **conversation-level**
sampling (separate from the per-run rates above, default 50% —
`conversation_evaluation_rate` in `semantic-sampling.yml`), bucketed on the
conversation's own id so a sampled conversation stays sampled for its whole
lifetime.

The **`USER_FRUSTRATION`** evaluator reads the last N runs in a sampled
conversation and evaluates frustration signals across the whole exchange —
something no single-run evaluator can see. See
[docs/dashboard.md](dashboard.md) for the conversation-detail view.

---

## What this is not

- Not a replacement for structural detection — semantic evaluation
  *augments* it (100% of structurally-flagged runs also get semantic
  context) and catches a different class of failure, but the 34 structural
  detectors remain the always-on, zero-cost, zero-LLM first line.
- Not a runtime guardrail — see the policies cross-link above.
- Not required — `SEMANTIC_WORKER_ENABLED` defaults to off; everything else
  in Dunetrace works identically without it.
- Not a substitute for an existing Langfuse/LangSmith/Braintrust setup if
  you already have one — see
  [docs/integrations/external-evaluation.md](integrations/external-evaluation.md)
  to pull those results in instead of (or alongside) Dunetrace's own
  evaluators.
