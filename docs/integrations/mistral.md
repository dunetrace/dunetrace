# Mistral

Mistral shows up in Dunetrace in four independent places. You can use any one of
them without the others:

| Where | What it does | Turned on by |
|---|---|---|
| **SDK auto-instrumentation** | Records your agent's Mistral calls as `llm.called`/`llm.responded` events | `dt.auto_instrument()` (Python or TypeScript) |
| **Dunetrace's own LLM features** | Runs native explain, diff generation, detector translation and issue summaries on Mistral | `API_LLM_PROVIDER=mistral` |
| **Semantic evaluator provider** | Runs Tier 2 LLM evaluation *on* Mistral instead of OpenAI/Anthropic | `SEMANTIC_LLM_PROVIDER=mistral` |
| **Price tables** | Prices Mistral models for `cost_usd`, `COST_SPIKE`, and the `cost_usd` policy trigger | Always on |

---

## SDK auto-instrumentation

```bash
pip install 'dunetrace[mistral]'   # pulls mistralai>=2.0
```

```python
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_live_...")
dt.init(agent_id="my-agent")       # patches mistral along with everything else installed

with dt.run("my-agent", user_input=question):
    resp = client.chat.complete(model="mistral-large-latest", messages=[...])
```

`mistral` is one of the frameworks `auto_instrument()` patches — pass
`frameworks=["mistral"]` to patch only it. Like the `openai` and `anthropic`
patches, it never opens a run of its own: it records into whatever `dt.run()` is
already active and does nothing outside one. See
[auto-instrumentation.md](auto-instrumentation.md).

### What gets patched

| Class | Methods |
|---|---|
| `mistralai.client.chat.Chat` | `complete`, `complete_async`, `stream`, `stream_async` |
| `mistralai.client.embeddings.Embeddings` | `create`, `create_async` |
| `mistralai.client.fim.Fim` | `complete`, `complete_async`, `stream`, `stream_async` |
| `mistralai.azure.client.chat.Chat` | same four as chat |
| `mistralai.gcp.client.chat.Chat` | same four as chat |
| `mistralai.gcp.client.fim.Fim` | same four as chat |

`MistralAzure` and `MistralGCP` carry **their own** `Chat`/`Fim` classes in
separate modules — `mistralai.azure.client.chat.Chat` is not
`mistralai.client.chat.Chat` — so they need patching separately. They are, so a
hyperscaler-hosted deployment is instrumented the same as a direct one. Azure
ships chat only; GCP ships chat and fim; neither ships embeddings.

`Chat.parse` and `Chat.parse_stream` are deliberately **not** patched: they call
`complete`/`stream` internally, so patching both would record one API call twice.

**v2 only.** mistralai 2.0 moved every module under `mistralai.client` and
dropped the top-level `__init__.py`, so the import paths above are v2-only by
construction. v1 and the pre-1.0 `MistralClient` are not supported. If
`mistralai` isn't installed, the patcher logs one debug line and does nothing.

### Streaming

Streamed calls are wrapped in a proxy that counts tokens as chunks pass through
and emits one `llm.responded` when the stream ends. The proxy is transparent —
it is a real iterator and a context manager, and anything else falls through to
the wrapped stream, so provider-specific helpers keep working.

The response is recorded when you finish the stream, which means:

- **Draining it** (`for chunk in stream:`) emits at the end. This is the normal case.
- **Breaking out early** still emits — you paid for the chunks you read. The
  event is flushed when the run block exits, at the latest.
- **A stream that dies mid-flight** is recorded with `finish_reason="error"` and
  the exception text, not as a clean `stop`. This matters because a stream that
  fails before its first token would otherwise look like an empty response.

Mistral reports real token usage on the final chunk of a stream, so streamed
Mistral calls carry exact counts — no estimation. (Anthropic does too. OpenAI
only does when the caller passes `stream_options={"include_usage": True}`; see
[auto-instrumentation.md](auto-instrumentation.md#streaming) for what happens
otherwise.)

### Bedrock-hosted Mistral

Bedrock is reached through boto3's `bedrock-runtime`, not the mistralai SDK, so
the Mistral patcher never sees it — and botocore rides on urllib3, so the
`httpx`/`requests` patchers don't either. It has its own patcher:

```python
dt.init(agent_id="my-agent")   # includes botocore when boto3 is installed
```

That covers `Converse`, `ConverseStream`, `InvokeModel` and
`InvokeModelWithResponseStream` for **every** Bedrock-hosted model, not just
Mistral's. See
[auto-instrumentation.md](auto-instrumentation.md#auto-instrumentation) for what
each operation can report. Note that Bedrock model ids
(`mistral.mistral-large-2407-v1:0`) don't match the price tables' plain family
names, so those calls currently fall back to the default rate.

### TypeScript

`packages/sdk-ts` auto-instruments Mistral too:

```ts
import { Mistral } from "@mistralai/mistralai";
autoInstrument({ mistral: Mistral });
```

It patches `chat.complete` and `chat.stream`; `parse`/`parseStream` are left
alone because they call those internally and would double-count.

---

## Dunetrace's own LLM features

Native explain's root-cause analysis, fix diff generation, custom-detector
translation and issue summarisation each call an LLM on your behalf. They pick a
provider through `API_LLM_PROVIDER`:

```bash
API_LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
```

Unset, the first configured key wins — Anthropic, then OpenAI, then Mistral,
which is the behaviour these features have always had. Set explicitly, a missing
key for that provider is an error rather than a silent fall-through to another
vendor.

This is deliberately *not* `SEMANTIC_LLM_PROVIDER`: that selects the Tier 2
evaluator provider in a different service, and conflating them would mean
enabling semantic evaluation silently repointed these four features too. Set
both to keep everything inside one provider.

Mistral is reached here through the `openai` package pointed at Mistral's
OpenAI-compatible endpoint, rather than by adding `mistralai` as a dependency of
the API service.

---

## Mistral as the semantic evaluator provider

Tier 2 [semantic evaluation](../semantic-evaluation.md) is off by default. When
you turn it on, `SEMANTIC_LLM_PROVIDER` chooses which provider the evaluators
run on:

```bash
SEMANTIC_WORKER_ENABLED=true
SEMANTIC_LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
```

The default evaluator model is `mistral-small-latest` — the cost-conscious
default, matching `gpt-4o-mini` and `claude-haiku-4-5` for the other two
providers. Override per evaluator with `HALLUCINATION_MODEL`,
`TASK_COMPLETION_MODEL`, and the rest.

DeepEval 4.0.9 ships no Mistral model class, so Dunetrace provides its own
(`semantic_svc/evaluators/mistral_model.py`). It mirrors DeepEval's
`AnthropicModel`: same `(result, cost)` contract, same parse-the-text approach to
structured output. Token counts still flow into the evaluators' cost accounting,
and requests retry with exponential backoff on 429/5xx.

### No cross-provider fallback

**The selected provider is the only provider the primary evaluation path will
use.** If Mistral fails, the evaluation fails — it is never quietly retried
against OpenAI. An unrecognised `SEMANTIC_LLM_PROVIDER` is a startup error, not a
silent default to OpenAI.

The reason is data residency: a customer who selects Mistral has usually done so
to keep evaluation inside a European provider, and a fallback would ship the
run's text to a US API at exactly the moment they asked us not to.

### Second opinions stay in region

Evaluators with `require_second_opinion: true` in
[`semantic-evaluators.yml`](../config/semantic-evaluators.yml) confirm a
HIGH-confidence finding with a *second* model before trusting it. By default that
second model is a different **vendor** — which is fine for an OpenAI or Anthropic
primary, and wrong for a Mistral one.

So with `SEMANTIC_LLM_PROVIDER=mistral`, a `second_opinion_provider` naming
another vendor is **ignored**, with a warning, and replaced by a second Mistral
model (`mistral-large-latest`, or `mistral-medium-latest` if the primary is
already large). The shipped `HALLUCINATION` block names `anthropic`; that setting
is in effect only for an OpenAI/Anthropic primary.

To allow the cross-vendor second opinion anyway — for a deployment that chose
Mistral on cost rather than residency, and wants cross-vendor model diversity:

```bash
SEMANTIC_ALLOW_CROSS_PROVIDER_SECOND_OPINION=true
```

Nothing about this changes OpenAI/Anthropic primaries: they keep crossing to each
other as before.

---

## Pricing

Mistral models are priced in Dunetrace's model price tables, which feed
`cost_usd`, the `COST_SPIKE` detector, and the `cost_usd` policy trigger. Priced
families: `mistral-medium`, `mistral-large`, `mistral-small`, `ministral-3b`,
`ministral-8b`, `ministral-14b`, `codestral`, `mistral-embed`, and
`codestral-embed`. Version suffixes and `-latest` aliases resolve to the family
rate, and the two embedding models bill on input only.

Anything unlisted falls back to a default rate rather than erroring, so an
unrecognised model still produces a cost estimate — just an approximate one.
`magistral` and `devstral` are deliberately unpriced; add rows if you use them.

Prices are duplicated across three tables (SDK, explainer, Customer API) that are
kept in agreement by hand — see BACKLOG.md.

---

## What "Mistral support" does and doesn't cover

Worth being explicit, because the phrase can be read more broadly than it holds:

- ✅ Your agent's Mistral calls are recorded, priced, and run through all 29
  structural detectors and the policy engine.
- ✅ Tier 2 semantic evaluation can run entirely on Mistral, including second
  opinions, with no silent cross-provider fallback.
- ✅ Dunetrace's own LLM features (native explain, diff generation,
  custom-detector translation, issue summarisation) can run on Mistral via
  `API_LLM_PROVIDER=mistral`.
- ✅ Bedrock-hosted Mistral, through the `botocore` patcher, and the TypeScript
  SDK.
- ⚠️ Bedrock model ids don't resolve against the price tables yet, so those
  calls are costed at the default rate.
