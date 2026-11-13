# Integrating LiteLLM with Dunetrace

<!--dunetrace:instrument
framework: litellm
language: python
install: pip install dunetrace openai
primary_symbol: dunetrace.Dunetrace.agent
mechanism: decorator
opens_own_run: true
requires_run_context: false
target_file: the module holding the OpenAI client pointed at your LiteLLM Proxy
target_example: `app/agent.py`, `src/llm.py`
target_hints: ["localhost:4000", "litellm.completion(", "from litellm import"]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded]
verify_cmd: python main.py  # then check the dashboard
-->

## Quick Start

```bash
pip install dunetrace openai
```

```python
from openai import OpenAI
from dunetrace import Dunetrace

client = OpenAI(base_url="http://localhost:4000/v1", api_key="no-key")   # your LiteLLM Proxy

dt = Dunetrace()   # local dev, no API key needed
dt.init(agent_id="my-agent")
dt.auto_instrument(["openai"])

@dt.agent(model="gpt-4o-mini")
def answer(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or ""

print(answer("What is the capital of France?"))
```

Requires a running LiteLLM Proxy (`litellm --config litellm.yaml --port 4000`) and the Dunetrace backend (`docker compose up -d`).

## Where this goes

**First, check which LiteLLM you have. They need different integrations and
only one of them is covered by this guide.**

```bash
grep -rn "litellm.completion(\|from litellm import" --include=*.py .
```

**Any hit means direct-call LiteLLM, and this guide does not apply.** Calls to
`litellm.completion()` never construct an OpenAI client, so the `openai` patch
has nothing to attach to and you will get **zero events with no error**. Use
[integrate-custom-python-agent.md](./integrate-custom-python-agent.md): wrap the
calling function in `@dt.trace` and emit `run.llm_called()` /
`run.llm_responded()` around the call yourself.

No hits, and an OpenAI client pointed at a proxy on port 4000? Then this guide
is right. `dt.auto_instrument(["openai"])` goes once at startup, and `@dt.agent`
goes on the function that owns one agent turn, which is what opens the run the
patch attaches to.

```bash
grep -rn "base_url=.*4000\|litellm" --include=*.py .
```

There is no LiteLLM-specific code. Dunetrace patches the OpenAI client, and a
LiteLLM Proxy is an OpenAI-compatible endpoint, so pointing the client at the
proxy is the whole integration.

Direct `litellm.completion()` calls bypass the OpenAI client and are not
covered. Use `dt.run()` with manual events for those.

## What this does

LiteLLM Proxy exposes an OpenAI-compatible endpoint. Dunetrace already patches the OpenAI Python client (via `dt.auto_instrument(["openai"])`), so any OpenAI client pointed at your LiteLLM Proxy gets auto-tracked with zero LiteLLM-specific code — model name, tokens, latency, and raw prompt/completion are all captured. Switching models is just a name change (`model="claude-haiku"`, etc.) — Dunetrace records whatever model string you pass.

If you call `litellm.completion()` directly instead of going through an OpenAI client, use `dt.run()` and emit `run.llm_called()`/`run.llm_responded()` manually — this auto-instrumentation path only covers the OpenAI-compatible HTTP interface.

## Verification

Run your agent once, then check the dashboard at `http://localhost:3000` — it should appear within ~15 seconds. Repeat the same tool call or return an empty response to confirm detectors fire.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`agent`) actually ran, and ran **before** the first agent invocation.

2. **Did the process flush?** Events ship from a background thread. Call
   `dt.shutdown()` before the process exits, or the buffer dies with it.

3. **Is the backend reachable?** Both should return `{"status":"ok",...}`:

   ```bash
   curl -s localhost:8001/ready   # ingest
   curl -s localhost:8002/ready   # customer API
   ```

4. **Did anything land?** If your agent id is listed here, instrumentation is
   working and the problem is downstream:

   ```bash
   curl -s localhost:8002/v1/agents
   curl -s "localhost:8002/v1/agents/<your-agent-id>/runs?limit=5"
   ```

5. **Turn on debug logging.** `Dunetrace(debug=True)` logs every event as it is buffered and
   every batch as it ships.

**Runs appear but no signals?** That is usually correct, not a fault. The
detector polls every 5 seconds, and a healthy run produces no signals. Several
detectors also need cross-run baselines and stay dormant until the agent has
run history. Check `docker compose logs detector` if you expected one.

---

## Advanced (optional)

### Troubleshooting

- **No runs appear** — confirm `dt.auto_instrument(["openai"])` runs before the first call, and the client's `base_url` points at your LiteLLM Proxy
- **LiteLLM returns 401 / provider errors** — check the proxy config and provider keys; Dunetrace doesn't manage LiteLLM credentials
- **Direct `litellm.completion()` calls aren't tracked** — use the OpenAI-compatible proxy path above, or instrument manually with `dt.run()`
- **Token counts missing** — come from the OpenAI-compatible response; if the upstream provider omits usage, detectors still run on step counts and latency
