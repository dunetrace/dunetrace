# Integrating a LlamaIndex RAG Agent with Dunetrace

<!--dunetrace:instrument
framework: llamaindex
language: python
install: pip install dunetrace llama-index
primary_symbol: dunetrace.Dunetrace.trace
mechanism: decorator
opens_own_run: true
requires_run_context: false
target_file: the function that calls query_engine.query()
target_example: `app/rag.py`, `src/query.py`
target_hints: ["query_engine.query(", "as_query_engine(", ".aquery("]
emits: [run.started, run.completed, run.errored, retrieval.called, retrieval.responded]
verify_cmd: python -c "import app.rag"  # then run one query
-->

## Quick Start

```bash
pip install dunetrace llama-index
```

```python
# rag.py
import time

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from dunetrace import Dunetrace, get_current_run

dt = Dunetrace()   # local dev, no API key needed

# Any query engine works. This one indexes ./data so the file runs as written.
query_engine = VectorStoreIndex.from_documents(
    SimpleDirectoryReader("data").load_data()
).as_query_engine()

@dt.trace("llamaindex-agent", model="gpt-4o-mini", tools=["query-engine"])
def answer_question(question: str) -> str:
    run = get_current_run()
    run.retrieval_called(index_name="my-index", query=question)

    t0 = time.perf_counter()
    response = query_engine.query(question)
    nodes = list(getattr(response, "source_nodes", None) or [])

    run.retrieval_responded(
        index_name="my-index",
        result_count=len(nodes),
        top_score=max((n.score for n in nodes if n.score is not None), default=None),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return str(response)

if __name__ == "__main__":
    print(answer_question("What does the documentation say about setup?"))
    dt.shutdown()
```

```bash
mkdir -p data && echo "Setup: run docker compose up -d." > data/notes.txt
OPENAI_API_KEY=sk-... python rag.py
```

Start the backend once, locally, before running this: `docker compose up -d`.

## Where this goes

`@dt.trace` goes on the function that owns a whole question-and-answer
round trip, the one that calls `query_engine.query()`.

```bash
grep -rn "query_engine.query(\|as_query_engine(" --include=*.py .
```

Typically `app/rag.py` or `src/query.py`. Decorate the wrapping function, not
the query engine construction.

LlamaIndex has no callback integration in this SDK, so the retrieval events are
emitted by hand inside the traced function. That is the one place this guide
asks you to add calls rather than a decorator.

## What this does

`@dt.trace` opens and closes a Dunetrace run around your query function. Inside it, `retrieval_called()`/`retrieval_responded()` record the query text, result count, and top similarity score from LlamaIndex's `response.source_nodes` — enough for `RAG_EMPTY_RETRIEVAL` to catch a retrieval that returned nothing but still got answered. `get_current_run()` returns `None` outside a traced call, so guard with `if run:` if you reuse the helper elsewhere.

## Verification

Run your agent once, then check the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds. Ask a question with no matching indexed content and confirm `response.source_nodes == []` triggers `RAG_EMPTY_RETRIEVAL`.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`trace`) actually ran, and ran **before** the first agent invocation.

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

### Async query engines

Same pattern with `await query_engine.aquery(question)` inside an `async def` traced function.

### What's captured

Run boundaries and latency, raw query text, retrieval result count and top score, model/tool names declared in `@dt.trace`. Not captured by this integration (just not wired up): retrieved document text, the generated answer text, node metadata/embeddings.

### Troubleshooting

- **No runs appear** — confirm the function calling `query_engine.query(...)` is wrapped in `@dt.trace`, and `dt.shutdown()` is called
- **`result_count` always zero** — confirm you're reading `response.source_nodes`, and the query engine is configured to return them
- **`top_score` always `None`** — some retrievers/rerankers don't populate scores; `RAG_EMPTY_RETRIEVAL` still works from `result_count` alone
