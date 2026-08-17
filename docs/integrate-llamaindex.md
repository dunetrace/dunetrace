# Integrating a LlamaIndex RAG Agent with Dunetrace

## Quick Start

```bash
pip install dunetrace llama-index
```

```python
import time
from dunetrace import Dunetrace, get_current_run

dt = Dunetrace()   # local dev, no API key needed

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

answer_question("What does the documentation say about setup?")
dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

`@dt.trace` opens and closes a Dunetrace run around your query function. Inside it, `retrieval_called()`/`retrieval_responded()` record the query text, result count, and top similarity score from LlamaIndex's `response.source_nodes` — enough for `RAG_EMPTY_RETRIEVAL` to catch a retrieval that returned nothing but still got answered. `get_current_run()` returns `None` outside a traced call, so guard with `if run:` if you reuse the helper elsewhere.

## Verification

Run your agent once, then check the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds. Ask a question with no matching indexed content and confirm `response.source_nodes == []` triggers `RAG_EMPTY_RETRIEVAL`.

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
