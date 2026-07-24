"""
Verify that the trace exercise_agent.py pushed actually landed in Jaeger with
the right span names and GenAI conventions. Queries the Jaeger trace API by the
deterministic trace_id (derived from run_id), so it checks the exact run.

    python otel/verify.py

Exit code 0 if every expected span is present and the LLM span carries the
GenAI conventions; 1 otherwise. Also prints a per-span summary you can paste as
evidence.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

JAEGER = "http://localhost:16686"


def _fetch_trace(trace_id: str) -> dict:
    url = f"{JAEGER}/api/traces/{trace_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> int:
    meta_path = Path(__file__).with_name("last_run.json")
    if not meta_path.exists():
        print("No last_run.json — run exercise_agent.py first.")
        return 1
    meta = json.loads(meta_path.read_text())
    trace_id = meta["trace_id"]
    expected = set(meta["expected_spans"])

    try:
        data = _fetch_trace(trace_id)
    except Exception as exc:
        print(f"Could not fetch trace {trace_id} from Jaeger: {exc}")
        return 1

    traces = data.get("data") or []
    if not traces:
        print(f"Trace {trace_id} not found in Jaeger yet (give it a few seconds).")
        return 1

    spans = traces[0]["spans"]
    names = {s["operationName"] for s in spans}

    def attrs(span) -> dict:
        return {t["key"]: t["value"] for t in span.get("tags", [])}

    print(f"Trace {trace_id}: {len(spans)} spans")
    for s in sorted(spans, key=lambda s: s["startTime"]):
        print(f"  - {s['operationName']}")

    missing = expected - names
    ok = not missing

    # Spot-check GenAI conventions on the LLM span.
    llm = next((s for s in spans if s["operationName"].startswith("chat ")), None)
    if llm is None:
        print("FAIL: no LLM (chat) span found")
        ok = False
    else:
        a = attrs(llm)
        for key in ("gen_ai.provider.name", "gen_ai.request.model", "gen_ai.usage.input_tokens"):
            if key not in a:
                print(f"FAIL: LLM span missing {key}")
                ok = False
        if a.get("gen_ai.provider.name") not in (None, "openai"):
            print(f"FAIL: gen_ai.provider.name={a.get('gen_ai.provider.name')} (expected openai)")
            ok = False

    # HTTP-shaped tool should carry server.address.
    http_tool = next(
        (s for s in spans if s["operationName"] == "dunetrace.tool.api.shipping.com"), None
    )
    if http_tool and "server.address" not in attrs(http_tool):
        print("FAIL: HTTP tool span missing server.address")
        ok = False

    if missing:
        print("FAIL: missing spans: " + ", ".join(sorted(missing)))

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
