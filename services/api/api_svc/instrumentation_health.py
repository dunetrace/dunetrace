"""
Vendor-side fleet query: which agents are reporting broken instrumentation.

The in-run view of this condition is INSTRUMENTATION_DEGRADED (a detector). This
is the cross-run view, and it exists because the two answer different questions.
The detector answers "was this run measurable?". This answers "is this agent's
telemetry broken?", which is only visible in aggregate — a single blank LLM call
is unremarkable, and the same call on 100% of an agent's traffic is a broken
pipeline.

THE FINGERPRINT
---------------
    output_length = 0 AND finish_reason = 'stop'
    AND completion_tokens = 0 AND latency_ms > 0

A call that measurably took time and measurably produced nothing. `latency_ms > 0`
is what separates a real round-trip from a call that never happened. A genuinely
empty model response has exactly this shape too — which is the point: one such
call is a finding, and 30% of an agent's calls having it is not a model that
answered nothing 30% of the time, it is an extractor reading the wrong object.

The incident that motivated this: langchain_openai calls
`client.with_raw_response.create()`, so `Completions.create` returned a
`LegacyAPIResponse` rather than a `ChatCompletion`. The extractors hit their
fallback branches and substituted ("", "stop"), producing this fingerprint on
100% of calls — and firing EMPTY_LLM_RESPONSE, a HIGH-severity behavioural
alert, on every run including the control.

Note the fingerprint keys on `finish_reason = 'stop'`. Post-provenance the SDK
omits finish_reason entirely when it could not read one, so a *current* SDK
produces NULL there and is caught by the `finish_reason IS NULL` arm below. Both
arms are needed: the 'stop' arm catches events already in the database, and
every agent still running an SDK from before provenance existed.

PROVIDER
--------
`provider` is set on `llm.called`, not `llm.responded` (run_context.llm_called,
written by the auto-instrumentation patchers), and is rebuilt server-side into
LlmCall.provider. No wiring was needed — but the join is real: llm.responded is
emitted with advance=False, so it shares its llm.called's step_index, which
makes (org_id, run_id, step_index) a reliable join key even for events predating
call_id.
"""

from __future__ import annotations

# One template, two dialect renderings. The docs quote the Postgres rendering
# and the test exercises the SQLite one; both come from here, so a change to the
# fingerprint cannot silently apply to only one of them.
_TEMPLATE = """
WITH responded AS (
    SELECT
        e.org_id,
        e.agent_id,
        e.run_id,
        e.step_index,
        {j_output_length} AS output_length,
        {j_finish_reason} AS finish_reason,
        {j_completion_tokens} AS completion_tokens,
        {j_latency_ms} AS latency_ms
    FROM events e
    WHERE e.event_type = 'llm.responded'
      {time_filter}
),
called AS (
    SELECT e.org_id, e.run_id, e.step_index, {j_provider} AS provider
    FROM events e
    WHERE e.event_type = 'llm.called'
      {time_filter}
)
SELECT
    r.org_id,
    r.agent_id,
    COALESCE(c.provider, 'unknown') AS provider,
    COUNT(*) AS llm_calls,
    SUM(CASE
            WHEN CAST(r.output_length AS INTEGER) = 0
             AND (r.finish_reason = 'stop' OR r.finish_reason IS NULL)
             AND CAST(r.completion_tokens AS INTEGER) = 0
             AND CAST(r.latency_ms AS INTEGER) > 0
            THEN 1 ELSE 0
        END) AS blank_calls,
    CAST(SUM(CASE
            WHEN CAST(r.output_length AS INTEGER) = 0
             AND (r.finish_reason = 'stop' OR r.finish_reason IS NULL)
             AND CAST(r.completion_tokens AS INTEGER) = 0
             AND CAST(r.latency_ms AS INTEGER) > 0
            THEN 1 ELSE 0
        END) AS REAL) / COUNT(*) AS blank_fraction
FROM responded r
LEFT JOIN called c
       ON c.org_id = r.org_id
      AND c.run_id = r.run_id
      AND c.step_index = r.step_index
GROUP BY r.org_id, r.agent_id, COALESCE(c.provider, 'unknown')
HAVING COUNT(*) >= {min_calls}
ORDER BY blank_fraction DESC, llm_calls DESC
"""

# Above this fraction, the agent is not misbehaving — the telemetry is.
BROKEN_INSTRUMENTATION_THRESHOLD = 0.30

_POSTGRES_ACCESSORS = {
    "j_output_length": "e.payload->>'output_length'",
    "j_finish_reason": "e.payload->>'finish_reason'",
    "j_completion_tokens": "e.payload->>'completion_tokens'",
    "j_latency_ms": "e.payload->>'latency_ms'",
    "j_provider": "e.payload->>'provider'",
}

_SQLITE_ACCESSORS = {
    "j_output_length": "json_extract(e.payload, '$.output_length')",
    "j_finish_reason": "json_extract(e.payload, '$.finish_reason')",
    "j_completion_tokens": "json_extract(e.payload, '$.completion_tokens')",
    "j_latency_ms": "json_extract(e.payload, '$.latency_ms')",
    "j_provider": "json_extract(e.payload, '$.provider')",
}


def blank_response_rate_sql(
    dialect: str = "postgres",
    *,
    min_calls: int = 20,
    since_days: int | None = 7,
) -> str:
    """Render the fleet query.

    `min_calls` keeps a brand-new agent with three calls out of the report;
    a fraction over a tiny denominator is noise, not a signal.
    """
    accessors = _POSTGRES_ACCESSORS if dialect == "postgres" else _SQLITE_ACCESSORS
    if since_days is None:
        time_filter = ""
    elif dialect == "postgres":
        time_filter = f"AND e.received_at >= NOW() - INTERVAL '{int(since_days)} days'"
    else:
        time_filter = f"AND e.received_at >= datetime('now', '-{int(since_days)} days')"
    return _TEMPLATE.format(min_calls=int(min_calls), time_filter=time_filter, **accessors)
