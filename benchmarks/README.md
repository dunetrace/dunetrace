# Benchmarks

## τ²-bench harness (`bench_tau2`)

Measures Dunetrace's detectors against externally graded agent runs: when a
detector fires on a τ²-bench trajectory, is it right, and when a trajectory
failed, did anything fire.

Status: the corpus, converter and replay runner exist and are tested. There
are no results yet. Nothing here changes a detector: measurement and tuning
are separate exercises.

### Corpus

Sierra publishes graded τ²-bench leaderboard trajectories in the public S3
bucket `sierra-tau-bench-public`. The corpus is the selection arXiv 2606.09863
used: 9 submissions across airline, retail and telecom (GPT-5.2 appears twice,
reasoning high and reasoning none). After dropping the 132 simulations with no
reward or no messages, the counts reproduce the paper's Table 1 exactly:

| Domain | Trajectories | Success | Failure |
|---|---|---|---|
| airline | 1,797 | 1,404 | 393 |
| retail | 4,029 | 3,134 | 895 |
| telecom | 4,050 | 3,608 | 442 |
| total | 9,876 | 8,146 | 1,730 |

`bench_tau2/data/manifest.json` pins all 27 files by S3 key and SHA-256. The
files themselves (968 MB) are not committed: Sierra states no data license
beyond the repo's MIT license. `fetch` downloads them and fails if any byte
differs from the manifest.

### Labels

Success and failure come from τ²-bench's programmatic reward. Failures are
split into false success (FS), honest failure (HF) and ambiguous (AMB) by the
paper's Appendix A regexes on the closing message. The paper releases no
labels, and its printed patterns do not reproduce its counts:

| | FS | HF | AMB |
|---|---|---|---|
| arXiv 2606.09863 | 616 | 753 | 361 |
| `bench_tau2.labels` | 534 | 662 | 534 |

`labels.py` uses the most literal reading and records it as
`LABEL_RULE_VERSION`. It documents the readings tried, including the printed
`\bI\s+('m|am)\s+sorry`, which can never match "I'm sorry". Matching it moves 4
trajectories. Labels decide which trajectories are in the recall subset, never
what a detector sees.

### Conversion

`convert.py` drives the Python SDK's own `dt.run()` API with what each
trajectory recorded and captures what the SDK emits. Payload shapes, step
indices, `call_id` pairing, field caps, `agent_version` and the in-path
prompt-injection scan are the SDK's. Four values are overwritten afterwards
for determinism: `run_id`, `event_id`, `timestamp` and
`run.started.payload.source_file`.

- **One run per user turn**, with the trajectory as `conversation_id`. That is
  what the SDK's request middleware produces for a chat agent, including the
  `final_answer()` call at the end of each request.
- **Dropped:** τ²-bench's canned greeting (not an LLM call), the final user
  message carrying the stop token (the agent never answers it), and telecom's
  user-side device actions (46,144 tool calls the user made, not the agent).
- **LLM calls:** tokens from `usage`; finish_reason and reasoning tokens from
  the provider response in `raw_data`. When `raw_data` is missing,
  finish_reason is left unknown, which the SDK omits from the wire.
- **Tool calls:** emitted when their result arrives, in τ²-bench's execution
  order. Arguments are the raw provider string where the provider response
  has the call (59,318 calls) and τ²-bench's parsed dict otherwise (13,522), so
  MALFORMED_TOOL_CALL sees what the model actually produced.
- **Tools and prompt:** the files leave `tool_defs` null. Tool names, their
  read/write annotations and the agent prompt template come from τ²-bench
  source at the newest public commit the corpus ran on
  (`bench_tau2/data/tau2_tools.json`). A tool with a successful call in a file
  is added for that file, since it demonstrably existed there.
- **Timestamps:** from the trajectory. Latencies are the gaps between them.
  A missing or backwards timestamp takes the previous one and is counted.
- **Field cap:** the SDK's 8,192-character cap applies as it would in
  production. 30,447 fields are capped across the corpus: telecom's
  23k-character system prompt on every telecom run (30,392), 51 LLM outputs
  and 4 tool outputs.
- **Abnormal ends:** a trajectory that ended on `too_many_errors` or
  `max_steps` (21 of them) ends its last run with `run.errored`.

`check` converts all 9,876 trajectories (54,068 runs, 478,146 events) without a
database, rebuilds every run with the production `build_run_state`, and
compares it field by field with the trajectory. Every run is lossless.

### Replay

`replay.py` persists each run through ingest's `_persist` (the function
`POST /v1/ingest` calls after auth), then runs the detector worker's
`process_run` on it, unchanged. Baselines, cross-run checks, shadow gating,
the risk engine, co-occurrence boosts and issue tracking all run as in
production.

- **Sequential, fixed order.** Baselines read the agent's last clean runs by
  processing time, so concurrent processing would make them depend on
  scheduling. Within an agent, runs go in completion order, the order a live
  worker would see them. Each manifest file is one agent.
- **Clean runs only feed baselines.** A run with any live signal is excluded
  from every baseline. A detector that fires often therefore starves the
  baseline-dependent ones. That is production behaviour and part of what is
  measured.
- **One harness setting.** UNRESOLVED_AMBIGUITY ships with
  `irreversible_tools: []`, which makes it inert. The generated config declares
  τ²-bench's own `ToolType.WRITE` tools per domain. Every other value is the
  shipped `detectors.yml`. Replay checks this before it starts and refuses to
  run otherwise.
- **Resumable.** Runs already in `processed_runs` are skipped. Replay records a
  fingerprint (code, corpus, config, label rule, selection) in
  `replay_meta.json` and refuses to resume when any of it changed.
- **Outputs**, sorted and byte-identical across replays: `runs.jsonl`,
  `signals.jsonl` (failure type, step, the trajectory messages behind that
  step, shadow flag, confidence, evidence keys and hash, no content) and
  `signals_evidence.jsonl` (the evidence itself, kept out of git).

Two detectors abort their scan after 50 ms of wall clock
(UNRESOLVED_AMBIGUITY, UNGROUNDED_DESTINATION). The slowest scan seen so far
took 2.9 ms. The SDK's cost-budget downgrade only applies in-path, never in
the worker. Those are the only timing-dependent paths found.

Not replayed yet: the semantic evaluators, which need an LLM key and a budget
decision (Phase 2). The voice pack cannot be measured on a text corpus.

### Running

```bash
cd benchmarks
python -m bench_tau2 fetch            # 27 files into benchmarks/.cache/tau2
python -m bench_tau2 check            # counts, labels, lossless check (~80 s)
python -m bench_tau2 db up            # Postgres 16 on 127.0.0.1:55433 (docker)
python -m bench_tau2 replay --fresh   # full replay into database dunetrace_bench
python -m bench_tau2 replay           # resume after an interruption
python -m bench_tau2 replay --agent tau2.airline.gpt-5-2_sierra_2026-02-26 --limit 500
```

Replay refuses a non-local database unless given `--allow-remote`.

### Tests

```bash
make test-bench                       # offline: corpus, labels, tools, conversion
BENCH_TEST_DATABASE_URL=postgresql://postgres:bench@127.0.0.1:55433/postgres \
  make test-bench                     # plus replay against Postgres
BENCH_TAU2_DATA=benchmarks/.cache/tau2 make test-bench   # plus the Table 1 check on real files
```

The tests use synthetic trajectories in τ²-bench's own format, since the real
files are not committed. The replay tests run the CLI in a subprocess against
throwaway databases and check four things: every signal maps back to
trajectory messages; two fresh replays are byte-identical; an interrupted and
resumed replay equals an uninterrupted one, with the interruption inside
baseline formation; and resuming with a changed selection is refused.
