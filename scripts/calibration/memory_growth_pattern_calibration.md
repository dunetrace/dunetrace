# MEMORY_GROWTH_PATTERN calibration

Date: 2026-11-02

## Goal

Check the shipped thresholds against a labeled synthetic corpus, and measure
the false positives the issue's tool-name fallback produces. The thresholds
are not swept and the substring lists are not edited: `write_read_ratio` is 3
and `min_writes` is 3, and a tool name matches when it contains one of the
issue's markers, case-insensitively, with no token boundary.

Write markers: write, store, save, append, add, insert.
Read markers: read, retrieve, recall, search, query, fetch, get.

## Corpus

`scripts/calibrate_memory_growth_pattern.py` runs the shipped
`MemoryGrowthPatternDetector` class on 45 hand-labeled synthetic runs. No
network, LLM, third-party package, or production traffic. The rates below
are synthetic calibration results, not production estimates.

The boundary slice (32 runs) is the ratio and the write floor:

- Positives on `memory_events`: ratio exactly 3 (3/1, 6/2, 9/3), ratios above
  3, zero reads at 3, 5, and 10 writes, and a firing run that also clears.
- Negatives that clear `min_writes` and miss only the ratio: 6/3 (ratio 2),
  8/3 (2.67), 5/2 (2.5), 4/2, and balanced 10/10, 6/6, and 3/3, plus a
  read-heavy 3/5.
- Below the write floor: 2 writes and 0 reads (the shape that would otherwise
  fire), 2/1, 1/0, reads only, cleared only, and no memory use.
- Non-empty `memory_events` that must not fall through to tool names: clears
  plus `save_memory`, and one memory write plus five `write_file` calls.
- Fallback positives whose names really are memory tools (`save_memory`,
  `add_memory`, `store_note`, `insert_memory`, `SAVE_NOTE`), and one negative
  of ordinary names that contain no marker (`calculator`, `summarize`).

The collision slice (13 runs) is ordinary tool names on an empty
`memory_events` list, which is when the fallback runs. Twelve are labeled
negative: a coding or lookup agent is not growing memory. One is labeled
positive: three `save_memory` calls diluted by two `web_search` calls.

## Result

| Slice | n | TP | FP | FN | TN | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Boundary (memory events and memory-tool names) | 32 | 15 | 0 | 0 | 17 | 100% | 100% | 0% |
| Collision (substring fallback) | 13 | 0 | 6 | 1 | 6 | 0% | 0% | 50% |
| **All** | **45** | **15** | **6** | **1** | **23** | **71.4%** | **93.8%** | **20.7%** |

These are synthetic calibration results, not production estimates.

- Boundary slice: 100% precision, 100% recall, 0% false-positive rate.
- Collision slice: this is where the substring fallback is weak (6 false
  positives, 1 false negative, 50% false-positive rate).
- Overall synthetic result: precision 71.4% (15/21), recall 93.8% (15/16),
  false-positive rate 20.7% (6/29).

The boundary slice includes ratio-just-below-3 negatives that clear the
write floor, zero reads, and the refusal to fall through when
`memory_events` is non-empty. Every error is in the collision slice.

## Substring false positives

These names were scored with the heuristic left exactly as specified.

| Run (empty `memory_events`) | Label | Fired | Why |
|---|---|---|---|
| 3× `write_file` | negative | yes | `write` is a write marker; zero reads |
| 3× `write_file` + 1× `read_file` | negative | yes | ratio 3.0, the fire line |
| 6× `write_file` + 2× `read_file` | negative | yes | ratio 3.0 |
| 3× `write_file` + 1× `web_search` | negative | yes | `search` counts as the one read |
| 3× `write_file` + 1× `get_weather` | negative | yes | `get` counts as the one read |
| 3× `write_file` + 1× `fetch_url` | negative | yes | `fetch` counts as the one read |
| 3× `write_file` + 3× `read_file` | negative | no | ratio 1, both floors cleared |
| 2× `write_file` | negative | no | under `min_writes`, not a ratio decision |
| 4× `read_file` | negative | no | read marker only; writes stay 0 |
| 6× `web_search` | negative | no | read marker only |
| 4× `get_weather` | negative | no | read marker only |
| 4× `fetch_url` | negative | no | read marker only |
| 3× `save_memory` + 2× `web_search` | positive | no | false negative: unrelated `search` calls dilute a real memory-write run to ratio 1.5 |

`write_file` is the false-positive source. `read_file`, `web_search`,
`get_weather`, and `fetch_url` do not fire on their own. They fire only as
the read side of a write-marker run that still clears a 3× ratio, and
`web_search` can also hide a true memory-write run.

The same substring rule matches other ordinary names that were not given
their own rows (`address` contains `add`). The five names above are enough
to show the class.

## Recommendation

Keep `MEMORY_GROWTH_PATTERN` out of `LIVE_DETECTORS`.

The memory-event path is the detector that was asked for, and on this
synthetic corpus it separates cleanly. The fallback is not optional in the
issue, and it is the path an agent with no memory instrumentation always
takes. The synthetic 20.7% false-positive rate on the combined corpus, and
the 50% rate on the collision slice, are why it stays shadowed. They are
not a measurement of production traffic. Tightening the marker lists would
improve the table and would also stop implementing the issue. The detector
remains shadow-only.
