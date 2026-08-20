# MEMORY_POISONING calibration

Calibration of the `MEMORY_POISONING` detector before ship. Reproduce with
`python scripts/calibrate_memory_poisoning.py` (`--write` regenerates the cached
scores JSON). No LLM and no API key: the detector is structural (deterministic
pattern matching over written memory values), so the run is fully reproducible.

## What it detects

An injection / override directive persisted into the agent's own memory (a
conversation buffer, scratchpad, or long-term store) — content that re-steers
the agent every time that memory is read back, on a later step or turn. It reads
`state.memory_events` (the typed view built from `memory.*` events) and fires on
the first written value matching a marker. It is a distinct surface from
`PROMPT_INJECTION_SIGNAL` (user input at run-start) and
`RETRIEVED_CONTENT_INJECTION` (retrieval/tool content read once during the run) —
here the danger is *persistence*.

## Method

A labeled corpus of memory writes, each wrapped in a one-write run and scored by
the real `MemoryPoisonedDetector`:

- **24 signature positives** — real injection/override strings across five marker
  families (direct override, embedded role marker, instruction delimiter,
  jailbreak/DAN, safety bypass), phrased as content that gets *persisted* to
  memory (a summarized document, a saved note, a stored "fact").
- **6 paraphrase evasions** — attacks phrased *around* the marker vocabulary
  ("set aside all the earlier guidance", "your prior limitations are lifted").
  Included to measure and disclose the recall ceiling honestly, reported
  separately from headline signature-recall.
- **33 negatives**, weighted toward the FP-risk class:
  - **10 role-play personalizations** ("act as my travel planner", "you are now
    my coding assistant", "pretend to be…"). This is the whole reason the marker
    set is narrower than `PROMPT_INJECTION_SIGNAL`'s — agents routinely persist
    this language and it is benign.
  - **12 benign preferences** + **8 benign summaries** — ordinary stored memory.
  - **3 meta-narration** hard negatives — memory that *quotes* an injection while
    describing it ("the user tried to make me ignore previous instructions; I
    declined"). A substring matcher cannot separate narration from a live
    directive; these are the disclosed residual FP surface.

`fired = detector returns a signal`. There is no score threshold to sweep — the
knob is the marker vocabulary itself.

## Results

| Marker family | fired / n |
|---|---|
| direct_override | 10 / 10 |
| role_marker | 3 / 3 |
| delimiter | 4 / 4 |
| jailbreak | 3 / 3 |
| safety_bypass | 4 / 4 |

| Negative class | fired / n |
|---|---|
| roleplay_personalization | 0 / 10 |
| benign_preferences | 0 / 12 |
| benign_summaries | 0 / 8 |
| meta_narration | 3 / 3 |

| Metric | Value | Bar |
|---|---|---|
| Signature recall | **100%** | — |
| Overall recall (incl. paraphrase) | 80% | — |
| False-positive rate | **9%** | < 15% |
| Precision | 89% | — |

**Ship: yes.** Every signature positive fires; every benign preference,
summary, and — critically — every role-play personalization stays clean, which
is what the narrower marker set buys. The only false positives are the 3
meta-narration cases.

## Decisions

- **Role-play phrases are excluded from the marker set** (`act as`, `you are
  now`, `pretend`, `your new role is`). Including them (as `PROMPT_INJECTION_SIGNAL`
  does for raw user input) fires on all 10 role-play personalizations — an
  unacceptable FP rate on content agents legitimately store. The exclusion is
  the single highest-leverage precision decision and is validated by the 0/10
  role-play result.

## Known limitations (disclosed, not bugs)

- **Paraphrase recall gap.** Novel phrasings that avoid the known signatures are
  not caught (0/6 paraphrase evasions fired → overall recall 80%). A substring/
  regex detector has no way around this; catching semantic paraphrase is a
  future LLM-scored evaluator's job, not this one's.
- **Meta-narration false positives.** Memory that quotes an injection while
  describing it trips the substring match (3/3). This is inherent to structural
  matching and mirrors the same residual surface `RETRIEVED_CONTENT_INJECTION`
  accepts. Net FP rate stays at 9%, well under the 15% bar.
- **Provenance is best-effort.** Framework-auto-captured writes carry no
  `source`, so the untrusted-source severity escalation only applies to
  manually-instrumented writes (or when the poisoned key is later read, which
  escalates regardless of source). `require_untrusted_source: true` in
  `detectors.yml` trades recall for precision by firing only on
  attacker-controllable sources.

## Ship status

**Live** — listed in `LIVE_DETECTORS`
(`services/detector/detector_svc/db.py`). Shipped shadow first, matching the
SILENT_TRUNCATION / MODEL_FALLBACK_DRIFT precedent, and promoted once real
traffic confirmed the FP rate held outside the calibration corpus. To reverse,
remove it from `LIVE_DETECTORS`; `require_untrusted_source: true` in
`detectors.yml` is the first knob to reach for if it is noisier than expected.
