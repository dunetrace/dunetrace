#!/usr/bin/env python3
"""
Detector/evaluator documentation drift check (Phase 9).

Two independent checks, both grounded in code rather than in a hand-maintained
expectation:

NAMES. Loads the real detector, detector-pack, and semantic-evaluator names from
code, then scans the docs where those names are CLAIMED for backticked all-caps
names.

  - A name mentioned in docs but not found in code (and not in the allowlist of
    known non-detector tokens) → FAIL. This is the drift this whole check exists
    to stop: README/marketing claiming a detector that doesn't ship (the way
    MEMORY_POISONING / DELEGATION_LOOP were once claimed but never built).
  - A name in code but never mentioned in the scanned docs → WARN only (some
    detectors are intentionally undocumented; surfaced for visibility).

COUNTS. Every "<n> detectors" / "<n> structural detectors" claim in a scanned
doc must resolve to the number code actually ships. The name check alone never
caught a count: a doc could say "29 detectors" and every *name* in it still be
real. All five count drifts the 2026 audit found were exactly that — "Runs 29
detectors" in docs/architecture.md, stale totals in CLAUDE.md, and the website.
The expected numbers come from `detector_counts()`, so adding a detector moves
them and the docs have to follow.

  - a sentence saying "in-path" or "TIER1" is about the SDK's client-side
    battery (TIER1_DETECTORS), so it is checked against that count;
  - a sentence naming a pack ("the voice pack adds 9 detectors") is checked
    against that pack's own detector count;
  - everything else is the full structural battery.
  - "Tier 1 detector" is prose for *structural* (CLAUDE.md says so explicitly),
    not a count of one, so it is excluded.

Scope: the NAME scan is deliberately limited to the docs that enumerate
detectors and evaluators (README + the detector/evaluator/pack pages), not all
of docs/. That keeps the allowlist tiny and the signal precise — operational
docs are full of env-var-shaped all-caps tokens that have nothing to do with
detector coverage (CLAUDE.md and docs/architecture.md alone would add 46 of
them: AUTH_MODE, GITHUB_TOKEN, SHARD_COUNT, POST, ALTER…, and an allowlist that
big stops being a guard). The COUNT scan has no such cost — a number is not a
name — so it covers those two files as well, which is where the drift was.

Run: python scripts/validate_detector_docs.py   (exit 1 on drift)
"""

from __future__ import annotations

import pathlib
import re
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "sdk-py"))

# Docs that claim which detectors/evaluators exist. Add here if a new such page
# appears; do NOT broaden to all of docs/ (see the scope note in the docstring).
_SCAN_GLOBS = [
    "README.md",
    "docs/detectors.md",
    "docs/semantic-evaluation.md",
    "docs/detector-packs/*.md",
]

# The COUNT check (below) scans a wider set than the NAME check. Every count
# drift the 2026-09 audit found lived in a file outside _SCAN_GLOBS — but those
# files are full of env vars, HTTP verbs and constants that are legitimately
# all-caps and are not detector names, so scanning them for NAMES would drown
# the signal in allowlist entries. Numbers have no such ambiguity.
_COUNT_SCAN_GLOBS = _SCAN_GLOBS + [
    "CLAUDE.md",
    "docs/architecture.md",
    "docs/dashboard.md",
]

# All-caps-snake tokens that appear in the scanned docs but are legitimately NOT
# detector/evaluator/pack signal names (env vars, class attrs, operational
# signals). When a new one shows up the check fails loudly; add it here after
# confirming it isn't a mistyped or unbuilt detector name.
_ALLOWLIST = {
    "DUNETRACE_API_URL",
    "DUNETRACE_CUSTOM_DETECTORS_PATH",
    "SEMANTIC_LLM_PROVIDER",
    "SEMANTIC_WORKER_ENABLED",
    "SEMANTIC_ALLOW_CROSS_PROVIDER_SECOND_OPINION",
    # Per-provider evaluator credentials, named in docs/semantic-evaluation.md
    # alongside SEMANTIC_LLM_PROVIDER. Env vars, not detectors.
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "MISTRAL_API_KEY",
    "API_LLM_PROVIDER",
    "DUNETRACE_POLICY_SECRET",
    "POLICY_SIGNING_SECRET",
    "DUNETRACE_MCP_READONLY",
    "SEMANTIC_QUOTA_EXCEEDED",  # operational signal, not a structural detector
    "OTLP_MAX_ATTR_CHARS",  # ingest-side arg truncation cap, not a detector
    # Env vars named in docs/detectors.md's INSTRUMENTATION_DEGRADED note:
    # the output-text transmission opt-out (why absent text is not a fault) and
    # the event retention window (why a baseline can expire). Neither is a
    # detector.
    "DUNETRACE_OMIT_LLM_OUTPUT_TEXT",
    "EVENT_RETENTION_DAYS",
    "SHADOW_BY_DEFAULT",
    "LIVE_DETECTORS",  # the shadow/live gating set in detector_svc, not a detector
    "MIN_MESSAGE_LENGTH",
    "NULL",
    "SHADOW",
    # SDK constants that name *collections of* detectors, or detector tunables —
    # all defined in packages/sdk-py/dunetrace/detectors.py.
    "TIER1_DETECTORS",  # the 31-detector in-path battery, not a detector itself
    "SCAN_HEAD_CHARS",  # injection scan window bound
    "SCAN_TAIL_CHARS",  # injection scan window bound
    # Custom-detector translator vocabulary (detector_svc/custom_detector.py and
    # api_svc/custom_detector_translator.py) — the fields a content condition may
    # inspect, not a detector.
    "CONTENT_FIELDS",
    # Per-evaluator model overrides read by semantic_svc/worker.py. These are env
    # var names shaped like <EVALUATOR>_MODEL; the evaluator itself is already a
    # known name, the _MODEL suffix makes it a distinct token to this checker.
    "HALLUCINATION_MODEL",
    "TASK_COMPLETION_MODEL",
    "TASK_UNDERSTANDING_FAILURE_MODEL",
    "OFF_TOPIC_DRIFT_MODEL",
    "USER_FRUSTRATION_MODEL",
    "CONFUSION_LOOP_MODEL",
    "SYCOPHANCY_SIGNAL_MODEL",
}

# Backticked ALL-CAPS token: multi-segment snake (`TOOL_LOOP`) or a single word
# (`HALLUCINATION`). ≥3 chars, so it also catches a single-word detector name;
# short non-detector words that show up backticked go in _ALLOWLIST.
_TOKEN_RE = re.compile(r"`([A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*)`")


def code_names() -> set[str]:
    """The authoritative set of detector, detector-pack, and evaluator signal
    names, read from code. Detectors/packs are imported (lightweight, no
    deepeval); evaluator names are parsed from source so this check never pulls
    the heavy semantic-service dependencies."""
    import importlib
    import pkgutil

    import dunetrace.packs as packs_pkg
    from dunetrace.detectors import (
        DelegationLoopDetector,
        HandoffContextLossDetector,
        PROMPT_INJECTION_DETECTOR,
        TIER1_DETECTORS,
    )
    from dunetrace.packs.base import PACK_REGISTRY

    # Import every pack module so it registers itself (voice today, plus any
    # future pack) — the registry is then the single source for pack detectors.
    for mod in pkgutil.iter_modules(packs_pkg.__path__):
        if mod.name != "base":
            importlib.import_module(f"dunetrace.packs.{mod.name}")

    names = {d.name for d in TIER1_DETECTORS}
    names.add(PROMPT_INJECTION_DETECTOR.name)
    names.add(HandoffContextLossDetector().name)
    names.add(DelegationLoopDetector().name)
    for pack in PACK_REGISTRY.values():
        for cls in pack.detectors:
            names.add(cls.name)

    ev_dir = _REPO / "services" / "semantic" / "semantic_svc" / "evaluators"
    name_attr = re.compile(r'^\s{4}name = "([A-Z_]+)"', re.M)
    for f in ev_dir.glob("*.py"):
        names.update(name_attr.findall(f.read_text()))
    return names


def scan_text(text: str) -> set[str]:
    """Backticked all-caps tokens in a blob of markdown."""
    return set(_TOKEN_RE.findall(text))


def scan_docs(globs: list[str] | None = None) -> dict[str, list[str]]:
    """Map each backticked all-caps token to the doc files it appears in."""
    mentions: dict[str, list[str]] = {}
    for glob in globs or _SCAN_GLOBS:
        for path in sorted(_REPO.glob(glob)):
            for tok in scan_text(path.read_text()):
                mentions.setdefault(tok, []).append(str(path.relative_to(_REPO)))
    return mentions


def validate(
    names: set[str], mentions: dict[str, list[str]], allowlist: set[str] | None = None
) -> tuple[dict[str, list[str]], list[str]]:
    """Pure comparison. Returns (unknown, undocumented):
    unknown = doc-mentioned tokens not in code and not allowlisted (FAIL set);
    undocumented = code names never mentioned in the scanned docs (WARN set)."""
    allow = allowlist if allowlist is not None else _ALLOWLIST
    unknown = {t: fs for t, fs in mentions.items() if t not in names and t not in allow}
    undocumented = sorted(n for n in names if n not in mentions)
    return unknown, undocumented


# Counts the docs assert. Every one of these was wrong somewhere at the 2026-09
# audit — "Runs 29 detectors", "the 15 checks", "34 structural detectors,
# in-path" — and NONE was caught, because this script validated NAMES and never
# numbers, and because the files carrying them were outside _SCAN_GLOBS.
_TOTAL_DETECTORS = 34  # detector classes / _DETECTOR_CLASSES / detectors.yml sections
_IN_PATH_DETECTORS = 31  # TIER1_DETECTORS — the SDK's client-side subset
# Below this, a number is counting something other than the battery (a table
# row, a pack, a single detector), not asserting how many exist.
_MIN_BATTERY_CLAIM = 10

# "<n> detectors" / "<n> structural detectors" / "<n> zero-LLM detectors".
_COUNT_CLAIM = re.compile(r"\b(\d{1,3})\s+(?:[a-z-]+\s+){0,3}detectors?\b", re.IGNORECASE)
# Phrases that mean the sentence is talking about the in-path subset, not all 34.
_IN_PATH_MARKERS = ("in-path", "in path", "tier1_detectors", "client-side", "in the sdk")


def check_counts(globs: list[str] | None = None) -> list[str]:
    """Every "<n> detectors" claim must be 34, or 31 when the sentence says
    in-path. Returns one message per bad claim."""
    problems: list[str] = []
    for glob in globs or _SCAN_GLOBS:
        for path in sorted(_REPO.glob(glob)):
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                for m in _COUNT_CLAIM.finditer(line):
                    claimed = int(m.group(1))
                    # Only whole-battery claims. "1 detector" in a table row and
                    # "9 voice-agent detectors" (the pack really does add 9) are
                    # not assertions about the battery; a floor plus a pack
                    # exclusion separates them without an allowlist.
                    if claimed < _MIN_BATTERY_CLAIM:
                        continue
                    # A window, not the whole line: CLAUDE.md writes a whole
                    # paragraph on one line, so scanning it entire matched
                    # "client-side" from a different sentence and read a correct
                    # "runs all 34 structural detectors" as an in-path claim.
                    window = line[max(0, m.start() - 120) : m.end() + 120].lower()
                    if "pack" in window:
                        continue
                    # The correct, fully-qualified phrasing names BOTH numbers:
                    # "34 zero-LLM detectors (31 of them in-path)". That is the
                    # shape CLAUDE.md asks for — say which one you mean — so it
                    # passes whichever of the two the regex happened to match.
                    if str(_TOTAL_DETECTORS) in window and str(_IN_PATH_DETECTORS) in window:
                        continue
                    in_path = any(marker in window for marker in _IN_PATH_MARKERS)
                    expected = _IN_PATH_DETECTORS if in_path else _TOTAL_DETECTORS
                    if claimed in (_TOTAL_DETECTORS, _IN_PATH_DETECTORS) and claimed == expected:
                        continue
                    if claimed in (_TOTAL_DETECTORS, _IN_PATH_DETECTORS):
                        problems.append(
                            f"{path.relative_to(_REPO)}:{line_no}: says {claimed} where the "
                            f"sentence reads as {'in-path' if in_path else 'all'} "
                            f"({expected} expected) — {m.group(0)!r}"
                        )
                    else:
                        problems.append(
                            f"{path.relative_to(_REPO)}:{line_no}: {m.group(0)!r} — expected "
                            f"{expected} ({_TOTAL_DETECTORS} total, {_IN_PATH_DETECTORS} in-path)"
                        )
    return problems


def main() -> int:
    names = code_names()
    mentions = scan_docs()
    unknown, undocumented = validate(names, mentions)
    count_problems = check_counts(_COUNT_SCAN_GLOBS)

    if undocumented:
        print("WARN: in code but not mentioned in the scanned docs (informational):")
        for n in undocumented:
            print(f"  - {n}")

    if count_problems:
        print("\nFAIL: detector-count claims that disagree with the code:")
        for problem in count_problems:
            print(f"  - {problem}")
        print(
            "\nGround truth: "
            f"{_TOTAL_DETECTORS} detector classes, {_IN_PATH_DETECTORS} in TIER1_DETECTORS "
            "(the in-path client-side subset). Say which one you mean."
        )

    if unknown:
        print("\nFAIL: names mentioned in docs but not found in code:")
        for tok, files in sorted(unknown.items()):
            print(f"  - {tok}  (in {', '.join(files)})")
        print(
            "\nEach is one of: a typo / renamed detector, a claim for a detector "
            "that doesn't exist yet (remove it or build it), or a non-detector "
            "token that belongs in _ALLOWLIST in this script."
        )
        return 1

    if count_problems:
        return 1

    print(
        f"\nOK: all {len(mentions)} doc-mentioned names exist in code "
        f"({len(names)} detector/evaluator names known)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
