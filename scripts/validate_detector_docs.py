#!/usr/bin/env python3
"""
Detector/evaluator documentation drift check (Phase 9).

Loads the real detector, detector-pack, and semantic-evaluator names from code,
then scans the docs where those names are CLAIMED for backticked all-caps names.

  - A name mentioned in docs but not found in code (and not in the allowlist of
    known non-detector tokens) → FAIL. This is the drift this whole check exists
    to stop: README/marketing claiming a detector that doesn't ship (the way
    MEMORY_POISONING / DELEGATION_LOOP were once claimed but never built).
  - A name in code but never mentioned in the scanned docs → WARN only (some
    detectors are intentionally undocumented; surfaced for visibility).

Scope: the scan is deliberately limited to the docs that enumerate detectors and
evaluators (README + the detector/evaluator/pack pages), not all of docs/. That
keeps the allowlist tiny and the signal precise — operational docs are full of
env-var-shaped all-caps tokens that have nothing to do with detector coverage.

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
    "SHADOW_BY_DEFAULT",
    "LIVE_DETECTORS",  # the shadow/live gating set in detector_svc, not a detector
    "MIN_MESSAGE_LENGTH",
    "NULL",
    "SHADOW",
    # SDK constants that name *collections of* detectors, or detector tunables —
    # all defined in packages/sdk-py/dunetrace/detectors.py.
    "TIER1_DETECTORS",  # the 28-detector in-path battery, not a detector itself
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


def main() -> int:
    names = code_names()
    mentions = scan_docs()
    unknown, undocumented = validate(names, mentions)

    if undocumented:
        print("WARN: in code but not mentioned in the scanned docs (informational):")
        for n in undocumented:
            print(f"  - {n}")

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

    print(
        f"\nOK: all {len(mentions)} doc-mentioned names exist in code "
        f"({len(names)} detector/evaluator names known)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
