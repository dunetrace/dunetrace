#!/usr/bin/env python3
"""Generate the detector coverage manifest the dunetrace-setup skill reads.

The skill reports which detectors an instrumented agent can now trigger. That
report is only worth shipping if its counts are right, so nothing here is
hardcoded: every name and count is read from the live detector set.

Three sources, combined:

  1. ``detector_svc._DETECTOR_CLASSES`` and ``LIVE_DETECTORS`` for the
     structural set, its yaml keys, and shadow/live status.
  2. ``scripts/detector_event_requirements.json`` for the ``requires`` /
     ``enriches`` split, which is a human judgment (see that file's _README for
     why it cannot be derived).
  3. An AST pass over ``dunetrace/detectors.py`` that records every
     ``state.<field>`` each detector reads, used purely as an audit: if a
     detector touches a field that the curated entry does not account for,
     this fails rather than shipping a manifest that quietly understates it.

Voice pack detectors need no curation. They read ``EventType.*`` members
directly, so their requirements come straight from the AST.

Usage:
    python scripts/gen_detector_manifest.py --out <references-dir>
    python scripts/gen_detector_manifest.py --out <references-dir> --check
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CURATED = pathlib.Path(__file__).resolve().parent / "detector_event_requirements.json"
MANIFEST_NAME = "detectors.manifest.json"

# A RunState field maps to the capability tokens that could supply it. The
# audit passes when the curated entry names at least one of them. Identity
# fields carry no instrumentation meaning and are ignored.
FIELD_TOKENS: dict[str, set[str]] = {
    "tool_calls": {"tool", "tool.output"},
    "llm_calls": {"llm", "llm.tokens"},
    "retrievals": {"retrieval", "retrieval.content"},
    "memory_events": {"memory"},
    "external_signals": {"external"},
    "input_text": {"input_text"},
    "system_prompt": {"system_prompt"},
    "available_tools": {"declared_tools"},
    "exit_reason": {"run"},
    "dropped_events": {"run"},
    "current_step": {"run", "llm", "tool"},
    "step_durations_ms": {"llm", "tool"},
    "events": {"run", "llm", "tool", "retrieval", "memory", "external"},
    "baseline_p75_steps": {"baseline"},
    "baseline_p75_latency_tool": {"baseline"},
    "baseline_p75_latency_llm": {"baseline"},
    "baseline_p75_token_growth": {"baseline"},
    "baseline_p75_llm_tool_ratio": {"baseline"},
    "baseline_p75_total_tokens": {"baseline"},
    "baseline_p75_duration_s": {"baseline"},
}
IDENTITY_FIELDS = {"run_id", "agent_id", "agent_version"}

SEMANTIC_LEVELS = {
    "HALLUCINATION": "run",
    "TASK_COMPLETION": "run",
    "TASK_UNDERSTANDING_FAILURE": "run",
    "OFF_TOPIC_DRIFT": "run",
    "USER_FRUSTRATION": "conversation",
    "CONFUSION_LOOP": "conversation",
    "SYCOPHANCY_SIGNAL": "conversation",
}

GUIDE_FOR_TOKEN = {
    "tool": "tool calls (`run.tool_called` / `run.tool_responded`, or `@dt.tool`)",
    "tool.output": "tool output text (pass `output=` to `tool_responded`)",
    "llm": "LLM calls (`run.llm_called` / `run.llm_responded`)",
    "llm.tokens": "token counts on LLM responses",
    "retrieval": "retrieval calls (`run.retrieval_called` / `run.retrieval_responded`)",
    "retrieval.content": "retrieved text (pass `content=` to `retrieval_responded`)",
    "memory": "the agent memory channel (`run.memory_written` / `memory_read`)",
    "external": "infrastructure signals (`run.external_signal`)",
    "input_text": "`user_input=` on `dt.run()`",
    "system_prompt": "`system_prompt=` on `dt.run()`",
    "declared_tools": "`tools=[...]` on `dt.run()`",
    "parent_run_id": "nested runs (open a `dt.run()` inside another)",
    "run": "run lifecycle",
    "baseline": "nothing: this accrues automatically once the agent has run history",
    "config:IRREVERSIBLE_TOOLS": "nothing: declare irreversible tools in `detectors.yml`",
}


def derived_fields() -> dict[str, set[str]]:
    """Per detector class name, every ``state.<field>`` its methods read."""
    sys.path.insert(0, str(ROOT / "packages" / "sdk-py"))
    import dunetrace.detectors as D  # noqa: E402

    src = pathlib.Path(D.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    known = set(FIELD_TOKENS) | IDENTITY_FIELDS

    def walk(node: ast.AST, seen: set[str], depth: int = 0) -> set[str]:
        found: set[str] = set()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in ("state", "run_state")
                and sub.attr in known
            ):
                found.add(sub.attr)
            if isinstance(sub, ast.Call) and depth < 4:
                passes_state = any(
                    isinstance(a, ast.Name) and a.id in ("state", "run_state") for a in sub.args
                )
                fn = sub.func
                nm = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if passes_state and nm in helpers and nm not in seen:
                    seen.add(nm)
                    found |= walk(helpers[nm], seen, depth + 1)
        return found

    out: dict[str, set[str]] = {}
    for cname, node in classes.items():
        acc: set[str] = set()
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                acc |= walk(member, set())
        out[cname] = acc
    return out


def voice_requirements() -> dict[str, list[str]]:
    """Voice pack detector -> the voice event types it reads. Fully derived."""
    import dunetrace.packs.voice as V  # noqa: E402

    src = pathlib.Path(V.__file__).read_text(encoding="utf-8")
    from dunetrace.models import EventType

    out: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.ClassDef):
            continue
        seg = ast.get_source_segment(src, node) or ""
        m = re.search(r'name\s*=\s*"([A-Z_]+)"', seg)
        if not m:
            continue
        members = sorted(set(re.findall(r"EventType\.([A-Z_]+)", seg)))
        out[m.group(1)] = [EventType[x].value for x in members if x in EventType.__members__]
    return out


def build() -> dict:
    sys.path[:0] = [
        str(ROOT / "packages" / "schemas-py"),
        str(ROOT / "packages" / "sdk-py"),
        str(ROOT / "services" / "detector"),
    ]
    from detector_svc.db import LIVE_DETECTORS  # noqa: E402
    from detector_svc.detectors import _DETECTOR_CLASSES  # noqa: E402

    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    entries = curated["detectors"]
    tokens = curated["_TOKENS"]
    implies = {k: v for k, v in curated["_IMPLIES"].items() if not k.startswith("__")}
    reads = derived_fields()

    def expand(names: set[str]) -> set[str]:
        """Close a token set under _IMPLIES. `llm.tokens` also satisfies `llm`."""
        out = set(names)
        for name in names:
            out |= set(implies.get(name, ()))
        return out

    live_names = {n.upper() for n in LIVE_DETECTORS}
    by_name = {cls.name: (key, cls) for key, cls in _DETECTOR_CLASSES.items()}

    problems: list[str] = []
    missing = sorted(set(by_name) - set(entries))
    extra = sorted(set(entries) - set(by_name))
    if missing:
        problems.append(
            f"{len(missing)} detector(s) not described in {CURATED.name}: {', '.join(missing)}. "
            "Add a requires/enriches/why entry; the coverage report cannot count them otherwise."
        )
    if extra:
        problems.append(
            f"{len(extra)} entry in {CURATED.name} matches no detector: {', '.join(extra)}. "
            "Remove it or fix the name."
        )

    structural: dict[str, dict] = {}
    for name, (yaml_key, cls) in sorted(by_name.items()):
        entry = entries.get(name)
        if entry is None:
            continue
        declared: set[str] = set(entry["enriches"])
        for item in entry["requires"]:
            declared |= set(item) if isinstance(item, list) else {item}
        satisfied = expand(declared)

        unknown = sorted(declared - set(tokens))
        if unknown:
            problems.append(f"{name}: unknown token(s) {unknown}. Declare them in _TOKENS.")

        # Audit: every field the code reads must be accounted for.
        for field in sorted(reads.get(cls.__name__, set()) - IDENTITY_FIELDS):
            acceptable = FIELD_TOKENS.get(field)
            if acceptable is None:
                problems.append(f"{name}: reads unmapped RunState field {field!r}.")
            elif not (acceptable & satisfied):
                problems.append(
                    f"{name}: reads state.{field} but neither requires nor enriches names "
                    f"any of {sorted(acceptable)}. Add it to `enriches` if it is optional, "
                    f"or to `requires` if the detector cannot fire without it."
                )

        structural[name] = {
            "yaml_key": yaml_key,
            "requires": entry["requires"],
            "enriches": entry["enriches"],
            "live": name in live_names,
            "why": entry["why"],
            **({"not_runstate": True} if entry.get("not_runstate") else {}),
        }

    if problems:
        raise SystemExit("FAIL:\n  " + "\n  ".join(problems))

    voice = voice_requirements()
    return {
        "_generated_by": "scripts/gen_detector_manifest.py in dunetrace/dunetrace",
        "_do_not_edit": "Regenerate instead. CI fails on drift.",
        "schema": 1,
        "tokens": tokens,
        "implies": implies,
        "token_hint": GUIDE_FOR_TOKEN,
        "counts": {
            "structural": len(structural),
            "structural_live": sum(1 for v in structural.values() if v["live"]),
            "structural_shadow": sum(1 for v in structural.values() if not v["live"]),
            "semantic": len(SEMANTIC_LEVELS),
            "voice_pack": len(voice),
        },
        "structural": structural,
        "semantic": {
            name: {
                "level": level,
                "requires": ["llm"] + (["run"] if level == "conversation" else []),
                "note": (
                    "Opt-in Tier 2. Needs SEMANTIC_WORKER_ENABLED=true and an LLM key "
                    "on the semantic worker. Never triggers a policy."
                ),
            }
            for name, level in sorted(SEMANTIC_LEVELS.items())
        },
        "voice_pack": {
            name: {
                "requires_events": events,
                "note": "Needs the voice pack enabled for the org (dt.enable_pack('voice')).",
            }
            for name, events in sorted(voice.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="references/ directory in the skills repo")
    ap.add_argument("--check", action="store_true", help="fail on drift instead of writing")
    args = ap.parse_args()

    body = json.dumps(build(), indent=2, sort_keys=False) + "\n"
    target = pathlib.Path(args.out) / MANIFEST_NAME

    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != body:
            print(
                f"FAIL: {target} is stale or missing. Regenerate with "
                f"`python scripts/gen_detector_manifest.py --out {args.out}`.",
                file=sys.stderr,
            )
            return 1
        counts = json.loads(body)["counts"]
        print(f"OK: {MANIFEST_NAME} matches the live detector set {counts}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    counts = json.loads(body)["counts"]
    print(f"Wrote {target}")
    print(f"  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
