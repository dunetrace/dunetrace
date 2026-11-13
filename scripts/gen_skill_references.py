#!/usr/bin/env python3
"""Project docs/integrate-*.md into skill reference files.

The platform guides stay the single source of truth. This reads the
``<!--dunetrace:instrument ...-->`` block each guide carries plus three of its
sections, and writes one compact reference per framework into the skills repo.

A reference is what the `dunetrace-setup` skill reads AFTER it has matched a
framework. It is deliberately short: the skill loads exactly one, and every
extra line is context the agent pays for on every invocation.

Usage:
    python scripts/gen_skill_references.py --out ../dunetrace-skills/skills/dunetrace-setup/references
    python scripts/gen_skill_references.py --out <dir> --check    # CI: fail on drift

--check is the same contract as scripts/gen_enums.py: regenerate, compare, exit
1 with a diff if the committed files are stale.
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
BLOCK_RE = re.compile(r"<!--dunetrace:instrument\n(.*?)\n-->", re.S)
BANNER = (
    "<!-- GENERATED from docs/{src} by scripts/gen_skill_references.py.\n"
    "     Do not edit here. Edit the guide and regenerate. -->"
)
# Sections lifted verbatim. Order is the order the skill needs them in.
#
# "Known limitations" is here because a safety caveat that lives only under
# "Advanced" never reaches the skill: the projection drops it, and the agent
# instruments the repo without ever seeing it. Anything that can break the
# user's own code belongs in that section, not in Advanced.
SECTIONS = ["Quick Start", "Where this goes", "Known limitations", "Verification"]


def parse_block(text: str) -> dict[str, str] | None:
    m = BLOCK_RE.search(text)
    if not m:
        return None
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def split_sections(text: str) -> dict[str, str]:
    """Map H2 title -> body, stopping at the next H2 or a horizontal rule."""
    out: dict[str, str] = {}
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        body = re.split(r"^---\s*$", body, flags=re.M)[0]
        out[parts[i].strip()] = body.strip()
    return out


def render(src: str, meta: dict[str, str], sections: dict[str, str]) -> str:
    fw = meta["framework"]
    lines = [
        f"# {fw}: Dunetrace instrumentation reference",
        "",
        BANNER.format(src=src),
        "",
        "## Facts",
        "",
        f"- **Language**: {meta['language']}",
        f"- **Install**: `{meta['install']}`",
        f"- **Primary symbol**: `{meta['primary_symbol']}`",
        f"- **Mechanism**: {meta['mechanism']}",
        f"- **Opens its own run**: {meta['opens_own_run']}",
        f"- **Requires an open `dt.run()`**: {meta['requires_run_context']}",
    ]
    if meta.get("wrappers_require_run_context") == "true":
        lines.append(
            "- **Note**: the client wrappers and `autoInstrument()` DO require an "
            "open run, even though `dt.run()`/`dt.trace()` do not."
        )
    lines += [
        f"- **Target file**: {meta['target_file']}",
        f"- **Typical path**: {meta['target_example']}",
        f"- **Grep signals**: {meta['target_hints']}",
        f"- **Emits**: {meta['emits']}",
        "",
    ]
    if meta["requires_run_context"] == "true":
        lines += [
            "> **This integration emits nothing outside an open run.** No error, no",
            "> warning. If the plan does not include a run context, the plan is wrong.",
            "",
        ]
    for name in SECTIONS:
        if name in sections:
            lines += [f"## {name}", "", sections[name], ""]
    lines += [
        "## Source",
        "",
        f"Full guide: `docs/{src}` in the dunetrace/dunetrace repository.",
        "",
    ]
    return "\n".join(lines)


#: Marker every generated file carries. The orphan sweep below keys on it so a
#: hand-authored shared reference (verification.md, and anything like it) is
#: never deleted just for not being derived from a guide.
_GENERATED_MARK = "GENERATED from docs/"


def _is_generated(path: pathlib.Path) -> bool:
    """True only for a file this script produced."""
    try:
        return _GENERATED_MARK in path.read_text(encoding="utf-8")[:512]
    except OSError:
        return False


def build(require_all: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    missing: list[str] = []
    for path in sorted(DOCS.glob("integrate-*.md")):
        text = path.read_text(encoding="utf-8")
        meta = parse_block(text)
        if meta is None:
            missing.append(path.name)
            print(f"  skip (no instrument block): {path.name}", file=sys.stderr)
            continue
        out[f"{meta['framework']}.md"] = render(path.name, meta, split_sections(text))
    if require_all and missing:
        raise SystemExit(
            f"FAIL: {len(missing)} guide(s) have no <!--dunetrace:instrument--> block: "
            f"{', '.join(missing)}.\nA guide without one is invisible to the "
            f"dunetrace-setup skill. Add the block or rename the file out of "
            f"the integrate-*.md pattern."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="references/ directory in the skills repo")
    ap.add_argument("--check", action="store_true", help="fail on drift instead of writing")
    ap.add_argument(
        "--require-all",
        action="store_true",
        help="fail if any docs/integrate-*.md lacks an instrument block",
    )
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    generated = build(require_all=args.require_all)
    if not generated:
        print("ERROR: no guides carried an instrument block", file=sys.stderr)
        return 1

    if args.check:
        stale = []
        for name, body in generated.items():
            existing = outdir / name
            current = existing.read_text(encoding="utf-8") if existing.exists() else ""
            if current != body:
                stale.append(name)
                sys.stderr.writelines(
                    difflib.unified_diff(
                        current.splitlines(True),
                        body.splitlines(True),
                        fromfile=f"committed/{name}",
                        tofile=f"generated/{name}",
                    )
                )
        orphans = sorted(
            p.name for p in outdir.glob("*.md") if p.name not in generated and _is_generated(p)
        )
        if stale or orphans:
            print(
                f"\nFAIL: {len(stale)} stale, {len(orphans)} orphaned "
                f"({', '.join(orphans) or 'none'}). Regenerate with "
                f"`python scripts/gen_skill_references.py --out {args.out}`.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {len(generated)} skill reference files match docs/integrate-*.md")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    for name, body in generated.items():
        (outdir / name).write_text(body, encoding="utf-8")
    for orphan in outdir.glob("*.md"):
        if orphan.name not in generated and _is_generated(orphan):
            orphan.unlink()
            print(f"  removed orphan: {orphan.name}")
    print(f"Wrote {len(generated)} reference files to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
