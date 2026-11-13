#!/usr/bin/env python3
"""
Checks detectors.yml against the detector classes it is supposed to tune.

Three separate things can drift, and all three fail silently at runtime:

  1. A key in detectors.yml that reaches no constructor kwarg. The shared
     parser (dunetrace_schemas.detector_config._PARAM_MAP) maps YAML key ->
     class attribute; a key with no entry is dropped with a logger.warning at
     worker and ingest startup. Nobody reads startup warnings, so the operator
     sets a threshold, restarts, and gets the old behaviour with no error.
     This is not hypothetical: ScattershotToolUseDetector renamed
     MAX_DISTINCT_TOOLS -> MIN_DISTINCT_TOOLS (the >= comparison contradicted
     the MAX_ name) and gained MIN_REPEAT_RATIO and SCAN_LIMIT. detectors.yml
     was updated, _PARAM_MAP was not, so all four shipped scattershot tunables
     were silently discarded — as was instrumentation_degraded's min_calls,
     which never had a _PARAM_MAP entry at all.

  2. A _PARAM_MAP entry pointing at a class attribute that no longer exists.
     The override can never land, and because the YAML key still parses, no
     unknown-key warning fires either. The same rename left
     "max_distinct_tools" -> "MAX_DISTINCT_TOOLS" aimed at nothing.

  3. A value in the `default:` section that differs from the class default.
     detectors.yml's own header says "Omitted fields use the SDK default
     (shown below)", so the shipped baseline claims to BE the class defaults.
     When a class default is tuned and the file is not, every deployment using
     the shipped file keeps the old value while the docs describe the new one.

This is the same ratchet gen_enums.py --check applies to the generated enum
modules: one source of truth, verified in CI rather than by memory.

Per-agent-category sections (a section named after an agent_id) are checked
for 1 and 2 but NOT 3 — an override is meant to differ from the default; that
is the entire point of one.

A deliberate divergence for 3 is marked by putting

    # default-drift-ok: <reason>

on the line immediately above the key, the same escape hatch
check_endpoint_conventions.py uses for its own legitimate exceptions. Prefer
changing the class default so the two agree; use the marker only when the
shipped baseline genuinely should differ from the code default.

Usage:
    python scripts/check_detector_defaults.py
    python scripts/check_detector_defaults.py --config path/to/detectors.yml

Exit code 0: no drift.
Exit code 1: at least one violation (suitable for a CI check).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import the SDK and the shared schemas package from the tree, so the check
# runs against the working copy rather than whatever happens to be installed.
sys.path[:0] = [
    str(REPO_ROOT / "packages" / "sdk-py"),
    str(REPO_ROOT / "packages" / "schemas-py"),
]

from dunetrace.detectors import DETECTOR_KEYS  # noqa: E402
from dunetrace_schemas.detector_config import (  # noqa: E402
    _PARAM_MAP,
    _RESERVED_DETECTOR_KEYS,
    BUILTIN_DETECTOR_KEYS,
    _CUSTOM_DETECTORS_KEY,
)

DEFAULT_CATEGORY = "default"
MARKER = "# default-drift-ok:"

# severity and max_cost_ns are accepted for every detector (cross-cutting
# BaseDetector attributes, not per-detector tunables), and both are things an
# operator is expected to set. They are checked for existence on the class but
# never value-compared: `severity: high` is a string in YAML and a Severity
# enum on the class, so equality would be comparing two different types.
_NO_VALUE_COMPARE = frozenset({"severity", "max_cost_ns"})
_CROSS_CUTTING_ATTRS = {"severity": "SEVERITY", "max_cost_ns": "MAX_COST_NS"}

_CATEGORY_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*):\s*(?:#.*)?$")
_DETECTOR_RE = re.compile(r"^ {2}([a-z_][a-z0-9_]*):\s*(?:#.*)?$")
_KEY_RE = re.compile(r"^ {4}([a-z_][a-z0-9_]*):\s*(\S.*)$")


def scan_positions(text: str) -> Dict[Tuple[str, str, str], Tuple[int, bool]]:
    """(category, detector, key) -> (line number, carries the drift marker).

    PyYAML discards comments, so the marker and the line numbers have to come
    from a separate pass over the raw text. Only the three indent levels that
    matter are tracked; anything deeper (alert_policy's body) belongs to a
    different consumer and is skipped.
    """
    positions: Dict[Tuple[str, str, str], Tuple[int, bool]] = {}
    category: Optional[str] = None
    detector: Optional[str] = None
    prev_comment = ""

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            prev_comment = ""
            continue
        if stripped.startswith("#"):
            prev_comment = stripped
            continue

        if (m := _CATEGORY_RE.match(raw)) is not None:
            category, detector = m.group(1), None
        elif (m := _DETECTOR_RE.match(raw)) is not None:
            detector = m.group(1)
        elif (m := _KEY_RE.match(raw)) is not None and category and detector:
            positions[(category, detector, m.group(1))] = (
                lineno,
                prev_comment.startswith(MARKER),
            )
        prev_comment = ""

    return positions


def _class_default(detector: str, key: str) -> Tuple[bool, Any]:
    """(the class has an attribute for this key, its value)."""
    attr = _PARAM_MAP.get(detector, {}).get(key) or _CROSS_CUTTING_ATTRS.get(key)
    cls = DETECTOR_KEYS.get(detector)
    if cls is None or attr is None or not hasattr(cls, attr):
        return False, None
    return True, getattr(cls, attr)


def check(config_path: Path) -> list[str]:
    import yaml  # imported here so --help works without PyYAML installed

    text = config_path.read_text()
    raw = yaml.safe_load(text) or {}
    positions = scan_positions(text)
    rel = config_path.name
    violations: list[str] = []

    # --- 2. every _PARAM_MAP entry aims at an attribute that exists ----------
    for detector, params in sorted(_PARAM_MAP.items()):
        if detector not in BUILTIN_DETECTOR_KEYS:
            violations.append(
                f"_PARAM_MAP has an entry for {detector!r}, which is not a known "
                f"detector section (BUILTIN_DETECTOR_KEYS)."
            )
            continue
        cls = DETECTOR_KEYS.get(detector)
        if cls is None:
            violations.append(f"_PARAM_MAP entry {detector!r} has no class in DETECTOR_KEYS.")
            continue
        for key, attr in sorted(params.items()):
            if not hasattr(cls, attr):
                violations.append(
                    f"_PARAM_MAP[{detector!r}][{key!r}] -> {cls.__name__}.{attr}, which "
                    f"does not exist. The override can never land; the YAML key still "
                    f"parses, so no unknown-key warning fires either."
                )

    # --- 1 and 3. every key in the file lands somewhere, with the right value -
    for category, body in raw.items():
        if category == _CUSTOM_DETECTORS_KEY or not isinstance(body, dict):
            continue
        for detector, cfg in body.items():
            if detector not in BUILTIN_DETECTOR_KEYS:
                violations.append(
                    f"{rel}: {category}.{detector} is not a known detector section. "
                    f"One section per entry in detector_svc's _DETECTOR_CLASSES."
                )
                continue
            if not isinstance(cfg, dict):
                continue
            for key, value in cfg.items():
                if key in ("alert_policy", "destinations"):
                    continue  # read by alerts_svc, not by this parser
                lineno, marked = positions.get((category, detector, key), (0, False))
                where = f"{rel}:{lineno}" if lineno else rel

                has_attr, default = _class_default(detector, key)
                if key not in _PARAM_MAP.get(detector, {}) and key not in _RESERVED_DETECTOR_KEYS:
                    violations.append(
                        f"{where}: {category}.{detector}.{key} reaches no constructor "
                        f"kwarg — _PARAM_MAP has no entry for it, so the loader drops "
                        f"it with a startup warning and the setting does nothing."
                    )
                    continue
                if not has_attr:
                    violations.append(
                        f"{where}: {category}.{detector}.{key} maps to an attribute "
                        f"{DETECTOR_KEYS[detector].__name__} does not have."
                    )
                    continue

                if category != DEFAULT_CATEGORY or key in _NO_VALUE_COMPARE or marked:
                    continue
                if default != value:
                    violations.append(
                        f"{where}: {category}.{detector}.{key} is {value!r} but "
                        f"{DETECTOR_KEYS[detector].__name__}."
                        f"{_PARAM_MAP[detector][key]} defaults to {default!r}. "
                        f"detectors.yml's header says the shipped values ARE the SDK "
                        f"defaults. Change the class default, or mark the line with "
                        f"'{MARKER} <reason>'."
                    )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "detectors.yml",
        help="detectors.yml to check (default: the one at the repo root)",
    )
    # Accepted so this reads like its sibling checks in CI; the script only
    # ever checks, it never rewrites detectors.yml.
    parser.add_argument("--check", action="store_true", help="no-op, accepted for symmetry")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"No detectors.yml at {args.config}")
        return 1

    violations = check(args.config)
    for v in violations:
        print(v)

    if violations:
        print(
            f"\n{len(violations)} violation(s). detectors.yml, the classes in "
            f"packages/sdk-py/dunetrace/detectors.py, and _PARAM_MAP in "
            f"packages/schemas-py/dunetrace_schemas/detector_config.py have to "
            f"agree — see this script's docstring for why each case is silent."
        )
        return 1

    tunables = sum(len(p) for p in _PARAM_MAP.values())
    print(
        f"Checked {len(BUILTIN_DETECTOR_KEYS)} detector sections and {tunables} "
        f"mapped tunables against {args.config.name}. No drift."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
