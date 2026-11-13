#!/usr/bin/env python3
"""
Static safety checks for the dashboard (dashboard/mission-control.html) and
the nginx config that serves it (dashboard/nginx.conf).

The dashboard is one static HTML file with a single inline <script>, served by
nginx with a strict Content-Security-Policy whose script-src is ONLY the
SHA-256 hash of that script (see dashboard/nginx.conf). That policy is what
makes an XSS in the page inert: an injected <script> or inline handler has no
matching hash and is refused by the browser. Two things keep it honest, and
this script guards both:

  1. The page must not depend on anything the CSP forbids. Inline event
     handlers (onclick=…), javascript: URLs, eval() and new Function are all
     blocked by a hash-only script-src, so any that creep back in are dead code
     at best and a regression of the delegated ACTIONS dispatcher at worst.
     The old escaper jsVal() and the onclick="${…}" pattern it served are
     banned by name — they are how caller-controlled ids used to be spliced
     into JS strings inside HTML attributes.

  2. The hash in nginx.conf must match the script's current bytes. Any edit to
     the inline script — one character — changes the hash, and a stale hash
     means the browser refuses to run the dashboard at all (blank page, one
     console error). `--fix` rewrites the hash in place; CI runs the check
     without it so a PR that touched the script but not the hash fails.

It also checks that every action name the markup dispatches
(data-action="…" attributes and the first argument of actionAttrs('…', …)
calls) is a key of the `const ACTIONS = { … };` registry, so a renamed or
removed handler is caught here rather than as a console warning at click time.

Everything is regex-based on the file text — deliberately dependency-free
(stdlib only, no HTML/JS parser) so it runs in the lint job with nothing
installed. It flags what LOOKS like a violation for a human to check.

Usage:
    python scripts/check_dashboard_safety.py          # check, exit 1 on any violation
    python scripts/check_dashboard_safety.py --fix    # also rewrite the CSP hash in nginx.conf
    python scripts/check_dashboard_safety.py --root /path/to/repo

Exit code 0: no violations.
Exit code 1: at least one violation (suitable for a CI check).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import sys
from pathlib import Path

DASHBOARD_DIR = "dashboard"
HTML_FILE = "dashboard/mission-control.html"
NGINX_FILE = "dashboard/nginx.conf"
# Checked and --fix'ed as well when it exists and carries a CSP (see run_checks).
EXTRA_NGINX_FILE = "infra/nginx.conf"

# Files under dashboard/ that are scanned for forbidden patterns. Anything else
# (images, fonts) is skipped — the patterns are meaningful only in text served
# to, or interpreted by, the browser.
SCANNED_SUFFIXES = {".html", ".htm", ".js", ".mjs", ".css", ".conf", ".svg", ".json", ".txt"}

# (label, compiled regex). Every match is a violation.
FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # The removed JS-string escaper — its only job was splicing values into
    # inline handlers, so any reference means the old pattern is back.
    ("jsVal(", re.compile(r"jsVal\(")),
    # Template interpolation straight into an inline handler.
    ('onclick="${', re.compile(r'onclick\s*=\s*["\']\$\{')),
    # Any inline event-handler attribute: onclick="…", onclick='…', with or
    # without spaces around '=', and whether preceded by a space, a quote, a
    # '>' or a newline (the "unspaced" variants: <a href="x"onclick="…">).
    # The lookbehind excludes word characters, '.', '$' and '-' so that
    # `el.onclick = …` (property assignment in JS), `data-onclick="…"` and
    # identifiers like `buttonclick` are not matched.
    (
        "inline event handler attribute",
        re.compile(r"(?<![\w.$-])on[a-z]+\s*=\s*[\"']", re.IGNORECASE),
    ),
    # javascript: URLs (any case, optional whitespace before the colon) in
    # value position: after an attribute's '=' (quoted or not) or opening a
    # string/template literal. Prose in a comment ("rejects javascript:, data:")
    # is not a URL and is not matched — the URL guard that documents its own
    # threat model must not fail the check.
    ("javascript: URL", re.compile(r"""(?:=\s*["']?\s*|["'`])javascript\s*:""", re.IGNORECASE)),
    # Dynamic code evaluation. \b keeps `retrieval(` from matching.
    ("eval(", re.compile(r"\beval\s*\(")),
    ("new Function", re.compile(r"\bnew\s+Function\b")),
]

# An inline <script> (no src= attribute) and its body. Bytes, because the CSP
# hash is over the exact bytes the browser receives.
_INLINE_SCRIPT_RE = re.compile(
    rb"<script(?P<attrs>\s[^>]*)?>(?P<body>.*?)</script>", re.DOTALL | re.IGNORECASE
)
_SRC_ATTR_RE = re.compile(rb"\bsrc\s*=", re.IGNORECASE)

# The CSP header line in nginx.conf: add_header Content-Security-Policy "…" always;
_CSP_HEADER_RE = re.compile(r'add_header\s+Content-Security-Policy\s+"(?P<value>[^"]*)"')
_SHA256_TOKEN_RE = re.compile(r"'sha256-[A-Za-z0-9+/=]+'")

# Action names in markup: data-action="name" / data-action='name', and the
# literal first argument of actionAttrs('name', …). A name is identifier-shaped;
# the dispatcher's own template (data-action="${attr(name)}") and prose in
# comments (data-action="…") are not names and are not matched.
_DATA_ACTION_RE = re.compile(r"""data-action\s*=\s*(["'])(?P<name>[A-Za-z_$][\w$-]*)\1""")
_ACTION_ATTRS_CALL_RE = re.compile(r"""actionAttrs\(\s*(["'])(?P<name>[A-Za-z_$][\w$-]*)\1""")

# The registry block: from `const ACTIONS = {` to the matching `}`.
_ACTIONS_START_RE = re.compile(r"\bconst\s+ACTIONS\s*=\s*\{")
# A property at depth 1 of the object literal: shorthand `name,`, `name: fn`,
# method `name(…) {` (optionally `async`), or a quoted key `'name': fn`.
_ACTIONS_KEY_RE = re.compile(
    r"""^\s*(?:async\s+)?(?:(?P<ident>[A-Za-z_$][\w$]*)|(?P<q>["'])(?P<quoted>[^"']+)(?P=q))\s*[,:(]"""
)


# ── Rule 1: forbidden patterns ───────────────────────────────────────────────


def find_forbidden_patterns(text: str, label: str) -> list[str]:
    """Return one message per forbidden-pattern match in `text` (`label` names the file)."""
    violations: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in FORBIDDEN_PATTERNS:
            for _ in pattern.finditer(line):
                snippet = line.strip()
                if len(snippet) > 100:
                    snippet = snippet[:100] + "…"
                violations.append(f"{label}:{line_no}: forbidden pattern {name!r} — {snippet}")
    return violations


def scan_dashboard_dir(root: Path) -> list[str]:
    dashboard = root / DASHBOARD_DIR
    if not dashboard.is_dir():
        return [f"{DASHBOARD_DIR}/: directory not found under {root}"]
    violations: list[str] = []
    for path in sorted(p for p in dashboard.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        violations.extend(find_forbidden_patterns(text, str(path.relative_to(root))))
    return violations


# ── Rule 2: inline-script hash vs nginx CSP ──────────────────────────────────


def extract_inline_scripts(html: bytes) -> list[bytes]:
    """Bodies of every <script> without a src= attribute, as raw bytes — the
    exact bytes between the opening tag's '>' and '</script>'."""
    bodies: list[bytes] = []
    for m in _INLINE_SCRIPT_RE.finditer(html):
        attrs = m.group("attrs") or b""
        if _SRC_ATTR_RE.search(attrs):
            continue
        bodies.append(m.group("body"))
    return bodies


def csp_hash(script: bytes) -> str:
    """The CSP source expression for an inline script: 'sha256-<base64>'."""
    return "'sha256-" + base64.b64encode(hashlib.sha256(script).digest()).decode("ascii") + "'"


def expected_script_hashes(html: bytes) -> list[str]:
    return [csp_hash(body) for body in extract_inline_scripts(html)]


def read_csp(nginx_text: str) -> str | None:
    m = _CSP_HEADER_RE.search(nginx_text)
    return m.group("value") if m else None


def csp_directive(csp: str, name: str) -> str | None:
    """The value of directive `name` in a CSP string, or None if absent."""
    for part in csp.split(";"):
        tokens = part.split()
        if tokens and tokens[0].lower() == name:
            return " ".join(tokens[1:])
    return None


def script_src_hashes(nginx_text: str) -> list[str] | None:
    """The 'sha256-…' tokens in nginx.conf's CSP script-src, or None if there
    is no CSP header / no script-src directive."""
    csp = read_csp(nginx_text)
    if csp is None:
        return None
    script_src = csp_directive(csp, "script-src")
    if script_src is None:
        return None
    return _SHA256_TOKEN_RE.findall(script_src)


def check_script_hash(html: bytes, nginx_text: str) -> list[str]:
    expected = expected_script_hashes(html)
    if not expected:
        return [f"{HTML_FILE}: no inline <script> found — nothing to hash"]
    if len(expected) != 1:
        return [
            f"{HTML_FILE}: found {len(expected)} inline <script> blocks, expected exactly one "
            "(the CSP hash covers a single script)"
        ]
    actual = script_src_hashes(nginx_text)
    if actual is None:
        return [
            f'{NGINX_FILE}: no `add_header Content-Security-Policy "…"` with a script-src directive found'
        ]
    if actual != expected:
        return [
            f"{NGINX_FILE}: script-src hash does not match the inline <script> in {HTML_FILE}\n"
            f"    expected: {expected[0]}\n"
            f"    found:    {' '.join(actual) if actual else '(no sha256 token)'}\n"
            "    run `python scripts/check_dashboard_safety.py --fix` to update it"
        ]
    return []


def rewrite_script_hash(nginx_text: str, hashes: list[str]) -> str:
    """Return nginx_text with script-src's sha256 tokens replaced by `hashes`.
    Only the script-src directive inside the CSP header value is touched."""
    m = _CSP_HEADER_RE.search(nginx_text)
    if m is None:
        raise ValueError(f"{NGINX_FILE}: no Content-Security-Policy add_header to rewrite")
    csp = m.group("value")
    new_parts: list[str] = []
    found = False
    for part in csp.split(";"):
        tokens = part.split()
        if tokens and tokens[0].lower() == "script-src":
            found = True
            kept = [t for t in tokens[1:] if not _SHA256_TOKEN_RE.fullmatch(t)]
            leading_ws = part[: len(part) - len(part.lstrip())]
            new_parts.append(leading_ws + " ".join(["script-src", *kept, *hashes]))
        else:
            new_parts.append(part)
    if not found:
        raise ValueError(f"{NGINX_FILE}: CSP has no script-src directive to rewrite")
    new_csp = ";".join(new_parts)
    return nginx_text[: m.start("value")] + new_csp + nginx_text[m.end("value") :]


# ── Rule 3: data-action names vs the ACTIONS registry ────────────────────────


def _scan_js(text: str, start: int):
    """Yield (index, char, depth_before) for every character of `text` from
    `start` that is NOT inside a string, template literal or comment, tracking
    `{`/`}` depth. A small state machine, not a JS parser — enough for an
    object literal of handlers."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c in ("'", '"', "`"):
            # The opening quote itself is reported (so a quoted key at the
            # start of a line has a depth), then the literal is skipped.
            yield i, c, depth
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                elif quote == "`" and text[i] == "$" and i + 1 < n and text[i + 1] == "{":
                    # `${ … }` inside a template literal: skip to its closing brace.
                    inner = 1
                    i += 2
                    while i < n and inner:
                        if text[i] == "{":
                            inner += 1
                        elif text[i] == "}":
                            inner -= 1
                        i += 1
                    continue
                i += 1
            i += 1
            continue
        yield i, c, depth
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1


def extract_actions_registry(script: str) -> str | None:
    """The text of the `const ACTIONS = { … }` object literal (between the
    braces, exclusive), or None if there is no registry."""
    m = _ACTIONS_START_RE.search(script)
    if m is None:
        return None
    start = m.end()  # just after the opening '{'
    for idx, ch, depth in _scan_js(script, start):
        if ch == "}" and depth == 0:
            return script[start:idx]
    return None


def parse_actions_keys(script: str) -> set[str] | None:
    """Keys of the ACTIONS registry — shorthand properties, `name: fn` entries
    and `name(…) {` methods — or None if the registry is not found."""
    body = extract_actions_registry(script)
    if body is None:
        return None
    # Depth at the start of each line, relative to the registry object
    # (0 = a direct property; >0 = inside a method body).
    line_starts = [0] + [i + 1 for i, c in enumerate(body) if c == "\n"]
    depth_at: dict[int, int] = {}
    for idx, _ch, depth in _scan_js(body, 0):
        depth_at.setdefault(idx, depth)
    keys: set[str] = set()
    for ls in line_starts:
        le = body.find("\n", ls)
        line = body[ls:] if le == -1 else body[ls:le]
        first = ls + (len(line) - len(line.lstrip()))
        if first not in depth_at or depth_at[first] != 0:
            continue  # blank, comment-only, or inside a nested body
        m = _ACTIONS_KEY_RE.match(line)
        if m:
            keys.add(m.group("ident") or m.group("quoted"))
    return keys


def referenced_actions(html_text: str) -> list[tuple[int, str]]:
    """(line, action name) for every literal data-action attribute and every
    actionAttrs('name', …) call."""
    refs: list[tuple[int, str]] = []
    for line_no, line in enumerate(html_text.splitlines(), start=1):
        for m in _DATA_ACTION_RE.finditer(line):
            refs.append((line_no, m.group("name")))
        for m in _ACTION_ATTRS_CALL_RE.finditer(line):
            refs.append((line_no, m.group("name")))
    return refs


def check_actions(html_text: str) -> list[str]:
    keys = parse_actions_keys(html_text)
    if keys is None:
        return [f"{HTML_FILE}: `const ACTIONS = {{ … }};` registry not found"]
    if not keys:
        return [f"{HTML_FILE}: ACTIONS registry parsed but no keys found"]
    violations: list[str] = []
    for line_no, name in referenced_actions(html_text):
        if name not in keys:
            violations.append(
                f"{HTML_FILE}:{line_no}: data-action {name!r} is not a key of the ACTIONS registry"
            )
    return violations


# ── Driver ───────────────────────────────────────────────────────────────────


def _check_or_fix_hash(html_bytes: bytes, nginx_path: Path, label: str, fix: bool) -> list[str]:
    """check_script_hash for one nginx config, rewriting the hash when fix=True."""
    nginx_text = nginx_path.read_text(encoding="utf-8")
    hash_violations = [
        v.replace(NGINX_FILE, label) for v in check_script_hash(html_bytes, nginx_text)
    ]
    if hash_violations and fix:
        expected = expected_script_hashes(html_bytes)
        if len(expected) == 1:
            try:
                new_text = rewrite_script_hash(nginx_text, expected)
            except ValueError as exc:
                return [str(exc).replace(NGINX_FILE, label)]
            if new_text != nginx_text:
                nginx_path.write_text(new_text, encoding="utf-8")
                print(f"{label}: script-src hash updated to {expected[0]}")
            hash_violations = [
                v.replace(NGINX_FILE, label) for v in check_script_hash(html_bytes, new_text)
            ]
    return hash_violations


def run_checks(root: Path, fix: bool = False) -> list[str]:
    """Run every rule; return the list of violation messages (empty = clean).
    With fix=True, a stale script-src hash is rewritten in nginx.conf instead
    of being reported."""
    violations: list[str] = []
    html_path = root / HTML_FILE
    nginx_path = root / NGINX_FILE

    violations.extend(scan_dashboard_dir(root))

    if not html_path.is_file():
        violations.append(f"{HTML_FILE}: not found")
        return violations
    if not nginx_path.is_file():
        violations.append(f"{NGINX_FILE}: not found")
        return violations

    html_bytes = html_path.read_bytes()
    html_text = html_bytes.decode("utf-8", errors="replace")
    nginx_text = nginx_path.read_text(encoding="utf-8")

    violations.extend(_check_or_fix_hash(html_bytes, nginx_path, NGINX_FILE, fix))

    # The production reverse proxy (infra/nginx.conf) serves the same page and
    # carries the same CSP; keep its hash in step too. Optional: a checkout
    # without it, or one whose proxy has no CSP header, is not a violation.
    extra_path = root / EXTRA_NGINX_FILE
    if (
        extra_path.is_file()
        and script_src_hashes(extra_path.read_text(encoding="utf-8")) is not None
    ):
        violations.extend(_check_or_fix_hash(html_bytes, extra_path, EXTRA_NGINX_FILE, fix))

    violations.extend(check_actions(html_text))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent, help="repo root"
    )
    parser.add_argument(
        "--fix", action="store_true", help="rewrite the CSP script-src hash in nginx.conf in place"
    )
    args = parser.parse_args(argv)

    violations = run_checks(args.root, fix=args.fix)
    if violations:
        print(f"check_dashboard_safety: {len(violations)} violation(s)", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("check_dashboard_safety: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
