#!/usr/bin/env python3
"""
Tests for check_dashboard_safety.py itself — a static-analysis tool is only
useful if its detection logic is actually correct, so this isn't left untested
just because it lives in scripts/. Self-contained: writes a tiny synthetic
dashboard (mission-control.html + nginx.conf) into a temp directory, no
fixtures needed from the rest of the repo. The last test class runs the checker
on the real repo, so a stale hash or a stray inline handler fails here too.

Run: python scripts/test_check_dashboard_safety.py
     python -m unittest scripts.test_check_dashboard_safety
"""

from __future__ import annotations

import base64
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_dashboard_safety as cds  # noqa: E402

SCRIPT = """
const API = 'http://localhost:8002';
const withValue = fn => function (...rest) { return fn.call(this, this.value, ...rest); };
const ACTIONS = {
  // Direct handlers.
  showDetail,
  navTo,
  'quoted-name': navTo,
  setRunsSearch: withValue(setRunsSearch),
  switchTab(id) { switchTab(id, this); },
  async fetchThing(id) { await load(id); },
  viewAllRunsForAgent(agentId) {
    navTo('runs');            // a call inside a method body is NOT a key
    const s = { nested: 1 };  // braces inside a body
    if (s) { renderRuns(); }
  },
  searchInput(e) { if (e.type === 'blur') closeSearch(); else runSearch(this.value); },
  tpl() { return `x ${ {a: 1}.a } }`; },
};
function actionAttrs(name, ...args) {
  return `data-action="${attr(name)}" data-args="${attr(JSON.stringify(args))}"`;
}
function render() {
  return `<button ${actionAttrs('showDetail', id)}>x</button>` +
         `<button ${actionAttrs("switchTab", 'a')}>y</button>`;
}
"""

HTML_HEAD = """<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono" rel="stylesheet">
<style>body { margin: 0 }</style>
</head><body>
<button data-action="navTo" data-args='["runs"]'>Runs</button>
<select data-action='setRunsSearch' data-on="change"></select>
<div data-action="quoted-name"></div>
<!-- data-action="…" is prose, not a name -->
"""
HTML_TAIL = "</body></html>\n"


def build_html(script: str = SCRIPT, body_extra: str = "") -> str:
    return HTML_HEAD + body_extra + "<script>" + script + "</script>\n" + HTML_TAIL


def build_nginx(script_src: str) -> str:
    return (
        "server {\n"
        "    listen 80;\n"
        '    add_header X-Frame-Options "DENY" always;\n'
        "    add_header Content-Security-Policy \"default-src 'self'; "
        f"script-src {script_src}; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' http://localhost:8002; object-src 'none'\" always;\n"
        "}\n"
    )


def write_repo(root: Path, html: str, nginx: str) -> None:
    (root / "dashboard").mkdir(parents=True, exist_ok=True)
    (root / "dashboard" / "mission-control.html").write_text(html, encoding="utf-8")
    (root / "dashboard" / "nginx.conf").write_text(nginx, encoding="utf-8")


def hash_of(html: str) -> str:
    return cds.expected_script_hashes(html.encode("utf-8"))[0]


class TestForbiddenPatterns(unittest.TestCase):
    def _v(self, text: str) -> list[str]:
        return cds.find_forbidden_patterns(text, "f.html")

    def test_clean_markup_and_js_pass(self):
        clean = (
            '<button data-action="x" data-args=\'["a"]\'>x</button>\n'
            "el.onclick = handler;            // property assignment, not an attribute\n"
            "const v = excessive_retrieval(x);  // a longer name that happens to end in -eval\n"
            "// the URL guard rejects other schemes (javascript:, data:, vbscript:)\n"
            "const e = evaluate(y); const f = medieval (z);\n"
            "const url = 'https://example.com/javascript/page';\n"
            '<div data-onclick="not-a-handler"></div>\n'
            "// inline handlers (onclick=fn('${x}')) used to be rendered here\n"
        )
        self.assertEqual(self._v(clean), [])

    def test_jsval_is_flagged(self):
        v = self._v('`<a onclick="f(${jsVal(id)})">`')
        self.assertTrue(any("jsVal(" in m for m in v), v)

    def test_onclick_template_interpolation_is_flagged(self):
        v = self._v('`<a onclick="${handler}">`')
        self.assertTrue(any('onclick="${' in m for m in v), v)

    def test_inline_handler_attribute_variants_are_flagged(self):
        for markup in (
            '<a onclick="f()">',  # spaced, double-quoted
            "<a onclick='f()'>",  # single-quoted
            '<a href="x"onclick="f()">',  # unspaced after a quote
            '<a\n\tonmouseover = "f()">',  # newline/tab before, spaces around '='
            '<input ONCHANGE="f()">',  # upper-case
            '<body onload="f()">',
            "<img src=x onerror='f()'>",
        ):
            with self.subTest(markup=markup):
                v = self._v(markup)
                self.assertTrue(any("inline event handler" in m for m in v), (markup, v))

    def test_javascript_url_is_flagged(self):
        for markup in (
            '<a href="javascript:void(0)">',
            "<a href='JavaScript :alert(1)'>",
            "<a href=javascript:alert(1)>",
            "location.href = 'javascript:alert(1)';",
            "el.setAttribute('href', `javascript:${x}`);",
        ):
            with self.subTest(markup=markup):
                v = self._v(markup)
                self.assertTrue(any("javascript: URL" in m for m in v), (markup, v))

    def test_eval_and_new_function_are_flagged(self):
        v = self._v("const r = eval(code);")
        self.assertTrue(any("eval(" in m for m in v), v)
        v = self._v("const r = window.eval (code);")
        self.assertTrue(any("eval(" in m for m in v), v)
        v = self._v("const fn = new Function('a', 'return a');")
        self.assertTrue(any("new Function" in m for m in v), v)
        v = self._v("const fn = new  Function(body);")
        self.assertTrue(any("new Function" in m for m in v), v)

    def test_message_carries_file_and_line(self):
        v = self._v('ok\nok\n<a onclick="f()">')
        self.assertEqual(len(v), 1)
        self.assertTrue(v[0].startswith("f.html:3:"), v[0])

    def test_scan_covers_every_text_file_under_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo(root, build_html(), build_nginx("'sha256-x'"))
            (root / "dashboard" / "extra.js").write_text("eval(x)\n")
            (root / "dashboard" / "sub").mkdir()
            (root / "dashboard" / "sub" / "widget.html").write_text('<a onclick="f()">\n')
            (root / "dashboard" / "logo.png").write_bytes(b"eval(\x00")  # binary: not scanned
            v = cds.scan_dashboard_dir(root)
        self.assertEqual(len(v), 2, v)
        self.assertTrue(any(m.startswith("dashboard/extra.js:1:") for m in v), v)
        self.assertTrue(any(m.startswith("dashboard/sub/widget.html:1:") for m in v), v)


class TestInlineScriptHash(unittest.TestCase):
    def test_hash_is_over_exact_bytes_between_tags(self):
        html = b"<html><script>alert(1)</script></html>"
        expected = base64.b64encode(hashlib.sha256(b"alert(1)").digest()).decode()
        self.assertEqual(cds.expected_script_hashes(html), [f"'sha256-{expected}'"])

    def test_whitespace_inside_the_script_changes_the_hash(self):
        a = cds.expected_script_hashes(b"<script>x()</script>")
        b = cds.expected_script_hashes(b"<script>x() </script>")
        c = cds.expected_script_hashes(b"<script>\nx()</script>")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_attributes_on_the_tag_do_not_affect_the_hash_but_src_scripts_are_skipped(self):
        plain = cds.expected_script_hashes(b"<script>x()</script>")
        typed = cds.expected_script_hashes(b'<script type="module">x()</script>')
        self.assertEqual(plain, typed)
        self.assertEqual(cds.expected_script_hashes(b'<script src="a.js"></script>'), [])
        self.assertEqual(cds.extract_inline_scripts(b"<SCRIPT>x()</SCRIPT>"), [b"x()"])

    def test_matching_hash_passes(self):
        html = build_html()
        nginx = build_nginx(hash_of(html))
        self.assertEqual(cds.check_script_hash(html.encode(), nginx), [])

    def test_stale_hash_fails_with_expected_and_found(self):
        html = build_html()
        nginx = build_nginx("'sha256-STALE='")
        v = cds.check_script_hash(html.encode(), nginx)
        self.assertEqual(len(v), 1)
        self.assertIn(hash_of(html), v[0])
        self.assertIn("'sha256-STALE='", v[0])
        self.assertIn("--fix", v[0])

    def test_extra_hash_token_fails(self):
        """script-src must contain EXACTLY the page's hash — an extra token
        would keep some other (possibly injected) script runnable."""
        html = build_html()
        nginx = build_nginx(hash_of(html) + " 'sha256-EXTRA='")
        self.assertEqual(len(cds.check_script_hash(html.encode(), nginx)), 1)

    def test_missing_csp_or_script_src_fails(self):
        html = build_html()
        self.assertEqual(len(cds.check_script_hash(html.encode(), "server { listen 80; }\n")), 1)
        no_script_src = "add_header Content-Security-Policy \"default-src 'self'\" always;"
        self.assertEqual(len(cds.check_script_hash(html.encode(), no_script_src)), 1)

    def test_zero_or_two_inline_scripts_fail(self):
        nginx = build_nginx("'sha256-x'")
        self.assertEqual(len(cds.check_script_hash(b"<html></html>", nginx)), 1)
        two = b"<script>a()</script><script>b()</script>"
        v = cds.check_script_hash(two, nginx)
        self.assertEqual(len(v), 1)
        self.assertIn("2 inline", v[0])

    def test_rewrite_replaces_only_the_script_src_hash(self):
        html = build_html()
        nginx = build_nginx("'self' 'sha256-STALE='")
        new = cds.rewrite_script_hash(nginx, [hash_of(html)])
        self.assertEqual(cds.script_src_hashes(new), [hash_of(html)])
        # Other directives, other headers, and non-hash script-src tokens survive.
        self.assertIn("script-src 'self' " + hash_of(html) + ";", new)
        self.assertIn("connect-src 'self' http://localhost:8002", new)
        self.assertIn('add_header X-Frame-Options "DENY" always;', new)
        # Idempotent.
        self.assertEqual(cds.rewrite_script_hash(new, [hash_of(html)]), new)

    def test_rewrite_without_csp_raises(self):
        with self.assertRaises(ValueError):
            cds.rewrite_script_hash("server {}", ["'sha256-x'"])


class TestActionsRegistry(unittest.TestCase):
    def test_parses_shorthand_colon_method_async_and_quoted_keys(self):
        keys = cds.parse_actions_keys(SCRIPT)
        self.assertEqual(
            keys,
            {
                "showDetail",
                "navTo",
                "quoted-name",
                "setRunsSearch",
                "switchTab",
                "fetchThing",
                "viewAllRunsForAgent",
                "searchInput",
                "tpl",
            },
        )

    def test_calls_inside_method_bodies_are_not_keys(self):
        keys = cds.parse_actions_keys(SCRIPT)
        for not_a_key in ("renderRuns", "nested", "s", "closeSearch", "runSearch", "load", "a"):
            self.assertNotIn(not_a_key, keys)

    def test_missing_registry_returns_none(self):
        self.assertIsNone(cds.parse_actions_keys("const OTHER = { a, b };"))

    def test_registry_end_is_found_past_braces_in_strings_and_comments(self):
        script = "const ACTIONS = {\n  a() { return '}'; },  // } in a comment\n  b,\n};\nconst later = { c };"
        self.assertEqual(cds.parse_actions_keys(script), {"a", "b"})

    def test_references_collect_attributes_and_action_attrs_calls(self):
        html = build_html()
        names = sorted({n for _, n in cds.referenced_actions(html)})
        self.assertEqual(
            names, ["navTo", "quoted-name", "setRunsSearch", "showDetail", "switchTab"]
        )

    def test_template_and_prose_values_are_not_references(self):
        refs = cds.referenced_actions(
            'data-action="${attr(name)}" data-action="…" actionAttrs(name, x)'
        )
        self.assertEqual(refs, [])

    def test_known_actions_pass(self):
        self.assertEqual(cds.check_actions(build_html()), [])

    def test_unknown_attribute_action_fails(self):
        html = build_html(body_extra='<button data-action="doesNotExist">x</button>\n')
        v = cds.check_actions(html)
        self.assertEqual(len(v), 1)
        self.assertIn("'doesNotExist'", v[0])

    def test_unknown_action_attrs_call_fails(self):
        html = build_html(SCRIPT + "\nconst m = `<a ${actionAttrs('gone', 1)}>`;\n")
        v = cds.check_actions(html)
        self.assertEqual(len(v), 1)
        self.assertIn("'gone'", v[0])

    def test_missing_registry_fails(self):
        v = cds.check_actions(build_html("function f() {}"))
        self.assertEqual(len(v), 1)
        self.assertIn("ACTIONS", v[0])


class TestRunChecksAndFix(unittest.TestCase):
    def test_clean_synthetic_repo_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html()
            write_repo(root, html, build_nginx(hash_of(html)))
            self.assertEqual(cds.run_checks(root), [])

    def test_stale_hash_is_reported_without_fix_and_rewritten_with_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html()
            write_repo(root, html, build_nginx("'sha256-STALE='"))
            self.assertEqual(len(cds.run_checks(root)), 1)
            # nginx.conf untouched by a plain check.
            self.assertIn("'sha256-STALE='", (root / "dashboard" / "nginx.conf").read_text())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cds.run_checks(root, fix=True), [])
            self.assertEqual(
                cds.script_src_hashes((root / "dashboard" / "nginx.conf").read_text()),
                [hash_of(html)],
            )
            self.assertEqual(cds.run_checks(root), [])

    def test_fix_does_not_hide_other_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html(body_extra='<a onclick="f()">bad</a>\n')
            write_repo(root, html, build_nginx("'sha256-STALE='"))
            with redirect_stdout(io.StringIO()):
                v = cds.run_checks(root, fix=True)
            self.assertEqual(len(v), 1)
            self.assertIn("inline event handler", v[0])

    def test_missing_files_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(cds.run_checks(Path(tmp)))

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html()
            write_repo(root, html, build_nginx("'sha256-STALE='"))
            err, out = io.StringIO(), io.StringIO()
            with redirect_stderr(err), redirect_stdout(out):
                self.assertEqual(cds.main(["--root", str(root)]), 1)
                self.assertEqual(cds.main(["--root", str(root), "--fix"]), 0)
                self.assertEqual(cds.main(["--root", str(root)]), 0)
            self.assertIn("1 violation(s)", err.getvalue())


class TestRealRepo(unittest.TestCase):
    """The checker against the actual dashboard: a stale CSP hash, a stray
    inline handler or an unregistered data-action fails here as well as in
    the CI step."""

    ROOT = Path(__file__).resolve().parent.parent

    def test_real_dashboard_passes(self):
        if not (self.ROOT / cds.HTML_FILE).is_file():
            self.skipTest("not running inside the repo")
        self.assertEqual(cds.run_checks(self.ROOT), [])

    def test_real_dashboard_has_no_inline_handlers_and_one_script(self):
        if not (self.ROOT / cds.HTML_FILE).is_file():
            self.skipTest("not running inside the repo")
        html = (self.ROOT / cds.HTML_FILE).read_bytes()
        self.assertEqual(len(cds.extract_inline_scripts(html)), 1)
        keys = cds.parse_actions_keys(html.decode("utf-8"))
        self.assertIsNotNone(keys)
        self.assertGreater(len(keys), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
