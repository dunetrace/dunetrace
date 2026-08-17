"""
Guards the bundling of the docs served as `dunetrace://docs/*` MCP resources.

The failure this prevents is silent and invisible in development. The resources
read Markdown from `<repo-root>/docs/`, which lives outside the package, so a
plain build shipped three .py files and nothing else — and every doc resource
answered "(doc not found)" once installed from PyPI. An editable install resolves
back to the real repo, so it worked perfectly for anyone with a checkout.

Two things are asserted:

1. `setup.py:BUNDLED_DOCS` lists exactly the files `server.py` asks for. Adding a
   resource without adding it to that list would ship a wheel whose new resource
   is empty — this fails the build instead.
2. `_read_doc` falls back correctly, so the same code path works from a checkout
   (docs read live, local edits visible without a rebuild) and from an installed
   wheel (docs read from the packaged copy).

Run:
    cd packages/mcp-server
    python -m pytest tests/test_docs_packaging.py -v
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
_REPO_ROOT = _PKG_ROOT.parents[1]
_SERVER_PY = _PKG_ROOT / "dunetrace_mcp" / "server.py"


def _bundled_docs() -> list[str]:
    """Read BUNDLED_DOCS out of setup.py without executing setup()."""
    source = (_PKG_ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r"BUNDLED_DOCS = \[(.*?)\]", source, re.DOTALL)
    assert match, "BUNDLED_DOCS not found in setup.py"
    return re.findall(r'"([^"]+\.md)"', match.group(1))


def _requested_docs() -> list[str]:
    """Every filename server.py passes to _read_doc()."""
    source = _SERVER_PY.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'_read_doc\("([^"]+)"\)', source)))


class TestBundleList:
    def test_every_served_doc_is_bundled(self):
        missing = sorted(set(_requested_docs()) - set(_bundled_docs()))
        assert not missing, (
            f"server.py serves {missing} but setup.py's BUNDLED_DOCS doesn't list "
            f"them — a built wheel would return '(doc not found)' for those "
            f"resources. Add them to BUNDLED_DOCS."
        )

    def test_no_bundled_doc_is_unused(self):
        extra = sorted(set(_bundled_docs()) - set(_requested_docs()))
        assert not extra, (
            f"BUNDLED_DOCS lists {extra}, which no resource serves — drop them "
            f"rather than shipping dead weight in the wheel."
        )

    def test_every_bundled_doc_exists_in_the_repo(self):
        missing = [n for n in _bundled_docs() if not (_REPO_ROOT / "docs" / n).is_file()]
        assert not missing, (
            f"BUNDLED_DOCS names files absent from {_REPO_ROOT / 'docs'}: {missing}. "
            f"The build raises SystemExit on this; the test catches it earlier."
        )

    def test_the_list_is_not_empty(self):
        # A regex that silently stopped matching would make every assertion above
        # vacuously pass.
        assert len(_bundled_docs()) >= 8
        assert len(_requested_docs()) >= 8


class TestReadDocFallback:
    """`_read_doc` has to satisfy both install shapes from one code path."""

    def test_reads_from_the_repo_when_running_from_a_checkout(self):
        from dunetrace_mcp.server import _read_doc

        content = _read_doc("mcp-server.md")
        assert not content.startswith("(doc not found")
        assert len(content) > 1000

    def test_every_served_doc_resolves_here(self):
        from dunetrace_mcp.server import _read_doc

        broken = [n for n in _requested_docs() if _read_doc(n).startswith("(doc not found")]
        assert not broken, f"these resources return nothing: {broken}"

    def test_unknown_doc_reports_both_lookup_locations(self):
        """The message has to say where it looked, or a packaging regression is
        indistinguishable from a typo."""
        from dunetrace_mcp.server import _read_doc

        msg = _read_doc("no-such-doc.md")
        assert msg.startswith("(doc not found")
        assert "no-such-doc.md" in msg
        assert "not packaged" in msg

    def test_prefers_the_packaged_copy_over_the_repo(self, tmp_path, monkeypatch):
        """An installed wheel must not depend on a repo being present — the
        packaged copy is consulted first and is sufficient on its own."""
        from dunetrace_mcp import server

        packaged = pathlib.Path(server.__file__).parent / "_docs"
        if not packaged.is_dir():
            pytest.skip("no packaged _docs in this tree (build hasn't run)")

        # Point the repo fallback at nothing; the packaged copy must still answer.
        monkeypatch.setattr(server, "_DOCS", tmp_path / "absent")
        assert not server._read_doc("mcp-server.md").startswith("(doc not found")


class TestWheelContents:
    """Checks a built artifact if one is present.

    `dist/` is gitignored, so this is a local convenience rather than a CI gate —
    it skips when nothing is built. It is scoped to the *current* version on
    purpose: stale wheels from earlier versions accumulate in `dist/` and predate
    the packaging fix, and failing someone's test run over a leftover artifact
    would be noise, not signal.
    """

    def _current_version(self) -> str:
        text = (_PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert match, "version not found in pyproject.toml"
        return match.group(1)

    def test_wheel_for_the_current_version_contains_the_docs(self):
        import zipfile

        version = self._current_version()
        wheels = sorted((_PKG_ROOT / "dist").glob(f"*-{version}-*.whl"))
        if not wheels:
            pytest.skip(f"no wheel built for version {version}")

        names = zipfile.ZipFile(wheels[-1]).namelist()
        shipped = {n.rsplit("/", 1)[-1] for n in names if n.endswith(".md")}
        missing = sorted(set(_bundled_docs()) - shipped)
        assert not missing, (
            f"{wheels[-1].name} is missing {missing} — the build hook in setup.py "
            f"did not run or did not find the repo docs. Rebuild: python -m build --wheel"
        )
