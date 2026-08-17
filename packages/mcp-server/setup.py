"""Build hook that bundles the docs this server serves as MCP resources.

The 8 `dunetrace://docs/*` resources read Markdown that lives at `<repo-root>/docs/`,
outside this package. setuptools can only ship files from *inside* the package
directory, so a plain build produced a wheel of three .py files and every doc
resource answered "(doc not found)" once installed — invisible during development,
because an editable install resolves back to the repo and works fine.

This copies them into `dunetrace_mcp/_docs/` at build time. The copy is generated,
never committed (see .gitignore), so `docs/` stays the single source of truth and
there is no second copy to drift.

Everything else is declared in pyproject.toml; this file exists only for the hook.
"""

from __future__ import annotations

import pathlib
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

# Keep in sync with the _read_doc(...) calls in dunetrace_mcp/server.py.
# test_docs_packaging.py asserts these two lists match, so adding a resource
# without adding it here fails the suite rather than shipping a broken build.
BUNDLED_DOCS = [
    "detectors.md",
    "integrate-custom-python-agent.md",
    "integrate-haystack-agent.md",
    "integrate-langchain-agent.md",
    "integrate-langdock.md",
    "integrate-typescript-agent.md",
    "mcp-server.md",
    "policies.md",
]

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_DOCS = _HERE.parents[1] / "docs"
_DEST = _HERE / "dunetrace_mcp" / "_docs"


class BuildPyWithDocs(build_py):
    """Copy the served docs into the package tree before the normal build."""

    def run(self) -> None:
        _copy_docs()
        super().run()


def _copy_docs() -> None:
    if not _REPO_DOCS.is_dir():
        # Building from an sdist that didn't carry the repo layout. Warn loudly:
        # the wheel is still usable, but its doc resources will be empty.
        print(
            f"WARNING: {_REPO_DOCS} not found — building without bundled docs. "
            "The dunetrace://docs/* resources will be unavailable in this wheel.",
        )
        return

    _DEST.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in BUNDLED_DOCS:
        src = _REPO_DOCS / name
        if not src.is_file():
            missing.append(name)
            continue
        shutil.copy2(src, _DEST / name)

    if missing:
        raise SystemExit(
            f"Cannot build: BUNDLED_DOCS lists files that don't exist in "
            f"{_REPO_DOCS}: {', '.join(missing)}. Fix the list in setup.py."
        )


setup(cmdclass={"build_py": BuildPyWithDocs})
