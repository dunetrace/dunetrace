"""
The startup guard runs at import of api_svc.config, so the refusal has to be
observed from outside the interpreter: a subprocess that imports the module
under ENV=prod AUTH_MODE=dev must exit non-zero with the message on stderr.
The two sanctioned pairs are run as controls so a passing refusal is not an
ImportError in disguise.

Run: PYTHONPATH=packages/sdk-py:packages/schemas-py:services/explainer:services/api \
       python -m pytest services/api/tests/test_deploy_guard.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_SERVICE_DIR = _TESTS_DIR.parent  # services/api
_REPO_ROOT = _SERVICE_DIR.parent.parent


def _child_pythonpath() -> str:
    # The Makefile's PYTHONPATH is relative to the repo root, and the child
    # runs in an empty temp dir so no developer .env can leak into it — so
    # absolutise what pytest was given and prepend the two paths the import
    # needs regardless of how the suite was launched.
    entries = [str(_REPO_ROOT / "packages" / "schemas-py"), str(_SERVICE_DIR)]
    entries += [os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return os.pathsep.join(entries)


def _import_config(
    cwd: Path, env: str | None, auth_mode: str | None
) -> subprocess.CompletedProcess:
    child_env = {k: v for k, v in os.environ.items() if k not in ("ENV", "AUTH_MODE")}
    child_env["PYTHONPATH"] = _child_pythonpath()
    if env is not None:
        child_env["ENV"] = env
    if auth_mode is not None:
        child_env["AUTH_MODE"] = auth_mode
    return subprocess.run(
        [sys.executable, "-c", "import api_svc.config"],
        cwd=cwd,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_prod_with_dev_auth_refuses_to_import(tmp_path):
    result = _import_config(tmp_path, env="prod", auth_mode="dev")
    assert result.returncode != 0
    assert "dunetrace/api: refusing to start" in result.stderr
    assert "ENV=prod" in result.stderr
    assert "AUTH_MODE=dev" in result.stderr
    # The fix is in the message, and it is a refusal, not a crash: SystemExit
    # prints the text alone, with no traceback for an operator to wade through.
    assert "AUTH_MODE=prod" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "env,auth_mode",
    [
        ("dev", "dev"),  # the quickstart (docker-compose.yml defaults)
        ("prod", "prod"),  # docker-compose.prod.yml
        (None, None),  # bare process: ENV defaults to dev, AUTH_MODE fails closed
    ],
)
def test_sanctioned_pairs_import_cleanly(tmp_path, env, auth_mode):
    result = _import_config(tmp_path, env=env, auth_mode=auth_mode)
    assert result.returncode == 0, result.stderr
