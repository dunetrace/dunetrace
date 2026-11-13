"""
Startup guard against the one misconfiguration that silently opens the whole
deployment: ``ENV=prod`` together with ``AUTH_MODE=dev``.

``AUTH_MODE=dev`` disables authentication on every endpoint of the ingest and
customer APIs — anyone who can reach the port is an admin of every org and can
spend the LLM budget. That is the intended state of the local quickstart
(``docker compose up -d``, everything on loopback) and an accident anywhere
else. The two HTTP services already log a WARNING when dev mode is active, but
a warning in a container log is not a control; a deployment that declares
itself production and still has auth off must not come up at all.

Every service's ``config.py`` calls :func:`assert_safe_deployment` right after
building its ``settings`` object, so the refusal happens at import time —
before a pool is opened, before a port is bound. The workers authenticate
nothing, but they run the same check so a mis-set environment is visible on
every container in ``docker compose ps``, not just on two of them.

This module is deliberately stdlib-only and side-effect free; it lives here
because ``dunetrace_schemas`` is on every service's PYTHONPATH.
"""

from __future__ import annotations

# Values of ENV that mean "this is a real deployment".
PROD_ENVS: frozenset[str] = frozenset({"prod", "production"})

# Values of AUTH_MODE that disable authentication. Mirrors the ``is_dev``
# properties in api_svc/config.py and ingest_svc/config.py.
DEV_AUTH_MODES: frozenset[str] = frozenset({"dev", "local", "test"})


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def is_unsafe_deployment(env: str | None, auth_mode: str | None) -> bool:
    """True when ``env`` names production and ``auth_mode`` disables auth.

    Anything else passes: an unset or dev ``env`` (the quickstart), a prod
    ``env`` with ``AUTH_MODE=prod``, or an unrecognised value of either — the
    guard refuses one specific known-bad pair and does not try to validate
    the vocabulary.
    """
    return _norm(env) in PROD_ENVS and _norm(auth_mode) in DEV_AUTH_MODES


def format_refusal(service: str, env: str | None, auth_mode: str | None) -> str:
    """The multi-line message a refused service prints to stderr."""
    env_shown = env if env is not None else "<unset>"
    auth_shown = auth_mode if auth_mode is not None else "<unset>"
    return (
        f"dunetrace/{service}: refusing to start.\n"
        "\n"
        f"    ENV={env_shown}  (a production deployment)\n"
        f"    AUTH_MODE={auth_shown}  (authentication disabled)\n"
        "\n"
        "AUTH_MODE=dev skips authentication on every endpoint, so a production\n"
        "deployment with it set is open to anyone who can reach the port. Either:\n"
        "\n"
        "  * set AUTH_MODE=prod — the production override\n"
        "    (docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d)\n"
        "    sets it for every service, or\n"
        "  * set ENV=dev if this really is a local, loopback-only instance.\n"
        "\n"
        'See docs/operations.md, section "Deploying".\n'
    )


def assert_safe_deployment(service: str, env: str | None, auth_mode: str | None) -> None:
    """Raise ``SystemExit`` (status 1, message on stderr) for ENV=prod + AUTH_MODE=dev.

    ``service`` is a short name for the log line (``"ingest"``, ``"api"``, …).
    A no-op for every other combination.
    """
    if is_unsafe_deployment(env, auth_mode):
        raise SystemExit(format_refusal(service, env, auth_mode))


__all__ = [
    "PROD_ENVS",
    "DEV_AUTH_MODES",
    "is_unsafe_deployment",
    "format_refusal",
    "assert_safe_deployment",
]
