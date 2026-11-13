"""
One guard for test dependencies that are optional at *import* time.

Several modules degrade gracefully when an optional library is absent:
``dunetrace_schemas.metrics`` turns every metric into a no-op and sets
``AVAILABLE = False`` when ``prometheus_client`` is missing, and ``/metrics``
then serves a body that says so. That is the right runtime behaviour and the
wrong *test* behaviour — the suites covering that instrumentation used to hide
behind three different escape hatches:

    raise unittest.SkipTest(...)            # services/detector/tests
    pytest.importorskip("prometheus_client") # services/api/tests
    @unittest.skipUnless(metrics.AVAILABLE)  # services/alerts/tests

all of which report ``skipped`` and **exit code 0** under pytest (a module-level
``unittest.SkipTest`` included — verified). Dropping ``prometheus-client`` from
a service's ``requirements.txt``, or losing it to a transitive resolver
conflict, therefore deleted ~90 tests into a skip count nobody reads while CI
stayed green and the shipped image served a ``/metrics`` body saying the
library was not installed — with the compose healthcheck on ``/ready`` still
passing.

So: a missing dependency is a hard error. The comments in those files asserted
"CI installs requirements.txt"; this makes CI *prove* it instead.

A developer who genuinely wants to run a suite without the library sets::

    DUNETRACE_ALLOW_MISSING_TEST_DEPS=1 make test-alerts

and gets the old skip back — explicitly, at their own request, and never in CI
(no workflow sets it).

Usage from a test module (import at module scope, before the dependency is
used, so the failure is a collection error rather than a per-test one)::

    from scripts.require_test_deps import require_prometheus_client

    prometheus_client = require_prometheus_client("services/api/requirements.txt")
"""

from __future__ import annotations

import importlib
import os
import unittest
from types import ModuleType

#: Set to any truthy value to turn a missing dependency back into a skip.
OPT_OUT_ENV = "DUNETRACE_ALLOW_MISSING_TEST_DEPS"

_FALSEY = {"", "0", "false", "no", "off"}


class MissingTestDependency(RuntimeError):
    """Raised at import time when a required test dependency is absent.

    Deliberately *not* a ``SkipTest``: raised at module scope it is a pytest
    collection error, which is a non-zero exit.
    """


def opted_out() -> bool:
    """True when the developer has explicitly asked for the skip behaviour."""
    return os.environ.get(OPT_OUT_ENV, "").strip().lower() not in _FALSEY


def require(module: str, *, requirement: str, pinned_in: str) -> ModuleType:
    """Import ``module`` or fail the run.

    ``requirement`` is the pip requirement that provides it and ``pinned_in``
    the file that is supposed to guarantee it, so the failure message names the
    fix rather than the symptom.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        message = (
            f"{module} is not installed, so these tests cannot verify anything.\n"
            f"It is meant to be guaranteed by {requirement!r} in {pinned_in}; if it "
            f"is missing there, the shipped image serves a /metrics body saying the "
            f"library is not installed while /ready still passes.\n"
            f"Install it, or set {OPT_OUT_ENV}=1 to skip these tests deliberately."
        )
        if opted_out():
            raise unittest.SkipTest(message) from exc
        raise MissingTestDependency(message) from exc


def require_prometheus_client(pinned_in: str) -> ModuleType:
    """The only caller shape today: the metrics/readiness suites."""
    return require(
        "prometheus_client",
        requirement="prometheus-client>=0.20",
        pinned_in=pinned_in,
    )
