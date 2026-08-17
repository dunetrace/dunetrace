"""
Loads third-party Python-class custom detectors from disk.

A user can already write a custom detector by subclassing dunetrace.detectors.
BaseDetector — it's publicly exported and importable. What was missing: a way
for that subclass to actually reach the live server-side detector pipeline
without editing detector_svc's own source and redeploying. This closes that
gap: any *.py file dropped in the configured directory gets imported, and
BaseDetector.__init_subclass__ (see packages/sdk-py/dunetrace/detectors.py)
auto-registers every subclass it finds into CUSTOM_DETECTOR_REGISTRY —
detectors.py::get_detectors() merges registered-but-unlisted classes into the
detector battery on every call.

This is a plugin surface: malformed or malicious files must not crash the
service or gain more than ordinary Python import privileges (no sandboxing
beyond that — a Python file dropped here already runs with the same
privileges as detector_svc itself, same trust boundary as any other code an
operator chooses to deploy). Each file loads into its own module namespace
(a unique synthetic name, not the shared package namespace) so one file's
import-time error, name collision, or bad syntax is caught and logged without
preventing any other file in the directory from loading.
"""

from __future__ import annotations

import importlib.util
import logging
import os

logger = logging.getLogger("dunetrace.detector.custom_python")

DEFAULT_CUSTOM_DETECTORS_PATH = os.path.expanduser("~/.dunetrace/detectors/")


def load_custom_detector_plugins(path: str | None = None) -> int:
    """Import every *.py file in `path` (default DUNETRACE_CUSTOM_DETECTORS_PATH
    env var, else ~/.dunetrace/detectors/) into its own isolated module
    namespace. Returns the number of files successfully imported.

    Safe to call when the directory doesn't exist (nothing to load — not an
    error, since most installs have no custom Python detectors at all) and
    safe to call more than once (already-imported files are just re-imported
    under a fresh synthetic module name each time; BaseDetector.__init_subclass__
    re-registering the same class name is a harmless overwrite, not an error).
    """
    directory = (
        path or os.environ.get("DUNETRACE_CUSTOM_DETECTORS_PATH") or DEFAULT_CUSTOM_DETECTORS_PATH
    )

    if not os.path.isdir(directory):
        logger.debug("No custom detectors directory at %s — skipping.", directory)
        return 0

    loaded = 0
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue

        filepath = os.path.join(directory, filename)
        module_name = f"dunetrace_custom_detector_plugin_{filename[:-3]}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not create a module spec for {filepath}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded += 1
            logger.info("Loaded custom detector plugin: %s", filename)
        except Exception:
            # Broad except is deliberate: a plugin file can fail in essentially
            # any way (syntax error, missing import, exception at module level)
            # and none of them should be able to take down the detector worker.
            logger.exception(
                "Failed to load custom detector plugin %s — skipping it, "
                "other plugins are unaffected.",
                filepath,
            )

    return loaded
