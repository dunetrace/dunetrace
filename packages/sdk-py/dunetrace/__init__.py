from dunetrace.client import Dunetrace, DunetraceClient
from dunetrace.models import RunState, FailureType, Severity
from dunetrace.detectors import BaseDetector, TIER1_DETECTORS, run_detectors, PROMPT_INJECTION_DETECTOR

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("dunetrace")
except PackageNotFoundError:
    __version__ = "0.0.0"  # running from source without installing
__all__ = [
    "Dunetrace",
    "DunetraceClient",  # backwards-compatible alias
    "RunState",
    "FailureType",
    "Severity",
    "BaseDetector",
    "TIER1_DETECTORS",
    "run_detectors",
    "PROMPT_INJECTION_DETECTOR",
]
