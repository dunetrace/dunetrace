from dunetrace.client import Dunetrace, DunetraceClient
from dunetrace.models import RunState, FailureType, Severity
from dunetrace.detectors import BaseDetector, TIER1_DETECTORS, run_detectors, PROMPT_INJECTION_DETECTOR

__version__ = "0.1.2"
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
