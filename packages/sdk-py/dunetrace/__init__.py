from dunetrace.client import Dunetrace, DunetraceClient
from dunetrace.context import get_current_run
from dunetrace.middleware import DunetraceASGIMiddleware, DunetraceWSGIMiddleware
from dunetrace.models import (
    RunState,
    FailureType,
    Severity,
    RiskScore,
    Exporter,
    CallableExporter,
)
from dunetrace.detectors import (
    BaseDetector,
    TIER1_DETECTORS,
    run_detectors,
    PROMPT_INJECTION_DETECTOR,
)
from dunetrace.policies import Policy, PolicyViolation
from dunetrace.risk_engine import RiskEngine

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dunetrace")
except PackageNotFoundError:
    __version__ = "0.0.0"  # running from source without installing
__all__ = [
    "Dunetrace",
    "DunetraceClient",  # backwards-compatible alias
    "get_current_run",
    "DunetraceASGIMiddleware",
    "DunetraceWSGIMiddleware",
    "RunState",
    "FailureType",
    "Severity",
    "BaseDetector",
    "TIER1_DETECTORS",
    "run_detectors",
    "PROMPT_INJECTION_DETECTOR",
    "Policy",
    "PolicyViolation",
    "RiskEngine",
    "RiskScore",
    "Exporter",
    "CallableExporter",
]
