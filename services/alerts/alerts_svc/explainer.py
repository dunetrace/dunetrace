from __future__ import annotations

import os
import sys

_EXPLAINER_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../explainer")
)
if _EXPLAINER_ROOT not in sys.path:
    sys.path.insert(0, _EXPLAINER_ROOT)

from app.explainer import explain

__all__ = ["explain"]
