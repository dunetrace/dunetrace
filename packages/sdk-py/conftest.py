"""Make `dunetrace` importable when running pytest from packages/sdk-py/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
