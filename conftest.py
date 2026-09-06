"""Put pipeline/ and experiments/ on sys.path so tests import the stage and analysis modules the way the scripts do."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "pipeline"))
