"""Put pipeline/ on sys.path so tests import the stage modules the way the scripts do."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
