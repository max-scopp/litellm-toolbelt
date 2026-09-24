"""Make the shared fakes importable as `fakes` from every test module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
