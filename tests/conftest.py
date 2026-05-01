"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when running tests directly via `pytest`
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT))
