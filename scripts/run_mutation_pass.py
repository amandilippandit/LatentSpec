"""Focused mutation testing.

mutmut runs the test suite once per source-code mutation. We focus on
two modules where surviving mutations indicate weak tests:

  - `latentspec/checking/dispatch.py` + per-type checkers
  - `latentspec/canonicalization/canonicalizer.py`

A surviving mutation in either is a real signal: the test for that
behaviour either doesn't exist or doesn't actually exercise the
mutated branch.

This script wraps mutmut so we can re-run it from `make`.
"""

from __future__ import annotations

import subprocess
import sys


TARGETS = [
    "src/latentspec/checking",
    "src/latentspec/canonicalization",
]


def main() -> int:
    print("Running mutmut on:", TARGETS)
    cmd = [
        ".venv/bin/mutmut",
        "run",
        "--paths-to-mutate",
        ",".join(TARGETS),
        "--tests-dir",
        "tests/",
        "--runner",
        ".venv/bin/python -m pytest tests/property tests/differential -q -x",
    ]
    proc = subprocess.run(cmd, capture_output=False)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
