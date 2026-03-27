"""pytest exporter — Expansion-phase target (months 4-6).

Stub today; will emit a `test_invariants.py` that imports the
`latentspec.checking` runtime and asserts each invariant against trace
fixtures stored alongside the test file.
"""

from __future__ import annotations

from typing import Iterable

from latentspec.schemas.invariant import MinedInvariant


def export_pytest(invariants: Iterable[MinedInvariant]) -> str:
    rules = list(invariants)
    return (
        "# pytest export — coming in the Expansion phase (§10).\n"
        f"# {len(rules)} invariants pending translation.\n"
    )
