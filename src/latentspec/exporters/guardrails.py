"""Guardrails AI exporter — Expansion-phase target (months 4-6).

Stub today; the registry shape is in place so downstream tooling can call
`export_guardrails(invariants)` once the Expansion-phase template work
lands. We emit a placeholder validator skeleton so existing callers don't
break.
"""

from __future__ import annotations

from typing import Iterable

from latentspec.schemas.invariant import MinedInvariant


def export_guardrails(invariants: Iterable[MinedInvariant]) -> str:
    """Return a placeholder Guardrails AI validator file."""
    rules = list(invariants)
    return (
        "# Guardrails AI export — coming in the Expansion phase (§10).\n"
        f"# {len(rules)} invariants pending translation.\n"
    )
