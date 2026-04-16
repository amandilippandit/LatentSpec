"""Adversarial trace synthesis (§10.1 supporting capability).

Given an invariant + a violation hypothesis, ask Z3 for a concrete
`NormalizedTrace` that satisfies the bounded trace constraints AND violates
the rule. The synthesized trace is a real instance of `NormalizedTrace`,
suitable for:

  - feeding into the runtime guardrail to prove it actually blocks.
  - showing in a §4.3 violation analysis dashboard alongside the rule.
  - shipping as part of a SOC2-grade compliance bundle ("here are
    machine-generated examples that violate this rule").

The synthesizer reuses the symbolic compilation from `smt/symbolic.py`
and decodes the Z3 model into the §3.2 trace schema.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import z3

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    TraceStep,
    UserInputStep,
)
from latentspec.smt.compiler import Z3Compilation
from latentspec.smt.symbolic import _add_domain_constraints

log = logging.getLogger(__name__)


def synthesize_violating_trace(
    compilation: Z3Compilation,
    *,
    agent_id: str = "synth-agent",
    max_length: int = 8,
    max_latency_ms: int = 60_000,
    timeout_ms: int = 3000,
    seed_user_input: str | None = None,
) -> NormalizedTrace | None:
    """Synthesize a §3.2 trace that VIOLATES the invariant, or None.

    The result is decoded into the canonical schema using the same tool-id
    table the verifier uses, so the synthesized trace is checkable by the
    same `dispatch()` layer as a real trace.
    """
    if compilation.invariant_type == InvariantType.OUTPUT_FORMAT:
        # Output format violations are semantic; can't be Z3-synthesized.
        return None

    solver = z3.Solver()
    solver.set("timeout", int(timeout_ms))

    _add_domain_constraints(
        solver,
        compilation,
        max_length=max_length,
        max_tool_id=64,
        max_latency_ms=max_latency_ms,
    )
    # Force trace length >= 1 so we always get *some* steps.
    solver.add(compilation.symbols.n >= 1)
    solver.add(z3.Not(compilation.formula))

    try:
        if solver.check() != z3.sat:
            return None
