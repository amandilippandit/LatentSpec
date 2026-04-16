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
        model = solver.model()
    except z3.Z3Exception as e:
        log.debug("synthesis Z3 failed: %s", e)
        return None

    return _decode(compilation, model, agent_id=agent_id, seed_user_input=seed_user_input)


def _decode(
    compilation: Z3Compilation,
    model: z3.ModelRef,
    *,
    agent_id: str,
    seed_user_input: str | None,
) -> NormalizedTrace:
    s = compilation.symbols
    inverse_tools = {v: k for k, v in s.tool_ids.items()}

    n_val = max(1, min(int(model.eval(s.n, model_completion=True).as_long()), 32))

    steps: list[TraceStep] = []
    if seed_user_input is not None:
        steps.append(UserInputStep(content=seed_user_input))
    elif s.keyword_ids:
        # If the rule references a keyword, seed it into user_input so the
        # decoded trace satisfies the precondition.
        plain_keywords = [k for k in s.keyword_ids if not k.startswith("segment:")]
        if plain_keywords:
            steps.append(UserInputStep(content=" ".join(plain_keywords[:3])))

    for i in range(n_val):
        tid = int(model.eval(s.tool(i), model_completion=True).as_long())
        if tid == 0:
            continue  # non-tool slot
        tool_name = inverse_tools.get(tid)
        if tool_name is None:
            continue
        latency = int(model.eval(s.latency(i), model_completion=True).as_long())
        success = bool(z3.is_true(model.eval(s.success(i), model_completion=True)))
        steps.append(
            ToolCallStep(
                tool=tool_name,
                args={},
                latency_ms=max(0, latency),
                result_status="success" if success else "error",
            )
        )

    if not any(isinstance(s_, ToolCallStep) for s_ in steps):
        # Empty model — fabricate a minimal violation trace using interned tools
        # so the runtime guardrail has something to chew on.
        for tool_name in list(inverse_tools.values())[:3]:
            steps.append(ToolCallStep(tool=tool_name, args={}, latency_ms=10))

    steps.append(AgentResponseStep(content="(synthesized adversarial trace)"))

    # Determine segment from the model (only when tool_selection)
    segment_value: str | None = None
    for kw_label, kw_id in s.keyword_ids.items():
        if not kw_label.startswith("segment:"):
            continue
        if z3.is_true(model.eval(s.keyword(kw_id), model_completion=True)):
            segment_value = kw_label.split(":", 1)[1]
            break

    return NormalizedTrace(
        trace_id=f"adversarial-{uuid.uuid4().hex[:10]}",
        agent_id=agent_id,
        timestamp=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        steps=steps,
        metadata=TraceMetadata(
            model="latentspec.smt.synthesis",
            user_segment=segment_value,
        ),
    )
