"""Symbolic Z3 verification — prove `forall trace satisfying C: rule(trace)`.

Where the concrete verifier in `smt/verifier.py` instantiates one trace and
asks Z3 to check it, the symbolic verifier *quantifies over an abstract
trace* of bounded length and asks Z3 to prove the rule holds for ALL such
traces. This is the engine behind §10.1 enterprise certificates.

The model:
  - `n` is the trace length, an Int constrained `0 <= n <= max_length`.
  - `tool(i)`, `latency(i)`, `success(i)` are uninterpreted functions
    constrained over `0 <= i < n` only.
  - Tool ids range over `[0, max_tool_id]`. The interned ids in the
    compilation's `symbols.tool_ids` plus `0` (no-tool) anchor the domain.
  - Optional environment constraints (e.g. "user_input contains keyword K")
    are added by the caller — useful when proving conditional rules.

If Z3 returns `unsat` for `Not(formula)` under the constraints, the rule is
*proven* over the bounded trace space — a real `forall` proof, not a
sample evaluation. If `sat`, the model is decoded into a counter-example
trace the §4.3 violation analysis can show.

Bounded model checking is decidable for the predicates we use (FOL +
linear arithmetic over a finite domain), so within the bound the proof is
exact. The bound itself is an explicit assumption recorded on the proof.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import z3

from latentspec.models.invariant import InvariantType
from latentspec.smt.compiler import Z3Compilation


log = logging.getLogger(__name__)


@dataclass
class SymbolicAssumption:
    """One human-readable assumption attached to a proof."""

    name: str
    value: str


@dataclass
class SymbolicProof:
    """Output of a symbolic verification run."""

    invariant_type: InvariantType
    proven: bool
    duration_ms: float
    counter_example: dict[str, Any] | None = None
    error: str | None = None
    assumptions: list[SymbolicAssumption] = field(default_factory=list)
    max_trace_length: int = 0
    z3_statistics: dict[str, Any] = field(default_factory=dict)


def _add_domain_constraints(
    solver: z3.Solver,
    compilation: Z3Compilation,
    *,
    max_length: int,
    max_tool_id: int,
    max_latency_ms: int,
) -> list[SymbolicAssumption]:
    s = compilation.symbols
    assumptions: list[SymbolicAssumption] = []

    solver.add(s.n >= 0)
    solver.add(s.n <= max_length)
    assumptions.append(
        SymbolicAssumption(name="max_trace_length", value=str(max_length))
    )

    i = z3.Int("__dom_i")
    # in-range tool ids: 0 (no-tool) or one of the interned ids
    interned_ids = sorted(set(s.tool_ids.values()) | {0})
    in_domain = z3.Or(*(s.tool(i) == tid for tid in interned_ids)) if interned_ids else z3.BoolVal(True)

    solver.add(
        z3.ForAll(
            [i],
            z3.Implies(
                z3.And(0 <= i, i < s.n),
                z3.And(in_domain, s.latency(i) >= 0, s.latency(i) <= max_latency_ms),
            ),
        )
    )
    solver.add(
        z3.ForAll(
            [i],
            z3.Implies(
                z3.Or(i < 0, i >= s.n),
                z3.And(s.tool(i) == 0, s.latency(i) == 0),
            ),
        )
    )
    assumptions.append(
        SymbolicAssumption(
            name="tool_domain",
            value=f"|tools|={len(interned_ids)} (closed-world over interned ids)",
        )
    )
    assumptions.append(
        SymbolicAssumption(name="latency_range", value=f"[0, {max_latency_ms}] ms"),
    )
    return assumptions


def _model_to_counter_example(
    model: z3.ModelRef, compilation: Z3Compilation
) -> dict[str, Any]:
    s = compilation.symbols
    inverse_tools = {v: k for k, v in s.tool_ids.items()}
    try:
        n_val = model.eval(s.n, model_completion=True).as_long()
    except Exception:
        n_val = 0
    n_val = max(0, min(n_val, 32))
    steps: list[dict[str, Any]] = []
    for i in range(n_val):
        try:
            tid = model.eval(s.tool(i), model_completion=True).as_long()
            lat = model.eval(s.latency(i), model_completion=True).as_long()
            ok = bool(z3.is_true(model.eval(s.success(i), model_completion=True)))
        except Exception:
            continue
        steps.append(
            {
                "i": i,
                "tool": inverse_tools.get(tid, "" if tid == 0 else f"tool#{tid}"),
                "latency_ms": lat,
                "success": ok,
            }
        )
    return {"n": n_val, "steps": steps}


def verify_symbolic(
    compilation: Z3Compilation,
    *,
    max_length: int = 12,
    max_tool_id: int = 64,
    max_latency_ms: int = 60_000,
    timeout_ms: int = 5000,
    extra_assumptions: list[z3.BoolRef] | None = None,
) -> SymbolicProof:
    """Prove the invariant holds for *every* trace of length <= `max_length`.

    Returns:
        SymbolicProof with `proven=True` when Z3 returns unsat (i.e. there
        is no model where `Not(formula)` holds), `proven=False` with a
        decoded counter-example when sat. Z3 timeout / unknown returns
        `proven=False` with `error` set so callers can distinguish
        "definitely violatable" from "we couldn't decide".
    """
    start = time.perf_counter()

    # Output_format invariants are LLM-as-judge by design; symbolic
    # verification is meaningless for them.
    if compilation.invariant_type == InvariantType.OUTPUT_FORMAT:
        return SymbolicProof(
            invariant_type=compilation.invariant_type,
            proven=False,
            duration_ms=0.0,
            error="output_format invariants are not symbolically verifiable",
        )

    solver = z3.Solver()
    solver.set("timeout", int(timeout_ms))

    assumptions = _add_domain_constraints(
        solver,
        compilation,
        max_length=max_length,
        max_tool_id=max_tool_id,
        max_latency_ms=max_latency_ms,
    )

    if extra_assumptions:
        for c in extra_assumptions:
            solver.add(c)
            assumptions.append(
                SymbolicAssumption(name="extra", value=str(c)[:240])
            )

    # Look for a model that satisfies the constraints AND violates the rule.
    solver.add(z3.Not(compilation.formula))

    try:
        result = solver.check()
    except z3.Z3Exception as e:
        return SymbolicProof(
            invariant_type=compilation.invariant_type,
            proven=False,
            duration_ms=round((time.perf_counter() - start) * 1000, 3),
            error=f"z3_exception: {e}",
            assumptions=assumptions,
            max_trace_length=max_length,
        )

    duration_ms = round((time.perf_counter() - start) * 1000, 3)
    stats = _z3_stats(solver)

    if result == z3.unsat:
        return SymbolicProof(
            invariant_type=compilation.invariant_type,
            proven=True,
            duration_ms=duration_ms,
            assumptions=assumptions,
            max_trace_length=max_length,
            z3_statistics=stats,
        )
    if result == z3.sat:
        try:
            counter = _model_to_counter_example(solver.model(), compilation)
        except Exception:
            counter = None
        return SymbolicProof(
            invariant_type=compilation.invariant_type,
            proven=False,
            duration_ms=duration_ms,
            counter_example=counter,
            assumptions=assumptions,
            max_trace_length=max_length,
            z3_statistics=stats,
        )
    return SymbolicProof(
        invariant_type=compilation.invariant_type,
        proven=False,
        duration_ms=duration_ms,
        error="z3_unknown_or_timeout",
        assumptions=assumptions,
        max_trace_length=max_length,
        z3_statistics=stats,
    )


def _z3_stats(solver: z3.Solver) -> dict[str, Any]:
    try:
        st = solver.statistics()
        return {st.key(i): st.value(i) for i in range(len(st))}
    except Exception:
        return {}
