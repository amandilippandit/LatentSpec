"""Verify a §3.2 trace against a Z3 expression.

For each compiled invariant we instantiate the trace as a concrete model:
  - n        := len(trace.steps)
  - tool(i)  := id of step[i].tool, or 0 for non-tool steps
  - latency  := step[i].latency_ms or 0
  - success  := result_status == "success"
  - keyword(K) := keyword K appears in any user_input step (case-insensitive)

We then ask Z3: does NOT(formula) have a model? If yes (sat), the trace
violates the invariant — Z3 returns a counter-example we surface in the
result. If no (unsat), the formula holds for the trace.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import z3

from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import (
    NormalizedTrace,
    StepType,
    ToolCallStep,
    UserInputStep,
)
from latentspec.smt.compiler import Z3Compilation


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Z3 thread safety
# ----------------------------------------------------------------------
#
# Z3's default Solver shares global state and SIGSEGV's Python when
# touched from multiple OS threads — verified by
# `benches/bench_z3_concurrency.py` which crashed the interpreter on
# the first parallel run.
#
# Two-layer defense:
#   1. `_Z3_LOCK` serialises every direct in-thread call so adversarial
#      callers (running verify_trace synchronously from multiple threads)
#      are still safe.
#   2. `verify_trace_async()` pins ALL Z3 work to a single dedicated OS
#      thread via `_Z3_EXECUTOR`. This is the right interface for
#      anything calling Z3 from an async event loop (the streaming
#      detector, the certificate generator, and the bench).
#
# Future hardening: per-thread `z3.Context()` instances would let
# verification actually run in parallel. For now, correctness wins —
# Z3 is not on the streaming hot path (rule-based checkers handle that).
# ----------------------------------------------------------------------
_Z3_LOCK = threading.Lock()
_Z3_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="latentspec-z3"
)


@dataclass
class VerificationResult:
    invariant_type: InvariantType
    holds: bool
    duration_ms: float
    counter_example: dict[str, Any] | None = None
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _trace_token_set(trace: NormalizedTrace) -> set[str]:
    """All lower-case alphanumeric tokens from user_input steps."""
    tokens: set[str] = set()
    for step in trace.steps:
        if isinstance(step, UserInputStep):
            for raw in step.content.lower().split():
                tok = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
                if tok:
                    tokens.add(tok)
    return tokens


def _trace_metadata_segment(trace: NormalizedTrace) -> str | None:
    seg = trace.metadata.user_segment
    return seg.lower() if seg else None


def _instantiate(compilation: Z3Compilation, trace: NormalizedTrace) -> list[z3.BoolRef]:
    """Build the constraints describing this concrete trace in the symbol space."""
    s = compilation.symbols
    constraints: list[z3.BoolRef] = []
    n_value = len(trace.steps)
    constraints.append(s.n == n_value)

    user_tokens = _trace_token_set(trace)
    segment = _trace_metadata_segment(trace)

    for idx, step in enumerate(trace.steps):
        if isinstance(step, ToolCallStep):
            tool_id = s.tool_ids.get(step.tool, 0)
            constraints.append(s.tool(idx) == tool_id)
            constraints.append(s.latency(idx) == int(step.latency_ms or 0))
            constraints.append(
                s.success(idx)
                == ((step.result_status or "success") == "success")
            )
            constraints.append(s.has_step_type(s.step_type_ids["tool_call"], idx))
        else:
            constraints.append(s.tool(idx) == 0)
            constraints.append(s.latency(idx) == 0)
            constraints.append(s.success(idx) == True)
            type_value = step.type if isinstance(step.type, StepType) else StepType(step.type)
            constraints.append(
                s.has_step_type(s.step_type_ids.get(type_value.value, 0), idx)
            )

    # Out-of-range tool slots default to 0 so quantifiers can range freely
    bound = z3.Int("__bound")
    constraints.append(
        z3.ForAll(
            [bound],
            z3.Implies(
                z3.Or(bound < 0, bound >= n_value),
                z3.And(s.tool(bound) == 0, s.latency(bound) == 0),
            ),
        )
    )

    # Keyword facts
    for kw, kw_id in s.keyword_ids.items():
        if kw.startswith("segment:"):
            seg_value = kw.split(":", 1)[1]
            constraints.append(
                s.keyword(kw_id) == (segment is not None and segment == seg_value)
            )
        else:
            constraints.append(s.keyword(kw_id) == (kw.lower() in user_tokens))

    return constraints


def _extract_counter_example(
    model: z3.ModelRef, compilation: Z3Compilation
) -> dict[str, Any]:
    s = compilation.symbols
    out: dict[str, Any] = {}
    try:
        n = model.eval(s.n, model_completion=True).as_long()
    except Exception:
        return out
    out["n"] = n
    samples: list[dict[str, Any]] = []
    inv_tool_ids = {v: k for k, v in s.tool_ids.items()}
    for i in range(min(n, 16)):
        try:
            tid = model.eval(s.tool(i), model_completion=True).as_long()
            lat = model.eval(s.latency(i), model_completion=True).as_long()
            ok = bool(z3.is_true(model.eval(s.success(i), model_completion=True)))
        except Exception:
            continue
        samples.append(
            {
                "i": i,
                "tool": inv_tool_ids.get(tid, "" if tid == 0 else f"tool#{tid}"),
                "latency_ms": lat,
                "success": ok,
            }
        )
    out["steps"] = samples
    return out


def verify_trace(
    compilation: Z3Compilation,
    trace: NormalizedTrace,
    *,
    timeout_ms: int = 100,
) -> VerificationResult:
    """Run Z3 against (compilation, trace).

    Returns `holds=True` when the invariant is satisfied. `holds=False` plus
    `counter_example` when violated. Z3 timeout returns `holds=True` with
    `error="timeout"` so the streaming detector stays fail-open under load.
    """
    start = time.perf_counter()

    with _Z3_LOCK:
        solver = z3.Solver()
        solver.set("timeout", int(max(1, timeout_ms)))

        constraints = _instantiate(compilation, trace)
        for c in constraints:
            solver.add(c)

        # Ask: can the formula be FALSE under these constraints?
        solver.add(z3.Not(compilation.formula))

        try:
            result = solver.check()
        except z3.Z3Exception as e:
            log.debug("Z3 exception during verify: %s", e)
            return VerificationResult(
                invariant_type=compilation.invariant_type,
                holds=True,
                duration_ms=round((time.perf_counter() - start) * 1000, 3),
                error=f"z3_exception: {e}",
            )

        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        if result == z3.unsat:
            return VerificationResult(
                invariant_type=compilation.invariant_type,
                holds=True,
                duration_ms=duration_ms,
            )
        if result == z3.sat:
            try:
                counter = _extract_counter_example(solver.model(), compilation)
            except Exception:
                counter = None
            return VerificationResult(
                invariant_type=compilation.invariant_type,
                holds=False,
                duration_ms=duration_ms,
                counter_example=counter,
            )
        # unknown — treat as fail-open with explicit error
        return VerificationResult(
            invariant_type=compilation.invariant_type,
            holds=True,
            duration_ms=duration_ms,
            error="z3_unknown_or_timeout",
        )


async def verify_trace_async(
    compilation: Z3Compilation,
    trace: NormalizedTrace,
    *,
    timeout_ms: int = 100,
) -> VerificationResult:
    """Async wrapper that pins Z3 work to the dedicated single-thread executor.

    Use this from any asyncio context (streaming detector, certificate
    generator, FastAPI handlers). Calling `verify_trace` directly from
    `asyncio.to_thread` is unsafe because Python's default thread pool
    will rotate Z3 calls across multiple OS threads.
    """
    loop = asyncio.get_running_loop()
    fn = functools.partial(verify_trace, compilation, trace, timeout_ms=timeout_ms)
    return await loop.run_in_executor(_Z3_EXECUTOR, fn)
