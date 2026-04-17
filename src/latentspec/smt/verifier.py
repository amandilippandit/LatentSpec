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
