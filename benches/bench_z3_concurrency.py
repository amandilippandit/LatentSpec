"""Z3 concurrency stress test.

Z3 is not thread-safe out of the box; the SMT solver caches global state
that can corrupt under parallel access. The verifier sets per-call
`timeout_ms` but I had never tested what happens when N callers race for
verification simultaneously.

This harness:

  1. Builds N independent compiled invariants with disjoint symbol tables.
  2. Verifies M traces against each one in parallel (asyncio.gather over
     `asyncio.to_thread`-wrapped Z3 calls).
  3. Reports: how many succeeded, how many timed out, max wall time,
     wall_time / serial_time speedup.
  4. Asserts no exceptions escape the verifier (so a Z3 race can't kill
     the streaming detector).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from datetime import UTC, datetime

from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.smt.compiler import compile_invariant
from latentspec.smt.verifier import verify_trace, verify_trace_async


def _build_traces(n: int) -> list[NormalizedTrace]:
    return [
        NormalizedTrace(
            trace_id=f"t-{i}",
            agent_id="bench",
            timestamp=datetime.now(UTC),
            steps=[
                UserInputStep(content=f"req {i}"),
                ToolCallStep(tool="auth", args={}, latency_ms=100),
                ToolCallStep(tool="db_write", args={}, latency_ms=200),
                AgentResponseStep(content=f"ok {i}"),
            ],
            metadata=TraceMetadata(),
        )
        for i in range(n)
    ]


def _build_compilations(n: int):
    out = []
    for i in range(n):
        # Different params each time so each compile is genuinely distinct
        params = {"tool_a": f"auth_v{i}", "tool_b": f"db_write_v{i}"}
        out.append(compile_invariant(InvariantType.ORDERING, params))
    return out


async def stress(*, n_invariants: int, n_traces: int, timeout_ms: int) -> dict:
    """Verify n_invariants × n_traces pairs in parallel."""
    compilations = _build_compilations(n_invariants)
    traces = _build_traces(n_traces)

    pairs = [(c, t) for c in compilations for t in traces]
    n_total = len(pairs)

    start = time.perf_counter()

    async def _run_one(comp, trace) -> tuple[float, bool, str | None]:
        t0 = time.perf_counter()
        result = await verify_trace_async(comp, trace, timeout_ms=timeout_ms)
        t1 = time.perf_counter()
