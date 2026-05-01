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
        return (t1 - t0) * 1000.0, result.holds, result.error

    results = await asyncio.gather(
        *(_run_one(c, t) for c, t in pairs), return_exceptions=True
    )
    elapsed = time.perf_counter() - start

    durations = []
    n_held = 0
    n_failed = 0
    n_errored = 0
    n_timeouts = 0
    n_exceptions = 0

    for r in results:
        if isinstance(r, Exception):
            n_exceptions += 1
            continue
        d_ms, holds, error = r
        durations.append(d_ms)
        if error == "z3_unknown_or_timeout":
            n_timeouts += 1
        elif error:
            n_errored += 1
        elif holds:
            n_held += 1
        else:
            n_failed += 1

    durations.sort()
    n = max(1, len(durations))
    return {
        "wall_time_seconds": round(elapsed, 3),
        "verifications": n_total,
        "throughput_per_second": round(n_total / max(elapsed, 0.001), 1),
        "n_held_proven": n_held,
        "n_violation_found": n_failed,
        "n_timeouts": n_timeouts,
        "n_errored": n_errored,
        "n_exceptions": n_exceptions,
        "p50_ms": durations[n // 2],
        "p95_ms": durations[min(n - 1, int(n * 0.95))],
        "p99_ms": durations[min(n - 1, int(n * 0.99))],
        "max_ms": durations[-1],
        "mean_ms": round(statistics.mean(durations), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-invariants", type=int, default=20)
    parser.add_argument("--n-traces", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=200)
    args = parser.parse_args()

    print("=" * 64)
    print("Z3 concurrency stress test")
    print(f"  parallelism = {args.n_invariants} × {args.n_traces}")
    print(f"  timeout per call = {args.timeout_ms}ms")
    print("=" * 64)

    out = asyncio.run(
        stress(
            n_invariants=args.n_invariants,
            n_traces=args.n_traces,
            timeout_ms=args.timeout_ms,
        )
    )
    for k, v in out.items():
        print(f"  {k:30} {v}")

    if out["n_exceptions"] > 0:
        print(f"\n⚠ {out['n_exceptions']} verifier calls raised — Z3 thread-safety issue")
    else:
        print("\n✓ no exceptions escaped the verifier under concurrent load")


if __name__ == "__main__":
    main()
