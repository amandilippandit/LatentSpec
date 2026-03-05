"""Performance benchmark: streaming detector at scale.

Three measurements, real numbers:
  1. Per-trace check latency (rule-based, no LLM): p50, p95, p99 across N
     traces with M invariants per agent.
  2. Sustained ingest throughput: traces/sec we can push through
     `dispatch()` for one agent before fail-open kicks in.
  3. Per-checker breakdown: which check type costs how much.

Numbers come out of THIS machine on the synthetic agent — they are
reference values, not production guarantees. Real production scale
needs a load test against a deployed cluster, which this script
intentionally does not pretend to be.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter, defaultdict

from latentspec.checking.base import InvariantSpec
from latentspec.checking.dispatch import dispatch
from latentspec.checking.runner import check_trace
from latentspec.demo import generate_traces
from latentspec.mining.orchestrator import mine_invariants
from latentspec.models.invariant import InvariantType, Severity


def _build_invariants_from_demo(n_invariants: int) -> list[InvariantSpec]:
    """Use the synthetic booking-agent's discovered invariants as the
    benchmark workload — these are the kinds of rules production produces."""
    import uuid as _uuid

    traces = generate_traces(120, seed=11)
    result = asyncio.run(
        mine_invariants(
            agent_id=_uuid.uuid4(), traces=traces, session=None, persist=False
        )
    )
    specs: list[InvariantSpec] = []
    for inv in result.invariants:
        if inv.type == InvariantType.OUTPUT_FORMAT:
            continue
        if not inv.params:
            continue
        specs.append(
            InvariantSpec(
                id=inv.invariant_id,
                type=inv.type,
                description=inv.description,
                formal_rule=inv.formal_rule,
                severity=Severity(inv.severity),
                params=inv.params,
            )
        )
        if len(specs) >= n_invariants:
            break
    return specs


def bench_per_trace_latency(
    *, n_traces: int, n_invariants: int, seed: int
) -> dict[str, float]:
    traces = generate_traces(n_traces, seed=seed)
    invariants = _build_invariants_from_demo(n_invariants)
    if not invariants:
        return {"error": "no invariants"}

    latencies_us: list[float] = []
    for trace in traces:
        t0 = time.perf_counter_ns()
        check_trace(invariants, trace)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    latencies_us.sort()
    n = len(latencies_us)
    return {
        "n_traces": n,
        "n_invariants_per_trace": len(invariants),
        "p50_us": latencies_us[int(n * 0.50)],
        "p95_us": latencies_us[int(n * 0.95)],
        "p99_us": latencies_us[min(n - 1, int(n * 0.99))],
        "max_us": latencies_us[-1],
        "mean_us": statistics.mean(latencies_us),
        "stdev_us": statistics.stdev(latencies_us) if n > 1 else 0.0,
    }


def bench_sustained_throughput(
    *, duration_seconds: float, n_invariants: int
) -> dict[str, float]:
    traces = generate_traces(2000, seed=23)
    invariants = _build_invariants_from_demo(n_invariants)
