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
    if not invariants:
        return {"error": "no invariants"}

    n_processed = 0
    started = time.perf_counter()
    deadline = started + duration_seconds
    idx = 0
    while time.perf_counter() < deadline:
        check_trace(invariants, traces[idx % len(traces)])
        n_processed += 1
        idx += 1
    elapsed = time.perf_counter() - started

    return {
        "duration_seconds": elapsed,
        "n_invariants_per_trace": len(invariants),
        "traces_processed": n_processed,
        "traces_per_second": n_processed / elapsed,
    }


def bench_per_type_breakdown(*, n_traces: int) -> dict[str, dict[str, float]]:
    traces = generate_traces(n_traces, seed=37)
    invariants = _build_invariants_from_demo(64)
    by_type: dict[str, list[float]] = defaultdict(list)

    for trace in traces:
        for spec in invariants:
            t0 = time.perf_counter_ns()
            try:
                dispatch(spec, trace)
            except Exception:
                pass
            t1 = time.perf_counter_ns()
            by_type[spec.type.value].append((t1 - t0) / 1000.0)

    out: dict[str, dict[str, float]] = {}
    for type_name, samples in by_type.items():
        if not samples:
            continue
        samples.sort()
        n = len(samples)
        out[type_name] = {
            "n_calls": n,
            "p50_us": samples[int(n * 0.5)],
            "p99_us": samples[min(n - 1, int(n * 0.99))],
            "mean_us": statistics.mean(samples),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traces", type=int, default=2000)
    parser.add_argument("--n-invariants", type=int, default=24)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    print("=" * 64)
    print("LatentSpec streaming benchmark")
    print("=" * 64)

    print("\n[1] Per-trace latency")
    lat = bench_per_trace_latency(
        n_traces=args.n_traces,
        n_invariants=args.n_invariants,
        seed=args.seed,
    )
    for k, v in lat.items():
        if isinstance(v, float):
            print(f"  {k:30} {v:>10.2f}")
        else:
            print(f"  {k:30} {v}")

    print(f"\n[2] Sustained throughput ({args.duration}s window)")
    thr = bench_sustained_throughput(
        duration_seconds=args.duration,
        n_invariants=args.n_invariants,
    )
    for k, v in thr.items():
        if isinstance(v, float):
            print(f"  {k:30} {v:>10.2f}")
        else:
            print(f"  {k:30} {v}")

    print("\n[3] Per-type latency breakdown")
    breakdown = bench_per_type_breakdown(n_traces=200)
    print(f"  {'type':<16} {'n':>8} {'p50_us':>10} {'p99_us':>10} {'mean_us':>10}")
    for t, vals in sorted(breakdown.items()):
        print(
            f"  {t:<16} {int(vals['n_calls']):>8} "
            f"{vals['p50_us']:>10.2f} {vals['p99_us']:>10.2f} {vals['mean_us']:>10.2f}"
        )


if __name__ == "__main__":
    main()
