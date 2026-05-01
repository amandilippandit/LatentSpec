"""Distributional analysis for statistical invariants (§3.3).

Surfaces:
  - per-tool latency p99 thresholds
  - per-tool success-rate floors
  - per-step-type response-length distributions

Each becomes a STATISTICAL invariant of the form
`<metric> stays under <threshold> for <pXX>% of requests`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), p))


def mine_distributions(
    traces: list[NormalizedTrace],
    *,
    min_samples: int = 20,
    p_target: float = 99.0,
) -> list[InvariantCandidate]:
    """Discover statistical invariants on tool latency and success rate.

    For each tool with at least `min_samples` observations:
      - emit a latency-p99 invariant (with the observed p99 as the threshold).
      - emit a success-rate invariant if success rate ≥ 0.95.
    """
    if not traces:
        return []

    latencies: dict[str, list[float]] = defaultdict(list)
    statuses: dict[str, list[bool]] = defaultdict(list)
    evidence: dict[str, list[str]] = defaultdict(list)

    for trace in traces:
        for step in trace.steps:
            if not isinstance(step, ToolCallStep):
                continue
            if step.latency_ms is not None:
                latencies[step.tool].append(float(step.latency_ms))
            statuses[step.tool].append((step.result_status or "success") == "success")
            evidence[step.tool].append(trace.trace_id)

    candidates: list[InvariantCandidate] = []

    for tool, vals in latencies.items():
        if len(vals) < min_samples:
            continue
        threshold = _percentile(vals, p_target)
        # round threshold to a clean number for human description
        rounded = int(np.ceil(threshold / 50.0) * 50)
        below = sum(1 for v in vals if v <= rounded) / len(vals)
        if below < 0.95:
            continue
        candidates.append(
            InvariantCandidate(
                type=InvariantType.STATISTICAL,
                description=(
                    f"Tool `{tool}` latency stays under {rounded}ms "
                    f"for {p_target:g}% of requests"
                ),
                formal_rule=(
                    f"forall trace, step in trace.tool_calls where step.tool == '{tool}': "
                    f"latency_ms(step) <= {rounded} (p{p_target:g})"
                ),
                evidence_trace_ids=list(set(evidence[tool]))[:50],
                support=round(below, 4),
                consistency=round(below, 4),
                severity=Severity.MEDIUM,
                discovered_by="statistical",
                extra={
                    "metric": "latency_ms",
                    "tool": tool,
                    "threshold": rounded,
                    "percentile": p_target,
                    "samples": len(vals),
                },
            )
        )

    for tool, ok_list in statuses.items():
        if len(ok_list) < min_samples:
            continue
        rate = sum(ok_list) / len(ok_list)
        if rate < 0.95:
            continue
        candidates.append(
            InvariantCandidate(
                type=InvariantType.STATISTICAL,
                description=(
                    f"Tool `{tool}` succeeds for at least "
                    f"{int(rate * 100)}% of invocations"
                ),
                formal_rule=(
                    f"forall trace, step in trace.tool_calls where step.tool == '{tool}': "
                    f"success_rate(step) >= {rate:.3f}"
                ),
                evidence_trace_ids=list(set(evidence[tool]))[:50],
                support=round(rate, 4),
                consistency=round(rate, 4),
                severity=Severity.MEDIUM,
                discovered_by="statistical",
                extra={
                    "metric": "success_rate",
                    "tool": tool,
                    "rate": round(rate, 4),
                    "samples": len(ok_list),
                },
            )
        )

    return candidates
