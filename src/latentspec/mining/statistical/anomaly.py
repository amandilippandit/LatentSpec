"""Isolation-forest anomaly baselines (§3.2 Track A).

Builds a per-agent behavioral feature vector for every trace and runs
sklearn's IsolationForest to surface outliers. Each surfaced cluster of
similar anomalies becomes a candidate STATISTICAL invariant of the form
"behavioral feature X stays within the observed normal range".

The features are deliberately structural — counts, latencies, distinct-
tool counts — so the invariants the miner produces are directly checkable
by the runtime detector without needing the trained model at check time.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.ensemble import IsolationForest

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    StepType,
    ToolCallStep,
    UserInputStep,
)


_FEATURE_NAMES = [
    "step_count",
    "tool_call_count",
    "distinct_tool_count",
    "user_input_chars",
    "agent_response_chars",
    "total_latency_ms",
    "max_step_latency_ms",
    "error_step_count",
]


def _features(trace: NormalizedTrace) -> list[float]:
    tool_calls = [s for s in trace.steps if isinstance(s, ToolCallStep)]
    distinct = {s.tool for s in tool_calls}
    user_input_chars = sum(
        len(s.content) for s in trace.steps if isinstance(s, UserInputStep)
    )
    agent_response_chars = sum(
        len(s.content) for s in trace.steps if isinstance(s, AgentResponseStep)
    )
    latencies = [s.latency_ms or 0 for s in tool_calls]
    error_count = sum(
        1 for s in tool_calls if (s.result_status or "success") != "success"
    )
    return [
        float(len(trace.steps)),
        float(len(tool_calls)),
        float(len(distinct)),
        float(user_input_chars),
        float(agent_response_chars),
        float(sum(latencies)),
        float(max(latencies) if latencies else 0),
        float(error_count),
    ]


def mine_anomaly_baselines(
    traces: list[NormalizedTrace],
    *,
    min_traces: int = 30,
    contamination: float = 0.05,
    random_state: int = 17,
) -> list[InvariantCandidate]:
    """Fit an isolation forest over behavioral feature vectors and emit
    one statistical invariant per feature whose normal range is tight."""
    if len(traces) < min_traces:
        return []

    X = np.asarray([_features(t) for t in traces], dtype=float)
    if X.shape[0] == 0:
        return []

    model = IsolationForest(
        n_estimators=128,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    # Per-feature normal envelope = [p1, p99] over the inlier subset.
    inlier_mask = model.predict(X) == 1
    inlier_X = X[inlier_mask] if inlier_mask.any() else X

    candidates: list[InvariantCandidate] = []
    for col, feature in enumerate(_FEATURE_NAMES):
        values = inlier_X[:, col]
        if values.size < min_traces:
            continue
        p1, p99 = float(np.percentile(values, 1)), float(np.percentile(values, 99))
        median = float(np.median(values))

        # Skip features with no useful spread
        if p99 - p1 < 1e-6:
            continue
        # Skip features dominated by zero (e.g. error_count when nothing fails)
        if median == 0 and p99 < 5:
            continue

        within = float(((X[:, col] >= p1) & (X[:, col] <= p99)).mean())
        if within < 0.95:
            continue

        candidates.append(
            InvariantCandidate(
                type=InvariantType.STATISTICAL,
                description=(
                    f"Feature `{feature}` stays in the observed normal "
                    f"range [{p1:.0f}, {p99:.0f}] (p1-p99 over inliers; "
                    f"median {median:.0f})"
                ),
                formal_rule=(
                    f"forall trace: {p1:.0f} <= feature(trace, '{feature}') <= {p99:.0f} (p99)"
                ),
                evidence_trace_ids=[t.trace_id for t in traces[:50]],
                support=round(within, 4),
                consistency=round(within, 4),
                severity=Severity.LOW,
                discovered_by="statistical",
                extra={
                    "metric": "feature_envelope",
                    "feature": feature,
                    "p1": round(p1, 3),
                    "p99": round(p99, 3),
                    "median": round(median, 3),
                    "within_rate": round(within, 4),
                    "samples": int(values.size),
                },
            )
        )

    return candidates
