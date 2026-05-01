"""Tests for trace shape vectorization, workflow clustering, and routing."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from latentspec.mining.clustering import (
    TraceShapeVectorizer,
    cluster_workflows,
    split_by_cluster,
)
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def _trace(idx: int, family: str) -> NormalizedTrace:
    """Build a synthetic trace from one of three workflow families."""
    if family == "booking":
        steps = [
            UserInputStep(content="book a flight to Tokyo please"),
            ToolCallStep(tool="search_flights", args={}, latency_ms=200),
            ToolCallStep(tool="check_inventory", args={}, latency_ms=120),
            ToolCallStep(tool="create_order", args={}, latency_ms=180),
            ToolCallStep(tool="book_flight", args={}, latency_ms=300),
            AgentResponseStep(content="confirmed"),
        ]
    elif family == "refund":
        steps = [
            UserInputStep(content="refund my booking please cancel"),
            ToolCallStep(tool="escalate_human", args={}, latency_ms=120),
            AgentResponseStep(content="escalated"),
        ]
    else:  # pricing
        steps = [
            UserInputStep(content="what is the price of flights to Paris"),
            ToolCallStep(tool="lookup_pricing", args={}, latency_ms=110),
            AgentResponseStep(content="$300-$650"),
        ]
    return NormalizedTrace(
        trace_id=f"t-{idx}",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=steps,
        metadata=TraceMetadata(user_segment="US"),
    )


def _corpus(n_per_family: int = 30) -> list[NormalizedTrace]:
    out: list[NormalizedTrace] = []
    for i in range(n_per_family):
        out.append(_trace(i, "booking"))
    for i in range(n_per_family):
        out.append(_trace(i + 1000, "refund"))
    for i in range(n_per_family):
        out.append(_trace(i + 2000, "pricing"))
    return out


def test_vectorizer_produces_consistent_dim() -> None:
    traces = _corpus(20)
    vec = TraceShapeVectorizer()
    matrix = vec.fit_transform(traces)
    assert matrix.shape[0] == len(traces)
    # Same vocabulary applies to new traces
    new = vec.transform([_trace(99, "booking")])
    assert new.shape[1] == matrix.shape[1]
    # L2-normalized rows
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)


def test_cluster_recovers_three_workflow_families() -> None:
    traces = _corpus(30)
    result = cluster_workflows(traces, k_min=2, k_max=8)
    # Three distinct workflow families ⇒ silhouette favors k>=3
    assert result.k >= 3
    # Cluster sizes should be roughly balanced (each family has 30 traces)
    assert all(20 <= size <= 40 for size in result.cluster_sizes.values())


def test_route_new_trace_to_existing_cluster() -> None:
    traces = _corpus(25)
    result = cluster_workflows(traces, k_min=2, k_max=6)
    # Booking-shaped trace routes to a cluster dominated by other booking traces
    booking_label = result.predict([_trace(999, "booking")])[0]
    assigned_to_same = sum(
        1
        for trace, lbl in zip(traces, result.labels.tolist(), strict=True)
        if int(lbl) == int(booking_label)
        and trace.steps[1].tool == "search_flights"  # type: ignore[attr-defined]
    )
    cluster_size = result.cluster_sizes[int(booking_label)]
    # ≥ 70% of the routed cluster's members must actually be booking traces
    assert assigned_to_same / cluster_size >= 0.7


def test_split_by_cluster_groups_traces_correctly() -> None:
    traces = _corpus(15)
    result = cluster_workflows(traces, k_min=2, k_max=6)
    groups = split_by_cluster(traces, result.labels)
    assert sum(len(g) for g in groups.values()) == len(traces)
    assert set(groups) == set(result.cluster_sizes)


def test_falls_back_to_one_cluster_for_tiny_corpus() -> None:
    traces = _corpus(2)  # 6 traces total
    result = cluster_workflows(traces, k_min=2, k_max=6, min_traces_per_cluster=8)
    assert result.k == 1
