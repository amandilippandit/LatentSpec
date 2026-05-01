"""Tests for the isolation-forest anomaly baseline miner."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.demo import generate_traces
from latentspec.mining.statistical.anomaly import mine_anomaly_baselines
from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def test_anomaly_emits_envelopes_for_synthetic_agent() -> None:
    traces = generate_traces(120, seed=3)
    cands = mine_anomaly_baselines(traces, min_traces=30)
    # Should surface at least one feature envelope per behavioral dimension
    assert any(c.type == InvariantType.STATISTICAL for c in cands)
    assert all(c.discovered_by == "statistical" for c in cands)
    for c in cands:
        meta = c.extra
        assert meta["metric"] == "feature_envelope"
        assert "p1" in meta and "p99" in meta and "median" in meta


def test_anomaly_returns_empty_below_min_traces() -> None:
    cands = mine_anomaly_baselines(generate_traces(5, seed=3), min_traces=30)
    assert cands == []
