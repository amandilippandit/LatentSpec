"""Tests for the Track A statistical sub-miners."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.mining.statistical import (
    mine_associations,
    mine_distributions,
    mine_negatives,
    mine_sequences,
)
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def _trace(idx: int, steps) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=f"t-{idx}",
        agent_id="agent",
        timestamp=datetime.now(UTC),
        steps=list(steps),
        metadata=TraceMetadata(),
    )


def test_sequence_mining_recovers_implanted_ordering() -> None:
    traces = []
    for i in range(40):
        traces.append(
            _trace(
                i,
                [
                    UserInputStep(content="please book"),
                    ToolCallStep(tool="check_inventory", args={}, latency_ms=120,
                                 result_status="success"),
                    ToolCallStep(tool="create_order", args={}, latency_ms=240,
                                 result_status="success"),
                    AgentResponseStep(content="done"),
                ],
            )
        )
    candidates = mine_sequences(traces, min_support=0.6, min_directionality=0.95)
    assert any(
        c.type == InvariantType.ORDERING
        and "check_inventory" in c.description
        and "create_order" in c.description
        for c in candidates
    )


def test_distribution_mining_emits_latency_threshold() -> None:
    traces = []
    for i in range(40):
        traces.append(
            _trace(
                i,
                [
                    UserInputStep(content="hi"),
                    ToolCallStep(
                        tool="search_flights",
                        args={},
                        latency_ms=200 + (i % 10) * 5,
                        result_status="success",
                    ),
                ],
            )
        )
    candidates = mine_distributions(traces, min_samples=20)
    latency_inv = [
        c for c in candidates if c.extra.get("metric") == "latency_ms"
    ]
    assert latency_inv, "expected at least one latency invariant"
    assert latency_inv[0].extra["tool"] == "search_flights"


def test_associations_recover_keyword_to_tool() -> None:
    """MI-based association miner recovers keyword -> tool rules.

    Strong correlation (every refund mention -> escalate_human, never
    elsewhere) yields high mutual information, large chi-square, and a
    high lift, so the rule clears all three filters.
    """
    traces = []
