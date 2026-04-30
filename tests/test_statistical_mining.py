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
    for i in range(60):
        if i % 2 == 0:
            content = "I need a refund please"
            tool = "escalate_human"
        else:
            content = "show me cheap flights to Tokyo"
            tool = "search_flights"
        traces.append(
            _trace(
                i,
                [
                    UserInputStep(content=content),
                    ToolCallStep(tool=tool, args={}, latency_ms=150,
                                 result_status="success"),
                ],
            )
        )
    candidates = mine_associations(traces)
    refund_rule = next(
        (c for c in candidates if c.extra.get("keyword") == "refund"),
        None,
    )
    assert refund_rule is not None, "expected a 'refund' MI association"
    assert refund_rule.extra["tool"] == "escalate_human"
    # MI in bits should be substantial for this perfect-correlation case
    assert refund_rule.extra["mutual_information_bits"] > 0.5
    # chi-square far above the p<0.01 threshold
    assert refund_rule.extra["chi_square"] > 20.0


def test_negative_mining_emits_closed_world_repertoire() -> None:
    """The closed-world miner emits a single negative invariant whose
    `allowed_repertoire` is exactly the set of tools observed often
    enough to clear the support threshold."""
    from latentspec.mining.statistical.negative import CustomerPolicy

    traces = []
    for i in range(40):
        traces.append(
            _trace(
                i,
                [
                    UserInputStep(content="hi"),
                    ToolCallStep(
                        tool="search_flights", args={}, latency_ms=120,
                        result_status="success",
                    ),
                    ToolCallStep(
                        tool="book_flight", args={}, latency_ms=200,
                        result_status="success",
                    ),
                ],
            )
        )
    candidates = mine_negatives(traces, min_traces=20)
    closed_world = [
        c for c in candidates
        if c.extra.get("category") == "closed_world_repertoire"
    ]
    assert closed_world, "expected one closed-world repertoire invariant"
    repertoire = set(closed_world[0].extra["allowed_repertoire"])
    assert {"search_flights", "book_flight"} <= repertoire
    assert closed_world[0].severity == Severity.CRITICAL


def test_negative_mining_with_customer_denylist() -> None:
    """Customer-supplied denylists emit alongside the closed-world rule."""
    from latentspec.mining.statistical.negative import CustomerPolicy
    from latentspec.models.invariant import Severity as Sev

    traces = []
    for i in range(40):
        traces.append(
            _trace(
                i,
                [
                    UserInputStep(content="hi"),
                    ToolCallStep(tool="search_flights", args={}, latency_ms=120),
                ],
            )
        )
    policy = CustomerPolicy(
        denylist=[("delete_user", Sev.CRITICAL, "delete")],
    )
    candidates = mine_negatives(traces, min_traces=20, policy=policy)
    delete_rule = next(
        (c for c in candidates if c.extra.get("category") == "delete"), None
    )
    assert delete_rule is not None
    assert "delete_user" in delete_rule.extra["forbidden_patterns"]
