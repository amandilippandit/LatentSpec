"""Tests for vertical invariant packs (ecommerce / banking / healthcare)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from latentspec.models.invariant import InvariantType, Severity
from latentspec.packs import auto_fit_score, get_pack, install_pack, list_packs
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def test_list_packs_includes_all_three_verticals() -> None:
    packs = set(list_packs())
    assert {"ecommerce", "banking", "healthcare"} <= packs


def test_load_ecommerce_pack_validates() -> None:
    pack = get_pack("ecommerce")
    assert pack is not None
    assert pack.pack_id == "ecommerce"
    # Every invariant in the pack passed the per-type schema
    assert all(inv.params for inv in pack.invariants if inv.type != InvariantType.OUTPUT_FORMAT)
    types = {inv.type for inv in pack.invariants}
    # Pack covers diverse rule types
    assert {InvariantType.ORDERING, InvariantType.NEGATIVE} <= types


def test_install_pack_returns_pending_invariants() -> None:
    invariants = install_pack(agent_id=uuid.uuid4(), pack_id="banking")
    assert invariants
    assert all(inv.discovered_by == "pack" for inv in invariants)
    # All start in pending — auto-fit decides whether to promote/demote
    assert all(inv.status.value == "pending" for inv in invariants)
    # Each carries pack provenance
    assert all("pack_id" in inv.params for inv in invariants)
    assert all(inv.params["pack_id"] == "banking" for inv in invariants)


def _booking_trace(seg: str = "EU") -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="booking-good",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[
            UserInputStep(content="book a flight"),
            ToolCallStep(tool="check_inventory", args={}, latency_ms=120),
            ToolCallStep(tool="create_order", args={}, latency_ms=180),
            ToolCallStep(tool="payments_v2_psd2", args={}, latency_ms=260),
            AgentResponseStep(content="confirmed"),
        ],
        metadata=TraceMetadata(user_segment=seg),
    )


def _booking_trace_violating_ordering() -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="booking-bad",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[
            UserInputStep(content="book a flight"),
            ToolCallStep(tool="create_order", args={}, latency_ms=180),
            ToolCallStep(tool="check_inventory", args={}, latency_ms=120),
            AgentResponseStep(content="oops"),
        ],
        metadata=TraceMetadata(user_segment="EU"),
    )


def test_auto_fit_promotes_high_fit_pack_rule() -> None:
    invariants = install_pack(agent_id=uuid.uuid4(), pack_id="ecommerce")
    ordering_rule = next(
        inv for inv in invariants
        if inv.type == InvariantType.ORDERING
        and inv.params.get("tool_a") == "check_inventory"
    )
    traces = [_booking_trace() for _ in range(20)]
    score = auto_fit_score(invariant=ordering_rule, traces=traces)
    assert score.fit > 0.7
    assert score.applicability > 0.9
    assert score.pass_rate == 1.0


def test_auto_fit_drops_low_fit_pack_rule() -> None:
    invariants = install_pack(agent_id=uuid.uuid4(), pack_id="ecommerce")
    ordering_rule = next(
        inv for inv in invariants
        if inv.type == InvariantType.ORDERING
        and inv.params.get("tool_a") == "check_inventory"
    )
    traces = [_booking_trace_violating_ordering() for _ in range(20)]
    score = auto_fit_score(invariant=ordering_rule, traces=traces)
    # All traces violate ⇒ pass_rate is 0
    assert score.pass_rate == 0.0


def test_install_unknown_pack_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        install_pack(agent_id=uuid.uuid4(), pack_id="not-a-real-pack")
