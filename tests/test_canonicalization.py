"""Tests for tool-name canonicalisation."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.canonicalization.applier import apply_canonicalisation
from latentspec.canonicalization.canonicalizer import (
    ToolCanonicalizer,
    canonical_form,
    collect_tool_names,
)
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def test_canonical_form_strips_versioning_and_punctuation() -> None:
    assert canonical_form("Payments_v2") == "payments"
    assert canonical_form("payments-v1") == "payments"
    assert canonical_form("payments.v2.1") == "payments"
    assert canonical_form("payments_v2.execute") == "payments_v2_execute"  # version is suffix only
    assert canonical_form("CHECK_INVENTORY") == "check_inventory"
    assert canonical_form("search-flights") == "search_flights"


def test_canonicalizer_clusters_versioned_aliases() -> None:
    names = ["payments_v1", "Payments_v2", "payments-v3", "search_flights"]
    cano = ToolCanonicalizer().fit(names)
    # All payments variants collapse to one canonical
    payment_canonicals = {cano.canonical_for[n] for n in names if "payment" in n.lower()}
    assert len(payment_canonicals) == 1
    # search_flights stays alone
    assert cano.canonical_for["search_flights"] == "search_flights"


def test_canonicalizer_handles_token_reordering() -> None:
    names = ["book_flight", "flight.book", "search_flights"]
    cano = ToolCanonicalizer().fit(names)
    # book_flight and flight.book have token Jaccard = 1.0
    assert cano.canonical_for["book_flight"] == cano.canonical_for["flight.book"]
    # search_flights doesn't share enough — stays separate
    assert cano.canonical_for["search_flights"] != cano.canonical_for["book_flight"]


def test_canonicalizer_handles_morphology_via_edit_distance() -> None:
    names = ["delete_user", "delete_users"]  # singular vs plural
    cano = ToolCanonicalizer().fit(names)
    assert cano.canonical_for["delete_user"] == cano.canonical_for["delete_users"]


def test_canonicalizer_picks_shortest_canonical() -> None:
    names = ["payments_v1_legacy", "payments_v2", "payments_v3_new"]
    cano = ToolCanonicalizer().fit(names)
    canonical = cano.canonical_for["payments_v2"]
    # Among the cluster, the shortest normalised form wins
    assert canonical == "payments_v2"


def test_canonicalizer_emits_decisions_with_method() -> None:
    cano = ToolCanonicalizer().fit(["Payments_v1", "payments_v1"])
    methods = {d.method for d in cano.decisions}
    assert {"self", "exact"} <= methods
    for d in cano.decisions:
        assert 0.0 < d.confidence <= 1.0


def test_apply_canonicalisation_rewrites_trace_tool_names() -> None:
    trace = NormalizedTrace(
        trace_id="t",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[
            UserInputStep(content="hi"),
            ToolCallStep(tool="Payments_v1", args={}),
            ToolCallStep(tool="payments-v2", args={}),
        ],
        metadata=TraceMetadata(),
    )
    cano = ToolCanonicalizer().fit(["Payments_v1", "payments-v2"])
    rewritten = apply_canonicalisation(trace, cano)
    tool_names = [s.tool for s in rewritten.steps if isinstance(s, ToolCallStep)]
    # Both rewritten to the same canonical name
    assert tool_names[0] == tool_names[1]


def test_collect_tool_names_returns_sorted_unique() -> None:
    t1 = NormalizedTrace(
        trace_id="t1",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[ToolCallStep(tool="b", args={}), ToolCallStep(tool="a", args={})],
        metadata=TraceMetadata(),
    )
    t2 = NormalizedTrace(
        trace_id="t2",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[ToolCallStep(tool="a", args={})],
        metadata=TraceMetadata(),
    )
    assert collect_tool_names([t1, t2]) == ["a", "b"]


def test_canonicalizer_handles_empty_input() -> None:
    cano = ToolCanonicalizer().fit([])
    assert cano.decisions == []
    assert cano.clusters == {}
    assert cano.canonicalise("anything") == "anything"
