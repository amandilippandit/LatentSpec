"""Tests for the active-learning loop (synthetic generator + HITL queue)."""

from __future__ import annotations

import pytest

from latentspec.active_learning import (
    AgentSpec,
    ReviewDecision,
    ReviewQueue,
)
from latentspec.active_learning.synthesis import deterministic_synthetic_traces
from latentspec.schemas.trace import NormalizedTrace


def _spec() -> AgentSpec:
    return AgentSpec(
        name="booking-agent",
        purpose="book flights and trains",
        tools=[
            "search_flights",
            "check_inventory",
            "book_flight",
            "send_confirmation",
        ],
        sample_user_inputs=[
            "book a flight to Tokyo",
            "any cheap options to Paris?",
        ],
        user_segments=["US", "EU"],
        forbidden_actions=["delete_user"],
        typical_session_length=(3, 6),
    )


def test_deterministic_synthesizer_produces_valid_traces() -> None:
    traces = deterministic_synthetic_traces(_spec(), n_traces=10)
    assert len(traces) == 10
    for t in traces:
        assert isinstance(t, NormalizedTrace)
        assert t.steps[0].type.value == "user_input"
        # Only invokes tools from the spec — no hallucinated tool names
        for s in t.steps[1:-1]:
            if hasattr(s, "tool"):
                assert getattr(s, "tool") in _spec().tools


def test_review_queue_lifecycle() -> None:
    spec = _spec()
    traces = deterministic_synthetic_traces(spec, n_traces=5)
    queue = ReviewQueue()
    items = queue.submit_many(traces, spec_name=spec.name)
    assert len(items) == 5
    assert len(queue.pending()) == 5
    assert queue.approved_traces() == []

    queue.decide(items[0].id, decision=ReviewDecision.APPROVED, decided_by="alice")
    queue.decide(items[1].id, decision=ReviewDecision.REJECTED, decided_by="alice")
    queue.decide(
        items[2].id,
        decision=ReviewDecision.EDITED,
        decided_by="alice",
        replacement_trace=traces[2],
    )

    pending = queue.pending()
    assert len(pending) == 2  # items[3], items[4]
    approved = queue.approved_traces()
    assert len(approved) == 2  # APPROVED + EDITED


def test_review_queue_stats() -> None:
    spec = _spec()
    traces = deterministic_synthetic_traces(spec, n_traces=4)
    queue = ReviewQueue()
    items = queue.submit_many(traces, spec_name=spec.name)
    queue.decide(items[0].id, decision=ReviewDecision.APPROVED)
    stats = queue.stats()
    assert stats["total"] == 4
    assert stats["approved"] == 1
    assert stats["pending"] == 3


def test_review_queue_unknown_id_raises() -> None:
    queue = ReviewQueue()
    with pytest.raises(KeyError):
        queue.decide("does-not-exist", decision=ReviewDecision.APPROVED)
