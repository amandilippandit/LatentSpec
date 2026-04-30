"""Tests for session-level invariant mining."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.sessions.orchestrator import mine_session_invariants
from latentspec.sessions.schema import Session


def _turn(
    idx: int, *, content: str, tool: str
) -> NormalizedTrace:
    base = datetime.now(UTC)
    return NormalizedTrace(
        trace_id=f"turn-{idx}",
        agent_id="a",
        timestamp=base,
        ended_at=base + timedelta(seconds=1),
        steps=[
            UserInputStep(content=content),
            ToolCallStep(tool=tool, args={}, latency_ms=100),
            AgentResponseStep(content="ok"),
        ],
        metadata=TraceMetadata(),
    )


def _session(idx: int, turns: list[NormalizedTrace]) -> Session:
    return Session(
        session_id=f"s-{idx}",
        agent_id="a",
        started_at=datetime.now(UTC),
        turns=turns,
    )


def test_session_transitions_recover_followup_pattern() -> None:
    sessions: list[Session] = []
    for i in range(40):
        sessions.append(
            _session(
                i,
                [
                    _turn(i * 2, content="refund please", tool="escalate_human"),
                    _turn(i * 2 + 1, content="thanks", tool="customer_followup"),
                ],
            )
        )
    result = mine_session_invariants(sessions, min_support=0.5)
    transitions = [
        c
        for c in result.transitions
        if c.type == InvariantType.COMPOSITION
        and c.extra.get("transition_probability", 0) >= 0.85
    ]
    assert transitions, "expected at least one strong transition rule"


def test_session_aggregates_recover_per_session_caps() -> None:
    sessions: list[Session] = []
    for i in range(30):
        sessions.append(
            _session(
                i,
                [
                    _turn(i * 2, content="hi", tool="search_flights"),
                    _turn(i * 2 + 1, content="confirm", tool="book_flight"),
                ],
            )
        )
    result = mine_session_invariants(sessions, min_support=0.4)
    aggregates = [c for c in result.aggregates if c.extra.get("session_level")]
    # We expect at least a "≤ N book_flight per session" rule
    book_rule = next(
        (
            c
            for c in aggregates
            if "book_flight" in c.extra.get("feature", "")
        ),
        None,
    )
    assert book_rule is not None


def test_session_terminations_recover_closing_shape() -> None:
    sessions: list[Session] = []
    for i in range(30):
        sessions.append(
            _session(
                i,
                [
                    _turn(i * 2, content="hi", tool="search_flights"),
                    _turn(i * 2 + 1, content="bye", tool="session_close"),
                ],
            )
        )
    result = mine_session_invariants(sessions, min_support=0.5)
    assert any(c.type == InvariantType.STATE for c in result.terminations)


def test_empty_sessions_return_empty_result() -> None:
    result = mine_session_invariants([])
    assert result.n_sessions == 0
    assert not result.transitions and not result.aggregates and not result.terminations
