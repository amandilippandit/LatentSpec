"""Tests for the Z3 SMT compiler + verifier (§3.2 / §10.1)."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.smt.certificates import generate_certificate
from latentspec.smt.compiler import Z3CompilerError, compile_invariant
from latentspec.smt.verifier import verify_trace


def _trace(steps, *, segment: str | None = None) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="t",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=list(steps),
        metadata=TraceMetadata(user_segment=segment),
    )


def test_z3_ordering_holds_for_correct_trace() -> None:
    comp = compile_invariant(
        InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"}
    )
    good = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="auth", args={}),
            ToolCallStep(tool="db_write", args={}),
        ]
    )
    result = verify_trace(comp, good, timeout_ms=2000)
    assert result.holds, result


def test_z3_ordering_fails_with_counter_example() -> None:
    comp = compile_invariant(
        InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"}
    )
    bad = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="db_write", args={}),
            ToolCallStep(tool="auth", args={}),
        ]
    )
    result = verify_trace(comp, bad, timeout_ms=2000)
    assert not result.holds
    assert result.counter_example is not None
    assert result.counter_example.get("n") == 3


def test_z3_negative_holds_when_dangerous_absent() -> None:
    comp = compile_invariant(
        InvariantType.NEGATIVE, {"forbidden_patterns": ["delete", "drop"]}
    )
    benign = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="search_flights", args={}),
            ToolCallStep(tool="book_flight", args={}),
        ]
    )
    assert verify_trace(comp, benign, timeout_ms=2000).holds


def test_z3_negative_fails_when_forbidden_present() -> None:
    comp = compile_invariant(
        InvariantType.NEGATIVE, {"forbidden_patterns": ["delete"]}
    )
    bad = _trace(
        [
