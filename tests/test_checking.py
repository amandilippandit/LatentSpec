"""Per-type checker tests (§4 / §3.3)."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.checking import dispatch
from latentspec.checking.base import CheckOutcome, InvariantSpec
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    UserInputStep,
)


def _trace(steps) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="t",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=list(steps),
    )


def _spec(type_, params, *, severity=Severity.HIGH) -> InvariantSpec:
    return InvariantSpec(
        id="inv-test",
        type=type_,
        description="test rule",
        formal_rule="placeholder",
        severity=severity,
        params=params,
    )


def test_ordering_pass() -> None:
    inv = _spec(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
    trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="auth", args={}),
            ToolCallStep(tool="db_write", args={}),
        ]
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.PASS


def test_ordering_fail_when_b_before_a() -> None:
    inv = _spec(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
    trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="db_write", args={}),
            ToolCallStep(tool="auth", args={}),
        ]
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.FAIL


def test_ordering_not_applicable_when_b_absent() -> None:
    inv = _spec(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
    trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="auth", args={}),
        ]
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.NOT_APPLICABLE


def test_conditional_pass_and_fail() -> None:
    inv = _spec(
        InvariantType.CONDITIONAL,
        {"keyword": "refund", "tool": "escalate_human"},
    )
    pass_trace = _trace(
        [
            UserInputStep(content="please refund my ticket"),
            ToolCallStep(tool="escalate_human", args={}),
        ]
    )
    fail_trace = _trace(
        [
            UserInputStep(content="please refund my ticket"),
            ToolCallStep(tool="search_flights", args={}),
        ]
    )
    other_trace = _trace([UserInputStep(content="book a flight")])
    assert dispatch(inv, pass_trace).outcome == CheckOutcome.PASS
    assert dispatch(inv, fail_trace).outcome == CheckOutcome.FAIL
    assert dispatch(inv, other_trace).outcome == CheckOutcome.NOT_APPLICABLE


def test_negative_flags_forbidden_tool() -> None:
    inv = _spec(
        InvariantType.NEGATIVE,
        {"forbidden_patterns": ["delete", "drop"], "category": "delete"},
        severity=Severity.CRITICAL,
    )
    fail_trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="delete_user", args={}),
        ]
    )
    pass_trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="search_flights", args={}),
        ]
    )
    assert dispatch(inv, fail_trace).outcome == CheckOutcome.FAIL
    assert dispatch(inv, pass_trace).outcome == CheckOutcome.PASS


def test_statistical_latency_warns() -> None:
    inv = _spec(
        InvariantType.STATISTICAL,
        {"metric": "latency_ms", "tool": "search_flights", "threshold": 500.0},
    )
    fail_trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="search_flights", args={}, latency_ms=1200),
        ]
    )
    pass_trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="search_flights", args={}, latency_ms=200),
        ]
    )
    assert dispatch(inv, fail_trace).outcome == CheckOutcome.WARN
    assert dispatch(inv, pass_trace).outcome == CheckOutcome.PASS


def test_statistical_success_rate_fails() -> None:
    inv = _spec(
        InvariantType.STATISTICAL,
        {"metric": "success_rate", "tool": "book_flight", "rate": 0.99},
    )
    err_trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="book_flight", args={}, result_status="error"),
        ]
    )
    ok_trace = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="book_flight", args={}, result_status="success"),
        ]
    )
    assert dispatch(inv, err_trace).outcome == CheckOutcome.FAIL
    assert dispatch(inv, ok_trace).outcome == CheckOutcome.PASS


def test_dispatcher_records_duration_ms() -> None:
    inv = _spec(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
    trace = _trace([ToolCallStep(tool="auth", args={}), ToolCallStep(tool="db_write", args={})])
    result = dispatch(inv, trace)
    assert result.duration_ms >= 0.0
    # §4.1 budget: rule-based checks must be sub-100ms.
    assert result.duration_ms < 100.0
