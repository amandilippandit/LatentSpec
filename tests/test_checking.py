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
