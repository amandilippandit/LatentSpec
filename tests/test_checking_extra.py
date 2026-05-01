"""Tests for the three week-3.5 checkers: state, composition, tool_selection."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.checking import dispatch
from latentspec.checking.base import CheckOutcome, InvariantSpec
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def _trace(steps, *, segment: str | None = None) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="t",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=list(steps),
        metadata=TraceMetadata(user_segment=segment),
    )


def _spec(type_, params, severity=Severity.HIGH) -> InvariantSpec:
    return InvariantSpec(
        id="inv-test",
        type=type_,
        description="rule",
        formal_rule="placeholder",
        severity=severity,
        params=params,
    )


# ---- state ---------------------------------------------------------------


def test_state_checker_pass() -> None:
    inv = _spec(
        InvariantType.STATE,
        {
            "terminator_tool": "session_close",
            "forbidden_after": ["read_user_data", "write_user_data"],
        },
    )
    trace = _trace(
        [
            UserInputStep(content="get my data"),
            ToolCallStep(tool="read_user_data", args={}),
            ToolCallStep(tool="session_close", args={}),
        ]
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.PASS


def test_state_checker_fails_on_post_terminator_call() -> None:
    inv = _spec(
        InvariantType.STATE,
        {
            "terminator_tool": "session_close",
            "forbidden_after": ["read_user_data"],
        },
    )
    trace = _trace(
        [
            ToolCallStep(tool="session_close", args={}),
            ToolCallStep(tool="read_user_data", args={}),
        ]
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.FAIL


def test_state_checker_not_applicable_without_terminator() -> None:
    inv = _spec(
        InvariantType.STATE,
        {
            "terminator_tool": "session_close",
            "forbidden_after": ["read_user_data"],
        },
    )
    trace = _trace([ToolCallStep(tool="read_user_data", args={})])
    assert dispatch(inv, trace).outcome == CheckOutcome.NOT_APPLICABLE


# ---- composition ---------------------------------------------------------


def test_composition_pass_and_fail() -> None:
    inv = _spec(
        InvariantType.COMPOSITION,
        {"upstream_tool": "agent_a_emit", "downstream_tool": "agent_b_consume"},
    )
    ok = _trace(
        [
            ToolCallStep(tool="agent_a_emit", args={}),
            ToolCallStep(tool="agent_b_consume", args={}),
        ]
    )
    bad = _trace(
        [
            ToolCallStep(tool="agent_b_consume", args={}),
            ToolCallStep(tool="agent_a_emit", args={}),
        ]
    )
    assert dispatch(inv, ok).outcome == CheckOutcome.PASS
    assert dispatch(inv, bad).outcome == CheckOutcome.FAIL


# ---- tool_selection ------------------------------------------------------


def test_tool_selection_pass_for_matching_segment() -> None:
    inv = _spec(
        InvariantType.TOOL_SELECTION,
        {
            "segment": "EU",
            "expected_tool": "payments_v2",
            "forbidden_tool": "payments_v1",
        },
    )
    trace = _trace(
        [
            UserInputStep(content="book flight"),
            ToolCallStep(tool="payments_v2", args={}),
        ],
        segment="EU",
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.PASS


def test_tool_selection_fails_when_forbidden_used_in_segment() -> None:
    inv = _spec(
        InvariantType.TOOL_SELECTION,
        {
            "segment": "EU",
            "expected_tool": "payments_v2",
            "forbidden_tool": "payments_v1",
        },
    )
    trace = _trace(
        [ToolCallStep(tool="payments_v1", args={})],
        segment="EU",
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.FAIL


def test_tool_selection_not_applicable_for_other_segment() -> None:
    inv = _spec(
        InvariantType.TOOL_SELECTION,
        {
            "segment": "EU",
            "expected_tool": "payments_v2",
            "forbidden_tool": "payments_v1",
        },
    )
    trace = _trace(
        [ToolCallStep(tool="payments_v1", args={})],
        segment="US",
    )
    assert dispatch(inv, trace).outcome == CheckOutcome.NOT_APPLICABLE
